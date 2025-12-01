import os
import json
from typing import Dict, Tuple
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# =========================
# 配置
# =========================
NEURAL_DATA_DIR = "./neural_data"
MODEL_DIR = "./router_neural_models"

DATASET_FILES = {
    "cnli_short": "cnli_short_neural.jsonl",
    "coqa_short": "coqa_short_neural.jsonl",
    "qasper_short": "qasper_short_neural.jsonl",
    "narrative_qa_short": "narrative_qa_short_neural.jsonl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 模型定义（要和训练时一致）
# =========================
class RouterMLP(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 输出 logit
            nn.ReLU(),
            nn.Linear(hidden_dim,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

def load_router(model_path: str) -> Tuple[RouterMLP, torch.Tensor, torch.Tensor]:
    """加载保存好的 router 模型和标准化参数 mean/std"""
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = RouterMLP(input_dim=3, hidden_dim=64).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["input_mean"].to(DEVICE)
    std = checkpoint["input_std"].to(DEVICE)

    return model, mean, std


# =========================
# 数据加载
# =========================
def load_neural_jsonl_full(path: str):
    """
    读取 jsonl:
    {
      "slm_score": ...,
      "llm_score": ...,
      "avg_logp": ...,
      "avg_entropy_nats": ...,
      "avg_entropy_bits": ...,
      "label": 0/1
    }
    """
    avg_logp_list = []
    avg_ent_nats_list = []
    avg_ent_bits_list = []
    slm_scores = []
    llm_scores = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            avg_logp_list.append(obj["avg_logp"])
            avg_ent_nats_list.append(obj["avg_entropy_nats"])
            avg_ent_bits_list.append(obj["avg_entropy_bits"])
            slm_scores.append(obj["slm_score"])
            llm_scores.append(obj["llm_score"])
            labels.append(obj["label"])

    # 特征矩阵 [N, 3]
    X = torch.tensor(
        list(zip(avg_logp_list, avg_ent_nats_list, avg_ent_bits_list)),
        dtype=torch.float32,
    )
    slm = torch.tensor(slm_scores, dtype=torch.float32)
    llm = torch.tensor(llm_scores, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    return X, slm, llm, labels_t


# =========================
# 评估函数
# =========================
def evaluate_router_on_dataset(
    dataset_name: str,
    data_path: str,
    model_path: str,
    threshold: float = 0.9,
    batch_size: int = 64,
) -> Dict[str, float]:
    print(f"\n=== Evaluating router on dataset: {dataset_name} ===")
    print(f"Data:  {data_path}")
    print(f"Model: {model_path}")

    # 1. 加载模型 + 标准化参数
    model, mean, std = load_router(model_path)

    # 2. 加载数据
    X, slm, llm, labels = load_neural_jsonl_full(data_path)
    N = X.shape[0]
    print(f"Number of examples: {N}")

    # 3. 标准化特征（和训练时一样）
    X = X.to(DEVICE)
    X_norm = (X - mean) / std

    # 4. 用 DataLoader 批量跑 router 预测
    ds = TensorDataset(X_norm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for (xb,) in tqdm(loader, desc=f"[{dataset_name}] Routing", leave=False):
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits)  # [B]
            preds = (probs >= threshold).float()  # [B]

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    probs = torch.cat(all_probs, dim=0)      # [N]
    preds = torch.cat(all_preds, dim=0)      # [N]
    labels = labels  # already on cpu
    slm = slm
    llm = llm

    # 5. 计算最终系统表现
    # 如果 router 预测 1 → 用 llm_score；否则用 slm_score
    chosen_scores = torch.where(preds == 1.0, llm, slm)
    final_acc = chosen_scores.mean().item()  # 如果 score 是 0/1，这就是准确率

    # router 本身是否选对：与 label (llm好=1/slm好=0) 对比
    router_decision_acc = (preds == labels).float().mean().item()

    # baseline：永远用小模型 / 永远用大模型
    slm_only_acc = slm.mean().item()
    llm_only_acc = llm.mean().item()

    # 6. 计算平均 cost
    # slm -> cost=1, llm -> cost=20
    cost_slm = torch.ones_like(preds)
    cost_llm = torch.ones_like(preds) * 20.0
    costs = torch.where(preds == 1.0, cost_llm, cost_slm)
    avg_cost = costs.mean().item()

    print(
        f"[{dataset_name}] final_acc={final_acc:.4f} | "
        f"router_decision_acc={router_decision_acc:.4f} | "
        f"slm_only_acc={slm_only_acc:.4f} | "
        f"llm_only_acc={llm_only_acc:.4f} | "
        f"avg_cost={avg_cost:.2f}"
    )

    return {
        "final_acc": final_acc,
        "router_decision_acc": router_decision_acc,
        "slm_only_acc": slm_only_acc,
        "llm_only_acc": llm_only_acc,
        "avg_cost": avg_cost,
    }


def main():
    results = {}

    for dataset_name, filename in DATASET_FILES.items():
        data_path = os.path.join(NEURAL_DATA_DIR, filename)
        model_path = os.path.join(MODEL_DIR, f"{dataset_name}_router.pt")

        if not os.path.exists(data_path):
            print(f"WARNING: data file {data_path} does not exist, skip.")
            continue
        if not os.path.exists(model_path):
            print(f"WARNING: model file {model_path} does not exist, skip.")
            continue

        metrics = evaluate_router_on_dataset(dataset_name, data_path, model_path)
        results[dataset_name] = metrics

    print("\n=== Summary ===")
    for name, m in results.items():
        print(
            f"{name}: "
            f"final_acc={m['final_acc']:.4f}, "
            f"router_decision_acc={m['router_decision_acc']:.4f}, "
            f"slm_only_acc={m['slm_only_acc']:.4f}, "
            f"llm_only_acc={m['llm_only_acc']:.4f}, "
            f"avg_cost={m['avg_cost']:.2f}"
        )


if __name__ == "__main__":
    main()
