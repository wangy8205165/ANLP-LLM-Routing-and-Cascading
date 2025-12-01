import os
import json
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



# ===============================
# 配置
# ===============================
NEURAL_DATA_DIR = "./neural_data"
OUTPUT_MODEL_DIR = "./router_neural_models"

DATASET_FILES = {
    "cnli_short": "cnli_short_neural.jsonl",
    "coqa_short": "coqa_short_neural.jsonl",
    "qasper_short": "qasper_short_neural.jsonl",
    "narrative_qa_short": "narrative_qa_short_neural.jsonl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU is available: {torch.cuda.is_available()}")

# ===============================
# 模型定义
# ===============================
class RouterMLP(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
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


# ===============================
# 数据加载函数
# ===============================
def load_neural_jsonl(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 jsonl 文件中读取:
      - 特征: [avg_logp, avg_entropy_nats, avg_entropy_bits]
      - 标签: label (0/1)
    返回:
      X: [N, 3] float32
      y: [N] float32 (0 或 1)
    """
    xs = []
    ys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            x = [
                obj["avg_logp"],
                obj["avg_entropy_nats"],
                obj["avg_entropy_bits"],
            ]
            y = obj["label"]
            xs.append(x)
            ys.append(y)

    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return X, y


def standardize_features(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对特征做标准化: (X - mean) / std
    返回:
      X_norm, mean, std
    """
    mean = X.mean(dim=0, keepdim=False)
    std = X.std(dim=0, keepdim=False)
    # 防止除零
    std_clamped = torch.clamp(std, min=1e-6)
    X_norm = (X - mean) / std_clamped
    return X_norm, mean, std_clamped


# ===============================
# 训练函数
# ===============================
def train_single_router(
    dataset_name: str,
    data_path: str,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    print(f"\n=== Training router for dataset: {dataset_name} ===")
    print(f"Loading data from {data_path}")

    # 1. 读数据
    X, y = load_neural_jsonl(data_path)
    X, mean, std = standardize_features(X)

    # 2. 划分 train / val（简单 80/20）
    N = X.shape[0]
    indices = torch.randperm(N)
    train_size = int(0.8 * N)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. 模型 & 优化器
    model = RouterMLP(input_dim=3, hidden_dim=64).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. 训练循环
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0.0
        total_train = 0.0
        # for xb, yb in train_loader:
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)            # [B]
            loss = criterion(logits, yb)  # yb 需要 float [B]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

            # 计算当前 batch 的 train acc
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct_train += (preds == yb).sum().item()
            total_train += yb.numel()


        avg_train_loss = total_loss / len(train_ds)
        train_acc = correct_train / total_train if total_train > 0 else 0.0


        # 验证
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_val_loss += loss.item() * xb.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()

        avg_val_loss = total_val_loss / len(val_ds)
        val_acc = correct / total if total > 0 else 0.0

        # if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    # 5. 保存模型和标准化参数
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_MODEL_DIR, f"{dataset_name}_router.pt")

    save_obj: Dict[str, torch.Tensor] = {
        "model_state_dict": model.state_dict(),
        "input_mean": mean,
        "input_std": std,
    }
    torch.save(save_obj, save_path)
    print(f"Saved router model to: {save_path}")


def main():
    for dataset_name, filename in DATASET_FILES.items():
        data_path = os.path.join(NEURAL_DATA_DIR, filename)
        if not os.path.exists(data_path):
            print(f"WARNING: {data_path} does not exist, skip.")
            continue
        train_single_router(dataset_name, data_path)


if __name__ == "__main__":
    main()
