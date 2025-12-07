import os
import json
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# =========================
# 配置
# =========================

NEURAL_DATA_DIR = "./neural_data"

DATASET_FILES = {
    "cnli_short": "cnli_short_neural.jsonl",
    "coqa_short": "coqa_short_neural.jsonl",
    "qasper_short": "qasper_short_neural.jsonl",
    "narrative_qa_short": "narrative_qa_short_neural.jsonl",
}

# cost: keep=1, route=20; lambda 控制 accuracy 与 cost 的 trade-off
LAMBDA = 0.01  # 你可以调这个值看看对 tau / policy 的影响
LLM_COST = 100

# =========================
# 工具函数：读数据
# =========================

def load_neural_dataset(path: str):
    """
    从 jsonl 读取一整个数据集。

    每一行的格式类似：
    {
        "slm_score": 1.0,
        "llm_score": 1.0,
        "avg_logp": -0.38,
        "avg_entropy_nats": 0.85,
        "avg_entropy_bits": 1.23,
        "label": 0
    }

    返回:
        X: (N, 3) 观测特征
        y_state: (N,) 状态标签 s (0/1) —— 这里就是 label
        slm_score: (N,) 小模型正确与否
        llm_score: (N,) 大模型正确与否
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

    X = np.stack(
        [avg_logp_list, avg_ent_nats_list, avg_ent_bits_list],
        axis=1,
    )  # (N,3)
    y_state = np.array(labels, dtype=np.int64)      # s=label
    slm_score = np.array(slm_scores, dtype=np.float32)
    llm_score = np.array(llm_scores, dtype=np.float32)

    return X, y_state, slm_score, llm_score


# =========================
# Step 2: 拟合观测模型 P(s=1|o)
# =========================

def fit_logistic_posterior(X: np.ndarray, y_state: np.ndarray) -> Pipeline:
    """
    拟合判别式观测模型 P(s=1 | o)：
        使用 StandardScaler + LogisticRegression

    返回一个 sklearn Pipeline，后续可以直接:
        probs = model.predict_proba(X)[:, 1]
    """
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",  # 数据可能不平衡，这样更稳一点
                max_iter=1000,
            )),
        ]
    )

    clf.fit(X, y_state)
    return clf


# =========================
# Step 3: 估计 reward 表 R(s,a)
# =========================

def estimate_reward_table(
    y_state: np.ndarray,
    slm_score: np.ndarray,
    llm_score: np.ndarray,
    lam: float = LAMBDA,
) -> Dict[Tuple[int, str], float]:
    """
    根据数据估计每个 (s,a) 的 reward:
      R(s, keep)  = E[slm_score | s] - lam * 1
      R(s, route) = E[llm_score | s] - lam * 20

    返回:
        R[(s, "keep")]  = ...
        R[(s, "route")] = ...
    """
    R = {}

    for s in [0, 1]:
        mask = (y_state == s)
        if mask.sum() == 0:
            # 极端情况: 没有某个 state 的样本，简单兜底
            alpha_slm = slm_score.mean()
            alpha_llm = llm_score.mean()
        else:
            alpha_slm = slm_score[mask].mean()
            alpha_llm = llm_score[mask].mean()

        R[(s, "keep")] = alpha_slm - lam * 1.0
        R[(s, "route")] = alpha_llm - lam * LLM_COST

    return R


# =========================
# Step 4: 根据 R 计算阈值 tau
# =========================

def compute_threshold_tau(R: Dict[Tuple[int, str], float]) -> float:
    """
    根据 reward 表 R(s,a) 计算阈值 tau，使得:
        若 p = P(s=1 | o) > tau 则选择 route，否则 keep。

    推导不等式:
        Q(route|o) > Q(keep|o)
    得到:
        p > tau

    其中:
        tau = (R(0,keep) - R(0,route)) /
              [(R(1,route) - R(1,keep)) + (R(0,keep) - R(0,route))]

    如果分母为负，则要翻转不等式:
        p < tau

    这里返回 tau，同时返回一个标记 sign，后面会根据 sign 决定 > 还是 <。
    """
    R0_keep = R[(0, "keep")]
    R0_route = R[(0, "route")]
    R1_keep = R[(1, "keep")]
    R1_route = R[(1, "route")]

    numerator = R0_keep - R0_route
    denominator = (R1_route - R1_keep) + (R0_keep - R0_route)

    if np.isclose(denominator, 0.0):
        # 非常极端：两种动作几乎一样好，直接设 tau=1.0（几乎永远 keep）
        tau = 1.0
        sign = ">"
    else:
        tau = numerator / denominator
        # 决定比较方向
        sign = ">" if denominator > 0 else "<"

    return tau, sign


# =========================
# Step 5: 在数据上应用 policy 并评估
# =========================

def evaluate_pomdp_policy(
    dataset_name: str,
    filename: str,
):
    path = os.path.join(NEURAL_DATA_DIR, filename)
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Loading data from {path}")

    # 1. 读数据
    X, y_state, slm_score, llm_score = load_neural_dataset(path)
    N = X.shape[0]
    print(f"Number of examples: {N}")

    # 2. 拟合 P(s=1|o) 的 logistic 模型
    clf = fit_logistic_posterior(X, y_state)

    # 3. 估计奖励表 R(s,a)
    R = estimate_reward_table(y_state, slm_score, llm_score, lam=LAMBDA)

    print("Estimated rewards R(s,a):")
    for s in [0, 1]:
        print(
            f"  s={s}: "
            f"R(keep)={R[(s,'keep')]:.4f}, "
            f"R(route)={R[(s,'route')]:.4f}"
        )

    # 4. 根据 R 计算阈值 tau
    tau, sign = compute_threshold_tau(R)
    print(f"Computed threshold tau = {tau:.4f}, compare with '{sign}'")

    # 5. 在整套数据上得到 P(s=1|o) 和 policy 决策
    probs = clf.predict_proba(X)[:, 1]  # p = P(s=1 | o)
    # policy: 根据 p 和 tau 决定 action
    if sign == ">":
        action_route = (probs > tau).astype(np.int64)  # 1=route, 0=keep
    else:
        action_route = (probs < tau).astype(np.int64)

    # 6. 评估 “是否需要 LLM” 这个决策层面的指标
    #    ground truth = y_state (1=LLM 比较好, 0=SLM 不差)
    decision_acc = accuracy_score(y_state, action_route)
    decision_f1 = f1_score(y_state, action_route, zero_division=0)

    # 7. 根据 policy 选择最终得分 & cost
    #    若 route -> 采用 llm_score, cost=20
    #    若 keep  -> 采用 slm_score, cost=1
    chosen_score = np.where(action_route == 1, llm_score, slm_score)
    final_task_acc = chosen_score.mean()

    cost = np.where(action_route == 1, LLM_COST, 1.0)
    avg_cost = cost.mean()

    # 8. 同时算两个 baseline 方便对比
    baseline_slm_acc = slm_score.mean()
    baseline_slm_cost = 1.0  # 永远用 SLM，cost 恒为 1

    baseline_llm_acc = llm_score.mean()
    baseline_llm_cost = LLM_COST  # 永远用 LLM，cost 恒为 20

    print("Policy metrics on full dataset:")
    print(f"  Decision accuracy (route vs keep vs label): {decision_acc:.4f}")
    print(f"  Decision F1 (positive=route where LLM better): {decision_f1:.4f}")
    print(f"  Final task accuracy (chosen slm/llm score):   {final_task_acc:.4f}")
    print(f"  Average cost (1 for SLM, {LLM_COST} for LLM):         {avg_cost:.2f}")

    print("Baselines:")
    print(
        f"  Always SLM: acc={baseline_slm_acc:.4f}, avg_cost={baseline_slm_cost:.2f}"
    )
    print(
        f"  Always LLM: acc={baseline_llm_acc:.4f}, avg_cost={baseline_llm_cost:.2f}"
    )

    # 9. 打印出 policy 的解析形式（便于你写到 paper / 报告里）
    #    logistic 回归: p = sigma(w^T o + b)
    #    policy:       if p {sign} tau -> route else keep
    logreg = clf.named_steps["logreg"]
    scaler = clf.named_steps["scaler"]

    # 真正作用在原始特征上的权重是 w' = w / std, b' = b - w·mean/std
    w = logreg.coef_[0]  # shape (3,)
    b = logreg.intercept_[0]
    mean = scaler.mean_
    scale = scaler.scale_

    w_prime = w / scale
    b_prime = b - np.sum(w * mean / scale)

    print("\nAnalytical policy:")
    print("  Observation features o = (avg_logp, avg_entropy_nats, avg_entropy_bits)")
    print(
        "  Posterior p = sigma(w'^T o + b'), with:"
        f"\n    w' = {w_prime}"
        f"\n    b' = {b_prime:.4f}"
    )
    print(
        f"  Policy: route if p {sign} {tau:.4f}, otherwise keep."
    )

    # 返回指标，方便主函数汇总（可选）
    return {
        "decision_acc": decision_acc,
        "decision_f1": decision_f1,
        "final_task_acc": final_task_acc,
        "avg_cost": avg_cost,
        "baseline_slm_acc": baseline_slm_acc,
        "baseline_slm_cost": baseline_slm_cost,
        "baseline_llm_acc": baseline_llm_acc,
        "baseline_llm_cost": baseline_llm_cost,
        "tau": tau,
        "sign": sign,
    }


def main():
    summary = {}
    for dataset_name, filename in DATASET_FILES.items():
        path = os.path.join(NEURAL_DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skip.")
            continue
        metrics = evaluate_pomdp_policy(dataset_name, filename)
        summary[dataset_name] = metrics

    print("\n=== Summary over datasets ===")
    for name, m in summary.items():
        print(
            f"{name}: "
            f"decision_acc={m['decision_acc']:.4f}, "
            f"decision_f1={m['decision_f1']:.4f}, "
            f"final_task_acc={m['final_task_acc']:.4f}, "
            f"avg_cost={m['avg_cost']:.2f}"
        )


if __name__ == "__main__":
    main()
