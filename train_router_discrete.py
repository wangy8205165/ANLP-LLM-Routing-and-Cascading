import os
import json
import argparse
import random
import numpy as np
import pandas as pd

V_BINS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Load the router data for the dataset
def load_router_data(dataset_name: str) -> pd.DataFrame:
    path = os.path.join("router_data", f"router_data_{dataset_name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Router data not found: {path}")
    df = pd.read_json(path, lines=True, orient="records")
    needed = {"slm_confidence", "perf_slm", "perf_llm"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in router_data: {missing}")
    return df

# Split the router data into train and test
def split_train_test(df: pd.DataFrame, train_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    if train_size > len(df):
        raise ValueError(f"train_size={train_size} > dataset size={len(df)}")
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

def safe_mean(x):
    x = [v for v in x if pd.notnull(v)]
    return float(np.mean(x)) if len(x) > 0 else np.nan


def estimate_bin_stats(train_df: pd.DataFrame):
    """
    estimate the average performance with each bin：
      P_slm(v) = mean(perf_slm | conf=v)
      P_llm(v) = mean(perf_llm | conf=v)
    """
    stats = {}
    global_pslm = safe_mean(train_df["perf_slm"].tolist())
    global_pllm = safe_mean(train_df["perf_llm"].tolist())
    for v in V_BINS:
        sub = train_df[train_df["slm_confidence"] == v] # get all the samples that has the confidence values
        p_slm = safe_mean(sub["perf_slm"].tolist())
        p_llm = safe_mean(sub["perf_llm"].tolist())
        if np.isnan(p_slm):
            p_slm = global_pslm
        if np.isnan(p_llm):
            p_llm = global_pllm
        stats[v] = {"P_slm": p_slm, "P_llm": p_llm, "n": int(len(sub))}
    return stats

def build_discrete_policy(bin_stats, lam: float, c_slm: float, c_llm: float):
    """
    for each singel V：
      keep  : R_keep(v)  = P_slm(v) - lambda * C_slm
      route : R_route(v) = P_llm(v) - lambda * (C_slm + C_llm)
    """
    policy = {}
    rewards = {}
    for v in V_BINS:
        p_slm = bin_stats[v]["P_slm"]
        p_llm = bin_stats[v]["P_llm"]
        r_keep = p_slm - lam * c_slm
        r_route = p_llm - lam * (c_slm + c_llm)
        action = "keep" if r_keep >= r_route else "route"
        policy[v] = action
        rewards[v] = {"R_keep": r_keep, "R_route": r_route}
    return policy, rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., cnli / coqa / narrativeqa / qasper / quality)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.01,
                        help="Trade-off coefficient λ for reward = perf - λ * cost")
    parser.add_argument("--c_slm", type=float, default=1.0, help="Cost of SLM")
    parser.add_argument("--c_llm", type=float, default=60.0, help="Cost of LLM (incremental over SLM)")
    parser.add_argument("--train_size", type=int, default=50, help="Number of samples to train router")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    lam = args.lam
    c_slm = args.c_slm
    c_llm = args.c_llm

    print(f"[INFO] Loading router data for dataset={dataset}") # load the data
    df = load_router_data(dataset)

    print(f"[INFO] Splitting train_size={args.train_size}, seed={args.seed}") # split the data
    train_df, test_df = split_train_test(df, train_size=args.train_size, seed=args.seed)

    print("[INFO] Estimating per-bin stats on training set ...") # estimate performance within each bin
    bin_stats = estimate_bin_stats(train_df)
    print("="*50)
    print(bin_stats)

    print("[INFO] Building discrete per-bin policy ...")
    disc_policy, disc_rewards = build_discrete_policy(bin_stats, lam, c_slm, c_llm)

    print("="*50)
    print(disc_policy)
    print("="*50)
    print(disc_rewards)

    out_dir = "policy"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"policy_{dataset}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(disc_policy, f, ensure_ascii=False, indent=4)
    
    print(f"File saved!")

if __name__ == "__main__":
    main()
