import os
import json
import argparse
import pandas as pd

def apply_router(dataset: str):
    name = dataset.lower()

    # path
    policy_path = os.path.join("policy", f"policy_{name}.jsonl")
    router_data_path = os.path.join("router_data", f"router_data_{name}.jsonl")
    out_dir = "pomdp"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{name}.json")

    # 1. read policy
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
    policy = {float(k): v for k, v in policy.items()}
    print(policy)

    # 2. read router_data（slm_pred, llm_pred, slm_confidence ）
    df = pd.read_json(router_data_path, lines=True, orient="records")

    results = []
    for i, row in df.iterrows():
        v = float(row["slm_confidence"])
        v_key = min(policy.keys(), key=lambda x: abs(x - v))
        action = policy[v_key]
        final_ans = row["slm_pred"] if action == "keep" else row["llm_pred"]

        results.append({
            "id": row.get("id", i),
            "dataset": row.get("dataset", name),
            "slm_confidence": v,
            "action": action,
            "gold_answer":row.get("gold_output"),
            "final_answer": final_ans
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Applied policy for {name}, results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,help="Dataset name (e.g., cnli / coqa / qasper / narrativeqa / quality)")
    args = parser.parse_args()
    apply_router(args.dataset)
