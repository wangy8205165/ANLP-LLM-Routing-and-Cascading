import os
import json
import argparse
import pandas as pd
from metric_compute import f1_score, normalize_cnli_label

F1_DATASETS = {"coqa_short", "qasper_short", "narrative_qa_short"}
ACC_DATASETS = {"cnli_short"}

def is_f1_dataset(name: str) -> bool:
    return name.lower() in F1_DATASETS


def is_acc_dataset(name: str) -> bool:
    return name.lower() in ACC_DATASETS

def compute_perf(dataset_name: str, pred: str, gold: str) -> float:
    """
    compute the performance score for a single sample in a dataset
    CNLI: Accuracy
    Others: F1 score
    """
    name = dataset_name.lower()
    if name in F1_DATASETS:
        return f1_score(pred,gold)
    
    if name == "cnli_short":
        return 1.0 if normalize_cnli_label(pred) == normalize_cnli_label(gold) else 0.0
    else:
        raise ValueError("Wrong Dataset!")
    
### ============== Main Logics ================== ###
"""
For each datset:
1. Read dataset/{name}.jsonl
2. read outputs/baseline_{name}.json
3. read verification/self_ver_{name}.json
4. generate a table including: id, dataset, gold_output,slm_pre,llm_pre,slm_confidence,per_slm,per_llm
5. save the results to router_data/router_data_{name}.jsonl
"""

def prepare_router_data(dataset_name: str):
    # Get the path for dataset, baseline outputs, and verification results. 
    name = dataset_name.lower()
    dataset_path = os.path.join("dataset", f"{name}.jsonl")
    baseline_path = os.path.join("outputs", f"baseline_output_{name}.json")
    verif_path = os.path.join("verification", f"self_ver_{name}.json")

    # Read the original dataset
    print(f"[INFO] Loading dataset from {dataset_path}")
    df = pd.read_json(dataset_path, lines=True, orient="records")
    
    # Read the baseline outputs
    print(f"[INFO] Loading baseline preds from {baseline_path}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    llama3_preds = baseline["llama3_pred"]
    gpt_preds = baseline["gpt_pred"]

    # Read the verification results
    print(f"[INFO] Loading self-verification from {verif_path}")
    with open(verif_path, "r", encoding="utf-8") as f:
        verif = json.load(f)
    slm_conf = verif["slm_ver_confidence"]

    # Check if the length is consistent
    n_data = len(df)
    assert len(llama3_preds) == n_data, f"llama3_pred length mismatch: {len(llama3_preds)} vs {n_data}"
    assert len(gpt_preds) == n_data, f"gpt_pred length mismatch: {len(gpt_preds)} vs {n_data}"
    assert len(slm_conf) == n_data, f"self-ver length mismatch: {len(slm_conf)} vs {n_data}"
    
    gold_outputs = df["output"].tolist()
    ids = df.get("id", pd.Series(range(n_data))).tolist()

    router_records = []

    print(f"[INFO] Computing per-sample performance for dataset={name} ...")
    for i in range(n_data):
        gold = gold_outputs[i]
        slm_pred = llama3_preds[i]
        llm_pred = gpt_preds[i]
        conf = slm_conf[i]

        perf_slm = compute_perf(name, slm_pred, gold)
        perf_llm = compute_perf(name, llm_pred, gold)

        router_records.append(
            {
                "id": ids[i],
                "dataset": name,
                "gold_output": gold,
                "slm_pred": slm_pred,
                "llm_pred": llm_pred,
                "slm_confidence": conf,   # self-ver confidence -> {0,0.25,...,1}
                "perf_slm": perf_slm,     # 0~1: F1 or Accuracy
                "perf_llm": perf_llm,     # 0~1
            }
        )
    
    out_dir = "router_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"router_data_{name}.jsonl")

    print(f"[INFO] Saving router data to {out_path}")
    out_df = pd.DataFrame(router_records)
    out_df.to_json(out_path, lines=True, orient="records", force_ascii=False)
    print("[DONE] Router data prepared.")

#=================== CLI ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name: cnli_short / coqa_short / narrativeqa_short / qasper_short",
    )
    args = parser.parse_args()

    prepare_router_data(args.dataset)