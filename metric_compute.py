'''
This project aims to compute the metrics for different datasets
'''

#================================
# Import all required libraries
#================================
import json
import pandas as pd
import re
import string
from typing import List, Dict

#================================
# Define all util functions
#================================
def load_predictions(pred_path: str) -> Dict[str, List[str]]:
    """
    load baseline_output_{dataset}.json
    expected format:
    {
        "llama3_pred": [...],
        "gpt_pred": [...]
    }
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_gold_outputs(gold_path: str) -> List[str]:
    """
    load dataset/{dataset}.jsonl
    Everyline has output element as the standard answer
    """
    df = pd.read_json(gold_path, lines=True, orient="records")
    return df["output"].tolist()

#================================
# Compute Accuracy for cnli/quality
#================================

def normalize_cnli_label(text: str) -> str:
    """
    Possible answers for CNLI:
    Entailment / Contradiction / Not mentioned / Neutral
    """
    text = text.strip().lower() # lowercase the text
    if "entail" in text:
        return "entailment"
    if "contradict" in text:
        return "contradiction"
    # Sometimes model outputs neutral, sometimes outputs not mentioned
    if "not mentioned" in text:
        return "not mentioned"
    if "neutral" in text:
        return "not mentioned"
    return text

def normalize_quality_label(text: str) -> str:
    """
    QUALITY is single choice question A/B/C/D
    we catch first A/B/C/D in the text
    """
    text = text.strip()
    m = re.search(r"[ABCD]", text, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    return text.upper()

def compute_accuracy(golds: List[str], preds: List[str], normalize_fn=None) -> float:
    assert len(golds) == len(preds), "gold and pred length mismatch"
    correct = 0
    for g, p in zip(golds, preds):
        if normalize_fn is not None:
            g = normalize_fn(g)
            p = normalize_fn(p)
        if g == p:
            correct += 1
    return correct / len(golds) if golds else 0.0

def eval_cnli(pred_path: str, gold_path: str):
    golds = load_gold_outputs(gold_path)
    preds_dict = load_predictions(pred_path)

    llama_acc = compute_accuracy(golds, preds_dict["llama3_pred"], normalize_cnli_label)
    gpt_acc = compute_accuracy(golds, preds_dict["gpt_pred"], normalize_cnli_label)

    print("=== CNLI Accuracy ===")
    print(f"Llama3  Accuracy: {llama_acc:.4f}")
    print(f"GPT     Accuracy: {gpt_acc:.4f}")

def eval_quality(pred_path: str, gold_path: str):
    golds = load_gold_outputs(gold_path)
    preds_dict = load_predictions(pred_path)

    llama_acc = compute_accuracy(golds, preds_dict["llama3_pred"], normalize_quality_label)
    gpt_acc = compute_accuracy(golds, preds_dict["gpt_pred"], normalize_quality_label)

    print("=== QuALITY Accuracy ===")
    print(f"Llama3  Accuracy: {llama_acc:.4f}")
    print(f"GPT     Accuracy: {gpt_acc:.4f}")

#================================
# Compute F1 for coqa, qasper, narrativeqa
#================================

def _normalize_text_for_f1(s: str) -> List[str]:
    """
    normalize the predicted text
    - lowercase
    - remove punctuations
    - remove space
    - split into tokens
    """
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation)) # remove all the punctuations
    s = " ".join(s.split())
    return s.split()

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text_for_f1(pred)
    gold_tokens = _normalize_text_for_f1(gold)

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    # compute the number of tokens occured and overlapped
    common_tokens = {}
    for t in set(pred_tokens):
        common_tokens[t] = min(pred_tokens.count(t), gold_tokens.count(t))

    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) # out of all predicts tokens, how many are actually positive. 
    recall = num_same / len(gold_tokens) # out of all golden tokens, how many are actually predicted.
    return 2 * precision * recall / (precision + recall)

def compute_average_f1(golds: List[str], preds: List[str]) -> float:
    assert len(golds) == len(preds), "gold and pred length mismatch"
    scores = [f1_score(p, g) for g, p in zip(golds, preds)]
    return sum(scores) / len(scores) if scores else 0.0

def eval_f1_dataset(name: str, pred_path: str, gold_path: str):
    """
    General F1 calculator for coqa, aqsper, narrativeqa
    """
    golds = load_gold_outputs(gold_path)
    preds_dict = load_predictions(pred_path)

    llama_f1 = compute_average_f1(golds, preds_dict["llama3_pred"])
    gpt_f1 = compute_average_f1(golds, preds_dict["gpt_pred"])

    print(f"=== {name} F1 ===")
    print(f"Llama3  F1: {llama_f1:.4f}")
    print(f"GPT     F1: {gpt_f1:.4f}")

if __name__ == "__main__":
    # 1) CNLI Accuracy
    eval_cnli(
        pred_path="outputs/baseline_output_cnli_short.json",
        gold_path="dataset/cnli_short.jsonl",
    )

    # 2) QuALITY Accuracy
    # eval_quality(
    #     pred_path="baseline_output_quality.json",
    #     gold_path="dataset/quality.jsonl",
    # )

    #3) CoQA F1
    # eval_f1_dataset(
    #     name="CoQA",
    #     pred_path="outputs/baseline_output_coqa_short.json",
    #     gold_path="dataset/coqa_short.jsonl",
    # )

    # 4) QASPER F1
    # eval_f1_dataset(
    #     name="QASPER",
    #     pred_path="baseline_output_qasper.json",
    #     gold_path="dataset/qasper.jsonl",
    # )

    # 5) NarrativeQA F1
    # eval_f1_dataset(
    #     name="NarrativeQA",
    #     pred_path="baseline_output_narrativeqa.json",
    #     gold_path="dataset/narrativeqa.jsonl",
    # )