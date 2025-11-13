# This is evaluation script that evaluates the performance of pomdp results.
import json
import argparse
import numpy as np
from metric_compute import normalize_cnli_label, compute_accuracy, f1_score

def read_data(dataset):
    results_path = f"pomdp/results_{dataset}.json"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gold_answers= [item["gold_answer"] for item in data]
    final_answers=[item["final_answer"]for item in data]

    return gold_answers, final_answers

def compute_perf(dataset, golds, finals):
    if dataset == "cnli_short":
        # Compute accuracy
        acc = compute_accuracy(golds, finals, normalize_fn=normalize_cnli_label)
        return acc
    else:
        # Compute F1 score
        score_list=[]
        for gold, final in zip(golds, finals):
            score = f1_score(final,gold)
            score_list.append(score)
        return np.mean(score_list), np.var(score_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,required=True,help="Enter the dataset")
    args = parser.parse_args()
    dataset = args.dataset
    golds, finals = read_data(dataset)

    if args.dataset == "cnli_short":
        acc = compute_perf(dataset,golds,finals)
        print(f"Accuracy:{acc}")
    else:
        mean, var = compute_perf(dataset,golds,finals)
        print(f"F1 score mean: {mean}, var: {var}")

if __name__ == "__main__":
    main()