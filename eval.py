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

def compute_IBC(pm,pslm,pllm,cm,cslm = 5,cllm = 25):
    ibc_m = (pm-pslm)/(cm-cslm)
    ibc_base = (pllm-pslm)/(cllm-cslm)
    # print(f"ibc_m is {ibc_m}")
    # print(f"ibc_base is {ibc_base}")
    delta_ibc = ((ibc_m-ibc_base)/(ibc_base))*100
    return delta_ibc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,required=True,help="Enter the dataset")
    parser.add_argument("--cost",type=float,required=True,help="Enter the cost")

    args = parser.parse_args()
    dataset = args.dataset
    golds, finals = read_data(dataset)
    CSLM = 5
    CLLM = 25


    if args.dataset == "cnli_short":
        acc = compute_perf(dataset,golds,finals)
        print(f"Accuracy:{acc}")
        delta_ibc = compute_IBC(acc,0.579,0.763,18.32,CSLM,CLLM)
        print(f"delta IBC is {delta_ibc}")
    else:
        mean, var = compute_perf(dataset,golds,finals)
        print(f"F1 score mean: {mean}, var: {var}")
        
        if args.dataset == "coqa_short":
            delta_ibc = compute_IBC(mean,0.4624,0.5939,args.cost)
        elif args.dataset == "narrative_qa_short":
            delta_ibc = compute_IBC(mean,0.3019,0.3853,args.cost)
        elif args.dataset == "qasper_short":
            delta_ibc = compute_IBC(mean, 0.2039,0.3352,args.cost)
        print(f"delta IBC is {delta_ibc}")


if __name__ == "__main__":
    main()