import json
import os
from typing import List, Optional, Callable
import pandas as pd
from metric_compute import compute_accuracy,normalize_cnli_label,compute_average_f1 # 修改成你实际的 import 路径


# ========= 配置部分 =========
LOGITS_PATH = "./logits_entropy/qasper_short_answerlogits.jsonl"
LARGE_OUTPUT_PATH = "./outputs/baseline_output_qasper_short.json"
GOLD_PATH = "./dataset/qasper_short.jsonl"
OUT_NEURAL_PATH = "./neural_data/coqa_qasper_neural.jsonl"

# ========= 工具函数 =========
def load_small_model_logits(path: str):
    """读取小模型 logits_entropy jsonl 文件。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records  # list[dict]


def load_llm_preds(path: str) -> List[str]:
    """读取大模型输出文件，返回 gpt_pred 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 假设结构为 {"gpt_pred": [ans1, ans2, ...]}
    preds = data["gpt_pred"]
    return preds



def load_gold_outputs(gold_path: str) -> List[str]:
    """
    load dataset/{dataset}.jsonl
    Everyline has output element as the standard answer
    """
    df = pd.read_json(gold_path, lines=True, orient="records")
    return df["output"].tolist()


def per_example_scores(
    golds: List[str],
    preds: List[str],
    normalize_fn = normalize_cnli_label,
) -> List[float]:
    """
    使用 compute_accuracy 对每个样本单独打分。
    compute_accuracy 接受列表，这里每次传 [gold_i], [pred_i]，得到 0/1。
    """
    assert len(golds) == len(preds)
    scores = []
    for g, p in zip(golds, preds):
        # score = compute_accuracy([g], [p], normalize_fn=normalize_fn)
        score = compute_average_f1([g], [p])
        scores.append(score)
    return scores



def main(logits_path: str = LOGITS_PATH,
    large_output_path: str = LARGE_OUTPUT_PATH,
    gold_path: str = GOLD_PATH,
    out_path: str = OUT_NEURAL_PATH,
    normalize_fn: Optional[Callable[[str], str]] = None,
):
    small_records = load_small_model_logits(logits_path)
    llm_preds = load_llm_preds(large_output_path)
    golds = load_gold_outputs(gold_path)

    n = len(small_records)
    assert n == len(llm_preds), f"长度不匹配: small={n}, llm={len(llm_preds)}"
    assert n == len(golds), f"长度不匹配: data={n}, golds={len(golds)}"

    # 2. 拿到小模型/大模型的文本答案
    slm_preds = [rec["generated_text"] for rec in small_records]

    # 3. 用 compute_accuracy 计算每个样本的小模型/大模型分数
    slm_scores = per_example_scores(golds, slm_preds, normalize_fn=normalize_cnli_label)
    llm_scores = per_example_scores(golds, llm_preds, normalize_fn=normalize_cnli_label)


    # 4. 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 5. 写 jsonl
    with open(out_path, "w", encoding="utf-8") as fout:
        for i in range(n):
            rec = small_records[i]

            slm_score = slm_scores[i]
            llm_score = llm_scores[i]

            # label: 大模型分数 > 小模型分数 → 1，否则 0
            label = 1 if llm_score > slm_score else 0

            out_obj = {
                "slm_score": slm_score,
                "llm_score": llm_score,
                "avg_logp": rec["avg_logp"],
                "avg_entropy_nats": rec["avg_entropy_nats"],
                "avg_entropy_bits": rec["avg_entropy_bits"],
                "label": label,
            }

            json_line = json.dumps(out_obj, ensure_ascii=False)
            fout.write(json_line + "\n")

    print(f"Done. Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()