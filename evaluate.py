import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config
from item import ItemCF

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]

# Parse dữ liệu
def parse_history(value):
    return value.split() if isinstance(value, str) else []

def parse_impressions(value):
    """'N1-0 N2-1' -> (['N1', 'N2'], [0, 1])."""
    candidates, labels = [], []
    for token in str(value).split():
        news_id, label = token.rsplit("-", 1)
        candidates.append(news_id)
        labels.append(int(label))
    return candidates, labels

# Các chỉ số xếp hạng 
# labels: nhãn đúng/sai (1/0) của các tin sau khi đã sắp xếp theo điểm model.
def mrr(labels):
    """Reciprocal rank của tin đúng đầu tiên."""
    for rank, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0

def dcg(labels):
    return sum(label / np.log2(rank + 1) for rank, label in enumerate(labels, start=1))

def ndcg(labels, k):
    """DCG@k chuẩn hoá theo thứ tự lý tưởng."""
    labels = labels[:k]
    ideal = dcg(sorted(labels, reverse=True))
    return dcg(labels) / ideal if ideal > 0 else 0.0

def auc(labels, scores):
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = rankdata(scores)  # hạng tăng dần, ties -> hạng trung bình
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

# Lưu kết quả
def save_results(output, results):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                runs = existing.get("runs", [])
        except json.JSONDecodeError:
            runs = []

    run = {
        "run_id": len(runs) + 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **results,
    }
    runs.append(run)

    path.write_text(
        json.dumps({"latest": run, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu run #{run['run_id']} vào {path}")

# đánh giá 
def evaluate(limit=None, batch_size=config.EVAL_BATCH_SIZE, output="results.json"):
    df = pd.read_csv(config.DEV_BEHAVIORS, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
    if limit:
        df = df.head(limit)

    model = ItemCF().fit()

    auc_scores, mrr_scores, ndcg5_scores, ndcg10_scores = [], [], [], []
    rows = list(zip(df["history"], df["impressions"]))

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        histories = [parse_history(history) for history, _ in chunk]
        parsed = [parse_impressions(impressions) for _, impressions in chunk]
        candidate_lists = [candidates for candidates, _ in parsed]
        label_lists = [labels for _, labels in parsed]

        batch_scores = model.score(histories, candidate_lists)

        for candidates, labels, scores in zip(candidate_lists, label_lists, batch_scores):
            score_values = [scores[c] for c in candidates]

            # AUC dùng điểm gốc của toàn danh sách (không phụ thuộc thứ tự).
            impression_auc = auc(labels, score_values)
            if impression_auc is not None:
                auc_scores.append(impression_auc)

            # Sắp xếp nhãn theo điểm model giảm dần, rồi đo chất lượng thứ hạng.
            order = sorted(range(len(candidates)), key=lambda i: score_values[i], reverse=True)
            ranked_labels = [labels[i] for i in order]

            mrr_scores.append(mrr(ranked_labels))
            ndcg5_scores.append(ndcg(ranked_labels, 5))
            ndcg10_scores.append(ndcg(ranked_labels, 10))

    metrics = {
        "auc": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg_5": float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0,
        "ndcg_10": float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
    }
    results = {
        "num_impressions": len(mrr_scores),
        "limit": limit,
        "batch_size": batch_size,
        "metrics": metrics,
    }

    print(f"AUC     = {metrics['auc']:.4f}")
    print(f"MRR     = {metrics['mrr']:.4f}")
    print(f"nDCG@5  = {metrics['ndcg_5']:.4f}")
    print(f"nDCG@10 = {metrics['ndcg_10']:.4f}")

    if output:
        save_results(output, results)
    return results

def main():
    parser = argparse.ArgumentParser(description="Đánh giá ItemCF trên dev set")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chỉ đánh giá N impression đầu.")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE,
                        help="Số impression mỗi batch chấm trên GPU.")
    parser.add_argument("--output", default="results.json",
                        help="File JSON lưu kết quả.")
    args = parser.parse_args()

    evaluate(limit=args.limit, batch_size=args.batch_size, output=args.output)

if __name__ == "__main__":
    main()
