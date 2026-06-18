import argparse
import pandas as pd
import config
from item import ItemCF

NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]

def load_titles(path=config.TRAIN_NEWS):
    """Trả về dict {news_id: title}. Rỗng nếu không đọc được file."""
    try:
        news = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLUMNS)
        return dict(zip(news["news_id"], news["title"]))
    except FileNotFoundError:
        print("Không tìm thấy news.tsv — sẽ không hiển thị tiêu đề.")
        return {}

def recommend(model, history, top_k, candidates=None):
    """Trả về list [(news_id, score)] đã sắp xếp giảm dần, lấy top-K."""
    if not history:
        return []
    if candidates is None:
        candidates = list(model.idx2news.values())

    scores = model.score_one(history, candidates)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]

def show(model, titles, history, top_k):
    recs = recommend(model, history, top_k)
    print(f"\nHistory: {' '.join(history)}")
    if not recs:
        print("Không có gợi ý.")
        return

    print(f"{'Score':<10}{'News ID':<12}Title")
    print("-" * 50)
    for news_id, score in recs:
        print(f"{score:<10.4f}{news_id:<12}{titles.get(news_id, '-')}")

def interactive(model, titles, top_k):
    print('\nNhập news IDs cách nhau bằng space (q để thoát). Vd: N1 N2 N3')
    while True:
        try:
            line = input("History: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in {"q", "quit", "exit", "e"}:
            break
        if line:
            show(model, titles, line.split(), top_k)

def main():
    parser = argparse.ArgumentParser(description="News recommendation demo")
    parser.add_argument("--history", help='History của user, vd: "N1 N2 N3"')
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    model = ItemCF().fit()
    titles = load_titles()

    if args.history:
        show(model, titles, args.history.split(), args.top_k)
    else:
        interactive(model, titles, args.top_k)

if __name__ == "__main__":
    main()
