import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import config

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]

def load_behaviors(path):
    """Đọc behaviors.tsv thành DataFrame."""
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
    print(f"Đã đọc {len(df)} dòng behaviors")
    return df

def extract_interactions(behaviors):
    interactions = []
    for user_id, history, impressions in zip(
        behaviors["user_id"], behaviors["history"], behaviors["impressions"]
    ):
        if isinstance(history, str):
            for news_id in history.split():
                interactions.append((user_id, news_id))

        if isinstance(impressions, str):
            for token in impressions.split():
                news_id, label = token.split("-")
                if label == "1":
                    interactions.append((user_id, news_id))

    print(f"Đã trích {len(interactions)} tương tác")
    return interactions

def build_user_item_matrix(interactions):
    users = sorted({user_id for user_id, _ in interactions})
    news = sorted({news_id for _, news_id in interactions})

    user2idx = {user_id: i for i, user_id in enumerate(users)}
    news2idx = {news_id: i for i, news_id in enumerate(news)}

    rows = [user2idx[user_id] for user_id, _ in interactions]
    cols = [news2idx[news_id] for _, news_id in interactions]
    data = np.ones(len(interactions), dtype=np.float32)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(news)),
        dtype=np.float32,
    )
    # csr_matrix cộng dồn các tương tác trùng nhau -> về nhị phân (chỉ 0/1).
    matrix.data[:] = 1.0

    print(f"Ma trận user × news: {matrix.shape}")
    return matrix, user2idx, news2idx


def save(matrix, user2idx, news2idx):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    save_npz(str(config.USER_ITEM_MATRIX), matrix)
    with open(config.USER2IDX_PATH, "wb") as f:
        pickle.dump(user2idx, f)
    with open(config.NEWS2IDX_PATH, "wb") as f:
        pickle.dump(news2idx, f)

    print(f"Đã lưu vào {config.PROCESSED_DIR}")

def main():
    behaviors = load_behaviors(config.TRAIN_BEHAVIORS)
    interactions = extract_interactions(behaviors)
    matrix, user2idx, news2idx = build_user_item_matrix(interactions)
    save(matrix, user2idx, news2idx)

if __name__ == "__main__":
    main()
