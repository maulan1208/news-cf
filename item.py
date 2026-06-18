"""
Item-based
- Mỗi tin được mô tả bằng một vector trong không gian "user":
  thành phần thứ i = 1 nếu user i đã đọc tin đó, ngược lại = 0.
  => Hai tin được cùng một nhóm user đọc thì vector của chúng gần nhau.
- sự giống nhau được đo bằng Cosine similarity giữa hai vector. Nếu ta
  chuẩn hoá mọi vector về độ dài 1 (L2-normalize), thì cosine similarity
  chính là tích vô hướng (dot product).
"""

import pickle
import numpy as np
import torch
from scipy.sparse import load_npz
import config

def l2_normalize_rows(sparse_matrix):
    """Chia mỗi DÒNG của sparse matrix cho độ dài L2 của nó."""
    norms = np.sqrt(sparse_matrix.multiply(sparse_matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0  # tránh chia cho 0 với dòng rỗng
    return sparse_matrix.multiply((1.0 / norms)[:, None]).tocsr()

def l2_normalize_columns(dense_tensor):
    """Chia mỗi CỘT của dense tensor cho độ dài L2 của nó."""
    norms = dense_tensor.norm(dim=0, keepdim=True).clamp(min=1e-12)
    return dense_tensor / norms

def to_torch_sparse(sparse_matrix, device):
    coo = sparse_matrix.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()

class ItemCF:
    def __init__(self):
        self.device = "cuda"

        # item_vectors:   (n_items, n_users) — mỗi dòng là 1 tin, đã L2-normalize
        # item_vectors_t: chuyển vị, dùng để gom history thành profile
        self.item_vectors = None
        self.item_vectors_t = None
        self.n_items = None
        self.n_users = None

        self.news2idx = None
        self.idx2news = None

    def fit(self):
        with open(config.NEWS2IDX_PATH, "rb") as f:
            self.news2idx = pickle.load(f)
        self.idx2news = {idx: news_id for news_id, idx in self.news2idx.items()}

        # user × news -> news × user: giờ mỗi DÒNG là 1 tin.
        user_item = load_npz(str(config.USER_ITEM_MATRIX))
        item_user = user_item.T.tocsr().astype(np.float32)

        # Chuẩn hoá để dot product = cosine similarity.
        item_user = l2_normalize_rows(item_user)

        self.n_items, self.n_users = item_user.shape
        self.item_vectors = to_torch_sparse(item_user, self.device)
        self.item_vectors_t = self.item_vectors.transpose(0, 1).coalesce()

        print(f"ItemCF: {self.n_items} tin, {self.n_users} user")
        return self

    def score_one(self, history, candidates):
        """Chấm điểm các candidate cho 1 user. Trả về dict {news_id: score}."""
        return self.score([history], [candidates])[0]

    def score(self, histories, candidate_lists):
        if self.item_vectors is None:
            raise RuntimeError("Phải gọi fit() trước.")

        batch_size = len(histories)

        # One-hot history: cột j đánh dấu các tin user j từng đọc.
        history_onehot = self._build_history_onehot(histories, batch_size)

        with torch.no_grad():
            # Profile = tổng vector (trong không gian user) của tin trong history.
            # (n_users, n_items) @ (n_items, batch) -> (n_users, batch)
            profiles = torch.sparse.mm(self.item_vectors_t, history_onehot)

            # Chuẩn hoá profile về độ dài 1 (vì tổng nhiều vector làm mất chuẩn).
            profiles = l2_normalize_columns(profiles)

            # Điểm = cosine(mọi tin, profile).
            # (n_items, n_users) @ (n_users, batch) -> (n_items, batch)
            scores = torch.sparse.mm(self.item_vectors, profiles)

        return self._gather_candidate_scores(scores, candidate_lists)

    def _build_history_onehot(self, histories, batch_size):
        onehot = torch.zeros(
            (self.n_items, batch_size), dtype=torch.float32, device=self.device
        )
        for col, history in enumerate(histories):
            indices = [self.news2idx[n] for n in history if n in self.news2idx]
            if indices:
                rows = torch.tensor(indices, dtype=torch.long, device=self.device)
                onehot[rows, col] = 1.0
        return onehot

    def _gather_candidate_scores(self, scores, candidate_lists):
        results = []
        for col, candidates in enumerate(candidate_lists):
            row = {news_id: 0.0 for news_id in candidates}  # tin lạ -> mặc định 0

            known = [(n, self.news2idx[n]) for n in candidates if n in self.news2idx]
            if known:
                idx = torch.tensor([i for _, i in known], dtype=torch.long, device=self.device)
                values = scores.index_select(0, idx)[:, col].cpu().numpy()
                for (news_id, _), value in zip(known, values):
                    row[news_id] = float(value)

            results.append(row)
        return results
