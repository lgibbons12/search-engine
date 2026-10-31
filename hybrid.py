"""Hybrid retrieval: weighted RRF of field-weighted BM25 and dense (bge-small) candidates.

    score(d) = W_BM25 / (rrf_k + rank_bm25(d)) + w_dense / (rrf_k + rank_dense(d))

Ranks are 1-based over the top DEPTH of each retriever; a product missing from one list
gets nothing from it. BM25 weight is the anchor at 1.0, so w_dense is the relative weight.

    python hybrid.py    # grid-search w_dense x rrf_k on dev, log dense-only + hybrid
"""
from __future__ import annotations

import itertools

import numpy as np
from sentence_transformers import SentenceTransformer

import eval as ev
from bm25 import get_bm25, top_k
from embeddings import MODEL_NAME, encode_query, load_embeddings
from split import dev_test_split

DEPTH = 100
W_BM25 = 1.0
# Tuned on dev (see main()).
W_DENSE = 1.0
RRF_K = 10
W_DENSE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
RRF_K_GRID = (10, 30, 60, 100)


class Hybrid:
    def __init__(self):
        self.bm25 = get_bm25()
        ids, self.embeddings = load_embeddings()
        if not np.array_equal(ids, self.bm25.product_ids):
            raise ValueError("embeddings out of sync with products.parquet; rerun embeddings.py")
        self.product_ids = self.bm25.product_ids
        self.model = SentenceTransformer(MODEL_NAME)

    def dense_scores(self, query: str) -> np.ndarray:
        return self.embeddings @ encode_query(self.model, query)

    def candidates(self, query: str, depth: int = DEPTH,
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Top-`depth` row indices from BM25 and from dense, each best first.

        `mask` (bool per product) restricts the search space: products where it's False are never returned.
        """
        bm25, dense = self.bm25.scores(query), self.dense_scores(query)
        if mask is not None:
            bm25, dense = np.where(mask, bm25, -np.inf), np.where(mask, dense, -np.inf)
            depth = min(depth, int(mask.sum()))
        return top_k(bm25, depth), top_k(dense, depth)

    def fuse(self, bm25_idx: np.ndarray, dense_idx: np.ndarray, k: int,
             w_dense: float = W_DENSE, rrf_k: int = RRF_K) -> list[str]:
        fused: dict[int, float] = {}
        for w, idx in ((W_BM25, bm25_idx), (w_dense, dense_idx)):
            if w == 0:
                continue
            for rank, i in enumerate(idx.tolist(), start=1):
                fused[i] = fused.get(i, 0.0) + w / (rrf_k + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return self.product_ids[best].tolist()

    def search(self, query: str, k: int = 100, w_dense: float = W_DENSE, rrf_k: int = RRF_K,
               mask: np.ndarray | None = None) -> list[str]:
        return self.fuse(*self.candidates(query, mask=mask), k, w_dense, rrf_k)

    def dense_search(self, query: str, k: int = 100) -> list[str]:
        return self.product_ids[top_k(self.dense_scores(query), k)].tolist()


_hybrid: Hybrid | None = None


def get_hybrid() -> Hybrid:
    global _hybrid
    if _hybrid is None:
        _hybrid = Hybrid()
    return _hybrid


def search(query: str, k: int = 100) -> list[str]:
    return get_hybrid().search(query, k)


def _exact_recall(ranked: list[str], gains: dict[str, int]) -> float:
    exact = {p for p, g in gains.items() if g == 3}
    return len(exact & set(ranked)) / len(exact)


def tune(hybrid: Hybrid, query_ids: list[int]) -> tuple[float, int]:
    """Grid-search (w_dense, rrf_k) on mean nDCG@10 over query_ids; also prints recall@100."""
    qtext = ev._queries.set_index("query_id")["query"]
    cached = {qid: hybrid.candidates(qtext[qid]) for qid in query_ids}
    best, best_ndcg = None, -1.0
    for w, rk in itertools.product(W_DENSE_GRID, RRF_K_GRID):
        ndcgs, recalls = [], []
        for qid, (b, d) in cached.items():
            ranked = hybrid.fuse(b, d, 100, w, rk)
            ndcgs.append(ev._ndcg(ranked, ev._gains[qid]))
            recalls.append(_exact_recall(ranked, ev._gains[qid]))
        ndcg = float(np.mean(ndcgs))
        print(f"  w_dense={w:<5} rrf_k={rk:<4} nDCG@10={ndcg:.4f}  recall@100={np.mean(recalls):.4f}")
        if ndcg > best_ndcg:
            best, best_ndcg = (w, rk), ndcg
    print(f"best on dev: w_dense={best[0]} rrf_k={best[1]} nDCG@10={best_ndcg:.4f}")
    return best


def main() -> None:
    dev, test = dev_test_split()
    hybrid = get_hybrid()
    ev.evaluate(hybrid.dense_search, name="dense_bge_small dev", query_ids=dev, show_worst=0)
    w, rk = tune(hybrid, dev)
    run = lambda q, k: hybrid.search(q, k, w, rk)
    ev.evaluate(run, name=f"hybrid_rrf dev w_dense={w} rrf_k={rk}", query_ids=dev, show_worst=0)
    ev.evaluate(run, name=f"hybrid_rrf test w_dense={w} rrf_k={rk}", query_ids=test)


if __name__ == "__main__":
    main()
