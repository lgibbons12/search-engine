"""Cross-encoder rerank of the hybrid (BM25 + dense RRF) top 50.

ms-marco-MiniLM-L-6-v2 scores each (query, product text) pair. The top RERANK_DEPTH are
reordered; positions after that keep their hybrid order, so recall@100 is unchanged.

Optionally blends CE with the hybrid order (RRF of the two rankings) instead of trusting
the CE alone: score = w_ce / (BLEND_K + ce_rank) + 1 / (BLEND_K + hybrid_rank).

    python rerank.py    # grid-search doc text + w_ce on dev, log dev and test
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

import eval as ev
from bm25 import PRODUCTS_PATH
from hybrid import get_hybrid
from split import dev_test_split

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MAX_LENGTH = 256
RERANK_DEPTH = 50
BLEND_K = 10
BULLETS_CAP = 300
# Which product fields the CE sees, in order.
TEXT_VARIANTS = {
    "title": ("title",),
    "brand+title": ("brand", "title"),
    "brand+title+bullets": ("brand", "title", "bullets"),
}
# Tuned on dev (see main()); math.inf = pure CE order.
TEXT_VARIANT = "title"
W_CE = math.inf
W_CE_GRID = (1.0, 2.0, 4.0, math.inf)


class Reranker:
    def __init__(self):
        self.hybrid = get_hybrid()
        self.products = pd.read_parquet(PRODUCTS_PATH).set_index("product_id")
        self.products["bullets"] = self.products["bullets"].str.slice(0, BULLETS_CAP)
        self.ce = CrossEncoder(CE_MODEL, max_length=CE_MAX_LENGTH)

    def doc_text(self, pid: str, variant: str) -> str:
        row = self.products.loc[pid]
        return " ".join(t for t in (row[f].strip() for f in TEXT_VARIANTS[variant]) if t)

    def ce_scores(self, query: str, pids: list[str], variant: str) -> np.ndarray:
        pairs = [(query, self.doc_text(p, variant)) for p in pids]
        return np.asarray(self.ce.predict(pairs, batch_size=64, show_progress_bar=False))

    @staticmethod
    def reorder(pids: list[str], ce: np.ndarray, w_ce: float) -> list[str]:
        """Reorder the head by CE (w_ce=inf) or by RRF of CE rank and hybrid rank."""
        ce_order = np.argsort(-ce)
        if math.isinf(w_ce):
            return [pids[i] for i in ce_order]
        ce_rank = np.empty(len(pids), dtype=int)
        ce_rank[ce_order] = np.arange(1, len(pids) + 1)
        hybrid_rank = np.arange(1, len(pids) + 1)
        blended = w_ce / (BLEND_K + ce_rank) + 1.0 / (BLEND_K + hybrid_rank)
        return [pids[i] for i in np.argsort(-blended, kind="stable")]

    def candidates(self, query: str, k: int = 100, variant: str = TEXT_VARIANT,
                   mask: np.ndarray | None = None) -> tuple[list[str], np.ndarray, list[str]]:
        """Hybrid results split into (head, CE scores for head, tail)."""
        ranked = self.hybrid.search(query, max(k, RERANK_DEPTH), mask=mask)
        head = ranked[:RERANK_DEPTH]
        return head, self.ce_scores(query, head, variant), ranked[RERANK_DEPTH:]

    def search(self, query: str, k: int = 100, variant: str = TEXT_VARIANT, w_ce: float = W_CE,
               mask: np.ndarray | None = None) -> list[str]:
        head, ce, tail = self.candidates(query, k, variant, mask)
        return (self.reorder(head, ce, w_ce) + tail)[:k]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def search(query: str, k: int = 100) -> list[str]:
    return get_reranker().search(query, k)


def tune(rr: Reranker, query_ids: list[int]) -> tuple[str, float]:
    """Grid-search (text variant, w_ce) on mean nDCG@10; CE scores cached per variant."""
    qtext = ev._queries.set_index("query_id")["query"]
    heads = {qid: rr.hybrid.search(qtext[qid], RERANK_DEPTH) for qid in query_ids}
    base = np.mean([ev._ndcg(heads[q], ev._gains[q]) for q in query_ids])
    print(f"  hybrid (no rerank)                        nDCG@10={base:.4f}")
    best, best_ndcg = None, -1.0
    for variant in TEXT_VARIANTS:
        ce = {qid: rr.ce_scores(qtext[qid], heads[qid], variant) for qid in query_ids}
        for w in W_CE_GRID:
            ndcg = np.mean([ev._ndcg(rr.reorder(heads[q], ce[q], w), ev._gains[q]) for q in query_ids])
            print(f"  text={variant:<20} w_ce={w:<4}  nDCG@10={ndcg:.4f}")
            if ndcg > best_ndcg:
                best, best_ndcg = (variant, w), ndcg
    print(f"best on dev: text={best[0]} w_ce={best[1]} nDCG@10={best_ndcg:.4f}")
    return best


def main() -> None:
    dev, test = dev_test_split()
    rr = get_reranker()
    variant, w = tune(rr, dev)
    run = lambda q, k: rr.search(q, k, variant, w)
    ev.evaluate(run, name=f"hybrid+ce_rerank50 dev text={variant} w_ce={w}", query_ids=dev, show_worst=0)
    ev.evaluate(run, name=f"hybrid+ce_rerank50 test text={variant} w_ce={w}", query_ids=test)


if __name__ == "__main__":
    main()
