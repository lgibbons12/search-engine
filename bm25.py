"""Field-weighted BM25: separate indexes for title / brand / bullets, scores summed with tuned weights.

BM25 is precomputed into a sparse doc x term weight matrix per field, so a query is just
a column-slice + sum (fast enough to grid-search weights over the dev split).

    python bm25.py      # grid-search weights on dev, log equal/tuned on dev and tuned on test
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import eval as ev
from split import dev_test_split

PRODUCTS_PATH = "data/products.parquet"
FIELDS = ("title", "brand", "bullets")
K1, B = 1.2, 0.75
# Tuned on dev (see main()); title is the anchor at 1.0.
WEIGHTS = {"title": 1.0, "brand": 0.25, "bullets": 0.0}
BRAND_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
BULLETS_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumerics + naive plural stripping ("shoes" -> "shoe", keeps "glass")."""
    toks = _TOKEN_RE.findall(text.lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]


class FieldIndex:
    """BM25 for one text field, stored as a CSC matrix of per-(doc, term) BM25 weights."""

    def __init__(self, texts: list[str], k1: float = K1, b: float = B):
        vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = vec.fit_transform(texts).tocsr().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() or 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        norm = k1 * (1 - b + b * dl[rows] / avgdl)
        tf.data = (tf.data * (k1 + 1) / (tf.data + norm) * idf[tf.indices]).astype(np.float32)
        self.weights = tf.tocsc()
        self.vocab = vec.vocabulary_

    def scores(self, tokens: list[str]) -> np.ndarray:
        cols = [self.vocab[t] for t in set(tokens) if t in self.vocab]
        if not cols:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()


def combine(field_scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[f] * field_scores[f] for f in FIELDS)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Row indices of the k highest scores, best first."""
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


class FieldBM25:
    def __init__(self, df: pd.DataFrame, k1: float = K1, b: float = B):
        self.product_ids = df["product_id"].to_numpy(dtype=str)
        self.indexes = {f: FieldIndex(df[f].fillna("").astype(str).tolist(), k1, b) for f in FIELDS}

    def field_scores(self, query: str) -> dict[str, np.ndarray]:
        toks = tokenize(query)
        return {f: idx.scores(toks) for f, idx in self.indexes.items()}

    def scores(self, query: str, weights: dict[str, float] | None = None) -> np.ndarray:
        """Weighted total BM25 score for every product (row order = product_ids)."""
        return combine(self.field_scores(query), weights or WEIGHTS)

    def rank(self, field_scores: dict[str, np.ndarray], weights: dict[str, float], k: int) -> list[str]:
        return self.product_ids[top_k(combine(field_scores, weights), k)].tolist()

    def search(self, query: str, k: int = 100, weights: dict[str, float] | None = None) -> list[str]:
        return self.rank(self.field_scores(query), weights or WEIGHTS, k)


_bm25: FieldBM25 | None = None


def get_bm25() -> FieldBM25:
    global _bm25
    if _bm25 is None:
        _bm25 = FieldBM25(pd.read_parquet(PRODUCTS_PATH))
    return _bm25


def search(query: str, k: int = 100) -> list[str]:
    return get_bm25().search(query, k)


def tune(bm25: FieldBM25, query_ids: list[int]) -> dict[str, float]:
    """Grid-search brand/bullets weights (title fixed at 1) on mean nDCG@10 over query_ids."""
    qtext = ev._queries.set_index("query_id")["query"]
    cached = {qid: bm25.field_scores(qtext[qid]) for qid in query_ids}
    best, best_ndcg = None, -1.0
    for wb, wu in itertools.product(BRAND_GRID, BULLETS_GRID):
        w = {"title": 1.0, "brand": wb, "bullets": wu}
        ndcg = np.mean([ev._ndcg(bm25.rank(fs, w, 10), ev._gains[qid]) for qid, fs in cached.items()])
        print(f"  brand={wb:<5} bullets={wu:<5} nDCG@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best, best_ndcg = w, ndcg
    print(f"best on dev: {best} nDCG@10={best_ndcg:.4f}")
    return best


def main() -> None:
    dev, test = dev_test_split()
    bm25 = get_bm25()
    equal = {f: 1.0 for f in FIELDS}
    ev.evaluate(lambda q, k: bm25.search(q, k, equal), name="bm25_fields_equal dev", query_ids=dev, show_worst=0)
    tuned = tune(bm25, dev)
    ev.evaluate(lambda q, k: bm25.search(q, k, tuned), name=f"bm25_fields_tuned dev {tuned}", query_ids=dev, show_worst=0)
    ev.evaluate(lambda q, k: bm25.search(q, k, tuned), name=f"bm25_fields_tuned test {tuned}", query_ids=test)


if __name__ == "__main__":
    main()
