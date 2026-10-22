"""Hybrid search: fuse BM25 and embedding rankings with Reciprocal Rank Fusion (RRF).

Each document gets sum(1 / (RRF_K + rank)) over both rankings, where rank starts at 1.
Runs on data-subset.csv, the rows embeddings.npy was built from (run embed_search.py first).

Usage:
  uv run rrf.py "remote machine learning job"   # one query
  uv run rrf.py                                 # interactive prompt
"""

import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bm25_search import build_index, tokenize
from embed_search import EMBEDDINGS_PATH, MODEL_NAME, SUBSET_PATH

TOP_K = 20
RRF_K = 60  # standard constant from the original RRF paper; dampens the gap between top ranks


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank of each document (1 = best) under the given scores."""
    order = np.argsort(scores)[::-1]
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def search(bm25, model, doc_embs: np.ndarray, df: pd.DataFrame, query: str, k: int = TOP_K) -> pd.DataFrame:
    bm25_scores = bm25.get_scores(tokenize(query))
    emb_scores = doc_embs @ model.encode(query, normalize_embeddings=True)

    bm25_rank, emb_rank = ranks(bm25_scores), ranks(emb_scores)
    rrf_scores = 1 / (RRF_K + bm25_rank) + 1 / (RRF_K + emb_rank)

    top = np.argsort(rrf_scores)[::-1][:k]
    results = df.iloc[top][["title", "location"]].copy()
    results.insert(0, "rrf", rrf_scores[top].round(4))
    results.insert(1, "bm25_rank", bm25_rank[top])
    results.insert(2, "emb_rank", emb_rank[top])
    return results


def main():
    df = pd.read_csv(SUBSET_PATH)
    doc_embs = np.load(EMBEDDINGS_PATH)
    if len(doc_embs) != len(df):
        sys.exit(f"{EMBEDDINGS_PATH} has {len(doc_embs):,} rows but {SUBSET_PATH} has {len(df):,}. Rerun embed_search.py.")

    print(f"Indexing {len(df):,} postings...")
    bm25 = build_index(df)
    model = SentenceTransformer(MODEL_NAME)

    if len(sys.argv) > 1:
        print(search(bm25, model, doc_embs, df, " ".join(sys.argv[1:])).to_string())
        return

    while True:
        try:
            query = input("query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query:
            print(search(bm25, model, doc_embs, df, query).to_string())


if __name__ == "__main__":
    main()
