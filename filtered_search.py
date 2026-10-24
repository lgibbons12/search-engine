"""Filtered hybrid search: hard filters from Claude, then weighted RRF over BM25 and embeddings.

Steps:
  1. Claude turns the query into a SearchFilter (see hard_filter.py).
  2. The filter's mask keeps only the postings that pass every hard filter.
  3. BM25 and embeddings each rank the remaining postings by the filter's soft_query.
  4. Weighted RRF: each posting gets w_bm25 / (RRF_K + bm25_rank) + w_emb / (RRF_K + emb_rank),
     with the weights read from rrf_weights.json.
  5. Claude reranks the top RERANK_K (see rerank.py).

Runs on data-subset.csv, the rows embeddings.npy was built from (run embed_search.py first).

Usage:
  uv run filtered_search.py "remote senior python engineer"   # one query
  uv run filtered_search.py                                   # interactive prompt
"""

import json
import os
import sys

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from bm25_search import build_index, tokenize
from embed_search import EMBEDDINGS_PATH, MODEL_NAME, SUBSET_PATH
from hard_filter import SearchFilter, parse_search_filter, search_filter_mask, unknown_level
from rerank import RERANK_K, rerank
from rrf import RRF_K, ranks

TOP_K = 20
WEIGHTS_PATH = "rrf_weights.json"


def load_weights() -> dict[str, float]:
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        weights = json.load(f)
    missing = {"bm25", "embeddings"} - weights.keys()
    if missing:
        sys.exit(f"{WEIGHTS_PATH} is missing weights for: {', '.join(sorted(missing))}")
    total = weights["bm25"] + weights["embeddings"]
    if not np.isclose(total, 1.0):
        sys.exit(f"{WEIGHTS_PATH} weights must sum to 1, got {total:g}")
    return weights


def weighted_rrf(bm25, model, doc_embs: np.ndarray, weights: dict[str, float], query: str,
                 rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted RRF scores, BM25 ranks and embedding ranks for `rows`, ranked only among themselves."""
    # Score against the whole index so BM25's IDF reflects the full corpus, then rank within `rows`.
    bm25_scores = bm25.get_scores(tokenize(query))[rows]
    emb_scores = doc_embs[rows] @ model.encode(query, normalize_embeddings=True)

    bm25_rank, emb_rank = ranks(bm25_scores), ranks(emb_scores)
    rrf_scores = weights["bm25"] / (RRF_K + bm25_rank) + weights["embeddings"] / (RRF_K + emb_rank)
    return rrf_scores, bm25_rank, emb_rank


def tiered_order(rrf_scores: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Positions sorted best first: known level matches before unknown-level postings, then by RRF score."""
    return np.lexsort((-rrf_scores, unknown))


def search(client: anthropic.Anthropic, bm25, model, doc_embs: np.ndarray, df: pd.DataFrame,
           weights: dict[str, float], query: str, k: int = TOP_K) -> tuple[SearchFilter, pd.DataFrame]:
    search_filter = parse_search_filter(client, query)
    rows = np.flatnonzero(search_filter_mask(df, search_filter).to_numpy())
    rrf_scores, bm25_rank, emb_rank = weighted_rrf(bm25, model, doc_embs, weights, search_filter.soft_query, rows)

    top = tiered_order(rrf_scores, unknown_level(df, search_filter).to_numpy()[rows])[:RERANK_K]
    candidates = df.iloc[rows[top]].copy()
    candidates.insert(0, "rrf", rrf_scores[top].round(4))
    candidates.insert(1, "bm25_rank", bm25_rank[top])
    candidates.insert(2, "emb_rank", emb_rank[top])

    results = rerank(client, query, candidates)
    columns = ["rerank", "rrf", "bm25_rank", "emb_rank", "title", "location", "formatted_experience_level", "work_type"]
    return search_filter, results[columns].head(k)


def main():
    df = pd.read_csv(SUBSET_PATH)
    doc_embs = np.load(EMBEDDINGS_PATH)
    if len(doc_embs) != len(df):
        sys.exit(f"{EMBEDDINGS_PATH} has {len(doc_embs):,} rows but {SUBSET_PATH} has {len(df):,}. Rerun embed_search.py.")
    weights = load_weights()

    print(f"Indexing {len(df):,} postings...")
    bm25 = build_index(df)
    model = SentenceTransformer(MODEL_NAME)
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_TOKEN") or os.getenv("ANTHROPIC_API_KEY"))

    def run(query: str) -> None:
        search_filter, results = search(client, bm25, model, doc_embs, df, weights, query)
        print(f"filter: {search_filter.model_dump_json()}")
        if results.empty:
            print("No postings pass the filters.")
        else:
            print(results.to_string())

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
        return

    while True:
        try:
            query = input("query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query:
            run(query)


if __name__ == "__main__":
    main()
