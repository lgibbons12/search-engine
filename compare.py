"""Compare weighted RRF search with and without hard filters on the queries in hard_filter.py.

Steps:
  1. No filter: weighted RRF over every posting, using the raw query.
     Filter: Claude parses the query into a SearchFilter, then weighted RRF over the postings that
     pass it, using the filter's soft_query.
     Filter + rerank: Claude reranks the filter method's top RERANK_K (see rerank.py).
  2. Pooling: the union of every method's top EVAL_K per query.
  3. Labeling: filtered_judge.py labels each pooled posting for relevance and for whether it
     meets the query's hard constraints (cached in filtered_labels.json).
  4. Scoring:
       nDCG@k: a posting counts as relevant only if it is relevant AND meets the constraints.
       violation rate: share of the top k that break at least one hard constraint.

Run embed_search.py first so data-subset.csv and embeddings.npy exist.
"""

import os
import sys

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from bm25_search import build_index
from embed_search import EMBEDDINGS_PATH, MODEL_NAME, SUBSET_PATH
from evaluate import EVAL_K, ndcg_at_k
from filtered_judge import label_pools
from filtered_search import load_weights, tiered_order, weighted_rrf
from hard_filter import QUERIES, parse_search_filter, search_filter_mask, unknown_level
from rerank import RERANK_K, rerank


def top_rows(bm25, model, doc_embs: np.ndarray, weights: dict[str, float], query: str,
             rows: np.ndarray, unknown: np.ndarray, k: int = EVAL_K) -> np.ndarray:
    """Row indices of the top k of `rows` under weighted RRF, best first, unknown-level rows last."""
    rrf_scores, _, _ = weighted_rrf(bm25, model, doc_embs, weights, query, rows)
    return rows[tiered_order(rrf_scores, unknown[rows])[:k]]


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
    all_rows = np.arange(len(df))
    no_unknown = np.zeros(len(df), dtype=bool)
    job_ids = df["job_id"].astype(str).to_numpy()

    print("\nParsed filters:")
    tops = {}
    for q in QUERIES:
        query = q["query"]
        search_filter = parse_search_filter(client, query)
        rows = np.flatnonzero(search_filter_mask(df, search_filter).to_numpy())
        candidates = top_rows(bm25, model, doc_embs, weights, search_filter.soft_query, rows,
                              unknown_level(df, search_filter).to_numpy(), k=RERANK_K)
        tops[query] = {
            "no_filter": top_rows(bm25, model, doc_embs, weights, query, all_rows, no_unknown),
            "filter": candidates[:EVAL_K],
            # df has a RangeIndex, so the reranked index is the row indices.
            "filter_rerank": rerank(client, query, df.iloc[candidates]).index.to_numpy()[:EVAL_K],
        }
        print(f"  {query}\n    {search_filter.model_dump_json()} -> {len(rows):,} postings")

    pools = {query: list(dict.fromkeys(i for order in t.values() for i in order)) for query, t in tops.items()}
    print(f"\nPooled {sum(len(p) for p in pools.values())} postings across {len(QUERIES)} queries")
    labels = label_pools(client, df, pools)

    rows = []
    for query, t in tops.items():
        judged = labels[query]
        relevant = {job_id for job_id, l in judged.items() if l["relevant"] and l["meets_constraints"]}
        for method, order in t.items():
            ranked_ids = list(job_ids[order])
            violations = sum(not judged[job_id]["meets_constraints"] for job_id in ranked_ids)
            rows.append({
                "query": query,
                "method": method,
                f"ndcg@{EVAL_K}": ndcg_at_k(ranked_ids, relevant, EVAL_K),
                "violation_rate": violations / len(ranked_ids) if ranked_ids else 0.0,
                "results": len(ranked_ids),
            })
    results = pd.DataFrame(rows)

    metrics = [f"ndcg@{EVAL_K}", "violation_rate"]
    for metric in metrics:
        print(f"\nPer query, {metric}:")
        print(results.pivot(index="query", columns="method", values=metric).round(3).to_string())
    print("\nMean by method:")
    print(results.groupby("method")[metrics].mean().round(3).to_string())

    short = results.loc[results["results"] < EVAL_K]
    for r in short.itertuples():
        print(f"\nNote: {r.method} returned only {r.results} results for {r.query!r}.")


if __name__ == "__main__":
    main()
