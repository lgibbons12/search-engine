"""Evaluate BM25, embedding and RRF search on a fixed set of queries over data-subset.csv.

Steps:
  1. Pooling: for each query, take the union of the top POOL_K from BM25, embeddings and RRF.
  2. Labeling: Claude judges each pooled posting relevant (1) or not (0). Labels are cached in
     labels.json, so reruns only pay for (query, posting) pairs that haven't been labeled yet.
  3. Scoring: recall@k and nDCG@k for each method. Postings outside the pool count as not relevant.

Run embed_search.py first so data-subset.csv and embeddings.npy exist.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from bm25_search import build_index, tokenize
from embed_search import EMBEDDINGS_PATH, MODEL_NAME, SUBSET_PATH
from rrf import RRF_K, ranks

# Written by looking through data-subset.csv so each query has real matches in it.
# "keyword" queries use words a posting would contain; "natural" queries describe the job
# in everyday language, so they test whether search can match meaning, not just words.
QUERIES = [
    {"query": "registered nurse night shift ICU", "kind": "keyword"},
    {"query": "dental hygienist", "kind": "keyword"},
    {"query": "senior accountant CPA", "kind": "keyword"},
    {"query": "warehouse forklift operator", "kind": "keyword"},
    {"query": "construction project manager", "kind": "keyword"},
    {"query": "social media marketing coordinator", "kind": "keyword"},
    {"query": "CDL truck driver", "kind": "keyword"},
    {"query": "entry level job analyzing data with SQL and Excel", "kind": "natural"},
    {"query": "caring for pets at an animal hospital", "kind": "natural"},
    {"query": "helping employees fix computer and login problems", "kind": "natural"},
]

POOL_K = 10
EVAL_K = 10
LABELS_PATH = "labels.json"
RESULTS_SVG_PATH = "results.svg"
JUDGE_MODEL = "claude-opus-5-5"

JUDGE_PROMPT = """You are judging search results for a job search engine.

A job seeker searched for: "{query}"

For each job posting below, decide whether it is relevant: would this job seeker plausibly want \
to see it in their results? Judge the job itself (role, duties, level), not just shared words. \
A posting is relevant if it is the kind of job they asked for, even if it uses different words.

{postings}

Return one label per posting, using its job_id."""


class Label(BaseModel):
    job_id: int
    relevant: bool


class Labels(BaseModel):
    labels: list[Label]


# --- Retrieval -------------------------------------------------------------------------------

def rankings(bm25, model, doc_embs: np.ndarray, query: str) -> dict[str, np.ndarray]:
    """Full ranking (row indices, best first) for each method."""
    bm25_scores = bm25.get_scores(tokenize(query))
    emb_scores = doc_embs @ model.encode(query, normalize_embeddings=True)
    rrf_scores = 1 / (RRF_K + ranks(bm25_scores)) + 1 / (RRF_K + ranks(emb_scores))
    return {name: np.argsort(scores)[::-1] for name, scores in
            [("bm25", bm25_scores), ("embeddings", emb_scores), ("rrf", rrf_scores)]}


def pool(ranked: dict[str, np.ndarray]) -> list[int]:
    """Union of the top POOL_K from every method, in a stable order."""
    return list(dict.fromkeys(i for order in ranked.values() for i in order[:POOL_K]))


# --- Labeling --------------------------------------------------------------------------------

def load_labels() -> dict[str, dict[str, int]]:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def judge(client: anthropic.Anthropic, query: str, postings: pd.DataFrame) -> dict[str, int]:
    """Ask Claude to label every posting for one query in a single request."""
    listing = "\n\n".join(
        f"<posting job_id=\"{row.job_id}\">\n{row.text}\n</posting>" for row in postings.itertuples()
    )
    response = client.beta.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(query=query, postings=listing)}],
        output_format=Labels,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Judge refused query {query!r}: {response.stop_details}")

    labels = {str(l.job_id): int(l.relevant) for l in response.parsed_output.labels}
    missing = set(postings["job_id"].astype(str)) - labels.keys()
    if missing:
        raise RuntimeError(f"Judge skipped {len(missing)} postings for query {query!r}")
    return labels


def label_pools(df: pd.DataFrame, pools: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    labels = load_labels()
    todo = {}
    for query, rows in pools.items():
        known = labels.get(query, {})
        unlabeled = df.iloc[rows]
        unlabeled = unlabeled[~unlabeled["job_id"].astype(str).isin(known)]
        if len(unlabeled):
            todo[query] = unlabeled

    if todo:
        total = sum(len(p) for p in todo.values())
        print(f"Labeling {total} postings across {len(todo)} queries with {JUDGE_MODEL}...")
        load_dotenv()
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_TOKEN") or os.getenv("ANTHROPIC_API_KEY"))
        with ThreadPoolExecutor(max_workers=5) as pool_exec:
            results = pool_exec.map(lambda item: (item[0], judge(client, *item)), todo.items())
            for query, new in results:
                labels.setdefault(query, {}).update(new)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved labels to {LABELS_PATH}")
    else:
        print(f"All pooled postings already labeled in {LABELS_PATH}")
    return labels


# --- Metrics ---------------------------------------------------------------------------------

def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Share of all known relevant postings that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG: DCG of the top k divided by the DCG of a perfect ranking."""
    dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(ranked_ids[:k]) if doc in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


# --- Output ----------------------------------------------------------------------------------

def save_table_svg(summary: pd.DataFrame, path: str, title: str) -> None:
    """Write a small table (methods as rows, metrics as columns) to an SVG, best value per column in bold."""
    names = {"bm25": "BM25", "embeddings": "Embeddings", "rrf": "RRF"}
    col_w, first_w, row_h, pad = 130, 150, 36, 20
    width = first_w + col_w * len(summary.columns) + 2 * pad
    top = pad + 40
    height = top + row_h * (len(summary) + 1) + pad

    def text(x, y, s, anchor="start", weight="normal", fill="#1f2328", size=15):
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-weight="{weight}" '
                f'font-size="{size}" fill="{fill}">{s}</text>')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="15">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(pad, pad + 16, title, weight="600", size=17),
        text(pad, pad + 36, f"Mean over {len(QUERIES)} queries", fill="#656d76", size=13),
    ]
    y = top + row_h * 0.65
    parts.append(text(pad, y, "Method", weight="600", fill="#656d76"))
    for j, col in enumerate(summary.columns):
        parts.append(text(pad + first_w + col_w * (j + 1) - 10, y, col, anchor="end", weight="600", fill="#656d76"))
    parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{top + row_h}" y2="{top + row_h}" stroke="#d0d7de"/>')

    best = summary.max()
    for i, (method, row) in enumerate(summary.iterrows(), 1):
        y = top + row_h * i + row_h * 0.65
        parts.append(text(pad, y, names.get(method, method)))
        for j, col in enumerate(summary.columns):
            weight = "700" if row[col] == best[col] else "normal"
            parts.append(text(pad + first_w + col_w * (j + 1) - 10, y, f"{row[col]:.3f}", anchor="end", weight=weight))
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{top + row_h * (i + 1)}" '
                     f'y2="{top + row_h * (i + 1)}" stroke="#eaeef2"/>')
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved {path}")


# --- Main ------------------------------------------------------------------------------------

def main():
    df = pd.read_csv(SUBSET_PATH)
    doc_embs = np.load(EMBEDDINGS_PATH)
    if len(doc_embs) != len(df):
        raise SystemExit(f"{EMBEDDINGS_PATH} and {SUBSET_PATH} have different row counts. Rerun embed_search.py.")

    print(f"Indexing {len(df):,} postings...")
    bm25 = build_index(df)
    model = SentenceTransformer(MODEL_NAME)
    job_ids = df["job_id"].astype(str).to_numpy()

    all_rankings = {q["query"]: rankings(bm25, model, doc_embs, q["query"]) for q in QUERIES}
    pools = {query: pool(ranked) for query, ranked in all_rankings.items()}
    print(f"Pooled {sum(len(p) for p in pools.values())} postings across {len(QUERIES)} queries")

    labels = label_pools(df, pools)

    rows = []
    for q in QUERIES:
        query = q["query"]
        relevant = {job_id for job_id, rel in labels[query].items() if rel}
        for method, ranked in all_rankings[query].items():
            ranked_ids = list(job_ids[ranked[:EVAL_K]])
            rows.append({
                "query": query,
                "kind": q["kind"],
                "method": method,
                "relevant_in_pool": len(relevant),
                f"recall@{EVAL_K}": recall_at_k(ranked_ids, relevant, EVAL_K),
                f"ndcg@{EVAL_K}": ndcg_at_k(ranked_ids, relevant, EVAL_K),
                "unjudged": sum(doc not in labels[query] for doc in ranked_ids),
            })
    results = pd.DataFrame(rows)

    metrics = [f"recall@{EVAL_K}", f"ndcg@{EVAL_K}"]
    print("\nPer query:")
    print(results.pivot(index="query", columns="method", values=f"ndcg@{EVAL_K}").round(3).to_string())
    summary = results.groupby("method")[metrics].mean()
    print("\nMean by method:")
    print(summary.round(3).to_string())
    save_table_svg(summary, RESULTS_SVG_PATH, "BM25 vs embeddings vs RRF")
    print("\nMean by method and query kind:")
    print(results.groupby(["kind", "method"])[metrics].mean().round(3).to_string())

    unjudged = results.loc[results["method"] == "rrf", "unjudged"].sum()
    if unjudged:
        print(f"\nNote: {unjudged} RRF top-{EVAL_K} results fell outside the pool and were counted as not relevant.")


if __name__ == "__main__":
    main()
