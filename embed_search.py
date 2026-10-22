"""Semantic search over the `text` column of data-with-text.csv using sentence embeddings.

Embeddings are cached in embeddings.npy; delete it (or rerun preprocessing) to rebuild.

Usage:
  uv run embed_search.py "remote machine learning job"   # one query
  uv run embed_search.py                                 # interactive prompt
"""

import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from preprocessing import OUTPUT_PATH

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "embeddings.npy"
TOP_K = 20
N_DOCS = 5000  # subset for testing; set to None to embed every posting
SUBSET_PATH = "data-subset.csv"  # the exact rows that embeddings.npy was built from


def save_subset(df: pd.DataFrame) -> None:
    df.to_csv(SUBSET_PATH, index=False)
    print(f"Saved the {len(df):,} embedded postings to {SUBSET_PATH}")


def load_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # Rebuild if the CSV has changed since the embeddings were cached.
    if os.path.exists(EMBEDDINGS_PATH) and os.path.getmtime(EMBEDDINGS_PATH) > os.path.getmtime(OUTPUT_PATH):
        doc_embs = np.load(EMBEDDINGS_PATH)
        if len(doc_embs) == len(texts):
            return doc_embs

    print(f"Encoding {len(texts):,} postings (cached to {EMBEDDINGS_PATH} afterwards)...")
    doc_embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, doc_embs)
    return doc_embs


def search(model: SentenceTransformer, doc_embs: np.ndarray, df: pd.DataFrame, query: str, k: int = TOP_K) -> pd.DataFrame:
    query_emb = model.encode(query, normalize_embeddings=True)
    scores = doc_embs @ query_emb  # cosine similarity, since both sides are normalized
    top = np.argsort(scores)[::-1][:k]
    results = df.iloc[top][["title", "location", "formatted_experience_level"]].copy()
    results.insert(0, "score", scores[top].round(3))
    return results


def main():
    df = pd.read_csv(OUTPUT_PATH)
    if N_DOCS is not None:
        df = df.head(N_DOCS)
    model = SentenceTransformer(MODEL_NAME)
    doc_embs = load_embeddings(model, df["text"].tolist())
    save_subset(df)

    if len(sys.argv) > 1:
        print(search(model, doc_embs, df, " ".join(sys.argv[1:])).to_string())
        return

    while True:
        try:
            query = input("query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query:
            print(search(model, doc_embs, df, query).to_string())


if __name__ == "__main__":
    main()
