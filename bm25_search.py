"""BM25 keyword search over the `text` column of data-with-text.csv.

Usage:
  uv run bm25_search.py "python data engineer"   # one query
  uv run bm25_search.py                          # interactive prompt
"""

import sys

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from preprocessing import OUTPUT_PATH

TOP_K = 20


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index(df: pd.DataFrame) -> BM25Okapi:
    tokenized_docs = [tokenize(text) for text in df["text"]]
    return BM25Okapi(tokenized_docs)


def search(bm25: BM25Okapi, df: pd.DataFrame, query: str, k: int = TOP_K) -> pd.DataFrame:
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]
    results = df.iloc[top][["title", "location", "formatted_experience_level"]].copy()
    results.insert(0, "score", scores[top].round(2))
    return results


def main():
    df = pd.read_csv(OUTPUT_PATH)
    print(f"Indexing {len(df):,} postings...")
    bm25 = build_index(df)

    if len(sys.argv) > 1:
        print(search(bm25, df, " ".join(sys.argv[1:])).to_string())
        return

    while True:
        try:
            query = input("query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query:
            print(search(bm25, df, query).to_string())


if __name__ == "__main__":
    main()
