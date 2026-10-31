"""Run this BEFORE the clock starts (Mercor will hand you data ready to go).

Streams part of the Amazon ESCI shopping-queries test split (US only) and writes:
  data/products.parquet  product_id, title, brand, color, bullets, description
  data/queries.csv       query_id, query
  data/qrels.csv         query_id, product_id, label (E/S/C/I), gain (3/1/0/0)

The corpus is every product judged for ANY sampled query, so each query's
competitors include other queries' products plus its own hard negatives (I).
Unjudged products count as irrelevant.

    uv add datasets pandas pyarrow   (or pip install ...)
    python setup_data.py
"""
import os, random
import pandas as pd
from datasets import load_dataset

MAX_ROWS = 80_000        # rows to stream (US rows only are kept)
N_QUERIES = 300
MIN_JUDGED = 8
SEED = 0
GAIN = {"E": 3, "S": 1, "C": 0, "I": 0}

def main():
    os.makedirs("data", exist_ok=True)
    ds = load_dataset("tasksource/esci", split="test", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if i >= MAX_ROWS:
            break
        if r["product_locale"] == "us":
            rows.append(r)
    df = pd.DataFrame(rows)
    df["esci_label"] = df.esci_label.str[0]  # dataset uses "Exact"/"Substitute"/...; we want E/S/C/I
    print(f"streamed {MAX_ROWS:,} rows, kept {len(df):,} US rows, {df.query_id.nunique():,} queries")

    stats = df.groupby("query_id").agg(n=("product_id", "size"),
                                       n_exact=("esci_label", lambda s: (s == "E").sum()))
    good = stats[(stats.n >= MIN_JUDGED) & (stats.n_exact >= 1)].index.tolist()
    random.Random(SEED).shuffle(good)
    keep = set(good[:N_QUERIES])
    print(f"{len(good):,} queries qualify; sampling {len(keep)}")

    # Corpus = all judged products from all streamed US rows (more distractors than just the sample).
    prods = (df[["product_id", "product_title", "product_brand", "product_color",
                 "product_bullet_point", "product_description"]]
             .drop_duplicates("product_id")
             .rename(columns={"product_title": "title", "product_brand": "brand",
                              "product_color": "color", "product_bullet_point": "bullets",
                              "product_description": "description"})
             .fillna(""))
    prods.to_parquet("data/products.parquet", index=False)

    sub = df[df.query_id.isin(keep)]
    sub[["query_id", "query"]].drop_duplicates().to_csv("data/queries.csv", index=False)
    q = sub[["query_id", "product_id", "esci_label"]].rename(columns={"esci_label": "label"})
    q["gain"] = q.label.map(GAIN)
    q.to_csv("data/qrels.csv", index=False)
    print(f"products: {len(prods):,} | queries: {len(keep)} | qrels: {len(q):,}")
    print(q.label.value_counts().to_string())

if __name__ == "__main__":
    main()