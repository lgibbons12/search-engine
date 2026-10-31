"""Build and save dense product embeddings (stretch dense retriever).

Embeds one text blob per product: title + brand + color + bullets + description.
Brand/color stay in the blob for now; later we can drop them and treat as hard filters.

Tuned for CPU: short max_seq_length + field caps (full 512-token blobs were ~40 min).

    python embeddings.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
PRODUCTS_PATH = Path("data/products.parquet")
OUT_PATH = Path("data/product_embeddings.npz")
TEXT_COLS = ("title", "brand", "color", "bullets", "description")
# Cap long fields so we don't burn CPU on 512-token product pages.
FIELD_CAPS = {
    "title": 200,
    "brand": 40,
    "color": 40,
    "bullets": 400,
    "description": 300,
}
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 256

# bge-small-en-v1.5: prepend this to *queries* only (not documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_text(df: pd.DataFrame) -> list[str]:
    parts = []
    for c in TEXT_COLS:
        s = df[c].fillna("").astype(str).str.strip()
        cap = FIELD_CAPS.get(c)
        if cap is not None:
            s = s.str.slice(0, cap)
        parts.append(s)
    texts = []
    for row in zip(*parts):
        texts.append(" ".join(t for t in row if t))
    return texts


def encode_products(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode a shopper query for cosine search against saved product vectors."""
    return model.encode(
        QUERY_PREFIX + query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def load_embeddings(path: Path = OUT_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (product_ids, embeddings) from a prior run of this script."""
    data = np.load(path, allow_pickle=False)
    return data["product_ids"], data["embeddings"]


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"loading {PRODUCTS_PATH}")
    df = pd.read_parquet(PRODUCTS_PATH)
    texts = build_text(df)
    print(
        f"{len(texts):,} products | model={MODEL_NAME} | "
        f"max_seq={MAX_SEQ_LENGTH} | batch={BATCH_SIZE}"
    )

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    embeddings = encode_products(model, texts)
    # Fixed-width unicode, not object dtype, so np.load(allow_pickle=False) works.
    product_ids = df["product_id"].to_numpy(dtype=str)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        product_ids=product_ids,
        embeddings=embeddings,
    )
    print(
        f"wrote {OUT_PATH} | shape={embeddings.shape} | "
        f"dtype={embeddings.dtype} | ~{OUT_PATH.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
