"""Fixed dev/test split of the queries: seeded shuffle, half each. Tune on dev, report on test."""
from __future__ import annotations

import random

import pandas as pd

QUERIES_PATH = "data/queries.csv"
SEED = 0


def dev_test_split() -> tuple[list[int], list[int]]:
    qids = sorted(pd.read_csv(QUERIES_PATH).query_id.unique().tolist())
    random.Random(SEED).shuffle(qids)
    half = len(qids) // 2
    return qids[:half], qids[half:]
