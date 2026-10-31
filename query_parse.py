"""LLM query understanding: Claude extracts brands / colors / negations into a pydantic model.

Parses are cached in data/query_parses.json keyed by query text, so eval reruns don't re-call
the API. Uses only the query text (never labels), so parsing all 300 up front isn't leakage.

    python query_parse.py    # parse every query in data/queries.csv (parallel) and write the cache
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

MODEL = "claude-opus-5-5"
CACHE_PATH = Path("data/query_parses.json")
QUERIES_PATH = "data/queries.csv"
N_WORKERS = 8

SYSTEM = """You extract shopping constraints from a short, messy e-commerce search query.

- brands: brand names the shopper wants, lowercase, as written ("nike", "samsung").
- colors: colors the shopper wants, lowercase ("pink", "rose gold").
- excluded_terms: things the shopper explicitly does NOT want, from phrasings like "not X",
  "no X", "without X", "except X", "non X". Return only the short term: "not pink" -> "pink",
  "without valve" -> "valve", "wheel not cover" -> "cover".

Only extract what the query states; leave a list empty otherwise. Do not treat these as exclusions:
- words inside a product name, slogan, or book/album/movie title
  ("listen without prejudice shirt", "talk to plants not cops", "the one for whom food is not enough");
- "X free" / "X-free" (e.g. "sugar free", "bpa free") - the shopper wants products labelled that way."""


class QueryParse(BaseModel):
    brands: list[str] = Field(description="Brands the shopper wants; empty if none.")
    colors: list[str] = Field(description="Colors the shopper wants; empty if none.")
    excluded_terms: list[str] = Field(description="Short terms the shopper explicitly excludes; empty if none.")


EMPTY = QueryParse(brands=[], colors=[], excluded_terms=[])

_client: anthropic.Anthropic | None = None
_cache: dict[str, dict] | None = None
_lock = threading.Lock()


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_TOKEN"])
    return _client


def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    return _cache


def save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(_load_cache(), indent=1, sort_keys=True))


def call_llm(query: str) -> QueryParse:
    response = _get_client().beta.messages.parse(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": query}],
        output_format=QueryParse,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return EMPTY  # no signal -> search runs unchanged
    return response.parsed_output


def parse_query(query: str) -> QueryParse:
    cache = _load_cache()
    if query not in cache:
        parsed = call_llm(query)
        with _lock:
            cache[query] = parsed.model_dump()
    return QueryParse(**cache[query])


def _timed_parse(query: str) -> tuple[QueryParse, float]:
    t0 = time.perf_counter()
    return parse_query(query), time.perf_counter() - t0


def main() -> None:
    queries = pd.read_csv(QUERIES_PATH)["query"].drop_duplicates().tolist()
    with ThreadPoolExecutor(N_WORKERS) as pool:
        parses, secs = zip(*pool.map(_timed_parse, queries))
    save_cache()
    with_signal = [(q, p) for q, p in zip(queries, parses) if p != EMPTY]
    print(f"parsed {len(queries)} queries -> {CACHE_PATH} | {len(with_signal)} have a signal")
    # Eval ms/query reads the cache; real per-query cost adds this LLM latency on top.
    print(f"  LLM latency per query: median {1000 * np.median(secs):.0f} ms, p90 {1000 * np.percentile(secs, 90):.0f} ms")
    for field in ("brands", "colors", "excluded_terms"):
        print(f"  {field}: {sum(bool(getattr(p, field)) for p in parses)} queries")
    for q, p in with_signal:
        if p.excluded_terms:
            print(f"  {q!r} -> {p.model_dump()}")


if __name__ == "__main__":
    main()
