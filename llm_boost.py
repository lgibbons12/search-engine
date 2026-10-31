"""Stage 3: LLM query understanding (query_parse.py) applied on top of hybrid + CE rerank, two ways.

soft (task rule 3): over the reranked top 50,
    score = CE + B_BRAND * brand_match + B_COLOR * color_match - P_EXCL * excluded_match
    Nothing is removed; weights tuned on dev.
hard (comparison):  products failing brand/color or hitting an excluded term are masked out of the
    search space before BM25/dense retrieval. Falls back to unfiltered if nothing survives.

Matching is whole-word on normalized text: brand -> brand or title field, color -> color or title,
excluded -> title or color, but NOT when the title itself negates it ("syringe without needle").

    python llm_boost.py    # tune soft weights on dev; log soft + hard on dev and test
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

import eval as ev
from bm25 import PRODUCTS_PATH
from query_parse import QueryParse, parse_query
from rerank import RERANK_DEPTH, get_reranker
from split import dev_test_split

# Tuned on dev (see main()).
B_BRAND = 1.0
B_COLOR = 2.0
P_EXCL = 4.0
B_BRAND_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
B_COLOR_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
P_EXCL_GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
NEGATION_TEMPLATES = ("without {t}", "no {t}", "non {t}", "not {t}", "{t} free", "{t}less")


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _pad(col: pd.Series) -> pd.Series:
    """Normalize a text column and pad with spaces so ' term ' is a whole-word contains."""
    return " " + col.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip() + " "


class Matcher:
    """Per-query boolean match vectors over all products (row order = products.parquet)."""

    def __init__(self, df: pd.DataFrame):
        self.title, self.brand, self.color = _pad(df["title"]), _pad(df["brand"]), _pad(df["color"])

    @staticmethod
    def _contains(field: pd.Series, phrase: str) -> np.ndarray:
        return field.str.contains(f" {phrase} ", regex=False).to_numpy()

    def _any(self, terms: list[str], fields: tuple[pd.Series, ...]) -> np.ndarray | None:
        terms = [t for t in map(_norm, terms) if t]
        if not terms:
            return None
        hit = np.zeros(len(self.title), dtype=bool)
        for t, f in itertools.product(terms, fields):
            hit |= self._contains(f, t)
        return hit

    def _excluded(self, terms: list[str]) -> np.ndarray | None:
        hit = self._any(terms, (self.title, self.color))
        if hit is None:
            return None
        for t in filter(None, map(_norm, terms)):
            for tmpl in NEGATION_TEMPLATES:
                hit &= ~self._contains(self.title, _norm(tmpl.format(t=t)))
        return hit

    def matches(self, parse: QueryParse) -> dict[str, np.ndarray | None]:
        return {
            "brand": self._any(parse.brands, (self.brand, self.title)),
            "color": self._any(parse.colors, (self.color, self.title)),
            "excluded": self._excluded(parse.excluded_terms),
        }


def hard_mask(m: dict[str, np.ndarray | None]) -> np.ndarray | None:
    mask = None
    for key, keep in (("brand", True), ("color", True), ("excluded", False)):
        if m[key] is not None:
            part = m[key] if keep else ~m[key]
            mask = part if mask is None else mask & part
    if mask is None or not mask.any():
        return None  # no signal, or the filter would empty the search space
    return mask


def soft_adjust(ce: np.ndarray, rows: np.ndarray, m: dict[str, np.ndarray | None],
                b_brand: float, b_color: float, p_excl: float) -> np.ndarray:
    adj = ce.astype(np.float64).copy()
    for key, w in (("brand", b_brand), ("color", b_color), ("excluded", -p_excl)):
        if m[key] is not None and w:
            adj += w * m[key][rows]
    return adj


class LLMBoost:
    def __init__(self):
        self.rr = get_reranker()
        self.matcher = Matcher(pd.read_parquet(PRODUCTS_PATH))
        self.row = {pid: i for i, pid in enumerate(self.rr.hybrid.product_ids)}
        self._matches: dict[str, dict[str, np.ndarray | None]] = {}

    def matches(self, query: str) -> dict[str, np.ndarray | None]:
        if query not in self._matches:
            self._matches[query] = self.matcher.matches(parse_query(query))
        return self._matches[query]

    def soft_search(self, query: str, k: int = 100, b_brand: float = B_BRAND,
                    b_color: float = B_COLOR, p_excl: float = P_EXCL) -> list[str]:
        head, ce, tail = self.rr.candidates(query, k)
        return (self._soft_order(query, head, ce, b_brand, b_color, p_excl) + tail)[:k]

    def _soft_order(self, query, head, ce, b_brand, b_color, p_excl) -> list[str]:
        rows = np.array([self.row[p] for p in head])
        adj = soft_adjust(ce, rows, self.matches(query), b_brand, b_color, p_excl)
        return [head[i] for i in np.argsort(-adj, kind="stable")]

    def hard_search(self, query: str, k: int = 100) -> list[str]:
        return self.rr.search(query, k, mask=hard_mask(self.matches(query)))


_boost: LLMBoost | None = None


def get_boost() -> LLMBoost:
    global _boost
    if _boost is None:
        _boost = LLMBoost()
    return _boost


def search(query: str, k: int = 100) -> list[str]:
    return get_boost().soft_search(query, k)


def tune(lb: LLMBoost, query_ids: list[int]) -> tuple[float, float, float]:
    """Grid-search soft weights on mean nDCG@10 (CE scores cached); also reports the signal subset."""
    qtext = ev._queries.set_index("query_id")["query"]
    cached = {qid: lb.rr.candidates(qtext[qid], RERANK_DEPTH) for qid in query_ids}
    signal = [q for q in query_ids if any(v is not None for v in lb.matches(qtext[q]).values())]
    print(f"  {len(signal)}/{len(query_ids)} dev queries carry a brand/color/negation signal")
    best, best_ndcg = None, -1.0
    for w in itertools.product(B_BRAND_GRID, B_COLOR_GRID, P_EXCL_GRID):
        per_q = {q: ev._ndcg(lb._soft_order(qtext[q], cached[q][0], cached[q][1], *w), ev._gains[q])
                 for q in query_ids}
        ndcg = float(np.mean(list(per_q.values())))
        if w == (0.0, 0.0, 0.0) or ndcg > best_ndcg:
            print(f"  brand={w[0]:<4} color={w[1]:<4} excl={w[2]:<4} nDCG@10={ndcg:.4f}  "
                  f"signal-subset={np.mean([per_q[q] for q in signal]):.4f}")
        if ndcg > best_ndcg:
            best, best_ndcg = w, ndcg
    print(f"best on dev: brand={best[0]} color={best[1]} excl={best[2]} nDCG@10={best_ndcg:.4f}")
    return best


def main() -> None:
    dev, test = dev_test_split()
    lb = get_boost()
    w = tune(lb, dev)
    tag = f"brand={w[0]} color={w[1]} excl={w[2]}"
    soft = lambda q, k: lb.soft_search(q, k, *w)
    ev.evaluate(soft, name=f"+llm_soft dev {tag}", query_ids=dev, show_worst=0)
    ev.evaluate(soft, name=f"+llm_soft test {tag}", query_ids=test)
    ev.evaluate(lb.hard_search, name="+llm_hard_prefilter dev", query_ids=dev, show_worst=0)
    ev.evaluate(lb.hard_search, name="+llm_hard_prefilter test", query_ids=test)


if __name__ == "__main__":
    main()
