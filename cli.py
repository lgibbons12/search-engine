"""Configurable search pipeline: every stage and weight is a flag. Defaults = tuned per-stage values.

    field BM25 (+ dense, weighted RRF) -> CE rerank of top N -> LLM signals (soft boosts | hard pre-filter)

    python cli.py --split dev                                  # evaluate defaults on dev (logs to results.log)
    python cli.py --split test --no-dense --llm off            # ablation on test
    python cli.py --query "nike running shoes women not pink"  # top 10 for one query

The Pipeline caches BM25 field scores, dense scores, CE scores and LLM matches per query, so
sweep.py can score many configs cheaply.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

import bm25
import eval as ev
from bm25 import PRODUCTS_PATH, FieldBM25, combine, top_k
from embeddings import MODEL_NAME, encode_query, load_embeddings
from hybrid import DEPTH, RRF_K, W_DENSE
from llm_boost import B_BRAND, B_COLOR, P_EXCL, Matcher, hard_mask, soft_adjust
from query_parse import parse_query
from rerank import (BULLETS_CAP, CE_MAX_LENGTH, CE_MODEL, RERANK_DEPTH, TEXT_VARIANT, TEXT_VARIANTS,
                    W_CE, Reranker)
from split import dev_test_split

LLM_MODES = ("off", "soft", "hard")


@dataclass(frozen=True)
class Config:
    # retrieval: field BM25
    use_bm25: bool = True
    w_title: float = bm25.WEIGHTS["title"]
    w_brand: float = bm25.WEIGHTS["brand"]
    w_bullets: float = bm25.WEIGHTS["bullets"]
    k1: float = bm25.K1
    b: float = bm25.B
    # retrieval: dense + weighted RRF (BM25 list has weight 1)
    dense: bool = True
    w_dense: float = W_DENSE
    rrf_k: int = RRF_K
    depth: int = DEPTH
    # cross-encoder rerank
    rerank: bool = True
    rerank_depth: int = RERANK_DEPTH
    ce_text: str = TEXT_VARIANT
    w_ce: float = W_CE
    # LLM query understanding
    llm: str = "soft"
    b_brand: float = B_BRAND
    b_color: float = B_COLOR
    p_excl: float = P_EXCL

    def tag(self) -> str:
        """Fields that differ from the defaults, e.g. 'k1=0.9 w_dense=1.5' (or 'defaults')."""
        default = Config()
        diff = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                if getattr(self, f.name) != getattr(default, f.name)]
        return " ".join(diff) or "defaults"

    def validate(self) -> None:
        if not (self.use_bm25 or self.dense):
            raise ValueError("need at least one retriever (BM25 or dense)")
        if self.llm not in LLM_MODES:
            raise ValueError(f"llm must be one of {LLM_MODES}")
        if self.llm == "soft" and not self.rerank:
            raise ValueError("--llm soft adds boosts to CE scores, so it needs --rerank")
        if self.ce_text not in TEXT_VARIANTS:
            raise ValueError(f"ce_text must be one of {list(TEXT_VARIANTS)}")


class Pipeline:
    def __init__(self):
        df = pd.read_parquet(PRODUCTS_PATH)
        self.df = df
        self.product_ids = df["product_id"].to_numpy(dtype=str)
        ids, self.embeddings = load_embeddings()
        if not np.array_equal(ids, self.product_ids):
            raise ValueError("embeddings out of sync with products.parquet; rerun embeddings.py")
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.ce = CrossEncoder(CE_MODEL, max_length=CE_MAX_LENGTH)
        self.matcher = Matcher(df)
        cols = df[["title", "brand", "bullets"]].fillna("").astype(str)
        cols["bullets"] = cols["bullets"].str.slice(0, BULLETS_CAP)
        self.texts = {v: [" ".join(t.strip() for t in row if t.strip()) for row in cols[list(f)].itertuples(index=False)]
                      for v, f in TEXT_VARIANTS.items()}
        self._bm25_key: tuple[float, float] | None = None
        self._bm25: FieldBM25 | None = None
        self._field_scores: dict[str, dict[str, np.ndarray]] = {}
        self._dense: dict[str, np.ndarray] = {}
        self._ce: dict[tuple[str, str, int], float] = {}
        self._matches: dict[str, dict[str, np.ndarray | None]] = {}

    # ---- cached per-query signals ----
    def field_scores(self, query: str, cfg: Config) -> dict[str, np.ndarray]:
        if self._bm25_key != (cfg.k1, cfg.b):  # one index (and score cache) at a time to bound memory
            self._bm25 = FieldBM25(self.df, cfg.k1, cfg.b)
            self._bm25_key, self._field_scores = (cfg.k1, cfg.b), {}
        if query not in self._field_scores:
            self._field_scores[query] = self._bm25.field_scores(query)
        return self._field_scores[query]

    def dense_scores(self, query: str) -> np.ndarray:
        if query not in self._dense:
            self._dense[query] = self.embeddings @ encode_query(self.encoder, query)
        return self._dense[query]

    def ce_scores(self, query: str, rows: np.ndarray, variant: str) -> np.ndarray:
        missing = [r for r in rows.tolist() if (variant, query, r) not in self._ce]
        if missing:
            preds = self.ce.predict([(query, self.texts[variant][r]) for r in missing],
                                    batch_size=64, show_progress_bar=False)
            self._ce.update({(variant, query, r): float(s) for r, s in zip(missing, preds)})
        return np.array([self._ce[(variant, query, r)] for r in rows.tolist()])

    def matches(self, query: str) -> dict[str, np.ndarray | None]:
        if query not in self._matches:
            self._matches[query] = self.matcher.matches(parse_query(query))
        return self._matches[query]

    # ---- stages ----
    def retrieve(self, query: str, cfg: Config, n: int, mask: np.ndarray | None = None) -> np.ndarray:
        """Top-n product rows from weighted RRF of the enabled retrievers."""
        lists = []
        if cfg.use_bm25:
            w = {"title": cfg.w_title, "brand": cfg.w_brand, "bullets": cfg.w_bullets}
            lists.append((1.0, combine(self.field_scores(query, cfg), w)))
        if cfg.dense:
            lists.append((cfg.w_dense, self.dense_scores(query)))
        depth = max(cfg.depth, n)
        if mask is not None:
            lists = [(w, np.where(mask, s, -np.inf)) for w, s in lists]
            depth = min(depth, int(mask.sum()))
        fused: dict[int, float] = {}
        for w, scores in lists:
            if w == 0:
                continue
            for rank, i in enumerate(top_k(scores, depth).tolist(), start=1):
                fused[i] = fused.get(i, 0.0) + w / (cfg.rrf_k + rank)
        return np.array(sorted(fused, key=fused.get, reverse=True)[:n], dtype=int)

    def rank_rows(self, query: str, cfg: Config, k: int) -> list[int]:
        m = self.matches(query) if cfg.llm != "off" else None
        mask = hard_mask(m) if cfg.llm == "hard" else None
        rows = self.retrieve(query, cfg, max(k, cfg.rerank_depth) if cfg.rerank else k, mask)
        if not cfg.rerank:
            return rows[:k].tolist()
        head, tail = rows[:cfg.rerank_depth], rows[cfg.rerank_depth:]
        score = self.ce_scores(query, head, cfg.ce_text)
        if cfg.llm == "soft":
            score = soft_adjust(score, head, m, cfg.b_brand, cfg.b_color, cfg.p_excl)
        return (Reranker.reorder(head.tolist(), score, cfg.w_ce) + tail.tolist())[:k]

    def search(self, query: str, k: int = 100, cfg: Config = Config()) -> list[str]:
        return self.product_ids[self.rank_rows(query, cfg, k)].tolist()


def config_to_flags(cfg: Config) -> str:
    """CLI flags reproducing cfg (only fields that differ from the defaults)."""
    out, default = [], Config()
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if v == getattr(default, f.name):
            continue
        flag = "--" + f.name.replace("_", "-")
        out.append(flag if v is True else f"--no-{flag[2:]}" if v is False else f"{flag} {v}")
    return " ".join(out)


def parse_args() -> tuple[argparse.Namespace, Config]:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run = p.add_argument_group("run")
    run.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    run.add_argument("--query", help="search one query and print the top --k titles instead of evaluating")
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--name", default="cli", help="prefix for the results.log entry")

    g = p.add_argument_group("field BM25")
    g.add_argument("--use-bm25", action=argparse.BooleanOptionalAction, default=d.use_bm25)
    g.add_argument("--w-title", type=float, default=d.w_title)
    g.add_argument("--w-brand", type=float, default=d.w_brand)
    g.add_argument("--w-bullets", type=float, default=d.w_bullets)
    g.add_argument("--k1", type=float, default=d.k1)
    g.add_argument("--b", type=float, default=d.b)

    g = p.add_argument_group("dense + RRF fusion")
    g.add_argument("--dense", action=argparse.BooleanOptionalAction, default=d.dense)
    g.add_argument("--w-dense", type=float, default=d.w_dense, help="dense RRF weight (BM25 = 1)")
    g.add_argument("--rrf-k", type=int, default=d.rrf_k)
    g.add_argument("--depth", type=int, default=d.depth, help="candidates taken from each retriever")

    g = p.add_argument_group("cross-encoder rerank")
    g.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=d.rerank)
    g.add_argument("--rerank-depth", type=int, default=d.rerank_depth)
    g.add_argument("--ce-text", choices=list(TEXT_VARIANTS), default=d.ce_text)
    g.add_argument("--w-ce", type=float, default=d.w_ce, help="CE weight vs hybrid order in RRF blend; inf = pure CE")

    g = p.add_argument_group("LLM query understanding")
    g.add_argument("--llm", choices=LLM_MODES, default=d.llm)
    g.add_argument("--b-brand", type=float, default=d.b_brand)
    g.add_argument("--b-color", type=float, default=d.b_color)
    g.add_argument("--p-excl", type=float, default=d.p_excl)

    args = p.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    try:
        cfg.validate()
    except ValueError as e:
        p.error(str(e))
    return args, cfg


def main() -> None:
    args, cfg = parse_args()
    pipe = Pipeline()
    if args.query:
        for rank, pid in enumerate(pipe.search(args.query, args.k, cfg), start=1):
            print(f"{rank:>3}. {pid}  {pipe.df.loc[pipe.df.product_id == pid, 'title'].iat[0][:100]}")
        return
    dev, test = dev_test_split()
    qids = {"dev": dev, "test": test, "all": None}[args.split]
    ev.evaluate(lambda q, k: pipe.search(q, k, cfg), name=f"{args.name} {args.split} {cfg.tag()}", query_ids=qids)


if __name__ == "__main__":
    main()
