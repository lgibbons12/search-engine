"""Dev-only sweep over cli.Config, stage by stage (coordinate descent), then one dev + test eval of the winner.

Candidates are scored with eval internals, so results.log only gets the final winner (dev + test).
Retrieval knobs are chosen on Exact-recall@50 (what the reranker sees; tie-break recall@100);
rerank and LLM knobs on nDCG@10. Progress streams to sweep_log.txt; winner saved to sweep_best.json.

    python sweep.py
"""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, replace
from typing import Callable

import numpy as np

import eval as ev
from cli import Config, Pipeline, config_to_flags
from split import dev_test_split

LOG_PATH = "sweep_log.txt"
BEST_PATH = "sweep_best.json"
PREVIOUS_BEST_TEST = 0.4912  # +llm_soft test in results.log

K1_GRID = (0.9, 1.2, 1.6)
B_GRID = (0.5, 0.75, 0.9)
W_BRAND_GRID = (0.0, 0.25, 0.5)
W_BULLETS_GRID = (0.0, 0.1, 0.25)
W_DENSE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
RRF_K_GRID = (1, 5, 10, 30, 60)
DEPTH_GRID = (100, 200)
CE_TEXT_GRID = ("title", "brand+title")
W_CE_GRID = (2.0, 4.0, 8.0, math.inf)
RERANK_DEPTH_GRID = (30, 50, 75, 100)
B_BRAND_GRID = (0.0, 0.5, 1.0, 2.0)
B_COLOR_GRID = (0.0, 1.0, 2.0, 4.0)
P_EXCL_GRID = (0.0, 2.0, 4.0, 8.0, 16.0)

_log = open(LOG_PATH, "w", buffering=1)
_t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _t0:6.0f}s] {msg}"
    print(line, flush=True)
    _log.write(line + "\n")


def _exact(qid: int) -> set[str]:
    return {p for p, g in ev._gains[qid].items() if g == 3}


def retrieval_score(pipe: Pipeline, cfg: Config, qids: list[int], qtext) -> tuple[float, float]:
    r50, r100 = [], []
    for q in qids:
        ids = pipe.product_ids[pipe.retrieve(qtext[q], cfg, 100)]
        exact = _exact(q)
        r50.append(len(exact & set(ids[:50])) / len(exact))
        r100.append(len(exact & set(ids)) / len(exact))
    return float(np.mean(r50)), float(np.mean(r100))


def ranking_score(pipe: Pipeline, cfg: Config, qids: list[int], qtext) -> tuple[float, float]:
    ndcg, r100 = [], []
    for q in qids:
        ids = pipe.search(qtext[q], 100, cfg)
        ndcg.append(ev._ndcg(ids, ev._gains[q]))
        exact = _exact(q)
        r100.append(len(exact & set(ids)) / len(exact))
    return float(np.mean(ndcg)), float(np.mean(r100))


def best_of(stage: str, cands: list[Config], score: Callable[[Config], tuple[float, float]],
            labels: tuple[str, str]) -> Config:
    log(f"== {stage}: {len(cands)} configs ==")
    best, best_s = None, None
    for cfg in cands:
        s = score(cfg)
        if best_s is None or s > best_s:
            best, best_s = cfg, s
            log(f"  new best {labels[0]}={s[0]:.4f} {labels[1]}={s[1]:.4f}  [{cfg.tag()}]")
    log(f"-> {stage} winner {labels[0]}={best_s[0]:.4f} {labels[1]}={best_s[1]:.4f}  [{best.tag()}]")
    return best


def main() -> None:
    dev, test = dev_test_split()
    qtext = ev._queries.set_index("query_id")["query"]
    pipe = Pipeline()
    log("pipeline loaded")
    ret = lambda c: retrieval_score(pipe, c, dev, qtext)
    rank = lambda c: ranking_score(pipe, c, dev, qtext)
    R, N = ("recall@50", "recall@100"), ("nDCG@10", "recall@100")

    cfg = Config()
    base = rank(cfg)
    log(f"baseline (current defaults) dev nDCG@10={base[0]:.4f} recall@100={base[1]:.4f}")

    # 1. retrieval: BM25 saturation/length norm, then field weights x fusion
    cfg = best_of("1a BM25 k1/b", [replace(cfg, k1=k, b=b) for k, b in itertools.product(K1_GRID, B_GRID)], ret, R)
    cfg = best_of("1b fields x fusion", [
        replace(cfg, w_brand=wb, w_bullets=wu, w_dense=wd, rrf_k=rk, depth=d)
        for wb, wu, wd, rk, d in itertools.product(W_BRAND_GRID, W_BULLETS_GRID, W_DENSE_GRID, RRF_K_GRID, DEPTH_GRID)
    ], ret, R)

    # 2. rerank: CE text x blend, then how deep to rerank (LLM off so CE is judged alone)
    no_llm = replace(cfg, llm="off")
    no_llm = best_of("2a CE text x w_ce", [replace(no_llm, ce_text=t, w_ce=w)
                                          for t, w in itertools.product(CE_TEXT_GRID, W_CE_GRID)], rank, N)
    no_llm = best_of("2b rerank depth", [replace(no_llm, rerank_depth=d) for d in RERANK_DEPTH_GRID], rank, N)

    # 3. LLM signals: off vs hard pre-filter vs soft boost grid
    cfg = best_of("3 LLM mode/weights", [no_llm, replace(no_llm, llm="hard")] + [
        replace(no_llm, llm="soft", b_brand=bb, b_color=bc, p_excl=pe)
        for bb, bc, pe in itertools.product(B_BRAND_GRID, B_COLOR_GRID, P_EXCL_GRID)
    ], rank, N)

    final = rank(cfg)
    log(f"dev: baseline {base[0]:.4f} -> sweep winner {final[0]:.4f} nDCG@10")
    with open(BEST_PATH, "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    log(f"winner saved to {BEST_PATH}; reproduce with: python cli.py --split test {config_to_flags(cfg)}")

    # Only now touch test, once, through the logging harness.
    run = lambda q, k: pipe.search(q, k, cfg)
    ev.evaluate(run, name=f"sweep_best dev {cfg.tag()}", query_ids=dev, show_worst=0)
    summary = ev.evaluate(run, name=f"sweep_best test {cfg.tag()}", query_ids=test)
    log(f"TEST nDCG@10={summary.ndcg10.mean():.4f} recall@100={summary.recall100.mean():.4f} "
        f"(previous best test {PREVIOUS_BEST_TEST})")


if __name__ == "__main__":
    main()
