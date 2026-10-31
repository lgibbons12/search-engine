"""Provided eval harness. Don't edit; import it.

    from eval import evaluate
    evaluate(my_search, name="bm25")          # my_search(query: str, k: int) -> list[product_id]

Prints nDCG@10 (graded: E=3, S=1, C/I=0), Exact-recall@100, MRR (first Exact),
the 10 worst queries by nDCG@10, and appends one line to results.log.
Pass `query_ids=[...]` to evaluate on a subset (e.g. a dev split you tune on).
"""
import math, time, json
import pandas as pd

_queries = pd.read_csv("data/queries.csv")
_qrels = pd.read_csv("data/qrels.csv")
_gains = {}
for qid, pid, g in _qrels[["query_id", "product_id", "gain"]].itertuples(index=False):
    _gains.setdefault(qid, {})[pid] = g


def _ndcg(ranked, gains, k=10):
    dcg = sum((2 ** gains.get(p, 0) - 1) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(search_fn, name="run", query_ids=None, k=100, show_worst=10):
    qs = _queries if query_ids is None else _queries[_queries.query_id.isin(set(query_ids))]
    rows, t0 = [], time.time()
    for qid, text in qs[["query_id", "query"]].itertuples(index=False):
        ranked = list(search_fn(text, k))[:k]
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"duplicate product_ids returned for query {qid!r}")
        gains = _gains[qid]
        exact = {p for p, g in gains.items() if g == 3}
        rr = next((1 / (i + 1) for i, p in enumerate(ranked) if p in exact), 0.0)
        rows.append({"query_id": qid, "query": text,
                     "ndcg10": _ndcg(ranked, gains),
                     "recall100": len(exact & set(ranked)) / len(exact),
                     "mrr": rr})
    r = pd.DataFrame(rows)
    secs = time.time() - t0
    summary = {"name": name, "n": len(r), "ndcg10": round(r.ndcg10.mean(), 4),
               "recall100": round(r.recall100.mean(), 4), "mrr": round(r.mrr.mean(), 4),
               "ms_per_query": round(1000 * secs / max(len(r), 1), 1)}
    print(json.dumps(summary))
    if show_worst:
        print(r.nsmallest(show_worst, "ndcg10")[["query_id", "query", "ndcg10", "recall100"]]
              .to_string(index=False))
    with open("results.log", "a") as f:
        f.write(json.dumps(summary) + "\n")
    return r