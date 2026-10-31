# Search build: notes

`search(query, k)` lives in `llm_boost.py` (full pipeline, tuned defaults). `cli.py` exposes every
stage as a flag (`python cli.py --help`); `bm25_explained.svg` and `pipeline.svg` explain the code.

## Results (test half)

| Stage | nDCG@10 | Recall@100 | ms/query |
|---|---|---|---|
| Field-weighted BM25 | 0.403 | 0.747 | 0.8 |
| + dense bge-small, weighted RRF (stretch) | 0.440 | 0.798 | 60 |
| + cross-encoder rerank (top 50) | 0.488 | 0.798 | 196 |
| + soft LLM boosts | **0.491** | 0.798 | 203* |
| *comparison:* hard LLM pre-filter instead of soft | 0.474 | 0.736 | 192* |
| *sweep winner (all flags tuned jointly on dev)* | 0.494 | 0.798 | 221* |

\*LLM parses are cached (`data/query_parses.json`); a live query adds ~1.8 s median / 3.4 s p90
for the Claude call, which is 10x the rest of the pipeline.

**The sweep says the knobs are exhausted.** `sweep.py` searched ~550 configs over every `cli.py`
flag on dev (coordinate descent: retrieval on Exact-recall@50, rerank/LLM on nDCG@10). Dev went
0.5720 to 0.5725 and test 0.491 to 0.494, both inside noise. It raised recall@50 (0.762 to 0.770,
via k1 0.9, rrf_k 60, depth 200) but the CE didn't turn that into nDCG. More gains need a better
model (see Next), not more tuning. Winner: `python cli.py --split test --w-brand 0.5 --k1 0.9
--rrf-k 60 --depth 200 --w-ce 8.0 --p-excl 2.0`; I kept the per-stage defaults as the headline.

## Per stage

**Field-weighted BM25.** Three sparse BM25 indexes (title / brand / bullets) summed with tuned
weights; dev nDCG@10 went 0.415 (equal weights) to 0.447. Bullets at any weight hurt (long marketing
text adds matches, not precision) and brand above 0.5 double-counts brand words already in titles.
Still fails on negations ("ffp2 mask without valve") and long natural-language queries ("laptop
with large memory capacity and a well known brand").

**+ Dense retriever (stretch).** bge-small alone beat tuned BM25 on dev (0.458 vs 0.447), and RRF
fusion added ~5 points of recall@100 on test, so yes, it's worth it in the recall stage. Fusion
weight is flat from 0.75 to 1.5x; the gain is from fusing, not the exact weight. Costs ~60 ms/query
of CPU query encoding.

**+ Cross-encoder rerank.** The biggest single jump (+4.8 nDCG@10 on test). Title-only text matched
brand+title+bullets (within 0.003 on dev), and pure CE order beat every blend with the hybrid rank.
Fixed long descriptive queries; still ranks accessories over the product for "kindle fire" and
"extra memory foam" (MS MARCO learned topicality, not product-vs-accessory).

**+ Soft LLM boosts.** Claude parses each query into a pydantic `QueryParse` (brands, colors,
excluded terms); matches add `+1.0 brand +2.0 color -4.0 excluded` to the CE score. Only 48/150 dev
queries carry a signal; on those, nDCG@10 went 0.503 to 0.525, and test moved +0.004 nDCG / +0.012
MRR. Exclusions skip titles that negate the term themselves ("Syringe Without Needle"), but "ffp2
mask without valve" and "08 chevy tahoe wheel not cover" still score 0.

**Hard pre-filter (why rule 3 says soft).** Masking non-matching products out before retrieval cost
6 points of recall@100 and 1.4 of nDCG@10. "kindle fire" shows why: the LLM extracts brand
"kindle", but the exact products are titled "Fire HD 8 Tablet" with brand "Amazon", so recall
dropped 0.79 to 0.14. A wrong parse is a small penalty when soft and a lost result when hard.

## Dev/test split + weights

`split.py`: seeded shuffle (seed 0) of the 300 query ids, 150 dev / 150 test, fixed before any
tuning. Every grid search scores dev through the eval internals (no results.log noise); test was
run once per stage through `evaluate()`.

| Component | Tuned values |
|---|---|
| BM25 fields | title 1.0, brand 0.25, bullets 0.0; k1 1.2, b 0.75 |
| Fusion | dense weight 1.0 (BM25 = 1), rrf_k 10, depth 100 per retriever |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2`, top 50, title only, pure CE order |
| LLM boosts | brand +1.0, color +2.0, excluded -4.0 (`claude-opus-5-5`, low effort) |

Caveat: many dev grid cells sit within 0.003 nDCG of each other; on 150 queries that's noise, so
the tuned values are "a good region", not precise optima.

## Next with another day

- **Fine-tune the cross-encoder on the ESCI train split** (E/S/C/I graded labels). The remaining
  errors are product-vs-accessory and substitute-vs-exact, which is exactly what ESCI labels teach
  and MS MARCO doesn't.
- **Audit labels and add confidence intervals.** Several recall-0 queries are label noise
  ("listen without prejudice shirt" -> exact = unrelated sarcastic T-shirts; "clothes bins" ->
  Spanish-language titles). Bootstrap CIs would show which stage gains are real.
- **LLM query rewrite as a third RRF list, folded into the existing parse call** (zero extra
  latency), and, outside this rep's rules, an LLM listwise rerank of the top 20 for negation and
  product-vs-accessory.

## Process notes

- `setup_data.py` wrote empty `queries.csv` / `qrels.csv`: the dataset's labels are full words
  ("Exact"), the script compared to "E". Fixed by normalizing to the first letter; products file
  unchanged (same md5), so embeddings stayed valid.
- Product `color` is noisy (e.g. `"80 CFM"`), so color matches also check the title.
- AI assistance: code in this rep was written with Claude Code (a deviation from the task's
  docs-only default).
