# Search build: product search (75 min, solo)

You're building search for an online store. Shoppers type short, messy queries
("nike running shoes women not pink", "iphone 12 case with card holder").
Make the results as good as you can, and show the evidence.

## Provided
- `data/products.parquet`: product_id, title, brand, color, bullets, description
- `data/queries.csv`: 300 real shopper queries
- `data/qrels.csv`: graded human labels (E = exact, S = substitute, C = complement, I = irrelevant)
- `eval.py`: `evaluate(search_fn, name)` → nDCG@10, Exact-recall@100, MRR, worst queries, appends to results.log

## Deliverable at minute 75
1. A `search(query, k)` function, plus a results table: one line per stage with nDCG@10 and recall@100
2. Two or three sentences per stage: what it fixed, what it broke, and which queries it still handles badly
3. What you'd do next with another day

## Rules for this rep
- No vanilla single-text-column BM25 + embeddings + RRF this time. You've done that. Use these:
  1. **Field-weighted BM25:** separate indexes for title / brand / bullets, combined with weights you tune
  2. **Cross-encoder rerank** of the top 50 (`cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers), not an LLM
  3. **LLM query understanding as SOFT signals:** extract brand, color, and negations ("not pink", "without") → score boosts and penalties, not a hard pre-filter
  4. Stretch: a dense retriever (e.g. `BAAI/bge-small-en-v1.5`) as a second candidate source. Is it worth adding to the recall stage?
- Tune weights on half the queries (dev), and report on the other half (test). Fix the split up front.
- Never go more than 15 minutes without a new number in results.log.
- AI tools: ask at setup, like you will Monday. Default for this rep: docs only, no codegen.