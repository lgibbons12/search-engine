On that branch:
1. Your code. Each stage should be runnable on its own.
2. results.log, with nothing edited. I’ll read it as your timeline.
3. A NOTES.md in this shape:
     ## Results (test half)
     | Stage | nDCG@10 | Recall@100 | ms/query |
     |---|---|---|---|
     | Field-weighted BM25 | | | |
     | + cross-encoder rerank | | | |
     | + soft LLM boosts | | | |

     ## Per stage (2–3 sentences each)
     What it fixed / what it broke / a query it still fails on.

     ## Dev/test split + weights
     How you split, and what weights you landed on.

     ## Next with another day
     2–3 bullets.