# Job posting search

Keyword search, semantic search and hybrid search over LinkedIn job postings, with an evaluation comparing the three. On top of that, a filtered search pipeline: Claude turns the query into hard filters, weighted RRF ranks what passes, and Claude reranks the top 30 (see [Filtered search](#filtered-search)).

Data: [`xanderios/linkedin-job-postings`](https://huggingface.co/datasets/xanderios/linkedin-job-postings) on Hugging Face (33,246 postings).

## Setup

```sh
uv sync
```

`evaluate.py` and the filtered search scripts call Claude, so they need an API key in `.env`:

```
ANTHROPIC_API_TOKEN=sk-ant-...
```

(`ANTHROPIC_API_KEY` works too.)

## Pipeline

Run these in order. Each step saves its output, so later steps don't redo earlier work.

| Step | Command | Output |
|---|---|---|
| 1. Preprocess | `uv run preprocessing.py` | `data-with-text.csv` |
| 2. Embed | `uv run embed_search.py` | `embeddings.npy`, `data-subset.csv` |
| 3. Evaluate | `uv run evaluate.py` | `labels.json`, `results.svg` |

### 1. Preprocessing (`preprocessing.py`)

- Loads the dataset and converts the time columns from epoch milliseconds to dates.
- Builds a `text` column for search: `<title> in <location>. Description: <first 1000 chars of description>`. `company_id` is left out because it's a number with no company name attached.
- Drops postings whose `text` repeats an earlier one (545 reposts), leaving 32,701.

### 2. Search methods

All three print the top 20 results. Pass a query as an argument, or run with no arguments for an interactive prompt.

- **BM25** (`bm25_search.py`): keyword search. Tokenizes with `text.lower().split()`, builds `BM25Okapi`, and takes the top 20 scores with `np.argsort`. It runs on all 32,701 postings.
- **Embeddings** (`embed_search.py`): semantic search with `SentenceTransformer("all-MiniLM-L6-v2")`. Vectors are normalized, so cosine similarity is just `doc_embs @ query_emb`.
  - To keep testing fast, it embeds only the first 5,000 postings (`N_DOCS`, about 2 minutes on CPU; set it to `None` for all of them).
  - It saves those exact rows to `data-subset.csv`, and the embeddings are cached in `embeddings.npy`.
- **RRF** (`rrf.py`): hybrid search that combines the BM25 and embedding rankings with Reciprocal Rank Fusion. Each posting scores `1/(60 + bm25_rank) + 1/(60 + emb_rank)`. It runs on `data-subset.csv`.

`main.py` is a separate interactive explorer for the data: column overview, value counts, text search, filters and salary stats.

### 3. Evaluation (`evaluate.py`)

1. **Queries:** 10 test queries written by looking through the 5,000-row subset, so each one has real matches.
   - 7 are *keyword* queries that use words a posting would contain, e.g. "CDL truck driver".
   - 3 are *natural-language* queries that describe the job in everyday words, e.g. "caring for pets at an animal hospital".
2. **Pooling:** for each query, the union of the top 10 from BM25, embeddings and RRF. That's 170 postings to label instead of 50,000.
3. **Labeling:** `claude-opus-5-5` marks each pooled posting relevant (1) or not (0), one request per query, using structured output.
   - Labels are cached in `labels.json`, so reruns only pay for new postings.
4. **Metrics:**
   - `recall_at_k`: the share of all known relevant postings that appear in the top k.
   - `ndcg_at_k`: binary-gain nDCG, which rewards putting relevant postings near the top.
   - Postings outside the pool count as not relevant.

## Results

Mean over the 10 queries, k = 10:

| Method | recall@10 | nDCG@10 | nDCG@10, keyword queries | nDCG@10, natural-language queries |
|---|---|---|---|---|
| BM25 | 0.575 | 0.692 | 0.862 | 0.294 |
| Embeddings | **0.728** | **0.862** | 0.876 | **0.829** |
| RRF | 0.695 | 0.852 | **0.954** | 0.614 |

![Results table](results.svg)

- **Keyword queries:** RRF is best, since combining the two methods beats either one alone.
- **Natural-language queries:** BM25 nearly fails, because the postings share few words with the query. Embeddings handle them well. BM25's weak rankings pull RRF down here.
- **Caveats:** this is a small test. There are only 10 queries (3 natural-language), the subset is 5,000 postings, and the labels come from a model. I spot-checked them and they look reasonable, but a few seniority calls are debatable.

## Filtered search

The search methods above only rank by the query's words or meaning. A query like "entry level nursing jobs, full time" also states hard requirements (seniority, employment type), and ranking alone doesn't enforce them. The filtered pipeline pulls those requirements out of the query, applies them as filters, ranks what's left, and has an LLM rerank the top results.

It runs on `data-subset.csv` and `embeddings.npy`, so run steps 1 and 2 of the pipeline first.

```
query
  -> hard_filter.parse_search_filter    Claude Haiku 5.5 -> SearchFilter (structured output)
  -> hard_filter.search_filter_mask     boolean mask over the 5,000 postings
  -> filtered_search.weighted_rrf       BM25 + embeddings on soft_query, ranked within the mask
  -> filtered_search.tiered_order       known level matches first, unknown-level postings after
  -> rerank.rerank                      Claude Haiku 5.5 scores the top 30 from 0 to 3
  -> top results
```

### 1. Query to filters (`hard_filter.py`)

- **`SearchFilter`** is a Pydantic model: `remote`, `experience_levels`, `work_types`, `location`, `exclude_terms` and `soft_query`.
- **`ExperienceLevel` and `WorkType`** are `StrEnum`s whose values are the dataset's exact strings, so they can be compared to the data directly. They come from `formatted_experience_level` (e.g. `"Mid-Senior level"`) and `work_type` (e.g. `"FULL_TIME"`).
  - The data has no separate mid or senior level, only `Mid-Senior level`. So the prompt maps "a few years of experience" to `Associate` plus `Mid-Senior level`.
- **`FILTER_PROMPT`** is the prompt that turns a query into a filter. Read it in full. Its main rules:
  - Leave a filter empty when unsure. A wrong hard filter throws away good jobs, while an empty one only lets in a few extra.
  - Express exclusions by listing everything else. "Not an internship" becomes every level except `Internship`.
  - Each field's description says how it will be matched, so the model writes values that will actually match. For example, location `"New York"` rather than `"NYC"`, or `", TX"` for a whole state.
  - `soft_query` is the query with the filter-only parts removed. It's what BM25 and embeddings rank on.
- **`parse_search_filter`** calls `client.messages.parse(..., output_format=SearchFilter)` on `claude-haiku-5-5` at `effort: "low"`.
  - It's a non-beta call. Haiku has no server-side refusal fallback, so a refusal raises an error.
- **`search_filter_mask`** applies the filter:

  | Field | How it's matched |
  |---|---|
  | `remote` | `remote_allowed == 1` |
  | `experience_levels` | `isin`, but postings with no level also pass (see below) |
  | `work_types` | `isin` on `work_type` |
  | `location` | case-insensitive substring of `location` |
  | `exclude_terms` | whole-word regex over `text`; any match drops the posting |

  An empty field filters nothing.
- **`experience_level`** returns each posting's level, counting an `INTERNSHIP` work type as the `Internship` level.
  - Only 69 postings are tagged with the Internship level, and 36 more are marked internship only by work type.
- **`unknown_level`** marks the roughly 20% of postings that have no level.
  - They pass the level filter rather than being dropped for missing data.
  - They're ranked after the postings whose level actually matches (see `tiered_order`).
  - Without this, about 1,000 untagged postings crowded out the 69 real internships.

Run `uv run hard_filter.py` to parse the 8 test queries in `QUERIES` and print each filter.

### 2. Weighted RRF over the filtered postings (`filtered_search.py`)

- **`load_weights`** reads `rrf_weights.json` (`{"bm25": 0.3, "embeddings": 0.7}`). It stops with an error if a weight is missing or the two don't sum to 1.
  - Only the ratio between the weights changes the ranking. Making them sum to 1 just keeps scores comparable across runs.
- **`weighted_rrf`** scores the filtered rows only. Each posting scores `w_bm25 / (60 + bm25_rank) + w_emb / (60 + emb_rank)`.
  - BM25 still scores against the full index, so how rare a word counts as reflects all 5,000 postings rather than the filtered few.
  - Ranks are then computed within the filtered rows.
- **`tiered_order`** sorts with `np.lexsort((-rrf_scores, unknown))`. Postings whose level is known to match come first, then the untagged ones, each group by RRF score.
- **`search`** runs the full pipeline: parse, mask, weighted RRF, tiered order, top 30, rerank. It returns the parsed filter and the results table.

### 3. LLM rerank (`rerank.py`)

- **`RERANK_PROMPT`** has Claude score each of the top `RERANK_K = 30` postings:
  - 3: the right job and it meets every requirement
  - 2: the right job, but the posting doesn't say whether it meets one requirement
  - 1: a related job that meets the requirements
  - 0: the wrong job, or it breaks a requirement
- Claude sees the raw query, not `soft_query`, so it can check the constraints. It also sees each posting's structured fields via `filtered_judge.format_posting`.
  - The prompt says the description can show a field is wrong. This is what catches per-diem (PRN) postings that the data marks `FULL_TIME`.
- **`rerank`** uses `claude-haiku-5-5` with structured output (`Scores`, with `score: Literal[0, 1, 2, 3]`).
  - It sorts with `kind="stable"`, so tied postings keep their RRF order.
  - It raises an error if any posting comes back without a score.

### 4. Judging hard constraints (`filtered_judge.py`)

This is the same approach as the judge in `evaluate.py`: one Opus 5.5 request per query labels every pooled posting. The difference is that each posting gets two labels:

- **`relevant`:** the kind of job they asked for (role, duties, field).
- **`meets_constraints`:** it satisfies every hard requirement in the query.

Things to read:

- **`JUDGE_PROMPT`**: the two labels' definitions.
- **`format_posting`**: puts location, remote, experience level and work type above the posting text. Missing values are shown as "not stated", and the judge infers them from the description.
- **`judge`** and **`label_pools`**: mirror `evaluate.py`.

The judge decides constraints from the original query, never from the parsed `SearchFilter`, so a parsing mistake counts against the filter. Labels are cached in `filtered_labels.json`.

### 5. Comparison (`compare.py`)

`main` runs three methods on the 8 queries in `hard_filter.QUERIES`:

| Method | What it does |
|---|---|
| `no_filter` | weighted RRF over all postings on the raw query |
| `filter` | the filtered pipeline without the rerank, top 10 |
| `filter_rerank` | Claude's rerank of `filter`'s top 30, top 10 |

It then judges the union of all three top-10s (`filtered_judge.label_pools`) and scores two metrics:

- **`ndcg@10`** (`evaluate.ndcg_at_k`): a posting counts as relevant only if it is `relevant` **and** `meets_constraints`.
- **`violation_rate`**: the share of the top 10 with `meets_constraints == 0`.

`top_rows` returns the top-k row indices for a set of rows, using the same tiered ordering as `filtered_search`.

### Run it yourself

Each of these calls Claude. The Haiku calls cost fractions of a cent. The Opus judge in `compare.py` only labels postings missing from `filtered_labels.json`, so the first run costs the most.

```sh
# 1. See what the filter parser does with each test query, and how many postings each filter keeps.
uv run hard_filter.py

# 2. Run the whole pipeline on one query. Prints the parsed filter, then the top 20 with
#    each posting's rerank score, weighted RRF score, BM25 rank and embedding rank.
uv run filtered_search.py "entry level nursing jobs, full time"
uv run filtered_search.py "data analyst in New York, not an internship"

# 3. Or search interactively.
uv run filtered_search.py

# 4. Compare no_filter, filter and filter_rerank: per-query and mean nDCG@10 and violation rate.
uv run compare.py

# 5. See how the judge labeled one query's postings (job_id -> relevant, meets_constraints).
uv run python -c "import json; q = 'internship in software or data'; [print(k, v) for k, v in json.load(open('filtered_labels.json'))[q].items()]"
```

Things to try while reading the output:

- **Change `rrf_weights.json`** (e.g. `0.5` / `0.5`) and rerun `compare.py`. Only the `filter` and `no_filter` rankings change, and the rerank's top 30 may change with them. Any new postings get judged.
- **In `compare.py`'s output, compare `filter` with `filter_rerank`.** This shows where the rerank helps, typically where the data's own tags are wrong, such as per-diem jobs marked `FULL_TIME`.
- **In `filtered_search.py`'s output, look at the `rerank` column.** Lots of ties mean the order comes mostly from RRF.

Caveats when interpreting the numbers:

- There are only 8 queries.
- nDCG is measured against every posting judged relevant so far. As new runs judge more postings, scores for an unchanged method can move. Compare methods within one run, not across runs.
- The rerank prompt's scoring rules are close to the judge's definitions. Part of the rerank's gain therefore comes from optimising for what the judge measures, even though they use different models (Haiku vs Opus).
- The filter parse and the rerank aren't deterministic, so reruns can differ slightly.

## Ideas for next steps

- ~~Weighted RRF that gives embeddings more weight than BM25.~~ Done in `filtered_search.py`.
- Check the parsed filter against the description, not just the tags, since some tags are wrong (per-diem jobs marked `FULL_TIME`).
- Tighten location matching. `"New York"` currently also matches "Albany, New York Metropolitan Area".
- A better BM25 tokenizer, e.g. `re.findall(r"\w+", text.lower())`, so punctuation like "python," doesn't block matches.
- More natural-language queries, and embedding all 32,701 postings.

## Notes

- The generated files (`data-with-text.csv`, `data-subset.csv`, `embeddings.npy`, `labels.json`, `filtered_labels.json`) and `.env` are in `.gitignore`.
- On Windows, set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide a harmless Hugging Face warning.
