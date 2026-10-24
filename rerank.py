"""LLM rerank: Claude scores the top RERANK_K results of filtered search, and they're re-sorted by score.

Claude sees the raw query (not soft_query), so it can check the constraints too, and each
posting's structured fields alongside its text. Scores are coarse (0-3), so ties are common;
tied postings keep their weighted RRF order.
"""

from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel

from filtered_judge import format_posting

RERANK_MODEL = "claude-haiku-5-5"
RERANK_K = 30

RERANK_PROMPT = """You are reranking search results for a job search engine.

A job seeker searched for: "{query}"

Score each job posting below for how well it matches the search:
  3: the kind of job they asked for, and it meets every requirement the search states
  2: the kind of job they asked for, but the posting doesn't say whether it meets one of the requirements
  1: a related job they might still consider, and it meets the requirements
  0: not the kind of job they asked for, or it breaks a stated requirement

Requirements are things like remote or on-site, seniority or experience level, employment type \
(full time, part time, internship, contract), location, and anything they explicitly rule out. \
Judge the job itself (role, duties, level), not just shared words. Use the listed fields first; \
where a field says "not stated", go by the title and description, which can also show that a \
listed field is wrong (a "full time" posting that says it's per diem, for example).

{postings}

Return a score for every posting, using its job_id."""


class Score(BaseModel):
    job_id: int
    score: Literal[0, 1, 2, 3]


class Scores(BaseModel):
    scores: list[Score]


def rerank(client: anthropic.Anthropic, query: str, postings: pd.DataFrame) -> pd.DataFrame:
    """Re-sort postings (given in retrieval order) by Claude's score, adding a `rerank` column."""
    listing = "\n\n".join(format_posting(row) for row in postings.itertuples())
    response = client.messages.parse(
        model=RERANK_MODEL,
        max_tokens=8000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": RERANK_PROMPT.format(query=query, postings=listing)}],
        output_format=Scores,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Reranker refused query {query!r}: {response.stop_details}")

    scores = {str(s.job_id): s.score for s in response.parsed_output.scores}
    missing = set(postings["job_id"].astype(str)) - scores.keys()
    if missing:
        raise RuntimeError(f"Reranker skipped {len(missing)} postings for query {query!r}")

    reranked = postings.copy()
    reranked.insert(0, "rerank", reranked["job_id"].astype(str).map(scores))
    return reranked.sort_values("rerank", ascending=False, kind="stable")
