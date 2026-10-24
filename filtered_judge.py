"""Judge search results for relevance and for the query's hard constraints.

Same approach as the judge in evaluate.py (one Claude request labels every pooled posting for a
query), with a second label per posting:
  relevant:          is it the kind of job they asked for (role, duties, field)?
  meets_constraints: does it satisfy every hard requirement in the query (remote, level, work
                     type, location, exclusions)?
The judge sees the posting's structured fields as well as its text, and decides constraints from
the query itself, not from the parsed SearchFilter, so filter mistakes count against the filter.

Labels are cached in filtered_labels.json as {query: {job_id: {"relevant": 0|1, "meets_constraints": 0|1}}},
so reruns only pay for postings that haven't been labeled yet.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
from pydantic import BaseModel

from evaluate import JUDGE_MODEL

LABELS_PATH = "filtered_labels.json"

JUDGE_PROMPT = """You are judging search results for a job search engine.

A job seeker searched for: "{query}"

For each job posting below, give two labels.

relevant: would this job seeker plausibly want to see it in their results, judging only the kind \
of job? Judge the job itself (role, duties, field), not just shared words. A posting is relevant \
if it is the kind of job they asked for, even if it uses different words. Leave the requirements \
covered by meets_constraints out of this label.

meets_constraints: does the posting satisfy every hard requirement the search states? These are \
remote or on-site, seniority or experience level, employment type (full time, part time, \
internship, contract), location, and anything they explicitly rule out (like "no nights" or "not \
an internship"). Use the listed fields first; where a field says "not stated", infer it from the \
title and description. Mark false if the posting clearly or very likely breaks a requirement, \
true otherwise. If the search states no hard requirements, mark true.

{postings}

Return both labels for every posting, using its job_id."""


class Label(BaseModel):
    job_id: int
    relevant: bool
    meets_constraints: bool


class Labels(BaseModel):
    labels: list[Label]


def format_posting(row) -> str:
    def field(value) -> str:
        return "not stated" if pd.isna(value) else str(value)

    return (
        f"<posting job_id=\"{row.job_id}\">\n"
        f"Location: {field(row.location)}\n"
        f"Remote allowed: {'yes' if row.remote_allowed == 1 else 'not stated'}\n"
        f"Experience level: {field(row.formatted_experience_level)}\n"
        f"Work type: {field(row.formatted_work_type)}\n"
        f"{row.text}\n"
        f"</posting>"
    )


def load_labels() -> dict[str, dict[str, dict[str, int]]]:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def judge(client: anthropic.Anthropic, query: str, postings: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Ask Claude to label every posting for one query in a single request."""
    listing = "\n\n".join(format_posting(row) for row in postings.itertuples())
    response = client.beta.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(query=query, postings=listing)}],
        output_format=Labels,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Judge refused query {query!r}: {response.stop_details}")

    labels = {
        str(l.job_id): {"relevant": int(l.relevant), "meets_constraints": int(l.meets_constraints)}
        for l in response.parsed_output.labels
    }
    missing = set(postings["job_id"].astype(str)) - labels.keys()
    if missing:
        raise RuntimeError(f"Judge skipped {len(missing)} postings for query {query!r}")
    return labels


def label_pools(client: anthropic.Anthropic, df: pd.DataFrame,
                pools: dict[str, list[int]]) -> dict[str, dict[str, dict[str, int]]]:
    """Label every pooled row (by row index into df) that isn't already in the cache."""
    labels = load_labels()
    todo = {}
    for query, rows in pools.items():
        known = labels.get(query, {})
        unlabeled = df.iloc[rows]
        unlabeled = unlabeled[~unlabeled["job_id"].astype(str).isin(known)]
        if len(unlabeled):
            todo[query] = unlabeled

    if todo:
        total = sum(len(p) for p in todo.values())
        print(f"Labeling {total} postings across {len(todo)} queries with {JUDGE_MODEL}...")
        with ThreadPoolExecutor(max_workers=5) as pool_exec:
            results = pool_exec.map(lambda item: (item[0], judge(client, *item)), todo.items())
            for query, new in results:
                labels.setdefault(query, {}).update(new)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved labels to {LABELS_PATH}")
    else:
        print(f"All pooled postings already labeled in {LABELS_PATH}")
    return labels
