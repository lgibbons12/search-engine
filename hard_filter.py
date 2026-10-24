"""Turn a natural-language job search into hard filters with Claude, then apply them to data-subset.csv.

Steps:
  1. Claude reads the query and returns a SearchFilter (structured output).
  2. search_filter_mask turns the filter into a boolean mask over the postings.
  3. soft_query is what's left for BM25 / embedding ranking over the filtered rows.

Usage:
  uv run hard_filter.py   # parse every query in QUERIES and show how many postings each keeps
"""

import os
import re
from enum import StrEnum

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from embed_search import SUBSET_PATH

QUERIES = [
    {"query": "remote senior software engineer working with Python"},
    {"query": "entry level nursing jobs, full time"},
    {"query": "data analyst in New York, not an internship"},
    {"query": "part-time retail or customer service work"},
    {"query": "remote marketing role for someone with a few years of experience"},
    {"query": "full-time warehouse job, no nights"},
    {"query": "director level finance or accounting"},
    {"query": "internship in software or data"},
]


# Values match the dataset's formatted_experience_level column.
class ExperienceLevel(StrEnum):
    INTERNSHIP = "Internship"
    ENTRY = "Entry level"
    ASSOCIATE = "Associate"
    MID_SENIOR = "Mid-Senior level"
    DIRECTOR = "Director"
    EXECUTIVE = "Executive"


# Values match the dataset's work_type column.
class WorkType(StrEnum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    TEMPORARY = "TEMPORARY"
    INTERNSHIP = "INTERNSHIP"
    VOLUNTEER = "VOLUNTEER"
    OTHER = "OTHER"


class SearchFilter(BaseModel, frozen=True):
    remote: bool | None
    experience_levels: list[ExperienceLevel]
    work_types: list[WorkType]
    location: str | None
    exclude_terms: list[str]
    soft_query: str


FILTER_MODEL = "claude-haiku-5-5"

FILTER_PROMPT = """You turn a job seeker's search into filters for a job search engine.

The search: "{query}"

Hard filters remove postings outright, so only set one when the search clearly asks for it. \
Wrongly removing good jobs is worse than letting a few extra through. If you're unsure, leave the \
filter empty and keep the wording in soft_query, which is used to rank whatever passes the filters.

Fields:

- remote: true if they ask for remote work, false if they explicitly ask for on-site or \
in-office work, otherwise null. Hybrid is null.

- experience_levels: the seniority levels to keep. Empty if they don't mention seniority. \
Postings use LinkedIn's levels, so map everyday wording onto them:
  - "intern", "internship" -> Internship
  - "entry level", "junior", "new grad", "no experience" -> Entry level
  - "a few years of experience", "mid-level" -> Associate and Mid-Senior level
  - "senior", "experienced", "lead" -> Mid-Senior level
  - "director" -> Director
  - "VP", "head of", "C-level", "executive" -> Executive
  To exclude a level ("not an internship"), list every other level.
  A job title such as "manager" or "engineer" is not a seniority level; leave it to soft_query.

- work_types: the employment types to keep. Empty if they don't mention one. "full time" -> \
FULL_TIME, "part time" -> PART_TIME, "contract" or "freelance" -> CONTRACT, "temp" or "seasonal" \
-> TEMPORARY. An internship goes in experience_levels, not here. To exclude a type, list every \
other type.

- location: the place they want to work, or null. It's matched as a case-insensitive substring of \
the posting's location, which looks like "Chicago, IL", "New York, NY" or "Atlanta Metropolitan \
Area". Use the shortest name that will match: "New York", not "NYC" or "New York City". For a \
whole state, use ", " plus its two-letter code, like ", TX". "Remote" is not a location.

- exclude_terms: words or short phrases that mark postings they don't want. Each is matched as a \
whole word, case-insensitively, against the posting's title and description, and any match \
removes the posting. Pick terms a posting would actually use and that mostly appear in unwanted \
postings: for "no nights", ["night shift", "overnight"], not "nights", which also appears in \
"no nights or weekends". Empty if they don't rule anything out. Don't repeat what \
experience_levels or work_types already handle.

- soft_query: the search rewritten as the words a matching posting would contain: the role, \
skills and field. Drop what the filters already handle (remote, location, work type, exclusions) \
and words that don't describe the job ("jobs", "work", "role", "someone"). Keep seniority words \
like "senior", since some postings have no level and rely on ranking. Never empty.

Example: "on-site welding job in Texas, no contract work" -> remote false, experience_levels [], \
work_types every type except CONTRACT, location ", TX", exclude_terms [], soft_query "welder \
welding"."""


def parse_search_filter(client: anthropic.Anthropic, query: str) -> SearchFilter:
    """Ask Claude to turn one search query into a SearchFilter."""
    response = client.messages.parse(
        model=FILTER_MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": FILTER_PROMPT.format(query=query)}],
        output_format=SearchFilter,
    )
    if not response or not response.parsed_output:
        raise ValueError("Could not parse")
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Filter parse refused query {query!r}: {response.stop_details}")
    return response.parsed_output


def experience_level(df: pd.DataFrame) -> pd.Series:
    """Each posting's experience level, counting an INTERNSHIP work type as the Internship level."""
    return df["formatted_experience_level"].mask(df["work_type"] == WorkType.INTERNSHIP.value, ExperienceLevel.INTERNSHIP.value)


def unknown_level(df: pd.DataFrame, search_filter: SearchFilter) -> pd.Series:
    """True for postings that only pass the level filter because they have no level.

    Rank these after the postings whose level is known to match, so they fill in rather than crowd out real matches.
    """
    if not search_filter.experience_levels:
        return pd.Series(False, index=df.index)
    return experience_level(df).isna()


def search_filter_mask(df: pd.DataFrame, search_filter: SearchFilter) -> pd.Series:
    """Boolean mask of the postings that pass every hard filter. Empty filters keep everything."""
    mask = pd.Series(True, index=df.index)
    if search_filter.remote is not None: # type: ignore
        mask &= (df["remote_allowed"] == 1) == search_filter.remote
    if search_filter.experience_levels:
        # About 20% of postings have no level; keep them (see unknown_level) rather than drop them for missing data.
        level = experience_level(df)
        mask &= level.isna() | level.isin([l.value for l in search_filter.experience_levels])
    if search_filter.work_types:
        mask &= df["work_type"].isin([w.value for w in search_filter.work_types])
    if search_filter.location:
        mask &= df["location"].str.contains(search_filter.location, case=False, regex=False, na=False)
    if search_filter.exclude_terms:
        pattern = r"\b(?:" + "|".join(map(re.escape, search_filter.exclude_terms)) + r")\b"
        mask &= ~df["text"].str.contains(pattern, case=False, regex=True, na=False)
    return mask


def main():
    df = pd.read_csv(SUBSET_PATH)
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_TOKEN") or os.getenv("ANTHROPIC_API_KEY"))

    for q in QUERIES:
        search_filter = parse_search_filter(client, q["query"])
        mask = search_filter_mask(df, search_filter)
        print(f"\n{q['query']}")
        print(f"  {search_filter.model_dump_json()}")
        print(f"  keeps {mask.sum():,} of {len(df):,} postings")


if __name__ == "__main__":
    main()
