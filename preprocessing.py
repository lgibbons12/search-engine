"""Build data-with-text.csv from the LinkedIn job postings dataset.

Steps:
  1. Load xanderios/linkedin-job-postings from Hugging Face.
  2. Convert the epoch-millisecond time columns to datetimes.
  3. Build a `text` column: "<title> in <location>. Description: <first 1000 chars of description>".
     (company_id is a numeric ID with no name lookup, so it's left out.)
  4. Drop rows whose `text` duplicates an earlier row (reposted jobs).
  5. Save to data-with-text.csv.
"""

import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = "data-with-text.csv"
DESCRIPTION_CHARS = 1000


def load() -> pd.DataFrame:
    df = load_dataset("xanderios/linkedin-job-postings")["train"].to_pandas()
    for col in ["listed_time", "original_listed_time", "expiry", "closed_time"]:
        df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def add_text(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description"].fillna("").str[:DESCRIPTION_CHARS]
    df["text"] = df["title"] + " in " + df["location"] + ". Description: " + desc
    return df


def main():
    print("Loading dataset...")
    df = load()
    print(f"Loaded {len(df):,} rows")

    df = add_text(df)
    before = len(df)
    df = df.drop_duplicates("text").reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate texts, {len(df):,} rows remain")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
