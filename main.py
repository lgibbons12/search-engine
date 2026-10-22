import sys
import textwrap

import pandas as pd

from preprocessing import OUTPUT_PATH

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

LIST_COLS = ["title", "location", "formatted_experience_level", "formatted_work_type", "max_salary", "pay_period"]
DETAIL_COLS = [
    "title", "location", "formatted_experience_level", "formatted_work_type", "remote_allowed",
    "min_salary", "med_salary", "max_salary", "pay_period", "currency", "views", "applies",
    "listed_time", "job_posting_url", "application_url",
]

HELP = """
Commands:
  overview                  shape, columns, dtypes, missing values
  top <column> [n]          most common values in a column (e.g. top location 15)
  search <text>             find postings whose title or description contains <text>
  filter <column> <text>    narrow current results to rows where <column> contains <text>
  remote                    narrow current results to remote-allowed postings
  salary                    salary stats for current results, grouped by pay period
  list [n]                  show the first n current results (default 20)
  show <row>                full details + description for one posting (row index from list)
  reset                     clear search/filters
  help                      show this message
  quit                      exit
"""


def load() -> pd.DataFrame:
    try:
        return pd.read_csv(OUTPUT_PATH, parse_dates=["listed_time", "original_listed_time", "expiry", "closed_time"])
    except FileNotFoundError:
        sys.exit(f"{OUTPUT_PATH} not found. Run `uv run preprocessing.py` first.")


def overview(df: pd.DataFrame) -> None:
    print(f"{len(df):,} rows x {len(df.columns)} columns\n")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
    })
    print(summary.to_string())


def top(df: pd.DataFrame, col: str, n: int = 10) -> None:
    if col not in df.columns:
        print(f"Unknown column '{col}'. Try 'overview' to see columns.")
        return
    counts = df[col].value_counts(dropna=False).head(n)
    print(counts.to_string())


def contains(df: pd.DataFrame, col: str, text: str) -> pd.Series:
    return df[col].astype(str).str.contains(text, case=False, regex=False, na=False)


def salary(df: pd.DataFrame) -> None:
    has_pay = df[df["max_salary"].notna() | df["med_salary"].notna()]
    print(f"{len(has_pay):,} of {len(df):,} postings list a salary\n")
    if has_pay.empty:
        return
    stats = has_pay.groupby("pay_period")[["min_salary", "med_salary", "max_salary"]].agg(["count", "median", "mean"])
    print(stats.round(0).to_string())


def show(df: pd.DataFrame, idx: int) -> None:
    if idx not in df.index:
        print(f"Row {idx} isn't in the current results.")
        return
    row = df.loc[idx]
    for col in DETAIL_COLS:
        if pd.notna(row[col]):
            print(f"{col:>28}: {row[col]}")
    for col in ["skills_desc", "description"]:
        if pd.notna(row[col]):
            print(f"\n--- {col} ---")
            print(textwrap.fill(str(row[col])[:3000], width=100))


def main():
    print("Loading dataset...")
    df = load()
    current = df
    print(f"Loaded {len(df):,} job postings. Type 'help' for commands.")
    print(HELP)

    while True:
        try:
            line = input(f"[{len(current):,} rows] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()

        try:
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "overview":
                overview(current)
            elif cmd == "top":
                col, _, n = arg.partition(" ")
                top(current, col, int(n) if n else 10)
            elif cmd == "search":
                current = current[contains(current, "title", arg) | contains(current, "description", arg)]
                print(f"{len(current):,} matches")
            elif cmd == "filter":
                col, _, text = arg.partition(" ")
                if col not in current.columns:
                    print(f"Unknown column '{col}'.")
                    continue
                current = current[contains(current, col, text)]
                print(f"{len(current):,} matches")
            elif cmd == "remote":
                current = current[current["remote_allowed"] == 1]
                print(f"{len(current):,} matches")
            elif cmd == "salary":
                salary(current)
            elif cmd == "list":
                print(current[LIST_COLS].head(int(arg) if arg else 20).to_string())
            elif cmd == "show":
                show(current, int(arg))
            elif cmd == "reset":
                current = df
            else:
                print(f"Unknown command '{cmd}'. Type 'help'.")
        except ValueError as e:
            print(f"Bad argument: {e}")




if __name__ == "__main__":
    main()
