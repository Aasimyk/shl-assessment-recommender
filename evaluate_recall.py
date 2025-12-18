# evaluate_recall.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pandas as pd

from app.main import _search_catalog


ROOT = Path(__file__).resolve().parent
GEN_AI_XLSX = ROOT / "data" / "Gen_AI Dataset.xlsx"
OUTPUT_CSV = ROOT / "data" / "gen_ai_eval_predictions.csv"


def load_gen_ai_queries() -> List[str]:
    """Load UNIQUE queries from Gen_AI Dataset.xlsx (expects a 'Query' column)."""
    df = pd.read_excel(GEN_AI_XLSX)

    # Normalise column names (strip + lower)
    rename_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    if "query" not in df.columns:
        raise ValueError(
            f"Expected a 'Query' column in {GEN_AI_XLSX}, got: {list(df.columns)}"
        )

    # Drop empty and duplicate queries so we end up with unique ones (should be 9)
    df = df.dropna(subset=["query"])
    df = df.drop_duplicates(subset=["query"])

    queries = df["query"].astype(str).tolist()
    return [q.strip() for q in queries if q.strip()]


def main() -> None:
    queries = load_gen_ai_queries()
    top_k = 10

    print(f"Loaded {len(queries)} unique queries from Gen_AI Dataset.xlsx\n")

    out_rows = []

    for q in queries:
        results = _search_catalog(q, top_k=top_k)
        urls = [r["url"] for r in results]

        print(f"Query: {q}")
        for rank, url in enumerate(urls, start=1):
            print(f"  {rank}. {url}")
        print()

        row = {"Query": q}
        for i in range(top_k):
            col_name = f"Assessment{i+1}url"
            row[col_name] = urls[i] if i < len(urls) else ""
        out_rows.append(row)

    fieldnames = ["Query"] + [f"Assessment{i+1}url" for i in range(top_k)]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"Saved predictions for {len(out_rows)} queries to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
