#!/usr/bin/env python3
"""
Read from the questions directory (sibling of the code folder):
    "QuestionsAnswers_A2_10172025 - Finalized Results (Partial).csv"
Filter rows where Answer Type == "Answer - Reviewed" and aim for category diversity (default 1 per Category),
then export mapped columns: Question ID, Category, Question, Answer (Question=Question - Reviewed, Answer=Answer - Reviewed).

Output: deployment_package/rag_versions/rag_system/data/selected_questions_YYYYMMDD_HHMMSS.csv

Optional environment variables:
    SELECT_PER_CATEGORY  Number of items to select per category (default 1)
    SELECT_TOTAL         Max total items (default 0 = no limit)
    SELECT_SHUFFLE       Shuffle categories (1 to enable)
    SELECT_SEED          Seed for shuffle (optional)
"""
from __future__ import annotations

import os
import time
import pandas as pd


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    questions_dir = os.path.join(root, "questions")
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    filename = "QuestionsAnswers_A2_10172025 - Finalized Results (Partial).csv"
    csv_path = os.path.join(questions_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Question file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Required column names
    col_qid = "Question ID"
    col_cat = "Category"
    col_q = "Question - Reviewed"
    col_a = "Answer - Reviewed"

    for c in [col_qid, col_cat, col_q, col_a]:
        if c not in df.columns:
            raise KeyError(f"CSV missing required column: {c}. Actual columns: {list(df.columns)}")

    # Keep entries that have reviewed answers only
    df = df.copy()
    # First filter non-null, then strip blanks/"nan" strings
    df = df[df[col_a].notna() & df[col_q].notna()]
    df[col_a] = df[col_a].astype(str).map(lambda x: x.strip())
    df[col_q] = df[col_q].astype(str).map(lambda x: x.strip())
    invalid = {"", "nan", "none", "null"}
    df = df[~df[col_a].str.lower().isin(invalid) & ~df[col_q].str.lower().isin(invalid)]

    # Try to take some per category (default 1)
    per_category = int(os.environ.get("SELECT_PER_CATEGORY", "1") or 1)
    total_limit = int(os.environ.get("SELECT_TOTAL", "0") or 0)  # 0 means no limit
    shuffle = os.environ.get("SELECT_SHUFFLE", "0") == "1"
    seed_env = os.environ.get("SELECT_SEED")
    seed = int(seed_env) if seed_env is not None and seed_env != "" else None

    groups = list(df.groupby(col_cat))
    # For stable output, sort by category by default; enable SELECT_SHUFFLE=1 for random diversity
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(groups)
    else:
        groups.sort(key=lambda kv: str(kv[0]))

    selected_parts = []
    for cat, g in groups:
        g_sorted = g.sort_values(by=[col_qid]).head(per_category)
        selected_parts.append(g_sorted[[col_qid, col_cat, col_q, col_a]])
    # If total limit is set, stop when the concatenated items reach the limit
        if total_limit and sum(len(x) for x in selected_parts) >= total_limit:
            break

    if selected_parts:
        out_df = pd.concat(selected_parts, axis=0, ignore_index=True)
    else:
        out_df = df[[col_qid, col_cat, col_q, col_a]].copy()
    if total_limit:
        out_df = out_df.head(total_limit)

    # Rename and export columns
    out_df = out_df.rename(columns={
        col_qid: "Question ID",
        col_cat: "Category",
        col_q: "Question",
        col_a: "Answer",
    })

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(data_dir, f"selected_questions_{ts}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Exported: {out_path}  (rows {len(out_df)}, categories {out_df['Category'].nunique()})")


if __name__ == "__main__":
    main()
