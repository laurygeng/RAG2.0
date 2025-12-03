#!/usr/bin/env python3
"""
Select up to N questions from ground_truth_answers.csv where Answer Type == 'Answer - Reviewed'.
Pick at most one question per Category to ensure different types.
Save to data/selected_questions_<timestamp>.csv with columns:
Question ID,Category,Question,Answer,Answer Type

Usage:
  python3 select_reviewed_questions.py [--n 10] [--seed 0] [--in PATH] [--outdir PATH]
"""
import argparse
import os
import pandas as pd
from datetime import datetime
import random

DEFAULT_IN = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ground_truth_answers.csv')
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def select_questions(in_path, out_dir, n=10, seed=None):
    if seed is not None:
        random.seed(seed)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)

    # normalize column names if needed
    # We expect columns: 'Question ID','Category','Question','Answer','Answer Type'
    # but ground_truth_answers.csv uses 'Question ID','Category','Question','Answer','Provider','Notes','Answer Type'

    # Filter by Answer Type == 'Answer - Reviewed'
    reviewed = df[df['Answer Type'].astype(str).str.strip() == 'Answer - Reviewed'].copy()

    if reviewed.empty:
        print('No entries found with Answer Type == "Answer - Reviewed".')
        return None

    # Find unique categories
    categories = reviewed['Category'].dropna().astype(str).unique().tolist()
    random.shuffle(categories)
    k = min(n, len(categories))
    chosen_cats = categories[:k]

    selected_rows = []
    for cat in chosen_cats:
        rows = reviewed[reviewed['Category'].astype(str) == str(cat)]
        if rows.empty:
            continue
        # pick one row at random from this category
        row = rows.sample(n=1, random_state=seed).iloc[0]
        selected_rows.append({
            'Question ID': row.get('Question ID', ''),
            'Category': row.get('Category', ''),
            'Question': row.get('Question', ''),
            'Answer': row.get('Answer', ''),
            'Answer Type': row.get('Answer Type', '')
        })

    out_df = pd.DataFrame(selected_rows, columns=['Question ID','Category','Question','Answer','Answer Type'])

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'selected_questions_{timestamp}.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Generated file: {out_path} (total {len(out_df)} entries)')
    return out_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='Number of questions to pick (distinct categories)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional, for reproducibility)')
    parser.add_argument('--in', dest='in_path', default=DEFAULT_IN, help='Input ground_truth_answers.csv path')
    parser.add_argument('--outdir', default=DEFAULT_OUT_DIR, help='Output directory (data/)')
    args = parser.parse_args()

    try:
        select_questions(args.in_path, args.outdir, n=args.n, seed=args.seed)
    except Exception as e:
        print('Error during execution:', e)
        raise
