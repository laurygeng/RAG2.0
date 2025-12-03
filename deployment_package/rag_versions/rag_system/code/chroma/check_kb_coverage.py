#!/usr/bin/env python3
"""check_kb_coverage.py

Quick script to quantify whether the top-K retrieved passages (from results CSV)
contain or semantically support the ground-truth answer.

Usage:
  python3 check_kb_coverage.py path/to/results_with_retrieval.csv

Outputs a summary to stdout and writes a CSV with per-question flags to
deployment_package/rag_versions/rag_system/results/kb_coverage_<timestamp>.csv
"""

import json
import os
import sys
from datetime import datetime
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMB = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    HAS_EMB = False

def tokenize_simple(text):
    import re
    if not text or pd.isna(text):
        return []
    s = re.sub(r'[^0-9a-zA-Z]+', ' ', str(text).lower())
    toks = [t for t in s.split() if t]
    return toks

def load_passages(cell):
    if pd.isna(cell) or not cell:
        return []
    # try JSON
    try:
        docs = json.loads(cell)
        if isinstance(docs, list):
            return docs
    except Exception:
        pass
    # fallback split by ||
    try:
        return str(cell).split('||')
    except Exception:
        return [str(cell)]

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 check_kb_coverage.py path/to/results.csv')
        sys.exit(1)
    csvp = sys.argv[1]
    df = pd.read_csv(csvp)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_rows = []

    emb_model = None
    if HAS_EMB and SentenceTransformer is not None:
        try:
            emb_model = SentenceTransformer('all-MiniLM-L6-v2')
            print('Loaded embedding model for semantic checks.')
        except Exception:
            emb_model = None

    token_thresh = float(os.environ.get('KB_TOKEN_OVERLAP_THRESH', 0.5))
    sem_thresh = float(os.environ.get('KB_SEMANTIC_THRESH', 0.75))

    gt_in_token = 0
    gt_in_sem = 0
    total = 0

    for idx, row in df.iterrows():
        total += 1
        q = row.get('Question', '')
        gt = row.get('Ground_Truth_Answer', '')
        docs = load_passages(row.get('Retrieved_Passages', ''))

        gt_toks = set(tokenize_simple(gt))
        max_overlap = 0.0
        token_hit = False
        for d in docs:
            dtoks = set(tokenize_simple(d))
            if len(gt_toks) == 0:
                ratio = 0.0
            else:
                ratio = float(len(gt_toks & dtoks)) / float(len(gt_toks))
            if ratio > max_overlap:
                max_overlap = ratio
            if ratio >= token_thresh:
                token_hit = True

        sem_hit = False
        max_sem = 0.0
        if emb_model is not None and docs:
            try:
                g_emb = emb_model.encode(gt)
                d_embs = emb_model.encode(docs)
                sims = cosine_similarity([g_emb], d_embs)[0]
                max_sem = float(max(sims)) if len(sims) > 0 else 0.0
                if max_sem >= sem_thresh:
                    sem_hit = True
            except Exception:
                max_sem = 0.0

        if token_hit:
            gt_in_token += 1
        if sem_hit:
            gt_in_sem += 1

        out_rows.append({
            'Index': idx,
            'Question': q,
            'Ground_Truth': gt,
            'GT_in_TopK_TokenOverlap': int(token_hit),
            'Max_Token_Overlap': max_overlap,
            'GT_in_TopK_Semantic': int(sem_hit),
            'Max_Semantic_Similarity': max_sem,
            'Num_Retrieved': len(docs),
        })

    out_df = pd.DataFrame(out_rows)
    outp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', f'kb_coverage_{ts}.csv')
    out_df.to_csv(outp, index=False)

    print('\nKB coverage summary:')
    print(f'  Total questions checked: {total}')
    print(f'  GT present in top-K by token-overlap (threshold={token_thresh}): {gt_in_token} ({gt_in_token/total:.2%})')
    if emb_model is not None:
        print(f'  GT semantically present in top-K (threshold={sem_thresh}): {gt_in_sem} ({gt_in_sem/total:.2%})')
    else:
        print('  Embedding model unavailable; semantic checks skipped.')

    print(f'Per-question results written to: {outp}')

if __name__ == "__main__":
    main()
