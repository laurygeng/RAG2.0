#!/usr/bin/env python3
"""
rerank_experiment.py

Quick experiment: for each question in an augmented results CSV, fetch top-20 retrieved
passages (CSV-stored or from Chroma), rerank with a CrossEncoder (if available) and
compare Relevant_Retrieved_Proportion and Recall@5 before/after reranking.

Usage:
  python3 rerank_experiment.py /path/to/results_with_retrieval.csv

"""
import os
import sys
import json
from datetime import datetime

import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_ST = True
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    cosine_similarity = None
    HAS_ST = False


def load_chroma_collection():
    try:
        import chromadb
        from chromadb.config import Settings
        chroma_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'braincheck_vectordb')
        if os.path.exists(chroma_dir):
            settings = Settings(persist_directory=os.path.abspath(chroma_dir), is_persistent=True)
            client = chromadb.Client(settings=settings)
            try:
                col = client.get_collection('braincheck')
                return col
            except Exception:
                return None
    except Exception:
        return None
    return None


def tokenize(text):
    if pd.isna(text) or not text:
        return set()
    import re
    s = re.sub(r"[^0-9a-zA-Z]+", " ", str(text).lower())
    toks = [t for t in s.split() if t]
    return set(toks)


def lcs_length(a, b):
    # a, b: lists of tokens
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        for j in range(lb - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def is_relevant(doc, gt, gt_tokens, method='token_overlap', overlap_thresh=0.5, sem_thresh=0.6, embed_model=None):
    # returns True if doc is considered relevant to ground truth under chosen method
    if method == 'token_overlap':
        doc_toks = tokenize(doc)
        if len(gt_tokens) == 0:
            return False
        overlap = len(doc_toks & gt_tokens)
        return (overlap / len(gt_tokens)) >= overlap_thresh
    elif method == 'lcs':
        # LCS-based rope similar to ROUGE-L (token-level)
        a = list(gt_tokens)
        b = list(tokenize(doc))
        if len(a) == 0:
            return False
        lcs = lcs_length(a, b)
        return (lcs / len(a)) >= overlap_thresh
    elif method == 'semantic':
        if embed_model is None:
            return False
        try:
            import numpy as _np
            g_emb = embed_model.encode(gt)
            d_emb = embed_model.encode(doc)
            from sklearn.metrics.pairwise import cosine_similarity as _cs
            sim = float(_cs([g_emb], [d_emb])[0][0])
            return sim >= sem_thresh
        except Exception:
            return False
    else:
        return False


def count_relevant(topk_docs, gt, gt_tokens, method='token_overlap', overlap_thresh=0.5, sem_thresh=0.6, embed_model=None):
    if not topk_docs:
        return 0
    c = 0
    for doc in topk_docs:
        if is_relevant(doc, gt, gt_tokens, method=method, overlap_thresh=overlap_thresh, sem_thresh=sem_thresh, embed_model=embed_model):
            c += 1
    return c


def main(csv_path, top_k=20, relevance_method='token_overlap', overlap_thresh=0.5, sem_thresh=0.6):
    df = pd.read_csv(csv_path)
    # init models
    embed_model = None
    reranker = None
    if HAS_ST and SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            embed_model = None
    if HAS_ST and CrossEncoder is not None:
        try:
            # a small cross-encoder; may download if missing
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception:
            reranker = None

    chroma_col = load_chroma_collection()

    rows = []
    summaries = {'orig_relevant_prop': [], 'orig_recall5': [], 'rerank_relevant_prop': [], 'rerank_recall5': []}

    for idx, row in df.iterrows():
        q = row.get('Question', '')
        gt = row.get('Ground_Truth_Answer', '')
        gt_tokens = tokenize(gt)

        # get candidates: prefer CSV-stored JSON
        candidates = []
        if 'Retrieved_Passages' in row and pd.notna(row['Retrieved_Passages']):
            try:
                docs = json.loads(row['Retrieved_Passages'])
                if isinstance(docs, list):
                    candidates = docs[:top_k]
            except Exception:
                try:
                    docs = str(row['Retrieved_Passages']).split('||')
                    candidates = docs[:top_k]
                except Exception:
                    candidates = []
        # fallback to chroma
        if not candidates and chroma_col is not None and embed_model is not None:
            try:
                q_emb = embed_model.encode(q).tolist()
                res = chroma_col.query(query_embeddings=[q_emb], n_results=top_k, include=['documents'])
                if res and 'documents' in res and res['documents']:
                    candidates = [d for d in res['documents'][0]]
            except Exception:
                pass

        # original metrics (using the top_k candidates order)
        orig_topk = candidates[:top_k]
        orig_relevant_count = count_relevant(orig_topk, gt, gt_tokens, method=relevance_method, overlap_thresh=overlap_thresh, sem_thresh=sem_thresh, embed_model=embed_model)
        orig_relevant_prop = float(orig_relevant_count) / float(len(orig_topk)) if orig_topk else 0.0
        orig_recall5 = sum(1 for doc in orig_topk[:5] if is_relevant(doc, gt, gt_tokens, method=relevance_method, overlap_thresh=overlap_thresh, sem_thresh=sem_thresh, embed_model=embed_model)) if orig_topk and len(gt_tokens)>0 else 0

        # rerank
        reranked = orig_topk
        if reranker is not None and reranked:
            pairs = [[q, p] for p in orig_topk]
            try:
                scores = reranker.predict(pairs)
                scored = list(zip(orig_topk, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                reranked = [s[0] for s in scored]
            except Exception:
                reranked = orig_topk
        elif embed_model is not None and orig_topk:
            # fallback: compute embedding similarity between question and passage
            try:
                q_emb = embed_model.encode(q)
                p_embs = embed_model.encode(orig_topk)
                sims = cosine_similarity([q_emb], p_embs)[0]
                scored = list(zip(orig_topk, sims.tolist()))
                scored.sort(key=lambda x: x[1], reverse=True)
                reranked = [s[0] for s in scored]
            except Exception:
                reranked = orig_topk

        rerank_relevant_count = count_relevant(reranked, gt, gt_tokens, method=relevance_method, overlap_thresh=overlap_thresh, sem_thresh=sem_thresh, embed_model=embed_model)
        rerank_relevant_prop = float(rerank_relevant_count) / float(len(reranked)) if reranked else 0.0
        rerank_recall5 = sum(1 for doc in reranked[:5] if is_relevant(doc, gt, gt_tokens, method=relevance_method, overlap_thresh=overlap_thresh, sem_thresh=sem_thresh, embed_model=embed_model)) if reranked and len(gt_tokens)>0 else 0

        rows.append({
            'Index': idx,
            'Question': q,
            'Ground_Truth': gt,
            'Orig_TopK': len(orig_topk),
            'Orig_Relevant_Count': orig_relevant_count,
            'Orig_Relevant_Prop': orig_relevant_prop,
            'Orig_Recall_at_5': orig_recall5,
            'Rerank_Relevant_Count': rerank_relevant_count,
            'Rerank_Relevant_Prop': rerank_relevant_prop,
            'Rerank_Recall_at_5': rerank_recall5,
        })

        summaries['orig_relevant_prop'].append(orig_relevant_prop)
        summaries['orig_recall5'].append(orig_recall5)
        summaries['rerank_relevant_prop'].append(rerank_relevant_prop)
        summaries['rerank_recall5'].append(rerank_recall5)

    out_df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'rerank_experiment_{ts}.csv')
    out_df.to_csv(out_file, index=False)

    summary = {
        'orig_avg_relevant_prop': float(np.mean(summaries['orig_relevant_prop'])) if summaries['orig_relevant_prop'] else 0.0,
        'orig_avg_recall5': float(np.mean(summaries['orig_recall5'])) if summaries['orig_recall5'] else 0.0,
        'rerank_avg_relevant_prop': float(np.mean(summaries['rerank_relevant_prop'])) if summaries['rerank_relevant_prop'] else 0.0,
        'rerank_avg_recall5': float(np.mean(summaries['rerank_recall5'])) if summaries['rerank_recall5'] else 0.0,
    }

    print('Rerank experiment complete')
    print(f'Wrote per-question results to: {out_file}')
    print('Summary:')
    for k,v in summary.items():
        print(f'  {k}: {v}')

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='results csv with retrieval')
    parser.add_argument('top_k', nargs='?', type=int, default=20)
    parser.add_argument('--method', choices=['token_overlap', 'lcs', 'semantic'], default='token_overlap', help='relevance method')
    parser.add_argument('--overlap_thresh', type=float, default=0.5)
    parser.add_argument('--sem_thresh', type=float, default=0.6)
    args = parser.parse_args()
    main(args.csv, top_k=args.top_k, relevance_method=args.method, overlap_thresh=args.overlap_thresh, sem_thresh=args.sem_thresh)
