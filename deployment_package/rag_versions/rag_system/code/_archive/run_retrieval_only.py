#!/usr/bin/env python3
"""Run retrieval only for the first N questions and save results.

This avoids contacting the Ollama server and only exercises vector/keyword retrieval.
"""
import os
import sys
import json
from datetime import datetime
import pandas as pd

# Ensure code folder on path
sys.path.insert(0, os.path.dirname(__file__))
from generate_answers import RealModelAnswerGenerator


def main(n=10, top_k=3):
    # ensure KB loading is enabled (keyword fallback will load local KB)
    os.environ['RAG_ENABLE_LOAD_DATA'] = os.environ.get('RAG_ENABLE_LOAD_DATA', '1')

    # Avoid initializing Chroma client if the on-disk chroma folder exists but
    # chromadb/rust bindings are incompatible in this environment. We monkey-patch
    # os.path.exists temporarily so the generator doesn't try to open the local
    # `data/braincheck_vectordb` directory. This lets us exercise the keyword
    # retrieval fallback safely.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chroma_dir = os.path.join(base_dir, 'data', 'braincheck_vectordb')
    _orig_exists = os.path.exists
    def _patched_exists(p):
        try:
            if os.path.abspath(p) == os.path.abspath(chroma_dir):
                return False
        except Exception:
            pass
        return _orig_exists(p)

    os.path.exists = _patched_exists
    try:
        gen = RealModelAnswerGenerator()
    finally:
        os.path.exists = _orig_exists

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    qfile = os.path.join(base_dir, 'data', 'selected_questions.csv')
    if not os.path.exists(qfile):
        print('Questions file not found:', qfile)
        return 2

    df = pd.read_csv(qfile)
    df = df.head(n)

    rows = []
    for idx, row in df.iterrows():
        qid = row.get('Question ID') if 'Question ID' in row else row.get('Question_ID', idx)
        question = row['Question']
        passages, scores, distances = gen.get_retrieved_passages(question, top_k=top_k, truncate=300)
        rows.append({
            'Question_ID': qid,
            'Question': question,
            'Retrieved_Passages': json.dumps(passages, ensure_ascii=False),
            'Retrieved_Scores': json.dumps(scores, ensure_ascii=False),
            'Retrieved_Distances': json.dumps(distances, ensure_ascii=False),
            'Retrieved_Count': len(passages),
        })

    out_dir = os.path.join(base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(out_dir, f'retrieval_only_{ts}.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print('Wrote retrieval results to:', outpath)
    return 0


if __name__ == '__main__':
    sys.exit(main())
