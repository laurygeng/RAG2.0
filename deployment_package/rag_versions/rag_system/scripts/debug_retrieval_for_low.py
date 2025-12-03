#!/usr/bin/env python3
"""
Debug retrieval: for a given Question_ID, fetch top-10 candidates via EnsembleRetriever
and write a small report to `results/` for inspection.

Run: python3 scripts/debug_retrieval_for_low.py --qid AA352
"""
import os
import csv
import json
import argparse
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_CSV = os.path.join(BASE_DIR, 'results', 'real_answers_20251118_114302.csv')
OUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def find_question(qid):
    with open(RESULTS_CSV, encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('Question_ID') == qid:
                return row.get('Question')
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qid', required=True)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()

    qid = args.qid
    question = find_question(qid)
    if not question:
        print('Question ID not found in results CSV:', qid)
        return

    print('Found question for', qid)

    # Import generator class from file path (avoid package import issues)
    try:
        import importlib.util
        ga_path = os.path.join(BASE_DIR, 'code', 'generate_answers.py')
        spec = importlib.util.spec_from_file_location('generate_answers', ga_path)
        ga = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ga)
        RealModelAnswerGenerator = getattr(ga, 'RealModelAnswerGenerator')
    except Exception as e:
        print('Failed to load RealModelAnswerGenerator from file:', e)
        return

    gen = RealModelAnswerGenerator()
    # Ensure KB loading enabled (generator honors env var)
    os.environ['RAG_ENABLE_LOAD_DATA'] = os.environ.get('RAG_ENABLE_LOAD_DATA','1')

    try:
        passages, sources, distances = gen.retrieve_passages_via_ensemble(question, top_k=args.topk, truncate=None)
    except Exception as e:
        print('Ensemble retrieval failed:', e)
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_md = os.path.join(OUT_DIR, f'retrieval_debug_{qid}_{ts}.md')
    with open(out_md, 'w', encoding='utf-8') as mf:
        mf.write(f'# Retrieval debug for {qid} — {ts}\n\n')
        mf.write('Question:\n')
        mf.write('```\n')
        mf.write(question + '\n')
        mf.write('```\n\n')
        mf.write(f'Top {args.topk} passages returned by EnsembleRetriever:\n\n')
        for i,(p,src) in enumerate(zip(passages, sources), start=1):
            mf.write(f'## Passage {i} — source: `{src}`\n')
            mf.write('```\n')
            mf.write((p or '')[:4000] + '\n')
            mf.write('```\n\n')

    print('Wrote report to', out_md)


if __name__ == '__main__':
    main()
