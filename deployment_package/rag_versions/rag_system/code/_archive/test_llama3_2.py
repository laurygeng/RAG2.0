#!/usr/bin/env python3
"""Regenerate timed-out rows using mistral:7b-instruct and top_k=3.

Reads the latest `real_answers_*.csv` in ../results, finds rows where
`Real_LLaMA_Answer` contains 'Timed out', and for each row calls the
generator to fetch top_k=3 passages and ask `mistral:7b-instruct` to
generate an answer. Results are written to a new CSV alongside the
original file (timestamped).
"""
import os
import sys
import pandas as pd
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA_DIR = os.path.join(ROOT, 'data')

def find_latest_results():
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('real_answers_') and f.endswith('.csv')]
    if not files:
        return None
    files = sorted(files)
    return os.path.join(RESULTS_DIR, files[-1])

def load_generator():
    # import the class from generate_answers (safe because main runs under __main__)
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_answers import RealModelAnswerGenerator
    return RealModelAnswerGenerator()

def main(results_path=None, top_k=3, timeout_override=None):
    if results_path is None:
        results_path = find_latest_results()
    if results_path is None:
        print('No results CSV found in', RESULTS_DIR)
        return 2

    print('Using results file:', results_path)
    df = pd.read_csv(results_path)

    # identify timed-out rows (based on Real_LLaMA_Answer containing 'Timed out')
    mask = df['Real_LLaMA_Answer'].astype(str).str.contains('Timed out', case=False, na=False)
    if not mask.any():
        print('No timed-out rows found in the CSV. Nothing to do.')
        return 0

    timed_out_df = df.loc[mask].copy()
    print(f'Found {len(timed_out_df)} timed-out rows; regenerating with mistral:7b-instruct (top_k={top_k})')

    gen = load_generator()
    # optionally override timeouts via env before calling generator
    if timeout_override:
        os.environ['RAG_GEN_REQUEST_TIMEOUT'] = str(timeout_override.get('request', gen.request_timeout))
        os.environ['RAG_GEN_OVERALL_TIMEOUT'] = str(timeout_override.get('overall', gen.overall_timeout))

    regenerated_answers = {}
    # load questions reference
    qfile = os.path.join(DATA_DIR, 'selected_questions.csv')
    if not os.path.exists(qfile):
        print('Questions file not found:', qfile)
        return 3
    qdf = pd.read_csv(qfile)

    for i, row in timed_out_df.iterrows():
        qid = row.get('Question_ID') or row.get('Question ID') or i
        # find question row in qdf
        qrows = qdf[qdf['Question ID'].astype(str) == str(qid)]
        if qrows.empty:
            # try matching by index
            try:
                qrow = qdf.iloc[int(i)]
            except Exception:
                print(f'Could not locate question for Question_ID={qid} at index {i}; skipping')
                continue
        else:
            qrow = qrows.iloc[0]

        question = qrow['Question']

        # fetch top_k passages and scores
        passages, scores = gen.get_retrieved_passages(question, top_k=top_k, truncate=300)
        if passages:
            numbered = []
            for j, p in enumerate(passages, start=1):
                numbered.append(f'[{j}] {p}')
            context = '\n\n'.join(numbered)
        else:
            context = gen.get_retrieved_context(question, top_k=top_k, truncate=300)

        print(f'Regenerating QID={qid} (index={i}) — passages={len(passages)}')
        answer = gen.generate_answer(gen.mistral_model, question, context=context, max_retries=3)

        regenerated_answers[i] = {
            'Question_ID': qid,
            'Regenerated_Answer': answer,
            'Regenerated_Model': gen.mistral_model,
            'Regenerated_TopK': top_k,
            'Retrieved_Passages': passages,
            'Retrieved_Scores': scores,
        }

    # write results back to a new CSV — keep original df but add regeneration columns
    out_df = df.copy()
    # create regeneration columns with defaults
    out_df['Regenerated_Model'] = out_df.get('Regenerated_Model', '')
    out_df['Regenerated_Answer'] = out_df.get('Regenerated_Answer', '')
    out_df['Regenerated_TopK'] = out_df.get('Regenerated_TopK', '')

    for idx, meta in regenerated_answers.items():
        out_df.at[idx, 'Regenerated_Model'] = meta['Regenerated_Model']
        out_df.at[idx, 'Regenerated_Answer'] = meta['Regenerated_Answer']
        out_df.at[idx, 'Regenerated_TopK'] = meta['Regenerated_TopK']
        # also update Real_Mistral_Answer so downstream evaluator can see it
        if 'Real_Mistral_Answer' in out_df.columns:
            out_df.at[idx, 'Real_Mistral_Answer'] = meta['Regenerated_Answer']

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(RESULTS_DIR, f"real_answers_{ts}_regenerated_mistral_topk{top_k}.csv")
    out_df.to_csv(outpath, index=False)
    print('Regeneration complete. Wrote:', outpath)
    return 0

def test_llama3_2(indices, results_path, timeout_override=120):
    """Test LLaMA3.2 with RealModelAnswerGenerator for specific indices."""
    generator = load_generator()

    # Override timeout
    generator.overall_timeout = timeout_override

    # Load results CSV
    df = pd.read_csv(results_path)

    for idx in indices:
        row = df.iloc[idx]
        question = row['Question']
        context = row['Retrieved_Passages']

        print(f"\nTesting index {idx}...")
        print(f"Question: {question}")
        print(f"Context: {context}")

        # Generate answer
        response = generator.generate_answer(generator.llama_model, question, context)
        print(f"Response: {response}")

if __name__ == '__main__':
    sys.exit(main())
