#!/usr/bin/env python3
"""
Analyze Rerank_Scores in a results CSV and produce a reproducible report.

Outputs:
 - results/rerank_analysis_<ts>.csv : per-row metrics and passages
 - results/rerank_analysis_<ts>.md  : human-readable summary + recommendations

Run: from repo root run `python3 scripts/analyze_rerank_scores.py`
"""
import csv
import json
import os
from datetime import datetime
import statistics
import textwrap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_CSV = os.path.join(BASE_DIR, 'results', 'real_answers_20251118_114302.csv')
OUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_CSV = os.path.join(OUT_DIR, f'rerank_analysis_{TS}.csv')
OUT_MD = os.path.join(OUT_DIR, f'rerank_analysis_{TS}.md')


def extract_rerank_config():
    """Try to read generate_answers.py defaults for reranker config values."""
    cfg = {}
    ga_path = os.path.join(BASE_DIR, 'code', 'generate_answers.py')
    if not os.path.exists(ga_path):
        return cfg
    try:
        with open(ga_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'RAG_RERANK_MODEL' in line and '=' in line and 'os.environ' in line:
                    # crude parse of default value
                    parts = line.split('=')
                    if len(parts) >= 2:
                        right = parts[1]
                        # find quoted string
                        import re
                        m = re.search(r"'([A-Za-z0-9_\-/]+)'", right)
                        if m:
                            cfg['RAG_RERANK_MODEL'] = m.group(1)
                if 'RAG_RERANK_DEVICE' in line and '=' in line and 'os.environ' in line:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        import re
                        m = re.search(r"'([a-zA-Z0-9_]+)'", parts[1])
                        if m:
                            cfg['RAG_RERANK_DEVICE'] = m.group(1)
                if 'RAG_RERANK_PASSAGE_CHARS' in line:
                    # may be defined elsewhere; capture numeric literal if present
                    import re
                    m = re.search(r'RAG_RERANK_PASSAGE_CHARS\s*=\s*(\d+)', line)
                    if m:
                        cfg['RAG_RERANK_PASSAGE_CHARS'] = int(m.group(1))
    except Exception:
        pass
    return cfg


def analyze():
    rows = []
    scores_all = []
    low_entries = []
    if not os.path.exists(RESULTS_CSV):
        print('Results CSV not found:', RESULTS_CSV)
        return

    with open(RESULTS_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for lineno, row in enumerate(reader, start=2):
            raw = row.get('Rerank_Scores', '')
            try:
                scores = json.loads(raw)
                if not isinstance(scores, list):
                    scores = list(scores)
            except Exception:
                scores = []
            metrics = {}
            if scores:
                metrics['min'] = min(map(float, scores))
                metrics['max'] = max(map(float, scores))
                metrics['mean'] = statistics.mean(map(float, scores))
                metrics['count'] = len(scores)
                metrics['below_0_5'] = sum(1 for s in scores if float(s) < 0.5)
                scores_all.extend([float(s) for s in scores])
            else:
                metrics['min'] = metrics['max'] = metrics['mean'] = None
                metrics['count'] = 0
                metrics['below_0_5'] = 0

            passages = row.get('Retrieved_Passages','')
            sources = row.get('Retrieved_Sources','')

            rows.append({
                'line': lineno,
                'Question_ID': row.get('Question_ID'),
                'Question': row.get('Question'),
                'max_score': metrics['max'],
                'mean_score': metrics['mean'],
                'min_score': metrics['min'],
                'count_scores': metrics['count'],
                'num_below_0_5': metrics['below_0_5'],
                'passages': passages,
                'sources': sources,
            })

            if metrics['max'] is not None and metrics['max'] < 0.5:
                low_entries.append(rows[-1])

    # write output CSV
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as out:
        fieldnames = ['line','Question_ID','Question','max_score','mean_score','min_score','count_scores','num_below_0_5','passages','sources']
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # prepare markdown summary
    conf = extract_rerank_config()
    total = len(rows)
    low_count = len(low_entries)
    summary_lines = []
    summary_lines.append(f'# Rerank score analysis — {TS}')
    summary_lines.append('')
    summary_lines.append(f'- Results CSV analyzed: `{RESULTS_CSV}`')
    summary_lines.append(f'- Total questions processed: **{total}**')
    summary_lines.append(f'- Entries with max rerank score < 0.5: **{low_count}**')
    if scores_all:
        summary_lines.append(f"- Score stats (over all individual passage scores): min={min(scores_all):.4f}, median={statistics.median(scores_all):.4f}, mean={statistics.mean(scores_all):.4f}, max={max(scores_all):.4f}")
    summary_lines.append('')
    summary_lines.append('## Reranker configuration (inferred)')
    if conf:
        for k,v in conf.items():
            summary_lines.append(f'- `{k}`: `{v}`')
    else:
        summary_lines.append('- Could not infer reranker env defaults from `code/generate_answers.py`')

    summary_lines.append('')
    summary_lines.append('## Rows with low max score (< 0.5)')
    if not low_entries:
        summary_lines.append('- None')
    else:
        for r in low_entries:
            summary_lines.append(f"\n### Line {r['line']} — Question_ID `{r['Question_ID']}` — max={r['max_score']:.4f} — num_scores={r['count_scores']}")
            qshort = (r['Question'] or '').strip().replace('\n',' ')[:500]
            summary_lines.append(f'- **Question**: {qshort}')
            # include first passage preview
            try:
                passages = json.loads(r['passages'])
            except Exception:
                passages = []
            try:
                sources = json.loads(r['sources'])
            except Exception:
                sources = []
            if passages:
                for i,p in enumerate(passages[:3], start=1):
                    src = sources[i-1] if i-1 < len(sources) else ''
                    preview = (p or '').replace('\n',' ')[:800]
                    summary_lines.append(f'- Passage[{i}] source: `{src}`')
                    summary_lines.append('```')
                    summary_lines.append(preview)
                    summary_lines.append('```')
            else:
                summary_lines.append('- No retrieved passages recorded')

    summary_lines.append('')
    summary_lines.append('## Preliminary analysis — possible causes')
    summary_lines.append('')
    summary_lines.append('- Reranker truncation: if passages are heavily truncated before scoring, important context may be missing.')
    summary_lines.append('- Poor initial retrieval: EnsembleRetriever may return low-relevance passages so reranker has nothing strongly relevant to score highly.')
    summary_lines.append('- Reranker model mismatch: chosen cross-encoder may be weak or not suitable for the domain; CPU fallback or fallback model may be used at runtime.')
    summary_lines.append('- Passage length / tokenization issues: long passages or artifacts may confuse the cross-encoder.')
    summary_lines.append('- Evaluation metric mismatch: cross-encoder scores may not be calibrated to [0,1] with interpretable threshold; low absolute values can be expected depending on model.')

    summary_lines.append('')
    summary_lines.append('## Recommended next steps')
    summary_lines.append('')
    summary_lines.append('1. Re-run reranking locally for the low rows using the same cross-encoder interactively to inspect raw scores and logits.')
    summary_lines.append('2. Increase `RAG_RERANK_PASSAGE_CHARS` (or disable truncation) for a test to see if scores improve.')
    summary_lines.append('3. Check which reranker model was actually loaded at runtime (logs show model name and device).')
    summary_lines.append('4. Inspect initial retrieval (BM25 and vector) for those queries to ensure candidate set contains relevant passages.')
    summary_lines.append('5. Optionally try an alternative cross-encoder (ms-marco-MiniLM-L-6-v2) to compare scores.')

    # write markdown
    with open(OUT_MD, 'w', encoding='utf-8') as mf:
        mf.write('\n'.join(summary_lines))

    print('Wrote analysis CSV:', OUT_CSV)
    print('Wrote analysis MD :', OUT_MD)


if __name__ == '__main__':
    analyze()
