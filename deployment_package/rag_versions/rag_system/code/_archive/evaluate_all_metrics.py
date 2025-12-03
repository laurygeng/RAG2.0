#!/usr/bin/env python3

"""
evaluate_all_metrics.py

Aggregate legacy basic metrics (evaluate_answers.py) and the five custom RAG metrics
(evaluate_ragas.py), and compute Answer Correctness using:

  - Statement-level F1: split GT and Answer into statements, match by semantic
    similarity with greedy one-to-one matching at a threshold, then
      Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)
  - Combined Answer Correctness = 0.75 * F1 + 0.25 * Semantic_Similarity

The Semantic_Similarity term is taken from evaluate_answers.py output.

Output: evaluation/evaluate_all_metrics/all_metrics_<timestamp>.csv
"""

import os
import re
import sys
import glob
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import numpy as np
import subprocess
import sys as _sys


def _repo_root_from(base_dir: str) -> str:
    """Best-effort to resolve repository root from rag_system/ directory."""
    # base_dir points to .../rag_versions/rag_system
    return os.path.abspath(os.path.join(base_dir, '..', '..', '..'))


def _resolve_python_interpreter(base_dir: str) -> str:
    """Prefer repo-local virtualenv python (e.g., .venv311) over current sys.executable.

    Order:
      1) <repo_root>/.venv311/bin/python
      2) first match of <repo_root>/.venv*/bin/python (lexicographic)
      3) sys.executable
    """
    repo_root = _repo_root_from(base_dir)
    candidates = [
        os.path.join(repo_root, '.venv311', 'bin', 'python'),
    ]
    # discover other .venv* if present
    try:
        for name in sorted(os.listdir(repo_root)):
            if name.startswith('.venv') and name != '.venv311':
                cand = os.path.join(repo_root, name, 'bin', 'python')
                candidates.append(cand)
    except Exception:
        pass
    for p in candidates:
        if os.path.exists(p):
            print(f"Using preferred repo interpreter: {p}")
            return p
    print(f"Falling back to current interpreter: {_sys.executable}")
    return _sys.executable or 'python'


def _segment_text_into_statements(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    segs: List[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p:
                segs.append(p)
    # dedupe by normalized form
    seen = set()
    out: List[str] = []
    for s in segs:
        norm = re.sub(r"\s+", " ", s.lower()).strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(s)
    return out


def _load_latest_file(patterns: List[str]) -> str | None:
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _init_embedder():
    # Optional fast-disable to avoid heavy imports when env lacks torch/transformers
    if os.environ.get('EVAL_ALL_DISABLE_EMB', '0') == '1':
        print("Embedding disabled via EVAL_ALL_DISABLE_EMB=1; answer_correctness will use semantic-only proxy.")
        return None
    try:
        from sentence_transformers import SentenceTransformer
        # Prefer explicit, then custom, then legacy fallbacks
        model_name = os.environ.get(
            'EVAL_ALL_EMB_MODEL',
            os.environ.get('CUSTOM_EMB_MODEL', os.environ.get('SIMPLE_EMB_MODEL', os.environ.get('RAGAS_HF_EMB_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))),
        )
        trust = os.environ.get('CUSTOM_EMB_TRUST_REMOTE', os.environ.get('EVAL_ALL_EMB_TRUST_REMOTE', '0')) == '1'
        model = SentenceTransformer(model_name, trust_remote_code=trust)
        return model
    except Exception as e:
        print(f"Embedding init failed for evaluate_all_metrics: {e}")
        return None


def _embed(model, texts: List[str]) -> np.ndarray:
    import numpy as _np
    if model is None or not texts:
        return _np.zeros((0, 384), dtype=_np.float32)
    embs = model.encode(texts, normalize_embeddings=True)
    embs = _np.asarray(embs, dtype=_np.float32)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    return embs


def _greedy_match(sim_matrix: np.ndarray, thr: float) -> Tuple[int, int, int]:
    """Return TP, FP, FN via greedy one-to-one matching of rows (GT) to cols (ANS).

    sim_matrix: shape (num_gt, num_ans) with cosine similarities.
    thr: threshold to accept a match.
    """
    num_gt, num_ans = sim_matrix.shape if sim_matrix.size else (0, 0)
    if num_gt == 0 and num_ans == 0:
        return 0, 0, 0
    if sim_matrix.size == 0:
        # no pairs
        return 0, num_ans, num_gt
    # collect all pairs above threshold
    pairs: List[Tuple[int, int, float]] = []
    for i in range(num_gt):
        for j in range(num_ans):
            s = float(sim_matrix[i, j])
            if s >= thr:
                pairs.append((i, j, s))
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_gt = set()
    used_ans = set()
    tp = 0
    for i, j, s in pairs:
        if i in used_gt or j in used_ans:
            continue
        used_gt.add(i)
        used_ans.add(j)
        tp += 1
    fp = num_ans - len(used_ans)
    fn = num_gt - len(used_gt)
    return tp, fp, fn


def compute_answer_correctness(gt_text: str, ans_text: str, semantic_sim: float, model) -> float:
    thr = float(os.environ.get('EVAL_ALL_CORRECT_SIM_TH', '0.7'))
    w1 = float(os.environ.get('EVAL_ALL_W1', '0.75'))
    w2 = float(os.environ.get('EVAL_ALL_W2', '0.25'))

    # If embeddings are disabled/unavailable, fall back to semantic-only proxy
    if model is None:
        # Treat F1 as 0 when we cannot compute statement matches; keep semantic contribution
        return w2 * float(semantic_sim or 0.0)

    gt_segs = _segment_text_into_statements(gt_text)
    ans_segs = _segment_text_into_statements(ans_text)
    if not gt_segs and not ans_segs:
        return float('nan')
    if not gt_segs or not ans_segs:
        # no matches possible
        tp, fp, fn = 0, len(ans_segs), len(gt_segs)
    else:
        gt_embs = _embed(model, gt_segs)
        ans_embs = _embed(model, ans_segs)
        if gt_embs.size == 0 or ans_embs.size == 0:
            tp, fp, fn = 0, len(ans_segs), len(gt_segs)
        else:
            sims = gt_embs @ ans_embs.T
            tp, fp, fn = _greedy_match(sims, thr)

    p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return w1 * f1 + w2 * float(semantic_sim or 0.0)


def find_latest_outputs(base_eval_dir: str):
    """Locate latest five-metrics CSV from evaluate_ragas.py (custom) and basic CSV.

    We support both old ragas-named outputs and the new custom outputs.
    """
    # new custom output
    custom_csv = _load_latest_file([
        os.path.join(base_eval_dir, 'ragas_*', 'ragas_five_metrics_*.csv'),
        os.path.join(base_eval_dir, 'custom_*', 'custom_five_metrics_*.csv'),
    ])
    # old ragas output (kept for compatibility)
    legacy_ragas_csv = _load_latest_file([
        os.path.join(base_eval_dir, 'ragas_*', 'llama_ragas_evaluation_*.csv'),
    ])
    # legacy basic evaluator outputs
    basic_csv = _load_latest_file([
        os.path.join(base_eval_dir, '*', 'real_llama_evaluation_*.csv'),
    ])
    # prefer custom over legacy ragas
    ragas_like_csv = custom_csv or legacy_ragas_csv
    return ragas_like_csv, basic_csv


def _run_script_if_missing(target_csv: str | None, generator_cmd: List[str]) -> None:
    """If target_csv is missing, run the provided command to generate it."""
    if target_csv and os.path.exists(target_csv):
        return
    try:
        print("Generating missing evaluation via:", " ".join(generator_cmd))
        subprocess.run(generator_cmd, check=True)
    except Exception as e:
        print(f"Failed to auto-generate evaluation file: {e}")


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # rag_system/
    eval_root = os.path.join(base_dir, 'evaluation')

    # Allow explicit files via args: [ragas_csv] [basic_csv]
    ragas_csv = sys.argv[1] if len(sys.argv) > 1 else None
    basic_csv = sys.argv[2] if len(sys.argv) > 2 else None

    if not ragas_csv or not os.path.exists(ragas_csv) or not basic_csv or not os.path.exists(basic_csv):
        # Try to generate missing pieces by invoking the scripts with preferred interpreter
        py = _resolve_python_interpreter(base_dir)
        # Generate five-metrics if missing
        if not ragas_csv or not os.path.exists(ragas_csv):
            _run_script_if_missing(None, [py, os.path.join(os.path.dirname(__file__), 'evaluate_ragas.py')])
        # Generate basic if missing
        if not basic_csv or not os.path.exists(basic_csv):
            _run_script_if_missing(None, [py, os.path.join(os.path.dirname(__file__), 'evaluate_answers.py')])
        # Re-scan
        auto_ragas, auto_basic = find_latest_outputs(eval_root)
        if not ragas_csv or not os.path.exists(ragas_csv):
            ragas_csv = auto_ragas
        if not basic_csv or not os.path.exists(basic_csv):
            basic_csv = auto_basic

    if not ragas_csv or not os.path.exists(ragas_csv):
        raise FileNotFoundError("Could not locate five-metrics CSV (ragas_five_metrics_*.csv or custom_five_metrics_*.csv or llama_ragas_evaluation_*.csv)")
    if not basic_csv or not os.path.exists(basic_csv):
        raise FileNotFoundError("Could not locate basic evaluation CSV (real_llama_evaluation_*.csv)")

    print("=== Aggregate Metrics Evaluation ===")
    print(f"Using five-metrics CSV: {ragas_csv}")
    print(f"Using basic CSV: {basic_csv}")

    df_ragas = pd.read_csv(ragas_csv)
    df_basic = pd.read_csv(basic_csv)

    # Normalize column names in basic eval
    # Ensure basic has Index, Question_ID, Question, Ground_Truth, Real_LLaMA_Answer
    # evaluate_answers already writes those columns

    # Merge on Index primarily; fallback to Question_ID + Question if Index missing
    on_cols = []
    if 'Index' in df_ragas.columns and 'Index' in df_basic.columns:
        on_cols = ['Index']
    elif all(c in df_ragas.columns for c in ['Question_ID', 'Question']) and all(c in df_basic.columns for c in ['Question_ID', 'Question']):
        on_cols = ['Question_ID', 'Question']
    else:
        # last resort: try Question text
        on_cols = ['Question']

    merged = df_ragas.merge(df_basic, on=on_cols, how='left', suffixes=('', '_basic'))

    # Initialize embedder for statement matching
    embedder = _init_embedder()

    # Compute Answer Correctness per-row using Semantic_Similarity from basic
    ac_scores = []
    for _, row in merged.iterrows():
        gt_text = row.get('Ground_Truth', row.get('Ground_Truth_Answer', ''))
        ans_text = row.get('Real_LLaMA_Answer', '')
        semantic = row.get('Semantic_Similarity', None)
        try:
            ac = compute_answer_correctness(str(gt_text or ''), str(ans_text or ''), float(semantic) if semantic is not None else 0.0, embedder)
        except Exception:
            ac = float('nan')
        ac_scores.append(ac)
    merged['answer_correctness'] = ac_scores

    # Select and order columns: base identifiers + old four + new five (with recomputed answer_correctness)
    id_cols = ['Index', 'Question_ID', 'Question', 'Ground_Truth', 'Ground_Truth_Answer', 'Real_LLaMA_Answer', 'Retrieved_Passages']
    old_four = ['Semantic_Similarity', 'Jaccard_Similarity', 'Overlap_Similarity', 'Combined_Score']
    new_five = ['context_precision', 'context_recall', 'answer_relevancy', 'faithfulness', 'answer_correctness']
    cols = [c for c in id_cols + old_four + new_five if c in merged.columns]
    out_df = merged[cols]

    # Output
    out_root = os.path.join(eval_root, 'evaluate_all_metrics')
    os.makedirs(out_root, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_root, f'all_metrics_{ts}.csv')

    # Round metric columns (old four + new five) to 3 decimals where numeric
    metric_cols = ['Semantic_Similarity','Jaccard_Similarity','Overlap_Similarity','Combined_Score',
                   'context_precision','context_recall','answer_relevancy','faithfulness','answer_correctness']
    for c in metric_cols:
        if c in out_df.columns and pd.api.types.is_numeric_dtype(out_df[c]):
            out_df[c] = out_df[c].astype(float).round(3)

    out_df.to_csv(out_path, index=False)
    print(f"\n📁 Saved aggregated evaluation: {out_path}")


if __name__ == '__main__':
    try:
        main()
        print("\n🎉 Aggregate evaluation completed!")
    except Exception as e:
        print(f"Error during aggregate evaluation: {e}")
        import traceback
        traceback.print_exc()
