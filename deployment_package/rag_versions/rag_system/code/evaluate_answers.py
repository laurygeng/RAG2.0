#!/usr/bin/env python3

"""
evaluate_answers.py

Clean, corrected evaluator that computes semantic/jaccard/overlap/combined
and the requested retrieval-based metrics when retrieval data or a local
Chroma DB is available. Gracefully degrades when chromadb or embeddings
are unavailable.

Produces outputs under rag_versions/rag_system/evaluation/<timestamp>/
"""

import os
import sys
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np
import logging
from collections import Counter
import os

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDINGS = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    HAS_EMBEDDINGS = False

# Optional stopword removal (enabled via env EVAL_REMOVE_STOPWORDS=1)
REMOVE_STOPWORDS = os.environ.get('EVAL_REMOVE_STOPWORDS', '0') == '1'
STOP_WORDS = set()
if REMOVE_STOPWORDS:
    try:
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        # If nltk stopwords are unavailable, silently proceed without stopword removal
        STOP_WORDS = set()

# NOTE: Advanced token normalization, stopword removal, lemmatization, and subword
# handling have been removed as per request. Only raw lowercased text similarity
# metrics remain.


class RealEvaluator:
    """Simplified evaluator: ONLY basic similarity metrics (semantic, jaccard, overlap, combined)."""

    def __init__(self, embed_model_name=None):
        print("Initializing evaluation model (basic metrics mode)...")
        # Resolve model name: prefer EVAL_EMB_MODEL, then CUSTOM_EMB_MODEL, else default
        if not embed_model_name:
            embed_model_name = os.environ.get('EVAL_EMB_MODEL', os.environ.get('CUSTOM_EMB_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
        self.strict_embeddings = os.environ.get('EVAL_STRICT_EMBEDDINGS', '0') == '1'
        trust = os.environ.get('CUSTOM_EMB_TRUST_REMOTE', os.environ.get('EVAL_EMB_TRUST_REMOTE', '0')) == '1'
        if HAS_EMBEDDINGS and SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(embed_model_name, trust_remote_code=trust)
            except Exception as e:
                logging.error(f"Failed to load embedding model '{embed_model_name}': {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        if self.strict_embeddings and self.embedding_model is None:
            raise RuntimeError("Embedding model not available and EVAL_STRICT_EMBEDDINGS=1")

    @staticmethod
    def preprocess_text(text):
        if pd.isna(text) or text == '' or 'Failed to generate' in str(text):
            return ''
        # Lowercase, remove punctuation, normalize digits, collapse whitespace
        t = str(text).lower()
        t = re.sub(r'[^\w\s]', ' ', t)  # remove punctuation
        t = re.sub(r'\d+', 'NUM', t)     # normalize numbers
        t = ' '.join(t.strip().split())
        return t

    def _encode_text(self, text: str):
        """Encode text with chunking for long inputs. Returns a 1D numpy vector or None."""
        if not text or self.embedding_model is None:
            return None
        # Chunk by characters to avoid excessive sequence length
        try:
            max_chars = int(os.environ.get('EVAL_EMBED_MAX_CHARS', '500'))
        except Exception:
            max_chars = 500
        try:
            if len(text) <= max_chars:
                emb = self.embedding_model.encode(text)
                return np.asarray(emb, dtype=np.float32)
            # Split into non-overlapping chunks and average their embeddings
            chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            embs = self.embedding_model.encode(chunks)
            embs = np.asarray(embs, dtype=np.float32)
            if embs.ndim == 2:
                return embs.mean(axis=0)
            elif embs.ndim == 1:
                return embs
            else:
                return None
        except Exception as e:
            logging.error(f"Embedding encode failed: {e}")
            if self.strict_embeddings:
                raise
            return None

    def calculate_semantic_similarity(self, text1, text2):
        if not text1 or not text2 or self.embedding_model is None:
            return 0.0
        try:
            v1 = self._encode_text(text1)
            v2 = self._encode_text(text2)
            if v1 is None or v2 is None:
                return 0.0
            # Prefer sklearn cosine if available; else compute manually
            if cosine_similarity is not None:
                sim = float(cosine_similarity([v1], [v2])[0][0])
            else:
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
                sim = float(np.dot(v1, v2) / denom)
            return max(0.0, sim)
        except Exception as e:
            logging.error(f"Semantic similarity error: {e}")
            if self.strict_embeddings:
                raise
            return 0.0

    @staticmethod
    def calculate_jaccard_similarity(text1, text2):
        if not text1 or not text2:
            return 0.0
        w1 = RealEvaluator.preprocess_text(text1).split()
        w2 = RealEvaluator.preprocess_text(text2).split()
        if REMOVE_STOPWORDS and STOP_WORDS:
            w1 = [w for w in w1 if w not in STOP_WORDS]
            w2 = [w for w in w2 if w not in STOP_WORDS]
        w1 = set(w1)
        w2 = set(w2)
        if not w1 and not w2:
            return 1.0
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    @staticmethod
    def calculate_overlap_similarity(text1, text2):
        if not text1 or not text2:
            return 0.0
        w1 = RealEvaluator.preprocess_text(text1).split()
        w2 = RealEvaluator.preprocess_text(text2).split()
        if REMOVE_STOPWORDS and STOP_WORDS:
            w1 = [w for w in w1 if w not in STOP_WORDS]
            w2 = [w for w in w2 if w not in STOP_WORDS]
        s1 = set(w1)
        s2 = set(w2)
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        inter = s1 & s2
        denom = min(len(s1), len(s2))
        return len(inter) / denom if denom > 0 else 0.0

    @staticmethod
    def calculate_combined_score(semantic, jaccard, overlap):
        # Fixed weights; env override removed
        return 0.5 * semantic + 0.25 * jaccard + 0.25 * overlap

    def evaluate_answers(self, gt, pred):
        gt_p = RealEvaluator.preprocess_text(gt)
        pred_p = RealEvaluator.preprocess_text(pred)
        sem = 0.0
        if self.embedding_model is not None and gt_p and pred_p:
            try:
                sem = float(self.calculate_semantic_similarity(gt_p, pred_p))
            except Exception:
                sem = 0.0
        jac = RealEvaluator.calculate_jaccard_similarity(gt_p, pred_p)
        ov = RealEvaluator.calculate_overlap_similarity(gt_p, pred_p)
        comb = RealEvaluator.calculate_combined_score(sem, jac, ov)
        return {
            'Semantic_Similarity': sem,
            'Jaccard_Similarity': jac,
            'Overlap_Similarity': ov,
            'Combined_Score': comb,
        }
def evaluate_real_answers(csv_file):
    print("=== Basic Real Answers Evaluation (pruned) ===")
    print(f"Using file: {csv_file}")
    df = pd.read_csv(csv_file)
    evaluator = RealEvaluator()

    rows = []
    for idx, row in df.iterrows():
        question = row.get('Question', '')
        gt = row.get('Ground_Truth_Answer', '') or row.get('Ground_Truth', '')
        ans = row.get('Real_LLaMA_Answer', '')
        scores = evaluator.evaluate_answers(gt, ans)
        rows.append({
            'Index': idx,
            'Question_ID': row.get('Question_ID', ''),
            'Question': question,
            'Ground_Truth': gt,
            'Real_LLaMA_Answer': ans,
            **scores,
        })

    out_df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_eval_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation')
    eval_dir = os.path.join(base_eval_root, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    out_file = os.path.join(eval_dir, f'real_llama_evaluation_{timestamp}.csv')

    # Round metric columns to 3 decimal places for consistent reporting
    metric_cols = ['Semantic_Similarity', 'Jaccard_Similarity', 'Overlap_Similarity', 'Combined_Score']
    for c in metric_cols:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype(float).round(3)

    out_df.to_csv(out_file, index=False)
    print(f"\n📁 Saved basic evaluation: {out_file}")
    return {'llama_results': out_df}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        import glob
        pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'real_answers_*.csv')
        files = glob.glob(pattern)
        if files:
            csv_file = max(files, key=os.path.getmtime)
        else:
            print('No real answers file found (pattern: real_answers_*.csv)')
            sys.exit(1)
    try:
        evaluate_real_answers(csv_file)
        print('\n🎉 Basic evaluation completed!')
    except Exception as e:
        print(f'Error during basic evaluation: {e}')
        import traceback
        traceback.print_exc()
