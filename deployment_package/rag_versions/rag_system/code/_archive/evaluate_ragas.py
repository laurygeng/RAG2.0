#!/usr/bin/env python3
"""
evaluate_ragas.py (custom implementation WITHOUT ragas library)

Computes five metrics for RAG evaluation using local embeddings + heuristics:
    - context_precision: 按原始检索顺序计算的加权精度（类似 MAP）。
        对每个位置 i 计算 Precision@i，并仅在该位置为“相关”时计入权重，
        最终取加权平均：sum(P@i * rel_i) / K（K 为被评估的上下文数量）。
  - context_recall: coverage of GT sentences supported by any retrieved context.
    - answer_relevancy: 可选“反向生成问题(QG)”方案：从答案生成若干问题，与原问题做语义相似度并取平均；
        若未启用/不可用则回退为 question 与 answer 的整体相似度。
  - faithfulness: fraction of answer sentences supported by contexts.
  - answer_correctness: statement-level F1 + semantic similarity (0.75 * F1 + 0.25 * semantic).

Inputs: latest results CSV in results/ (real_answers_*.csv) unless path is passed as argv[1].
Outputs: evaluation/ragas_<timestamp>/ragas_five_metrics_<timestamp>.csv
Environment:
  CUSTOM_EMB_MODEL      (default sentence-transformers/all-MiniLM-L6-v2)
  RAGAS_SIM_THR         (default 0.6) similarity threshold for relevance / support.
  RAGAS_TOPK_CONTEXTS   (default 5) number of retrieved passages to consider.
  RAGAS_MAX_SAMPLES     (default 0 = all) truncate dataset for quick runs.
  RAGAS_AC_THR          (default 0.7) sentence match threshold for answer correctness F1.
    ANSWER_REL_USE_QG     (default 0) set 1 to enable reverse-QG for answer_relevancy.
    ANSWER_REL_QG_MODEL   (default google/flan-t5-small) HF model id for QG.
    ANSWER_REL_QG_NUM     (default 3) number of questions to generate.
    ANSWER_REL_QG_MAXTOK  (default 64) max new tokens for generation.

All numeric outputs rounded to 3 decimals.
"""
from __future__ import annotations
import os, sys, json, ast, re
from datetime import datetime
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

def _find_latest_results_csv(results_dir: str) -> str:
    import glob
    files = glob.glob(os.path.join(results_dir, 'real_answers_*.csv'))
    if not files:
        raise FileNotFoundError(f"No results CSV found in {results_dir}")
    return max(files, key=os.path.getmtime)

def _parse_contexts(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell,float) and pd.isna(cell)): return []
    s = str(cell).strip()
    if not s: return []
    try:
        data = json.loads(s)
    except Exception:
        try: data = ast.literal_eval(s)
        except Exception: return [s]
    out: List[str] = []
    if isinstance(data,(list,tuple)):
        for x in data:
            if x is None: continue
            out.append(str(x))
    else:
        out=[str(data)]
    return out

def _segment_sentences(text: str) -> List[str]:
    if not isinstance(text,str) or not text.strip(): return []
    segs: List[str] = []
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p=p.strip()
            if p: segs.append(p)
    seen=set(); out=[]
    for s in segs:
        norm=re.sub(r"\s+"," ",s.lower()).strip()
        if norm and norm not in seen:
            seen.add(norm); out.append(s)
    return out

def _init_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model_name=os.environ.get('CUSTOM_EMB_MODEL','sentence-transformers/all-MiniLM-L6-v2')
        trust = os.environ.get('CUSTOM_EMB_TRUST_REMOTE','0') == '1'
        return SentenceTransformer(model_name, trust_remote_code=trust)
    except Exception as e:
        print(f"Embedding init failed: {e}")
        return None
    
def _init_qg_pipeline():
    """Initialize a lightweight HF pipeline for reverse question generation (optional)."""
    use_qg = os.environ.get('ANSWER_REL_USE_QG', '0') == '1'
    if not use_qg:
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        model_id = os.environ.get('ANSWER_REL_QG_MODEL', 'google/flan-t5-small')
        max_new = int(os.environ.get('ANSWER_REL_QG_MAXTOK', '64'))
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        gen = pipeline('text2text-generation', model=mdl, tokenizer=tok, max_new_tokens=max_new, do_sample=False)
        return gen
    except Exception as e:
        print(f"Reverse-QG init failed, falling back to QA similarity: {e}")
        return None

def _embed(model, texts: List[str]) -> np.ndarray:
    if model is None or not texts: return np.zeros((0,384),dtype=np.float32)
    embs=model.encode(texts, normalize_embeddings=True)
    embs=np.asarray(embs,dtype=np.float32)
    if embs.ndim==1: embs=embs.reshape(1,-1)
    return embs

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size==0 or b.size==0: return 0.0
    d=(np.linalg.norm(a)*np.linalg.norm(b))+1e-12
    return float(np.dot(a,b)/d)

def _parse_qg_output(text: str, n: int) -> List[str]:
    # Split by lines and punctuation; keep short unique questions
    if not isinstance(text, str) or not text.strip():
        return []
    raw_lines = [t.strip('-• 	') for t in text.splitlines()]
    parts: List[str] = []
    for ln in raw_lines:
        parts.extend([p.strip() for p in re.split(r"\?\s+|\.|;", ln) if p.strip()])
    # Clean and dedupe
    out: List[str] = []
    seen = set()
    for p in parts:
        if len(p) < 3:
            continue
        norm = re.sub(r"\s+", " ", p.lower())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(p if p.endswith('?') else (p + '?'))
        if len(out) >= n:
            break
    return out

def _weighted_precision_avg(relevance: List[int]) -> float:
    """加权平均精度：sum(P@i * rel_i) / K，其中 P@i = (#前i个相关)/i。
    注意分母使用位置总数 K，符合用户提供示例公式。
    """
    K = len(relevance)
    if K == 0:
        return 0.0
    precisions = []
    seen = 0
    for i, r in enumerate(relevance, start=1):
        if r:
            seen += 1
            precisions.append(seen / i)
        else:
            precisions.append(0.0)
    # 仅在 rel_i=1 的位置计入权重，相当于按 rel 进行逐位加权后再 / K
    num = sum(p * r for p, r in zip(precisions, relevance))
    return float(num / K)

# ---- Answer correctness utilities (greedy sentence F1 + semantic) ----

def _greedy_match(sim_matrix: np.ndarray, thr: float) -> Tuple[int,int,int]:
    if sim_matrix.size==0:
        return 0, sim_matrix.shape[1] if sim_matrix.ndim==2 else 0, sim_matrix.shape[0] if sim_matrix.ndim==2 else 0
    g,a=sim_matrix.shape
    pairs=[]
    for i in range(g):
        for j in range(a):
            s=float(sim_matrix[i,j])
            if s>=thr: pairs.append((i,j,s))
    pairs.sort(key=lambda x:x[2], reverse=True)
    used_g=set(); used_a=set(); tp=0
    for i,j,s in pairs:
        if i in used_g or j in used_a: continue
        used_g.add(i); used_a.add(j); tp+=1
    fp=a-len(used_a); fn=g-len(used_g)
    return tp,fp,fn

def _compute_answer_correctness(model, gt: str, ans: str, semantic_sim: float, thr: float) -> float:
    gt_segs=_segment_sentences(gt); ans_segs=_segment_sentences(ans)
    if not gt_segs and not ans_segs: return float('nan')
    if not gt_segs or not ans_segs: tp,fp,fn=0,len(ans_segs),len(gt_segs)
    else:
        g_embs=_embed(model, gt_segs); a_embs=_embed(model, ans_segs)
        if g_embs.size==0 or a_embs.size==0: tp,fp,fn=0,len(ans_segs),len(gt_segs)
        else:
            sims=g_embs @ a_embs.T
            tp,fp,fn=_greedy_match(sims, thr)
    p=(tp/(tp+fp)) if (tp+fp)>0 else 0.0
    r=(tp/(tp+fn)) if (tp+fn)>0 else 0.0
    f1=(2*p*r)/(p+r) if (p+r)>0 else 0.0
    return 0.75*f1 + 0.25*float(semantic_sim or 0.0)

# ---- Core per-row metric computation ----

def compute_row(model, question: str, answer: str, gt: str, contexts: List[str], thr: float, ac_thr: float, qg_pipeline=None):
    q_emb=_embed(model,[question])[0] if question else np.zeros((384,),dtype=np.float32)
    a_emb=_embed(model,[answer])[0] if answer else np.zeros((384,),dtype=np.float32)
    g_emb=_embed(model,[gt])[0] if gt else np.zeros((384,),dtype=np.float32)
    ctx_embs=_embed(model, contexts) if contexts else np.zeros((0,384),dtype=np.float32)
    # answer_relevancy: reverse-QG if enabled; else fallback to sim(q,a)
    answer_relevancy: float
    if qg_pipeline is not None and answer:
        try:
            n = int(os.environ.get('ANSWER_REL_QG_NUM', '3'))
            prompt = (
                "Generate concise questions (one per line) that are answered by the following answer.\n"
                "Answer: " + answer + "\nQuestions:"
            )
            out = qg_pipeline(prompt)[0]['generated_text']
            qs = _parse_qg_output(out, n)
            if qs:
                q_embs = _embed(model, qs)
                sims = [ _cos(q_emb, q_embs[i]) for i in range(q_embs.shape[0]) ] if q_embs.size else []
                if sims:
                    answer_relevancy = float(np.mean([max(0.0, min(1.0, s)) for s in sims]))
                else:
                    answer_relevancy = max(0.0, min(1.0, _cos(q_emb, a_emb)))
            else:
                answer_relevancy = max(0.0, min(1.0, _cos(q_emb, a_emb)))
        except Exception as e:
            print(f"Reverse-QG failed, fallback to QA sim: {e}")
            answer_relevancy = max(0.0, min(1.0, _cos(q_emb, a_emb)))
    else:
        answer_relevancy = max(0.0, min(1.0, _cos(q_emb, a_emb)))
    # context_precision（保持原始检索顺序）：
    # 用 GT 分句与每个 context 的最大相似度判定相关性（>=thr），不改变原顺序
    relevance: List[int] = []
    if ctx_embs.size:
        if gt:
            gt_sents = _segment_sentences(gt)
            g_sent_embs = _embed(model, gt_sents) if gt_sents else np.zeros((0,384), dtype=np.float32)
            for i in range(ctx_embs.shape[0]):
                if g_sent_embs.size:
                    sims = g_sent_embs @ ctx_embs[i].reshape(1,-1).T  # (num_gt_sents, 1)
                    max_sim = float(sims.max()) if sims.size else 0.0
                else:
                    # 若无 GT 分句则回退为与 GT 整段相似度
                    max_sim = float(ctx_embs[i] @ g_emb) if g_emb.size else 0.0
                relevance.append(1 if max_sim >= thr else 0)
        else:
            # 无 GT 时，以与 Question 的相似度作相关性近似
            for i in range(ctx_embs.shape[0]):
                max_sim = float(ctx_embs[i] @ q_emb) if q_emb.size else 0.0
                relevance.append(1 if max_sim >= thr else 0)
    context_precision = _weighted_precision_avg(relevance)
    # context_recall
    gt_sents=_segment_sentences(gt); context_recall=0.0
    if gt_sents:
        gt_sent_embs=_embed(model, gt_sents)
        if gt_sent_embs.size and ctx_embs.size:
            sims=gt_sent_embs @ ctx_embs.T
            max_per= sims.max(axis=1) if sims.size else np.zeros((len(gt_sents),),dtype=np.float32)
            context_recall=float((max_per>=thr).sum()/len(gt_sents))
    # faithfulness
    ans_sents=_segment_sentences(answer); faithfulness=0.0
    if ans_sents:
        ans_sent_embs=_embed(model, ans_sents)
        if ans_sent_embs.size and ctx_embs.size:
            sims=ans_sent_embs @ ctx_embs.T
            max_per = sims.max(axis=1) if sims.size else np.zeros((len(ans_sents),),dtype=np.float32)
            faithfulness=float((max_per>=thr).sum()/len(ans_sents))
    # answer_correctness
    semantic_sim=_cos(g_emb,a_emb)
    answer_correctness=_compute_answer_correctness(model, gt, answer, semantic_sim, ac_thr)
    return context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness

# ---- Main flow ----

def main():
    base_dir=os.path.dirname(os.path.dirname(__file__))
    results_dir=os.path.join(base_dir,'results')
    # resolve CSV
    if len(sys.argv)>1:
        csv_file=sys.argv[1]
        if not os.path.exists(csv_file):
            cand=os.path.join(results_dir,csv_file)
            if os.path.exists(cand): csv_file=cand
            else: raise FileNotFoundError(f'CSV not found: {csv_file}')
    else:
        csv_file=_find_latest_results_csv(results_dir)
    print('=== Custom RAG Five Metrics (no ragas) ===')
    print('Using file:', csv_file)
    df=pd.read_csv(csv_file)
    topk=int(os.environ.get('RAGAS_TOPK_CONTEXTS', os.environ.get('CUSTOM_TOPK_CONTEXTS','5')))
    max_samples=int(os.environ.get('RAGAS_MAX_SAMPLES', os.environ.get('CUSTOM_MAX_SAMPLES','0')))
    thr=float(os.environ.get('RAGAS_SIM_THR', os.environ.get('CUSTOM_SIM_THR','0.6')))
    ac_thr=float(os.environ.get('RAGAS_AC_THR','0.7'))
    items=[]
    for idx,row in df.iterrows():
        q=str(row.get('Question','') or '')
        a=str(row.get('Real_LLaMA_Answer','') or '')
        gt=str(row.get('Ground_Truth_Answer', row.get('Ground_Truth','')) or '')
        ctxs=_parse_contexts(row.get('Retrieved_Passages',''))
        if topk>0: ctxs=ctxs[:topk]
        items.append((idx,row.get('Question_ID',''),q,a,gt,ctxs))
    if max_samples and len(items)>max_samples: items=items[:max_samples]
    model=_init_embedder()
    qg_pipeline=_init_qg_pipeline()
    out_rows=[]
    for idx,qid,q,a,gt,ctxs in items:
        try:
            cprec,crec,arel,faith,acorr=compute_row(model,q,a,gt,ctxs,thr,ac_thr,qg_pipeline=qg_pipeline)
        except Exception as e:
            print(f'Row {idx} error: {e}')
            cprec=crec=arel=faith=acorr=float('nan')
        out_rows.append({
            'Index': idx,
            'Question_ID': qid,
            'Question': q,
            'Ground_Truth': gt,
            'Real_LLaMA_Answer': a,
            'Retrieved_Passages': str(ctxs),
            'context_precision': cprec,
            'context_recall': crec,
            'answer_relevancy': arel,
            'faithfulness': faith,
            'answer_correctness': acorr,
        })
    out_df=pd.DataFrame(out_rows)
    for col in ['context_precision','context_recall','answer_relevancy','faithfulness','answer_correctness']:
        if col in out_df.columns and pd.api.types.is_numeric_dtype(out_df[col]):
            out_df[col]=out_df[col].astype(float).round(3)
    eval_root=os.path.join(base_dir,'evaluation')
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(eval_root,f'ragas_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    out_csv=os.path.join(out_dir, f'ragas_five_metrics_{ts}.csv')
    out_df.to_csv(out_csv, index=False)
    print(f'\n📁 Saved five-metrics evaluation: {out_csv}')
    print('\n🎉 Done!')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        import traceback; traceback.print_exc()
