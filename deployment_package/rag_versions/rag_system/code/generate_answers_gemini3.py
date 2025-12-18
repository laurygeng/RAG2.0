#!/usr/bin/env python3

import os
import sys
import pandas as pd
import requests
import json
import time
import concurrent.futures
import logging
from datetime import datetime
import shutil
import subprocess
import numpy as np
import hashlib
import pickle
from typing import Optional, List

# Ensure the code folder is on sys.path (we avoid importing heavy modules here).
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Defer importing `load_data` (and transitively heavy libs like langchain/pydantic)
# until it's explicitly needed. Provide placeholders so code can run without them.
SimpleBrainCheckLoader = None
load_knowledge_base = None

class RealModelAnswerGenerator:
    """Generate answers from Google Gemini 3 API."""

    def __init__(self):
        # Google Gemini API configuration
        self.api_key = "AIzaSyD7Gr761E9PZp-C3tmdCdzBT7nEUNRNads"
        self.gemini_model = "gemini-3-pro-preview"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.api_key}"

        # optional retrieval components
        self.kb = None
        self.chunks = None
        self.keyword_index = None
        self.loader = None
        self.chroma_collection = None
        self._embed_model = None
        # project base dir (used for lazy KB loading)
        try:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        except Exception:
            self.base_dir = os.getcwd()

        # NOTE: Do not attempt to import or load the offline KB here — that pulls in
        # heavy dependencies (langchain/pydantic) which can slow or block startup.
        # The KB will be loaded lazily inside `get_retrieved_context` if explicitly enabled
        # via the environment variable RAG_ENABLE_LOAD_DATA=1.

        # Prefer Chroma if available, but do not hard-fail; we can still run BM25-only.
        disable_chroma = os.environ.get('RAG_DISABLE_CHROMA', '0') == '1'
        if not disable_chroma:
            try:
                print("⏳ Initializing Chroma and SentenceTransformers (this may take a moment)...")
                import chromadb
                from chromadb.config import Settings
                from sentence_transformers import SentenceTransformer
                chroma_dir = os.path.join(self.base_dir, 'data', 'braincheck_vectordb')
                if os.path.exists(chroma_dir):
                    settings = Settings(persist_directory=os.path.abspath(chroma_dir), is_persistent=True)
                    try:
                        client = chromadb.Client(settings=settings)
                        try:
                            desired_metric = os.environ.get('RAG_CHROMA_METRIC', 'cosine')
                            collection_name = os.environ.get('RAG_CHROMA_COLLECTION', 'braincheck')
                            try:
                                self.chroma_collection = client.get_collection(collection_name)
                            except Exception:
                                try:
                                    self.chroma_collection = client.create_collection(collection_name, metadata={"distance_metric": desired_metric})
                                except Exception:
                                    try:
                                        self.chroma_collection = client.create_collection(collection_name, metadata={"hnsw:space": desired_metric})
                                    except Exception:
                                        self.chroma_collection = client.create_collection(collection_name)
                            # Ensure we have an embedder for legacy paths
                            self._embed_model = self._embed_model or SentenceTransformer('all-MiniLM-L6-v2')
                            print(f"✅ Loaded Chroma from {chroma_dir} (collection={collection_name}, preferred metric={desired_metric})")
                        except Exception as e:
                            print(f"⚠️ Chroma collection not ready: {e}")
                            self.chroma_collection = None
                    except BaseException as e:
                        print(f"⚠️ Chroma client initialization failed (continuing without dense retrieval): {e}")
                        self.chroma_collection = None
                        self._embed_model = None
                else:
                    self.chroma_collection = None
            except Exception as e:
                print(f"ℹ️ Chroma dependencies not available (dense retrieval will be skipped): {e}")
                self.chroma_collection = None
                self._embed_model = None

        # timeouts and retry configuration (can be adjusted via env vars)
        # Support RAG_MODEL_READ_TIMEOUT as an alias for RAG_GEN_REQUEST_TIMEOUT
        self.request_timeout = int(os.environ.get('RAG_MODEL_READ_TIMEOUT', os.environ.get('RAG_GEN_REQUEST_TIMEOUT', '120')))
        # overall timeout per generation attempt (including retries)
        self.overall_timeout = int(os.environ.get('RAG_GEN_OVERALL_TIMEOUT', '600'))
        # Max retries for generation
        self.max_retries = int(os.environ.get('RAG_MODEL_ATTEMPTS', '3'))
        # Final top-k after optional reranking
        self.retrieval_top_k = int(os.environ.get('RAG_RETRIEVAL_TOP_K', '3'))
        # Pre-retrieval size before reranking (if enabled)
        # Default lowered to 10 to reduce CPU load during rerank
        self.retrieval_pre_k = int(os.environ.get('RAG_RETRIEVE_PRE_K', os.environ.get('RAG_PRE_RERANK_K', '10')))
        # Cross-encoder reranker toggle and settings
        self.rerank_enabled = os.environ.get('RAG_RERANK_ENABLE', '1') != '0'
        self.rerank_model_name = os.environ.get('RAG_RERANK_MODEL', 'BAAI/bge-reranker-base')
        # Force reranker device; default to CPU to leave GPU/MPS to the LLM
        self.rerank_device = os.environ.get('RAG_RERANK_DEVICE', 'cpu')
        # Safety truncation for scoring text (characters)
        self.rerank_passage_chars = int(os.environ.get('RAG_RERANK_PASSAGE_CHARS', '1200'))
        # how many times to poll /api/tags / try lightweight checks when model load errors occur
        self.model_poll_attempts = int(os.environ.get('RAG_MODEL_POLL_ATTEMPTS', '10'))
        self.model_poll_interval = int(os.environ.get('RAG_MODEL_POLL_INTERVAL', '3'))

        # configure basic logging to stdout (module-level callers can reconfigure)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

        # Lazy-built hybrid retriever (BM25 + Vector via RRF)
        self._ensemble_retriever = None
        # Lazy-loaded cross-encoder reranker
        self._cross_encoder = None

    def generate_retrieval_sample(self, n: int = 3, top_k: Optional[int] = None, md_truncate: Optional[int] = None) -> Optional[str]:
        """Create a qualitative retrieval report (Markdown) for N reviewed questions.

        - Selects questions from ground_truth_answers.csv where Answer Type == 'Answer - Reviewed'
        - Uses the EnsembleRetriever (BM25 + Vector via RRF) to fetch top-k passages per question
        - Writes a markdown report under code/test/

        Returns the path to the generated markdown file or None on failure.
        """
        try:
            import pandas as pd
        except Exception as e:
            print(f"❌ pandas not available: {e}")
            return None

        questions_path = os.path.join(self.base_dir, 'ground_truth_answers.csv')
        if not os.path.exists(questions_path):
            print(f"❌ Reviewed questions file not found: {questions_path}")
            return None

        try:
            df = pd.read_csv(questions_path)
        except Exception as e:
            print(f"❌ Failed to read {questions_path}: {e}")
            return None

        reviewed = df[(df.get('Answer Type') == 'Answer - Reviewed') & df.get('Question').notna()].copy()
        if reviewed.empty:
            print("⚠️ No 'Answer - Reviewed' rows found in ground_truth_answers.csv")
            return None

        # Determine parameters
        if top_k is None:
            try:
                top_k = int(self.retrieval_top_k)
            except Exception:
                top_k = 3
        if md_truncate is None:
            try:
                md_truncate = int(os.environ.get('RAG_MD_TRUNCATE', '800'))
            except Exception:
                md_truncate = 800
        # also determine pre-retrieval K when reranking
        pre_k = self.retrieval_pre_k if self.rerank_enabled else (top_k or self.retrieval_top_k)

        # Take first N reviewed questions deterministically
        sampled = reviewed.head(max(1, int(n))).reset_index(drop=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(os.path.dirname(__file__), 'test')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'retrieval_sample_rrf_rerank_topk{top_k}_n{len(sampled)}_{ts}.md')

        lines = []
        lines.append(f"# Retrieval qualitative sample (RRF BM25+Vector + Cross-Encoder Rerank)\n")
        lines.append(f"- Timestamp: {ts}")
        lines.append(f"- Questions source: ground_truth_answers.csv (Answer Type == 'Answer - Reviewed')")
        lines.append(f"- Retrieve top-{pre_k}, rerank, return top-{top_k}")
        if self.rerank_enabled:
            lines.append(f"- Reranker: {self.rerank_model_name}")
            lines.append(f"- Note: LangChain EnsembleRetriever does not expose per-document distances; we report cross-encoder rerank scores instead.")
        lines.append(f"- Printed passage truncate (for this report only): {md_truncate} characters")
        lines.append("")

        for i, row in sampled.iterrows():
            qid = row.get('Question ID') or row.get('Question_ID') or ''
            cat = row.get('Category', '')
            question = row.get('Question', '')
            gt = row.get('Answer', '')

            lines.append(f"## Q{i+1}. {question}")
            meta_bits = []
            if qid:
                meta_bits.append(f"ID: {qid}")
            if cat:
                meta_bits.append(f"Category: {cat}")
            if meta_bits:
                lines.append(f"- {', '.join(meta_bits)}")
            if isinstance(gt, str) and gt.strip():
                gt_show = gt if (md_truncate is None) else gt[:md_truncate]
                lines.append("- Ground truth (reviewed):")
                lines.append("")
                lines.append(gt_show)
                lines.append("")

            # Retrieve full passages for prompt; we will truncate only for printing
            passages, scores, _distances, sources = self.get_retrieved_passages(question, top_k=top_k, truncate=None)
            if not passages:
                lines.append("- Retrieved: (none)")
                lines.append("")
                continue

            lines.append("- Retrieved passages:")
            for j, p in enumerate(passages, start=1):
                p_show = p if (md_truncate is None) else p[:md_truncate]
                src = sources[j-1] if sources and len(sources) >= j else None
                score_txt = ''
                try:
                    if scores and len(scores) >= j and scores[j-1] is not None:
                        score_txt = f" (rerank_score={scores[j-1]:.4f})"
                except Exception:
                    score_txt = ''
                lines.append("")
                lines.append(f"[{j}]{score_txt} {p_show}")
                if src:
                    lines.append(f"Source: {src}")
            lines.append("")

        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"✅ Retrieval sample saved to: {out_path}")
            return out_path
        except Exception as e:
            print(f"❌ Failed to write report: {e}")
            return None

    def generate_answer(self, model_name, question, context=None, max_retries=3):
        """Run a generation with an overall timeout. Uses a thread to ensure we can
        abort if the generation blocks for too long."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._generate_answer_with_context, model_name, question, context, max_retries)
                return fut.result(timeout=self.overall_timeout)
        except concurrent.futures.TimeoutError:
            logging.warning(f"Timeout occurred for question: {question}. Retrying with extended timeout.")
            # Extend timeout and retry once more
            extended_timeout = self.overall_timeout + 30
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(self._generate_answer_with_context, model_name, question, context, max_retries)
                    return fut.result(timeout=extended_timeout)
            except concurrent.futures.TimeoutError:
                return f"Timed out after {extended_timeout}s"
        except Exception as e:
            logging.error(f"Failed to generate answer for question: {question}. Error: {e}")
            return f"Failed to generate answer: {e}"

    def _generate_answer_with_context(self, model_name, question, context: str = None, max_retries=3):
        """Call the Google Gemini API with a short prompt. Returns string answer or failure message.

        The prompt is evidence-first: the model is instructed to only use provided contexts
        and to respond 'Insufficient evidence to answer.' when contexts are insufficient.
        """
        if context:
            logging.info(f"Context length for question '{question[:30]}...': {len(context)} chars")
            # DEBUG: Print context to stdout for user inspection
            print(f"\n--- DEBUG: PROMPT CONTEXT ---\n{context[:2000]}...\n-----------------------------")
        else:
            logging.warning(f"No context provided for question '{question[:30]}...'")

        prompt = self._build_prompt(question, context)
        # Generation constraints: max words
        try:
            max_words = int(os.environ.get('GEN_MAX_WORDS', '130'))
        except Exception:
            max_words = 130
        # Append explicit instruction to help steer the model; we'll also truncate post-hoc as a safety
        prompt += f"\n\nNOTE: Limit your answer to at most {max_words} words."

        # Gemini API payload structure
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800,  # Adjust as needed
            }
        }

        start = time.monotonic()
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    status = getattr(getattr(e, 'response', None), 'status_code', 'unknown')
                    body = ''
                    try:
                        body = (e.response.text or '')[:600] if e.response is not None else ''
                    except Exception:
                        body = ''
                    logging.error(f"HTTPError from Gemini API (status={status}) on attempt {attempt+1}/{max_retries}: {body}")
                    raise
                
                result = response.json()
                # Parse Gemini response
                try:
                    candidates = result.get('candidates', [])
                    if not candidates:
                        return "Model error: No candidates returned"
                    
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if not parts:
                        finish_reason = candidates[0].get('finishReason', 'UNKNOWN')
                        return f"Model error: No content parts returned (finishReason={finish_reason})"
                        
                    answer = parts[0].get('text', '')
                except Exception as e:
                    return f"Model error: Failed to parse response: {e}"

                if isinstance(answer, str):
                    # Truncate to max_words as a safety-net
                    try:
                        words = answer.strip().split()
                        if len(words) > max_words:
                            answer = ' '.join(words[:max_words])
                    except Exception:
                        pass
                    return answer.strip()
                return str(answer)
            except requests.exceptions.Timeout as e:
                logging.warning(f"Request timeout on attempt {attempt+1}/{max_retries}: {e}")
                # retry until overall timeout exceeded
                if time.monotonic() - start > self.overall_timeout:
                    return f"Timed out after {self.overall_timeout}s"
                time.sleep(2)
                continue
            except requests.exceptions.RequestException as e:
                # log error details if available
                try:
                    status = getattr(getattr(e, 'response', None), 'status_code', 'unknown')
                    body = (e.response.text or '')[:600] if getattr(e, 'response', None) is not None else ''
                except Exception:
                    status, body = 'unknown', ''
                logging.error(f"RequestException from Gemini API (status={status}) on attempt {attempt+1}/{max_retries}: {e} {body}")
                # give a short backoff and retry until overall timeout
                if time.monotonic() - start > self.overall_timeout:
                    return f"Failed to generate answer after {max_retries} attempts"
                time.sleep(2)
                continue

        return f"Failed to generate answer after {max_retries} attempts"

    def _build_prompt(self, question: str, context: Optional[str]) -> str:
        """Construct a RAG prompt."""
        has_ctx = bool(context and context.strip())
        
        if not has_ctx:
             return f"""Question: {question}
Context: No context available.
Instruction: Answer the question to the best of your ability.
Answer:"""

        prompt = f"""You are an expert assistant. Answer the question using the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based on the context.
2. If the answer is not explicitly stated, infer it from the context if possible.
3. Cite passage numbers [1], [2] etc.
4. Do NOT say "Insufficient evidence". If you are unsure, provide the most relevant information from the context.
"""
        return prompt

    def _is_model_load_error(self, text: str) -> bool:
        # Gemini API doesn't have "model load" errors in the same way as local Ollama
        return False

    def get_retrieved_context(self, question: str, top_k: int = 3, truncate: int = 300) -> str:
        # vector retrieval first
        if self.chroma_collection is not None and self._embed_model is not None:
            try:
                return self.get_vector_context(question, top_k=top_k, truncate=truncate)
            except Exception:
                pass
        # fallback to simple keyword index (lazy & opt-in)
        if os.environ.get('RAG_ENABLE_LOAD_DATA', '0') != '1':
            # By default we avoid importing load_data/langchain to keep startup fast.
            return None

        # Attempt lazy import of load_data (may pull heavy dependencies)
        try:
            try:
                from load_data import SimpleBrainCheckLoader, load_knowledge_base
            except Exception:
                # try parent folder
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from load_data import SimpleBrainCheckLoader, load_knowledge_base
        except Exception:
            return None

    def _build_ensemble_retriever(self, top_k: int = 3):
        """Build a hybrid retriever using LangChain's EnsembleRetriever (BM25 + Vector).

        Strict requirement: this code relies on LangChain's official EnsembleRetriever.
        If the class isn't available in your installed langchain version, this will raise
        a RuntimeError with an installation/upgrade hint. No manual RRF fallback is used.
        """
        if self._ensemble_retriever is not None:
            return self._ensemble_retriever

        # Try importing components (BM25 + Vector). EnsembleRetriever may or may not exist.
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain_community.vectorstores import Chroma as LCChroma
            from langchain_community.embeddings import SentenceTransformerEmbeddings
        except Exception as e:
            logging.info(f"LangChain community retrievers/vectorstores unavailable: {e}")
            self._ensemble_retriever = None
            return None

        # Load chunks for BM25
        chunks = None
        try:
            try:
                from load_data import load_knowledge_base
            except Exception:
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from load_data import load_knowledge_base

            # Prefer data/ path; fall back to code/ path for backward compatibility
            kb_candidates = [
                os.path.join(self.base_dir, 'data', 'kb_semantic_chunks', 'braincheck_knowledge_base.pkl'),
                os.path.join(self.base_dir, 'data', 'braincheck_knowledge_base.pkl'),
                os.path.join(os.path.dirname(__file__), 'braincheck_knowledge_base.pkl'),
                os.path.join(os.path.dirname(__file__), 'braincheck_knowledge_base_with_emb.pkl'),
            ]
            kb_path = next((p for p in kb_candidates if os.path.exists(p)), None)
            if kb_path:
                kb = load_knowledge_base(kb_path)
                chunks = kb.get('chunks')
        except Exception as e:
            logging.info(f"KB load for BM25 failed or unavailable: {e}")
            chunks = None

        # Build BM25 retriever (with cache)
        bm25 = None
        try:
            if chunks:
                bm25 = self._load_or_build_bm25(chunks, top_k, BM25Retriever)
        except Exception as e:
            logging.info(f"BM25 retriever init failed: {e}")
            bm25 = None

        # Build vector retriever from persisted Chroma
        vector = None
        try:
            chroma_dir = os.path.join(self.base_dir, 'data', 'braincheck_vectordb')
            if os.path.exists(chroma_dir):
                embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
                # Allow switching collection via env (e.g., kb_semantic_chunks_v2_hash)
                collection_name = os.getenv('RAG_CHROMA_COLLECTION', 'braincheck')
                vs = LCChroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=chroma_dir,
                )
                vector = vs.as_retriever(search_kwargs={'k': top_k})
        except Exception as e:
            logging.info(f"Vector retriever init failed: {e}")
            vector = None

        # Use LangChain's EnsembleRetriever
        # Import robustly across versions and package splits (langchain vs langchain_community)
        try:
            try:
                # Modern split often exposes retrievers in langchain_community
                from langchain_community.retrievers import EnsembleRetriever  # type: ignore
            except Exception:
                try:
                    from langchain.retrievers import EnsembleRetriever  # type: ignore
                except Exception:
                    from langchain.retrievers.ensemble import EnsembleRetriever  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LangChain EnsembleRetriever is not available in the current environment. "
                "Please install/upgrade langchain and langchain-community to versions that provide an EnsembleRetriever. "
                f"Import error: {e}"
            )

        retrievers = [r for r in [bm25, vector] if r is not None]
        if not retrievers:
            raise RuntimeError("No retrievers available to build EnsembleRetriever (both BM25 and Vector are unavailable).")
        # Parse env-based weights, e.g., RAG_ENSEMBLE_WEIGHTS="3,1" for Vector:BM25
        weights_env = os.getenv('RAG_ENSEMBLE_WEIGHTS')
        weights: list[float]
        if weights_env:
            try:
                parts = [p.strip() for p in weights_env.split(',') if p.strip()]
                weights = [float(p) for p in parts]
                # If length mismatch, fallback to equal weights but log
                if len(weights) != len(retrievers):
                    logging.info(
                        f"RAG_ENSEMBLE_WEIGHTS length {len(weights)} doesn't match retrievers {len(retrievers)}; using equal weights."
                    )
                    weights = [1.0] * len(retrievers)
            except Exception as e:
                logging.info(f"Failed to parse RAG_ENSEMBLE_WEIGHTS='{weights_env}': {e}; using equal weights.")
                weights = [1.0] * len(retrievers)
        else:
            weights = [1.0] * len(retrievers)

        ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
        if hasattr(ensemble, 'k'):
            try:
                setattr(ensemble, 'k', top_k)
            except Exception:
                pass
        self._ensemble_retriever = ensemble
        logging.info(
            "Using LangChain EnsembleRetriever (BM25 + Vector)"
            + f" with weights={weights}"
        )
        return self._ensemble_retriever

    # Removed manual RRF implementation: relying on LangChain's EnsembleRetriever exclusively

    def retrieve_passages_via_ensemble(self, question: str, top_k: int = 3, truncate: int = 300):
        """Retrieve passages using LangChain EnsembleRetriever (BM25 + Vector).

        Returns: passages, sources, distances (distances are None placeholders)
        """
        retriever = self._build_ensemble_retriever(top_k=top_k)
        # _build_ensemble_retriever is strict; it raises if not available
        docs = []
        try:
            # Newer LC retrievers implement .invoke; fall back to get_relevant_documents
            if hasattr(retriever, 'invoke'):
                docs = retriever.invoke(question)
            else:
                docs = retriever.get_relevant_documents(question)
        except Exception as e:
            logging.info(f"Ensemble retrieval error: {e}")
            raise

        docs = docs[:top_k] if docs else []
        passages = []
        sources = []
        for d in docs:
            try:
                txt = getattr(d, 'page_content', str(d))
            except Exception:
                txt = str(d)
            # Do NOT truncate here: full content goes to the LLM prompt; truncation is only for CSV logging
            passages.append(txt)
            md = getattr(d, 'metadata', {}) or {}
            src = md.get('source') or md.get('source_file') or None
            sources.append(src)
        distances = [None] * len(passages)
        return passages, sources, distances

    def _ensure_reranker(self):
        """Lazily load the cross-encoder reranker model."""
        if not self.rerank_enabled:
            return
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                try:
                    # Force device as configured (default 'cpu')
                    self._cross_encoder = CrossEncoder(self.rerank_model_name, device=self.rerank_device)
                except Exception:
                    # Fallback to a very fast cross-encoder if preferred model isn't available
                    self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.rerank_device)
                # Attempt to report the actual target device
                dev = None
                try:
                    dev = getattr(self._cross_encoder, 'device', None) or getattr(self._cross_encoder, '_target_device', None)
                except Exception:
                    dev = None
                logging.info(
                    f"Using Cross-Encoder reranker: {getattr(self._cross_encoder, 'model_name', self.rerank_model_name)}"
                    + (f" on device: {dev}" if dev is not None else f" (requested device: {self.rerank_device})")
                )
            except Exception as e:
                logging.info(f"Cross-Encoder initialization failed (reranking disabled): {e}")
                self.rerank_enabled = False

    def _rerank_passages(self, query: str, passages: List[str], sources: Optional[List[str]], final_k: int):
        """Score and sort passages using the cross-encoder; return top-N along with sources and scores.

        If reranker fails for any reason, returns inputs truncated to final_k in original order.
        """
        self._ensure_reranker()
        if not self.rerank_enabled or not passages:
            return passages[:final_k], (sources[:final_k] if sources else []), []

        try:
            # Prepare (query, passage) pairs; apply light truncation for safety
            pairs = []
            trunc = max(0, int(self.rerank_passage_chars)) if isinstance(self.rerank_passage_chars, int) else 1200
            for p in passages:
                text = p if trunc <= 0 else (p[:trunc] if isinstance(p, str) else str(p))
                pairs.append((query, text))

            scores = self._cross_encoder.predict(pairs)
            # scores could be numpy array or list
            try:
                scores_list = [float(s) for s in list(scores)]
            except Exception:
                scores_list = [float(s) for s in scores]

            order = sorted(range(len(passages)), key=lambda i: scores_list[i], reverse=True)
            top_idx = order[:final_k]
            reranked_passages = [passages[i] for i in top_idx]
            reranked_sources = [sources[i] for i in top_idx] if sources else []
            reranked_scores = [scores_list[i] for i in top_idx]
            return reranked_passages, reranked_sources, reranked_scores
        except Exception as e:
            logging.info(f"Reranking failed; returning original order: {e}")
            return passages[:final_k], (sources[:final_k] if sources else []), []

    def _compute_chunks_signature(self, chunks) -> str:
        """Compute a lightweight signature for the current chunks set to validate BM25 cache.

        Uses sha256 over a rolling subset of contents to reduce memory overhead.
        """
        h = hashlib.sha256()
        try:
            h.update(str(len(chunks)).encode('utf-8'))
            # sample every Nth chunk to avoid O(n) concatenation cost on huge corpora while still changing when corpus changes
            step = max(1, len(chunks) // 5000)  # cap sampling to ~5k samples max
            for i in range(0, len(chunks), step):
                c = chunks[i]
                try:
                    txt = getattr(c, 'page_content', '')
                    src = ''
                    md = getattr(c, 'metadata', {}) or {}
                    if isinstance(md, dict):
                        src = md.get('source') or md.get('source_file') or ''
                    h.update(str(len(txt)).encode('utf-8'))
                    # add small prefix/suffix snippets to reflect content variation without hashing entire strings
                    h.update((txt[:256] + txt[-256:]).encode('utf-8', errors='ignore'))
                    if src:
                        h.update(src.encode('utf-8', errors='ignore'))
                except Exception:
                    continue
            return h.hexdigest()
        except Exception:
            return 'unknown'

    def _load_or_build_bm25(self, chunks, top_k, BM25RetrieverClass):
        """Load BM25 retriever from disk if signature matches; otherwise build and persist.

        Cache files: data/bm25_retriever.pkl and data/bm25_retriever.meta.json
        """
        index_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(index_dir, exist_ok=True)
        idx_pkl = os.path.join(index_dir, 'bm25_retriever.pkl')
        idx_meta = os.path.join(index_dir, 'bm25_retriever.meta.json')

        sig = self._compute_chunks_signature(chunks)
        # Try load
        try:
            if os.path.exists(idx_pkl) and os.path.exists(idx_meta):
                with open(idx_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if meta.get('signature') == sig:
                    with open(idx_pkl, 'rb') as f:
                        bm25 = pickle.load(f)
                    try:
                        bm25.k = top_k
                    except Exception:
                        pass
                    logging.info("Loaded cached BM25 retriever from disk")
                    return bm25
        except Exception as e:
            logging.info(f"BM25 cache load failed (will rebuild): {e}")

        # Build fresh
        bm25 = BM25RetrieverClass.from_documents(chunks)
        try:
            bm25.k = top_k
        except Exception:
            pass
        # Persist
        try:
            tmp_pkl = idx_pkl + '.tmp'
            with open(tmp_pkl, 'wb') as f:
                pickle.dump(bm25, f)
            os.replace(tmp_pkl, idx_pkl)
            meta = {'signature': sig, 'top_k_default': top_k}
            tmp_meta = idx_meta + '.tmp'
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
            os.replace(tmp_meta, idx_meta)
            logging.info("Persisted BM25 retriever cache to disk")
        except Exception as e:
            logging.info(f"BM25 cache persist skipped (non-fatal): {e}")

        return bm25

    def get_vector_context(self, question: str, top_k: int = 3, truncate: int = 300) -> str:
        # Only use Chroma (no fallback)
        if self.chroma_collection is not None:
            if self._embed_model is None:
                self._ensure_embed_model()
            q_emb = self._embed_model.encode(question).tolist()
            res = self.chroma_collection.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas', 'distances'])
            docs = []
            if res and 'documents' in res and res['documents']:
                for d in res['documents'][0]:
                    t = d
                    if truncate:
                        t = t[:truncate]
                    docs.append(t)
            return "\n\n".join(docs)
        return None

    def get_keyword_passages(self, question: str, top_k: int = 3, truncate: int = 300):
        """Perform a simple keyword-based retrieval using the local KB loader (lazy).

        Returns a list of passage strings (may be empty). This is intentionally
        lightweight and will return [] if the KB loader or data is not available.
        """
        # Only attempt keyword retrieval when explicitly enabled (to avoid heavy imports)
        if os.environ.get('RAG_ENABLE_LOAD_DATA', '0') != '1':
            return []

        try:
            try:
                from load_data import SimpleBrainCheckLoader, load_knowledge_base
            except Exception:
                parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from load_data import SimpleBrainCheckLoader, load_knowledge_base
        except Exception:
            return []

        try:
            candidate = os.path.join(self.base_dir, 'data', 'braincheck_knowledge_base.pkl')
            if not os.path.exists(candidate):
                return []

            # Load KB (may reuse cached self.kb if already loaded)
            if not self.kb:
                self.kb = load_knowledge_base(candidate)
                self.chunks = self.kb.get('chunks')
                self.keyword_index = self.kb.get('keyword_index')
                self.loader = SimpleBrainCheckLoader() if SimpleBrainCheckLoader is not None else None

            if not self.loader:
                return []

            results = self.loader.search_simple(self.chunks, self.keyword_index, question, top_k=top_k)
            texts = []
            for chunk in results:
                text = getattr(chunk, 'page_content', str(chunk))
                if truncate:
                    text = text[:truncate]
                texts.append(text)
            return texts
        except Exception:
            return []

    def get_retrieved_passages(self, question: str, top_k: int = 10, truncate: int = 300):
        """Retrieve Top-K with EnsembleRetriever, optionally rerank Top-20, return Top-k final.

        Returns: (passages, scores, raw_distances, sources) where raw_distances are None placeholders.
        """
        # Determine how many to pull before reranking
        pre_k = self.retrieval_pre_k if self.rerank_enabled else top_k
        pre_k = max(top_k, pre_k)
        # Retrieve full passages for prompting; truncation is applied later for CSV logging only
        passages_raw, sources_raw, distances = self.retrieve_passages_via_ensemble(question, top_k=pre_k, truncate=None)
        # Apply reranking if enabled
        passages, sources, scores = self._rerank_passages(question, passages_raw, sources_raw, final_k=top_k)
        return passages, scores, distances, sources

    def _ensure_embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                raise RuntimeError(f"Failed to initialize embedding model: {e}")

    def generate_all_answers(self, questions_df):
        results = []
        total = len(questions_df)
        # CSV logging truncate length (does NOT affect LLM prompt). Default 800 for readable logs.
        try:
            passage_truncate = int(os.environ.get('RAG_PASSAGE_TRUNCATE', '800'))
        except Exception:
            passage_truncate = 800
        for idx, row in questions_df.iterrows():
            qid = row.get('Question ID') or row.get('Question_ID') or idx
            category = row.get('Category', '')
            question = row.get('Question')
            ground_truth = row.get('Answer') or row.get('Ground Truth')

            # retrieve top-k passages (default set on the generator instance)
            top_k = int(self.retrieval_top_k)
            passages, scores, raw_distances, sources = self.get_retrieved_passages(question, top_k=top_k, truncate=None)
            # join passages into a context string for the prompt (numbered)
            if passages:
                numbered = []
                for i, p in enumerate(passages, start=1):
                    numbered.append(f"[{i}] {p}")
                context = "\n\n".join(numbered)
            else:
                # fallback to legacy context function (which may query simple KB)
                context = self.get_retrieved_context(question, top_k=top_k, truncate=None)

            # use generate_answer wrapper which enforces per-question overall timeout
            # pass the retrieved `context` so the evidence-first prompt is used
            llama = self.generate_answer(self.gemini_model, question, context=context, max_retries=self.max_retries)
            # Adaptive fallbacks if generation failed due to API errors/timeouts
            if isinstance(llama, str) and (llama.startswith("Failed to generate") or llama.startswith("Timed out") or llama.startswith("Model error")):
                # Try a shorter context: reduce number of passages and slice each passage shorter
                if passages:
                    reduced_count = max(3, min(len(passages), (len(passages) + 1) // 2))
                    reduced_passages = [p[:min(350, passage_truncate)] if isinstance(p, str) else p for p in passages[:reduced_count]]
                    reduced_ctx = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(reduced_passages, start=1)])
                    llama2 = self.generate_answer(self.gemini_model, question, context=reduced_ctx, max_retries=self.max_retries)
                    if not (isinstance(llama2, str) and (llama2.startswith("Failed to generate") or llama2.startswith("Timed out") or llama2.startswith("Model error"))):
                        llama = llama2
                    else:
                        # Final fallback: try without any context
                        llama3 = self.generate_answer(self.gemini_model, question, context=None, max_retries=max(1, self.max_retries - 1))
                        llama = llama3
                else:
                    # No passages; try a final no-context attempt
                    llama2 = self.generate_answer(self.gemini_model, question, context=None, max_retries=max(1, self.max_retries - 1))
                    llama = llama2
            if "Timed out" in str(llama):
                llama = "Default answer: Insufficient evidence to answer."

            # For CSV logging: store truncated copy to keep files compact
            passages_csv = [ (p[:passage_truncate] if isinstance(p, str) and passage_truncate else p) for p in passages ]

            results.append({
                'Question_ID': qid,
                'Category': category,
                'Question': question,
                'Ground_Truth_Answer': ground_truth,
                'Real_LLaMA_Answer': llama,
                'Retrieved_Passages': json.dumps(passages_csv, ensure_ascii=False),
                'Rerank_Scores': json.dumps(scores, ensure_ascii=False),
                'Retrieved_Sources': json.dumps(sources, ensure_ascii=False),
            })
            time.sleep(0.5)
        return results


def main():
    print("=== Real model answer generation (Gemini 3) ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Allow overriding the questions file via env var
    custom_questions_file = os.environ.get('RAG_QUESTIONS_FILE')
    if custom_questions_file:
        # If absolute path, use as is; otherwise relative to data dir
        if os.path.isabs(custom_questions_file):
            questions_file = custom_questions_file
        else:
            questions_file = os.path.join(base_dir, 'data', custom_questions_file)
    else:
        questions_file = os.path.join(base_dir, 'data', 'selected_questions.csv')

    if not os.path.exists(questions_file):
        print(f"ERROR: Questions file not found: {questions_file}")
        return None

    questions_df = pd.read_csv(questions_file)
    print(f"Loaded {len(questions_df)} questions")

    # Enable KB loading by default for this run so the generator can use the
    # local knowledge base. You can still override via env var RAG_ENABLE_LOAD_DATA=0.
    os.environ['RAG_ENABLE_LOAD_DATA'] = os.environ.get('RAG_ENABLE_LOAD_DATA', '1')
    if os.environ.get('RAG_ENABLE_LOAD_DATA') != '1':
        print("ℹ️ KB loading disabled for this run (RAG_ENABLE_LOAD_DATA!=1)")
    else:
        print("ℹ️ KB loading enabled for this run (RAG_ENABLE_LOAD_DATA=1)")
    generator = RealModelAnswerGenerator()

    # Retrieval-only qualitative sample mode: skip LLM connectivity and generation
    if os.environ.get('RAG_RETRIEVAL_SAMPLE', '0') == '1':
        try:
            n = int(os.environ.get('RAG_QUAL_SAMPLE_N', '3'))
        except Exception:
            n = 3
        try:
            top_k_env = os.environ.get('RAG_RETRIEVAL_TOP_K')
            top_k = int(top_k_env) if top_k_env is not None else int(generator.retrieval_top_k)
        except Exception:
            top_k = 3
        out = generator.generate_retrieval_sample(n=n, top_k=top_k)
        return out

    # Connectivity check for Gemini
    print(f"ℹ️ Testing connectivity to Gemini API ({generator.gemini_model})...")
    test = generator.generate_answer(generator.gemini_model, "What is dementia?", max_retries=2)
    
    if str(test).startswith("Failed to generate") or str(test).startswith("Timed out") or str(test).startswith("Model error"):
        print(f"❌ Model request failed: {test}")
        print("Please check your API key and internet connection.")
        return None
    else:
        print(f"✅ Connectivity test passed, sample output: {test[:200]}{'...' if len(test)>200 else ''}")

    try:
        results = generator.generate_all_answers(questions_df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        out = os.path.join(results_dir, f'real_answers_gemini_{timestamp}.csv')
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"Results saved to: {out}")
        return out
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return None


if __name__ == '__main__':
    main()
