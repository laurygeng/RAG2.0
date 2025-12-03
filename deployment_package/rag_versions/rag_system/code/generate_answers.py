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

# Ensure the code folder is on sys.path (we avoid importing heavy modules here).
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Defer importing `load_data` (and transitively heavy libs like langchain/pydantic)
# until it's explicitly needed. Provide placeholders so code can run without them.
SimpleBrainCheckLoader = None
load_knowledge_base = None

class RealModelAnswerGenerator:
    """Generate answers from local LLM endpoints (simple wrapper used in experiments)."""

    def __init__(self):
        # Use 127.0.0.1 instead of 'localhost' to avoid potential IPv6 vs IPv4
        # name-resolution differences that can make requests miss a server
        # that is listening only on the IPv4 loopback.
        self.api_url = "http://127.0.0.1:11434/api/generate"
        self.llama_model = "llama3.2:latest"  # Only use llama; remove mistral per user request

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
                            try:
                                self.chroma_collection = client.get_collection('braincheck')
                            except Exception:
                                try:
                                    self.chroma_collection = client.create_collection('braincheck', metadata={"distance_metric": desired_metric})
                                except Exception:
                                    try:
                                        self.chroma_collection = client.create_collection('braincheck', metadata={"hnsw:space": desired_metric})
                                    except Exception:
                                        self.chroma_collection = client.create_collection('braincheck')
                            # Ensure we have an embedder for legacy paths
                            self._embed_model = self._embed_model or SentenceTransformer('all-MiniLM-L6-v2')
                            print(f"✅ Loaded Chroma from {chroma_dir} (preferred metric={desired_metric})")
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
        self.request_timeout = int(os.environ.get('RAG_GEN_REQUEST_TIMEOUT', '45'))
        # overall timeout per generation attempt (including retries)
        self.overall_timeout = int(os.environ.get('RAG_GEN_OVERALL_TIMEOUT', '300'))
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

    def generate_retrieval_sample(self, n: int = 3, top_k: int | None = None, md_truncate: int | None = None) -> str | None:
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

    def get_available_models(self):
        """Return a set of model names registered on the ollama server, or empty set on error."""
        try:
            # self.api_url is usually 'http://127.0.0.1:11434/api/generate'.
            # Avoid constructing '/api/api/tags' by splitting at the '/api' segment.
            tags_url = self.api_url.rsplit('/api', 1)[0] + '/api/tags'
            resp = requests.get(tags_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = {m.get('name') for m in data.get('models', []) if m.get('name')}
            return models
        except Exception:
            return set()

    def register_local_manifests(self):
        """If the running Ollama doesn't expose models but ~/.ollama contains manifests,
        attempt to register those manifests with the running binary by invoking the
        local `ollama pull registry.ollama.ai/library/<name>:<tag>` commands.
        Returns the set of model names it attempted to register.
        """
        """Try to register manifests by invoking `ollama pull` for manifests found in ~/.ollama.
        This routine will try common ollama binary locations and a couple of pull command
        variants and will capture stdout/stderr so failures are visible in logs.
        Returns the set of model tags it attempted to register (e.g. 'llama3.2:latest')."""

        tried = set()
        try:
            home_manifests = os.path.expanduser('~/.ollama/models/manifests/registry.ollama.ai/library')
            if not os.path.isdir(home_manifests):
                logging.info(f"No local manifests dir found at {home_manifests}")
                return tried

            # common ollama binary candidates to try. Allow overriding the binary via
            # env var RAG_OLLAMA_BIN (useful when multiple installations exist).
            import shutil
            candidates = []
            # prefer user-provided binary path
            user_bin = os.environ.get('RAG_OLLAMA_BIN')
            if user_bin:
                candidates.append(user_bin)
            repo_local = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ollama'))
            if os.path.exists(repo_local):
                candidates.append(repo_local)
            which_ollama = shutil.which('ollama')
            if which_ollama:
                candidates.append(which_ollama)
            # Homebrew common locations (Intel / Apple Silicon)
            candidates.extend(['/usr/local/bin/ollama', '/opt/homebrew/bin/ollama'])

            # dedupe while preserving order
            seen = set()
            bins = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    bins.append(c)

            import subprocess

            # find model folders like 'llama3.2' or 'mistral'
            for model_name in os.listdir(home_manifests):
                model_dir = os.path.join(home_manifests, model_name)
                if not os.path.isdir(model_dir):
                    continue
                for tag_fname in os.listdir(model_dir):
                    tag = tag_fname
                    # prefer short form 'llama3.2:latest' but also try the fully-qualified registry form
                    short_tag = f"{model_name}:{tag}"
                    fq_tag = f"registry.ollama.ai/library/{model_name}:{tag}"
                    tried.add(short_tag)

                    env = os.environ.copy()
                    env['OLLAMA_DIR'] = os.path.expanduser('~/.ollama')

                    # try each binary and each tag format, capture output
                    for bin_path in bins:
                        if not os.path.exists(bin_path):
                            continue
                        for candidate_tag in (short_tag, fq_tag):
                            cmd = [bin_path, 'pull', candidate_tag]
                            try:
                                logging.info(f"Attempting to run: {cmd} (OLLAMA_DIR={env['OLLAMA_DIR']})")
                                proc = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True, timeout=180)
                                out = (proc.stdout or '').strip()
                                err = (proc.stderr or '').strip()
                                logging.info(f"ollama pull exit={proc.returncode}; stdout={out[:800]}{'...' if len(out)>800 else ''}")
                                if err:
                                    logging.info(f"ollama pull stderr: {err[:800]}{'...' if len(err)>800 else ''}")
                                    # If the error indicates the ollama daemon isn't running, provide a helpful hint
                                    if 'server not responding' in err.lower() or 'could not find ollama app' in err.lower():
                                        logging.info("Hint: the ollama binary ran but could not contact a running ollama server.\n" \
                                                     "Start the server with: OLLAMA_DIR=~/.ollama <path-to-ollama> serve &\n" \
                                                     "or ensure the running service uses the same OLLAMA_DIR. Then retry this script.")
                                # if exit code is 0 we consider the pull attempted/successful
                                if proc.returncode == 0:
                                    logging.info(f"Successfully executed pull: {candidate_tag} using {bin_path}")
                                    # once a pull succeeds for this tag, stop trying other candidates for it
                                    raise StopIteration
                            except StopIteration:
                                break
                            except Exception as e:
                                logging.info(f"pull command failed: {cmd} -> {e}")
                        else:
                            # inner loop didn't break; continue to next binary
                            continue
                        # inner loop broke due to StopIteration: break outer binary loop
                        break

            return tried
        except Exception as e:
            logging.info(f"register_local_manifests error: {e}")
            return tried

    def _generate_answer_with_context(self, model_name, question, context: str = None, max_retries=3):
        """Call the local generation API with a short prompt. Returns string answer or failure message.

        The prompt is evidence-first: the model is instructed to only use provided contexts
        and to respond 'Insufficient evidence to answer.' when contexts are insufficient.
        """
        prompt = self._build_prompt(question, context)
        # Generation constraints: max words
        try:
            max_words = int(os.environ.get('GEN_MAX_WORDS', '130'))
        except Exception:
            max_words = 130
        # Append explicit instruction to help steer the model; we'll also truncate post-hoc as a safety
        prompt += f"\n\nNOTE: Limit your answer to at most {max_words} words."

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            # Ollama's HTTP API rejects unknown option keys like `max_tokens`/`top_p`.
            # Keep options minimal and compatible with the server (temperature is accepted).
            "options": {"temperature": 0.3}
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
                    logging.error(f"HTTPError from Ollama API (status={status}) on attempt {attempt+1}/{max_retries}: {body}")
                    raise
                result = response.json()
                # Ollama may return an error string in an 'error' field or as the HTTP body
                if isinstance(result, dict) and result.get('error'):
                    return f"Model error: {result.get('error')}"
                answer = result.get('response', '')
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
                logging.error(f"RequestException from Ollama API (status={status}) on attempt {attempt+1}/{max_retries}: {e} {body}")
                # give a short backoff and retry until overall timeout
                if time.monotonic() - start > self.overall_timeout:
                    return f"Failed to generate answer after {max_retries} attempts"
                time.sleep(2)
                continue

        return f"Failed to generate answer after {max_retries} attempts"

    def _build_prompt(self, question: str, context: str | None) -> str:
        """Construct a stricter RAG prompt with role, golden rules, citation policy (English only)."""
        insuff = "Insufficient evidence to answer."
        has_ctx = bool(context and context.strip())
        role = "You are a professional medical advisor. Your sole goal is to produce an accurate, evidence-grounded answer strictly from the provided contexts."
        rules_intro = "Golden Rules:"
        rules = [
            "1. Use ONLY the provided [Context] passages; no external knowledge or speculation.",
            f"2. If evidence is insufficient, reply EXACTLY: '{insuff}'.",
            "3. Cite supporting passage indices (e.g. [1]) immediately after facts; indices must exist in the provided list.",
            "4. Do NOT hallucinate or add unsupported details; avoid vague hedging unless present in a passage.",
            "5. Answer format: 2-4 concise sentences, then a final line: Sources: [1], [2].",
            "6. Do NOT emit disclaimers, safety notices, or meta commentary.",
            "7. Do not cite an index you did not use; do not fabricate combined evidence.",
        ]
        question_label = "Question"
        ctx_header = "Contexts"

        rules_block = rules_intro + "\n" + "\n".join(rules)

        if has_ctx:
            prompt_parts = [
                role,
                rules_block,
                f"{ctx_header}:\n{context}",
                f"{question_label}: {question}",
                "Follow all rules precisely. If unsure, output the insufficiency phrase only.",
            ]
        else:
            # No context case: instruct conservative behavior
            if lang == 'zh':
                no_ctx_rules = (
                    f"无可用上下文。若无法仅凭证据安全回答，必须输出：'{insuff}'. 不要编造。"
                )
            else:
                no_ctx_rules = (
                    f"No contexts available. If you cannot answer safely without unsupported fabrication, reply EXACTLY: '{insuff}'. Do not invent."
                )
            prompt_parts = [role, rules_block, no_ctx_rules, f"{question_label}: {question}"]

        return "\n\n".join(prompt_parts).strip()

    def _is_model_load_error(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        checks = [
            'failed to load model',
            'incompatible',
            'model may be incompatible',
            'failed to load',
            'loader error',
        ]
        return any(c in t for c in checks)

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
                vs = LCChroma(
                    collection_name='braincheck',
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

        ensemble = EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))
        if hasattr(ensemble, 'k'):
            try:
                setattr(ensemble, 'k', top_k)
            except Exception:
                pass
        self._ensemble_retriever = ensemble
        logging.info("Using LangChain EnsembleRetriever (BM25 + Vector)")
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

    def _rerank_passages(self, query: str, passages: list[str], sources: list[str] | None, final_k: int):
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
            llama = self.generate_answer(self.llama_model, question, context=context, max_retries=3)
            # Adaptive fallbacks if generation failed due to API errors/timeouts
            if isinstance(llama, str) and (llama.startswith("Failed to generate") or llama.startswith("Timed out") or llama.startswith("Model error")):
                # Try a shorter context: reduce number of passages and slice each passage shorter
                if passages:
                    reduced_count = max(3, min(len(passages), (len(passages) + 1) // 2))
                    reduced_passages = [p[:min(350, passage_truncate)] if isinstance(p, str) else p for p in passages[:reduced_count]]
                    reduced_ctx = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(reduced_passages, start=1)])
                    llama2 = self.generate_answer(self.llama_model, question, context=reduced_ctx, max_retries=3)
                    if not (isinstance(llama2, str) and (llama2.startswith("Failed to generate") or llama2.startswith("Timed out") or llama2.startswith("Model error"))):
                        llama = llama2
                    else:
                        # Final fallback: try without any context
                        llama3 = self.generate_answer(self.llama_model, question, context=None, max_retries=2)
                        llama = llama3
                else:
                    # No passages; try a final no-context attempt
                    llama2 = self.generate_answer(self.llama_model, question, context=None, max_retries=2)
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
    print("=== Real model answer generation ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

    # quick connectivity check: ask the Ollama server which models are registered
    available = generator.get_available_models()
    if available:
        print(f"✅ Available models detected: {', '.join(sorted(available))}")
    else:
        print("⚠️ Ollama server is running but no registered models detected (will attempt to register local manifests or start a local ollama server).")
        # Optionally try to start a local ollama server automatically (can be disabled via RAG_AUTO_START_OLLAMA=0)
        auto_start = os.environ.get('RAG_AUTO_START_OLLAMA', '1')
        tried = set()
        if auto_start != '0':
            # determine binary to use (allow override via RAG_OLLAMA_BIN)
            bin_candidates = []
            user_bin = os.environ.get('RAG_OLLAMA_BIN')
            if user_bin:
                bin_candidates.append(user_bin)
            which_bin = shutil.which('ollama')
            if which_bin:
                bin_candidates.append(which_bin)
            bin_candidates.extend(['/usr/local/bin/ollama', '/opt/homebrew/bin/ollama'])

            # find the first existing binary
            bin_path = None
            for b in bin_candidates:
                if b and os.path.exists(b):
                    bin_path = b
                    break

            if bin_path:
                print(f"ℹ️ Attempting to start ollama server using: {bin_path}")
                # start in background and redirect logs to results/ollama_serve_<ts>.log
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs(os.path.join(generator.base_dir, 'results'), exist_ok=True)
                serve_log = os.path.join(generator.base_dir, 'results', f'ollama_serve_{ts}.log')
                env = os.environ.copy()
                env['OLLAMA_DIR'] = os.path.expanduser('~/.ollama')
                try:
                    lf = open(serve_log, 'a')
                    proc = subprocess.Popen([bin_path, 'serve'], env=env, stdout=lf, stderr=lf)
                    print(f"ℹ️ Launched ollama serve (pid={proc.pid}), logs: {serve_log}")
                    # give server a short time to initialize
                    time.sleep(5)
                except Exception as e:
                    print(f"⚠️ Failed to start ollama serve automatically: {e}")
            else:
                print("⚠️ No ollama binary found in candidates; set RAG_OLLAMA_BIN to point to your ollama binary if you want auto-start.")

        # Simplify: directly pull only llama3.2:latest (skip manifest scan & other models)
        print("ℹ️ Pulling llama3.2:latest (single-model mode)...")
        try:
            bin_path = os.environ.get('RAG_OLLAMA_BIN') or shutil.which('ollama') or '/usr/local/bin/ollama'
            if bin_path and os.path.exists(bin_path):
                env2 = os.environ.copy(); env2['OLLAMA_DIR'] = os.path.expanduser('~/.ollama')
                proc_pull = subprocess.run([bin_path, 'pull', generator.llama_model], check=False, env=env2, capture_output=True, text=True, timeout=600)
                if proc_pull.returncode == 0:
                    print("✅ pull llama3.2:latest success")
                else:
                    print(f"⚠️ pull llama3.2:latest exit={proc_pull.returncode} stderr={proc_pull.stderr[:200]}")
            else:
                print("⚠️ ollama binary not found; set RAG_OLLAMA_BIN to explicit path.")
        except Exception as e:
            print(f"⚠️ llama3.2 pull failed: {e}")
        # Poll for availability
        for i in range(5):
            time.sleep(2)
            available = generator.get_available_models()
            if generator.llama_model in available:
                print(f"✅ Detected {generator.llama_model} after attempt {i+1}")
                break

    # If no models are available, bail out immediately — stop further work so we
    # can diagnose and fix Ollama connectivity first.
    if not available:
        print("❌ No available models detected; aborting run so Ollama connectivity can be diagnosed. Check previous pull/log output and fix before retrying.")
        return None

    # choose a model for a short smoke-test (only use llama3.2:latest)
    test_model = None
    if generator.llama_model in available:
        test_model = generator.llama_model
    else:
        print(f"❌ Required model {generator.llama_model} not available; aborting connectivity test.")

    if test_model:
        test = generator.generate_answer(test_model, "What is dementia?", max_retries=2)
        # If we see model-load errors or timeouts, poll a few times to allow the server to finish loading
        if generator._is_model_load_error(str(test)) or str(test).startswith("Timed out") or str(test).startswith("Failed to generate"):
            logging.info(f"Initial test returned an error/timeout: {test}. Polling up to {generator.model_poll_attempts} times to wait for model readiness.")
            for i in range(generator.model_poll_attempts):
                time.sleep(generator.model_poll_interval)
                test = generator.generate_answer(test_model, "What is dementia?", max_retries=2)
                if not (generator._is_model_load_error(str(test)) or str(test).startswith("Timed out") or str(test).startswith("Failed to generate")):
                    break
                logging.info(f"Attempt {i+1}/{generator.model_poll_attempts} still failing: {test}")

        if str(test).startswith("Failed to generate") or str(test).startswith("Timed out") or generator._is_model_load_error(str(test)):
            print(f"❌ Model request failed: {test}")
        else:
            print(f"✅ Connectivity test passed (model: {test_model}), sample output: {test[:200]}{'...' if len(test)>200 else ''}")
    else:
        print("❌ No model found for connectivity test; skipping generation test.")

    try:
        results = generator.generate_all_answers(questions_df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        out = os.path.join(results_dir, f'real_answers_{timestamp}.csv')
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"Results saved to: {out}")
        return out
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return None


if __name__ == '__main__':
    main()
    # Load questions (from data folder)
