# RAG system architecture (as of 2025-11-10)

This document captures the end-to-end architecture and the concrete models, parameters, and environment toggles currently used in this workspace. It reflects the code under `deployment_package/rag_versions/rag_system/code` and the latest runs captured in `evaluation/` and `results/`.

## Diagram (Updated)

```mermaid
flowchart TB
  subgraph Ingestion[1. Ingestion & Cleaning]
    FS[downloaded_files/* PDF DOCX TXT] --> LDR[Loaders: PyMuPDF / PyPDF / Docx2txt / Text]
    LDR --> CLEAN[Pre-clean: headers/footers\nref pages removal\nURL noise filtering\nspacing normalization]
    CLEAN --> SPLIT[RecursiveCharacterTextSplitter\nchunk_size=RAG_CHUNK_TOKENS (500)\noverlap=RAG_CHUNK_TOKENS_OVERLAP (100)]
    SPLIT --> CHUNKS[Chunks (Document[])\nmetadata: chunk_id,size,preview]
    CHUNKS --> KWIDX[Keyword index]
  end

  subgraph Indexing[2. Indexing]
    CHUNKS --> BM25[BM25 retriever cache\nsha256 signature]
    CHUNKS --> EMBED[SentenceTransformer all-MiniLM-L6-v2]
    EMBED --> CHROMA[Chroma DB persistent\ncollection=braincheck metric=cosine]
  end

  subgraph Retrieval[3. Hybrid Retrieval]
    Q[Question] --> ENS[EnsembleRetriever\n(BM25 + Vector)]
    ENS --> PREK[Pre-retrieval Top-M (M=RAG_RETRIEVE_PRE_K)]
    PREK --> RERANK[Cross-Encoder (CPU)\nmodel=RAG_RERANK_MODEL]
    RERANK --> TOPK[Final Top-K passages\nK=RAG_RETRIEVAL_TOP_K]
  end

  subgraph Prompt[4. Prompt]
    TOPK --> PROMPT[Strict RAG Prompt\nrole + golden rules\ncitations [i]]
  end

  subgraph Generation[5. Generation]
    PROMPT --> OLLAMA[Ollama API /api/generate\nmodel=llama3.2:latest]
    OLLAMA --> FALLBACK[Timeout fallbacks]
    FALLBACK --> ANSWER[Answer]
  end

  subgraph Evaluation[6. Evaluation]
    ANSWER --> METRICS[semantic / jaccard / overlap / combined]
  end

  subgraph Outputs[7. Artifacts]
    TOPK --> OUT1[Retrieval sample MD]
    ANSWER --> OUT2[results CSV (Rerank_Scores)]
    METRICS --> OUT3[evaluation CSV]
  end
```

## Core configuration (env vars)

- Server and models

  - OLLAMA endpoint: `http://127.0.0.1:11434/api/generate`
  - Generation model: `llama3.2:latest` (default), alt `mistral:7b-instruct`
  - Auto-start Ollama: `RAG_AUTO_START_OLLAMA=1` (default), override binary via `RAG_OLLAMA_BIN`
  - Connectivity/model discovery: GET `<host>/api/tags`

- Timeouts and retries (generate_answers.py)

  - `RAG_GEN_REQUEST_TIMEOUT` default 45s (per HTTP request)
  - `RAG_GEN_OVERALL_TIMEOUT` default 300s (per question, including retries)
  - `RAG_MODEL_POLL_ATTEMPTS` default 10; `RAG_MODEL_POLL_INTERVAL` default 3s
  - `GEN_MAX_WORDS` default 130 for prompt word cap

- Retrieval

  - `RAG_RETRIEVAL_TOP_K` default 3 (used in generate_all_answers for prompt contexts)
  - Fusion weight `RAG_FUSION_ALPHA` default 0.7
  - Chroma metric `RAG_CHROMA_METRIC` default `cosine` (fallback metadata key `hnsw:space`)
  - Disable Chroma: `RAG_DISABLE_CHROMA=1` to force fallback
  - Enable simple KB loading: `RAG_ENABLE_LOAD_DATA=1` (default in generate_answers main)

- Embedding selection

  - Evaluation: `CUSTOM_EMB_MODEL` (e.g., `BAAI/bge-large-en-v1.5`), `CUSTOM_EMB_TRUST_REMOTE=1`
  - Basic evaluator override: `EVAL_EMB_MODEL`
  - Export embeddings: `RAG_EMBED_MODEL` (default `all-MiniLM-L6-v2`)
  - load_data embedding default is `'bge-m3'` (forced CPU); falls back to HF Inference API if configured

- Five-metrics evaluator (evaluate_ragas.py)
  - `RAGAS_TOPK_CONTEXTS` default 5 (contexts considered per row)
  - `RAGAS_SIM_THR` default 0.6 (similarity threshold for relevance/support)
  - `RAGAS_AC_THR` default 0.7 (sentence match threshold for answer_correctness F1)
  - `RAGAS_MAX_SAMPLES` default 0 (no limit)
  - Reverse-QG for answer_relevancy: `ANSWER_REL_USE_QG=1` + `ANSWER_REL_QG_MODEL` (default `google/flan-t5-small`) + `ANSWER_REL_QG_NUM` (default 3) + `ANSWER_REL_QG_MAXTOK` (default 64)

## Chunking settings (Updated)

- Splitter: `RecursiveCharacterTextSplitter.from_tiktoken_encoder`
- Parameters: `chunk_size=RAG_CHUNK_TOKENS (default 500)`, `chunk_overlap=RAG_CHUNK_TOKENS_OVERLAP (default 100)`
- Mixed-language separators: `["\n\n", "\n", "。", "！", "？", ".", " "]`
- Metadata: `chunk_id`, `chunk_size`, `preview`

## Embedding settings

- Index building (`build_chroma_from_kb.py`): `SentenceTransformer('all-MiniLM-L6-v2')`
- KB export (`export_kb_for_chroma.py`): `RAG_EMBED_MODEL` (default `'all-MiniLM-L6-v2'`)
- Loader embedding helper (`load_data.embed_chunks`): default `model_name='bge-m3'` (CPU forced). Remote Hugging Face fallbacks have been disabled; only local sentence-transformers is used.
- Runtime retrieval (generate_answers.py): embeds questions via `SentenceTransformer('all-MiniLM-L6-v2')` when querying Chroma or fallback vectors.
- Evaluation stages use the configured local embedding model (no remote inference).

Note: On this machine torch==2.2.2 blocks `BAAI/bge-m3` due to the `torch.load` security gate (requires torch≥2.6). Current successful runs use `BAAI/bge-large-en-v1.5`.

## Indexing settings (Chroma)

- Persist dir: `data/braincheck_vectordb`
- Collection: `braincheck`
- Distance metric: from env `RAG_CHROMA_METRIC` (default `'cosine'`); creation falls back to default if metadata keys not supported
- Import/export: `export_kb_for_chroma.py` → `chroma_import_*.jsonl` → `import_chroma_jsonl.py`
- Batch sizes: add in batches (default 4096 in build, 512 in import)

## Retrieval settings (Updated)

- EnsembleRetriever: official LangChain `EnsembleRetriever` combining BM25 + Vector (no manual RRF / custom fusion).
- Pre-retrieval size: `RAG_RETRIEVE_PRE_K` (default 10; adjustable: 3 / 5 / 20 scenarios).
- Final Top-K for prompt contexts: `RAG_RETRIEVAL_TOP_K` (default 3; can be 1 for minimal context tests).
- Cross-Encoder rerank: `RAG_RERANK_ENABLE=1`, `RAG_RERANK_MODEL=BAAI/bge-reranker-base` (CPU forced via `RAG_RERANK_DEVICE=cpu`).
- Passage text truncation for scoring only: `RAG_RERANK_PASSAGE_CHARS` (default 1200).
- Outputs now include `Rerank_Scores` instead of distances (EnsembleRetriever docs do not expose raw scores in current version).

## Answer generation flow

1. Build prompt with numbered contexts `[i]` if passages are available; otherwise use a no-context instruction.
2. Instruction forces evidence-only answers; must respond "Insufficient evidence to answer." when contexts don't support the question; request 2–4 sentences + citations.
3. Apply word cap `GEN_MAX_WORDS` (default 130) both in instruction and post-generation truncation.
4. Send to Ollama `/api/generate` with `{"model": "llama3.2:latest", "options": {"temperature": 0.3}, "stream": false}`.
5. Retries within `RAG_GEN_OVERALL_TIMEOUT` and `RAG_GEN_REQUEST_TIMEOUT`.
6. Save results with `Retrieved_Passages`, `Retrieved_Scores`, and `Retrieved_Distances` (raw distances from vector retrieval if present).

## Evaluation

An evaluation stage runs after answer generation to compute internal RAG quality metrics and persist results (details omitted here for simplicity).

### Latest successful run artifacts

- Five-metrics CSV: `evaluation/ragas_20251110_212550/ragas_five_metrics_20251110_212550.csv`
- Basic evaluation CSV: `evaluation/20251110_212735/real_llama_evaluation_20251110_212735.csv`
- Aggregated 9-metrics CSV: `evaluation/evaluate_all_metrics/all_metrics_20251110_212939.csv`
- Embedding model during these runs: `BAAI/bge-large-en-v1.5` (reverse-QG enabled)

## Improvement recommendations

1. Centralize config

   - Create a single `.env`/config and load in all scripts to avoid diverging defaults (e.g., unify embedding model across indexing/retrieval/evaluation).
   - Add automatic fallback: try `BAAI/bge-m3`; on torch<2.6 CVE error, fall back to `BAAI/bge-large-en-v1.5` with a clear warning.

2. Chunking enhancements

   - Switch to token-aware splitter (e.g., RecursiveTextSplitter with token counter) targeting ~800–1200 tokens; preserve headings and list boundaries to reduce semantic fragmentation.

3. Indexing/retrieval quality

   - Explicitly set collection metric at creation time and store it in metadata; add MMR diversification or a re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) after fusion.
   - Persist doc IDs and sources to enable per-source recall analysis.

4. Answer generation guardrails

   - Post-validate citations ([i] must exist); auto-switch to "Insufficient evidence" if no citations present or if faithfulness<τ.
   - Consider few-shot examples that demonstrate desired citation formatting.

5. Evaluation pipeline

   - Fix pandas `SettingWithCopyWarning` via `.loc[:, c] = ...` for rounding; add per-category aggregates; add a single runner that executes generation→five-metrics→basic→aggregate with pinned env.

6. Performance & reproducibility

   - Cache embeddings and enable batched encode with MPS; log all env vars into the output directories; pin package versions in `requirements*.txt`.

7. Torch ≥2.6 migration (for `bge-m3`)
   - When feasible, install a nightly/conda torch≥2.6 and switch `CUSTOM_EMB_MODEL=BAAI/bge-m3` across scripts. Until then, keep `bge-large-en-v1.5` as the default fallback.

---

If you want this diagram exported as an image for docs/slides, open this file in a Markdown preview with Mermaid enabled and export to PNG/SVG.

## How to export the diagram as PNG/SVG

Option A: Use the included script (requires mermaid-cli)

1. Ensure mermaid-cli is installed (via npm):
   npm install -g @mermaid-js/mermaid-cli
2. Run the render script from the repo root:
   bash deployment_package/rag_versions/rag_system/scripts/render_architecture.sh

Outputs will be written to:

- deployment_package/rag_versions/rag_system/docs/architecture.svg
- deployment_package/rag_versions/rag_system/docs/architecture.png

If mermaid-cli is not available, the script will print installation hints.
