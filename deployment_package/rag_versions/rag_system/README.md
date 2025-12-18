# RAG System Documentation

This repository contains a complete Retrieval-Augmented Generation (RAG) system designed for medical QA, specifically focused on dementia care. The system supports hybrid retrieval (BM25 + Vector), reranking, and answer generation using Google Gemini 3 (or local LLMs).

## 📂 Directory Structure

Here is a detailed overview of the project folders:

- **`code/`**: Contains the core source code for the system.

  - `generate_answers_gemini3.py`: **Main entry point** for generating answers using Google Gemini 3.
  - `generate_answers.py`: Legacy script for generating answers using local LLMs (Ollama).
  - `evaluate_answers_*.py`: Scripts for calculating specific evaluation metrics (BLEU, ROUGE, METEOR, F1, BERTScore).
  - `load_data.py`: Utilities for loading the knowledge base and vector database.
  - `evaluate_all_metrics.py`: Wrapper to run multiple evaluations (if configured).

- **`data/`**: Stores the knowledge base and input data.

  - `braincheck_vectordb/`: The ChromaDB vector database folder.
  - `selected_questions.csv`: The default input file containing questions to be answered.
  - `*.pkl`: Serialized knowledge base chunks for BM25 retrieval.

- **`docs/`**: Documentation and diagrams.

  - `architecture.md`: System architecture description.
  - `README.md`: This file.

- **`evaluation/`**: **Output folder** for evaluation reports.

  - Contains subfolders like `BLEU/`, `ROUGE/`, `F1/`, etc., where detailed per-question metric CSVs are saved after running evaluation scripts.

- **`logs/`**: Stores runtime logs (if file logging is enabled).

- **`questions/`**: Contains raw or intermediate question datasets used for selecting the final test set.

- **`results/`**: **Output folder** for generation results.

  - The `generate_answers_*.py` scripts save the final CSVs here (e.g., `real_answers_gemini_YYYYMMDD_HHMMSS.csv`).

- **`scripts/`**: Helper and utility scripts.
  - `analyze_rerank_scores.py`: Tools for analyzing retrieval quality.
  - `select_diverse_questions.py`: Scripts used to sample questions from the dataset.

---

## 🚀 How to Run the System

The standard workflow consists of two main steps: **Generation** and **Evaluation**.

### Prerequisites

Ensure you have the necessary Python dependencies installed.

```bash
pip install -r requirements.txt
```

_(Note: Ensure `google-generativeai`, `chromadb`, `sentence-transformers`, `pandas`, `nltk`, `rouge-score`, `bert-score` are installed)_

### Step 1: Generate Answers (Gemini 3)

To generate answers for the questions in `data/selected_questions.csv` (or a custom file), use the `generate_answers_gemini3.py` script.

**Configuration (Environment Variables):**
You can tune the system using the following environment variables:

- `RAG_CHROMA_COLLECTION`: Name of the ChromaDB collection (e.g., `braincheck_updated`).
- `RAG_RETRIEVAL_TOP_K`: Number of final passages to retrieve (default: 3).
- `RAG_RETRIEVE_PRE_K`: Number of passages to fetch before reranking (default: 10).
- `RAG_GEN_OVERALL_TIMEOUT`: Timeout for generation in seconds.
- `RAG_QUESTIONS_FILE`: (Optional) Filename in `data/` to use as input (default: `selected_questions.csv`).

**Execution Command:**

```bash
# Example: Run with specific retrieval settings
export RAG_CHROMA_COLLECTION=braincheck_updated
export RAG_RETRIEVAL_TOP_K=3
export RAG_RETRIEVE_PRE_K=10
export RAG_GEN_OVERALL_TIMEOUT=600

# Run the generation script
python3 code/generate_answers_gemini3.py
```

**Output:**

- The script will print progress to the console.
- The final result CSV will be saved in: `results/real_answers_gemini_<TIMESTAMP>.csv`.

### Step 2: Evaluate Results

Once you have a result CSV file (e.g., `results/real_answers_gemini_20251210_035619.csv`), you can run the evaluation scripts to calculate performance metrics.

**Available Metrics:**

- **BERTScore**: Semantic similarity.
- **ROUGE**: N-gram overlap (Recall-oriented).
- **BLEU**: N-gram precision.
- **METEOR**: Alignment-based metric with synonym matching.
- **F1**: Token-level precision/recall.

**Execution Commands:**

Replace `<YOUR_RESULT_FILE.csv>` with the actual filename generated in Step 1.

```bash
# 1. BERTScore (Semantic Quality)
python3 code/evaluate_answers_BERTSCORE.py results/<YOUR_RESULT_FILE.csv>

# 2. ROUGE (Recall)
python3 code/evaluate_answers_ROUGE.py results/<YOUR_RESULT_FILE.csv>

# 3. F1 Score (Token Overlap)
python3 code/evaluate_answers_F1.py results/<YOUR_RESULT_FILE.csv>

# 4. METEOR
python3 code/evaluate_answers_METEOR.py results/<YOUR_RESULT_FILE.csv>

# 5. BLEU
python3 code/evaluate_answers_BLEU.py results/<YOUR_RESULT_FILE.csv>
```

**Output:**

- Each script prints a summary to the console.
- Detailed per-question scores are saved in the `evaluation/<METRIC_NAME>/` folder.

---

## 🛠 Troubleshooting

- **Memory Errors**: If running local models (via `generate_answers.py`) causes OOM errors, switch to `generate_answers_gemini3.py` which uses the cloud API.
- **NLTK Errors**: If you see `LookupError: Resource punkt_tab not found`, the scripts are designed to auto-download it. If that fails, run `python3 -m nltk.downloader punkt_tab` manually.
- **Zero Scores**: If BLEU/ROUGE scores are 0, check if your input CSV has a valid `Answer` (Ground Truth) column. If the ground truth is empty, scores will be 0.
