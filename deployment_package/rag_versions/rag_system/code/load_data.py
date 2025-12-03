#!/usr/bin/env python3
from __future__ import annotations
"""
BrainCheck Knowledge Base Loader
Basic document processing without external ML dependencies
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING  # Ensure compatibility with type annotations
from collections import defaultdict
from datetime import datetime
import random
import csv
import textwrap
import re

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
# Optional improved PDF loader (PyMuPDF). Enabled via env RAG_PDF_LOADER=pymupdf
try:  # pragma: no cover
    from langchain_community.document_loaders import PyMuPDFLoader  # requires pymupdf (fitz)
    _HAS_PYMUPDF = True
except Exception:
    PyMuPDFLoader = None  # type: ignore
    _HAS_PYMUPDF = False
try:
    # Newer langchain releases provide text splitters under langchain.text_splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Some installations provide the standalone package langchain_text_splitters
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        # Last resort: re-raise the original import error to be visible to the caller
        raise
if TYPE_CHECKING:
    # Only import for type checking to avoid runtime import issues
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:  # pragma: no cover
        from typing import Any as Document  # fallback for type checkers
else:
    # At runtime, we don't need actual typing; treat Document as Any
    from typing import Any as Document

class SimpleBrainCheckLoader:
    """Simple document loader with basic text processing"""
    
    def __init__(self, local_folder_path: Optional[str] = None):
        # Default to folder co-located with this script to be robust to CWD
        if local_folder_path is None:
            self.local_folder_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "downloaded_files")
        else:
            self.local_folder_path = local_folder_path
        # Prefer token-aware recursive splitting: semantic-first (paragraph/sentence boundaries) while
        # enforcing token limits to stay within model context windows.
        # Updated defaults per best-practice: 500 token chunks with 100 token overlap.
        target_tokens = int(os.environ.get('RAG_CHUNK_TOKENS', '500'))
        overlap_tokens = int(os.environ.get('RAG_CHUNK_TOKENS_OVERLAP', '100'))
        # Allow overriding tiktoken encoding (defaults to cl100k_base, suitable for modern OpenAI models)
        encoding_name = os.environ.get('RAG_TIKTOKEN_ENCODING', 'cl100k_base')
        # Custom separators include Chinese punctuation to preserve sentence boundaries in mixed-language corpora.
        mixed_lang_separators = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        try:
            # Many langchain versions support this convenience ctor
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=target_tokens,
                chunk_overlap=overlap_tokens,
                encoding_name=encoding_name,
            )
            # Override separators to optimize for both English and Chinese sentence/paragraph boundaries
            try:
                self.text_splitter.separators = mixed_lang_separators  # type: ignore[attr-defined]
            except Exception:
                pass
            self._split_mode = 'tokens'
            print(f"🧩 Using token-recursive splitter (tiktoken) encoding={encoding_name}: chunk_size={target_tokens}, overlap={overlap_tokens}")
        except Exception:
            # Fallback to TokenTextSplitter if available
            try:
                from langchain.text_splitter import TokenTextSplitter  # type: ignore
                self.text_splitter = TokenTextSplitter(
                    chunk_size=target_tokens,
                    chunk_overlap=overlap_tokens,
                )
                self._split_mode = 'tokens'
                print(f"🧩 Using TokenTextSplitter (no recursive semantics): chunk_size={target_tokens}, overlap={overlap_tokens}")
            except Exception:
                # Final fallback: character-based splitter with larger defaults for mixed-language docs
                char_size = int(os.environ.get('RAG_CHUNK_CHARS', '3000'))
                char_overlap = int(os.environ.get('RAG_CHUNK_CHARS_OVERLAP', '600'))
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=char_size,
                    chunk_overlap=char_overlap,
                    separators=mixed_lang_separators
                )
                self._split_mode = 'chars'
                print(f"⚠️ Token-based splitter unavailable; falling back to character-based splitter: chunk_size={char_size}, overlap={char_overlap}")
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Pre-clean documents to remove references, headers/footers, and URL-only noise before splitting.

        Heuristics (page-level since PDF loader yields per-page docs):
        - Drop pages that look like a References/Bibliography section (by header or citation density).
        - Remove header/footer lines (page numbers, copyright).
        - Remove URL-only or URL-dense lines.
        """
        if not documents:
            return []

        kept: List[Document] = []
        total = len(documents)

        for d in documents:
            try:
                text = getattr(d, 'page_content', '') or ''
                cleaned = _preclean_text(text)
                if cleaned.strip():
                    # Mutate in place to retain metadata
                    d.page_content = cleaned
                    kept.append(d)
            except Exception:
                # If anything goes wrong, keep original page to avoid data loss
                kept.append(d)

        removed = total - len(kept)
        print(f"🧼 Pre-cleaned pages: kept {len(kept)}/{total} (removed {removed} noisy/reference pages)")
        return kept
        
    def get_download_instructions(self) -> str:
        """Instructions for downloading files"""
        return f"""
📁 DOWNLOAD INSTRUCTIONS:

1. 🌐 Visit: https://smu365-my.sharepoint.com/personal/xnluo_smu_edu/_layouts/15/onedrive.aspx
2. 📂 Navigate to 'BrainCheck knowledge base' folder
3. ☑️  Select all files you want to include
4. ⬇️  Click 'Download' to get a ZIP file
5. 📦 Extract ZIP contents to: {os.path.abspath(self.local_folder_path)}
6. ▶️  Run this script again

Supported file types: .pdf, .docx, .txt
        """
    
    def load_documents(self) -> List[Document]:
        """Load documents from local folder"""
        documents = []
        
        if not os.path.exists(self.local_folder_path):
            os.makedirs(self.local_folder_path, exist_ok=True)
            print(f"📁 Created folder: {self.local_folder_path}")
            print(self.get_download_instructions())
            return documents
        
        # PDF loader selection: prefer PyMuPDFLoader automatically if available; fallback to PyPDFLoader.
        if _HAS_PYMUPDF and PyMuPDFLoader is not None:
            chosen_pdf_loader = PyMuPDFLoader
            print("🧾 Using PyMuPDFLoader for PDFs (auto-detected). To force legacy parser uninstall pymupdf.")
        else:
            chosen_pdf_loader = PyPDFLoader
            print("🧾 Using PyPDFLoader for PDFs (PyMuPDF not available). Install with: pip install pymupdf")

        file_handlers = {
            '.pdf': chosen_pdf_loader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
        }
        
        # Optional cap for quick smoke tests
        try:
            max_files = int(os.environ.get('RAG_MAX_FILES', '0'))
        except Exception:
            max_files = 0
        processed_files = 0

        # Process files
        for file_path in Path(self.local_folder_path).rglob("*"):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                if file_ext in file_handlers:
                    if max_files and processed_files >= max_files:
                        break
                    try:
                        loader_class = file_handlers[file_ext]
                        loader = loader_class(str(file_path))
                        file_docs = loader.load()
                        
                        # Add metadata
                        for doc in file_docs:
                            doc.metadata.update({
                                'source_file': file_path.name,
                                'file_type': file_ext,
                                'file_path': str(file_path)
                            })
                        
                        documents.extend(file_docs)
                        print(f"✅ Loaded {len(file_docs)} documents from {file_path.name}")
                        processed_files += 1
                        
                    except Exception as e:
                        print(f"❌ Error loading {file_path.name}: {e}")
                else:
                    print(f"⏭️  Skipping unsupported file: {file_path.name}")
        
        print(f"📊 Total documents loaded: {len(documents)}")
        return documents
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'preview': chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
            })
        
        print(f"📝 Created {len(chunks)} text chunks")
        return chunks

    def create_keyword_index(self, chunks: List[Document]) -> Dict[str, List[int]]:
        """Create a simple keyword index for search"""
        keyword_index = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            # Simple keyword extraction
            text = chunk.page_content.lower()
            words = text.split()
            
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 2:  # Skip very short words
                    keyword_index[word].append(i)
        
        return dict(keyword_index)
        
    def save_knowledge_base(self, chunks: List[Document], keyword_index: Dict[str, List[int]], 
                            save_path: str = "./braincheck_knowledge_base.pkl"):
        """Save the knowledge base to a file"""
        knowledge_base = {
            'chunks': chunks,
            'keyword_index': keyword_index,
            'metadata': {
                'total_chunks': len(chunks),
                'total_keywords': len(keyword_index),
                'created_at': datetime.now().isoformat()
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(knowledge_base, f)
        
        print(f"💾 Knowledge base saved to: {save_path}")
        return save_path
        
    def search_simple(self, chunks: List[Document], keyword_index: Dict[str, List[int]], 
                       query: str, top_k: int = 3) -> List[Document]:
        """Simple keyword-based search"""
        query_words = [word.lower().strip('.,!?') for word in query.split()]
        chunk_scores = defaultdict(int)
        
        # Score chunks based on keyword matches
        for word in query_words:
            if word in keyword_index:
                for chunk_idx in keyword_index[word]:
                    chunk_scores[chunk_idx] += 1
        
        # Sort by score and return top results
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_idx, score in sorted_chunks[:top_k]:
            if score > 0:
                chunk = chunks[chunk_idx]
                chunk.metadata['search_score'] = score
                results.append(chunk)
        
        return results
        
    def test_search(self, chunks: List[Document], keyword_index: Dict[str, List[int]], 
                     query: str = "BrainCheck"):
        """Test the search functionality"""
        print(f"\n🔍 Testing search: '{query}'")
        results = self.search_simple(chunks, keyword_index, query)
        
        if not results:
            print("❌ No results found")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i} (Score: {result.metadata.get('search_score', 0)}):")
            print(f"Source: {result.metadata.get('source_file', 'Unknown')}")
            print(f"Content: {result.page_content[:200]}...")

def _approx_token_count(text: str) -> int:
    """Approximate token count using tiktoken if available, else fall back to words/4 heuristic."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Simple fallback heuristic: rough 1 token ~= 4 characters in English; for mixed languages use words
        return max(1, int(len(text) / 4))

# ===== Pre-cleaning helpers =====
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*p\.\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*©\s*(19|20)\d{2}.*$", re.IGNORECASE),
    re.compile(r"^\s*copyright\b.*$", re.IGNORECASE),
    re.compile(r"^\s*all\s+rights\s+reserved.*$", re.IGNORECASE),
]
_REF_HEADER_RE = re.compile(r"^\s*(references|bibliography|works\s+cited|参考文献)\s*$", re.IGNORECASE)
_CITATION_LINE_RE = re.compile(r"^\s*(\[\d+\]|\d{1,3}[\.)]\s)\S+.*")
_DOI_RE = re.compile(r"\bdoi:\s*\S+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _is_reference_page(lines: List[str]) -> bool:
    # Header within first ~20 lines
    for line in lines[:20]:
        if _REF_HEADER_RE.match(line.strip()):
            return True
    # Density heuristic: a lot of citation-like lines / DOIs / years
    if not lines:
        return False
    hits = 0
    for l in lines:
        if _CITATION_LINE_RE.match(l):
            hits += 1
            continue
        if _DOI_RE.search(l):
            hits += 1
            continue
        # Lines with many author-like tokens and a year
        if _YEAR_RE.search(l) and (',' in l or 'et al' in l.lower()):
            hits += 1
    return hits >= max(5, int(0.5 * len(lines)))

def _clean_line_remove_urls(line: str) -> str:
    # Remove raw URLs from line
    return _URL_RE.sub('', line).strip()

def _line_is_url_noise(original: str, cleaned: str) -> bool:
    # If the line was mostly URLs (or becomes too short), drop it
    if not original.strip():
        return True
    if _URL_RE.search(original):
        if len(cleaned) < 15:
            return True
    return False

def _remove_headers_footers(lines: List[str]) -> List[str]:
    out = []
    for l in lines:
        dropped = False
        for pat in _HEADER_FOOTER_PATTERNS:
            if pat.match(l.strip()):
                dropped = True
                break
        if not dropped:
            out.append(l)
    return out

def _preclean_text(text: str) -> str:
    # Split into lines first for structural heuristics
    raw_lines = text.splitlines()
    # Remove obvious headers/footers
    lines = _remove_headers_footers(raw_lines)
    # Early detect and drop reference-like pages entirely
    if _is_reference_page(lines):
        return ''
    # Drop URL-only or URL-heavy lines; strip URLs elsewhere
    kept_lines: List[str] = []
    for l in lines:
        cleaned = _clean_line_remove_urls(l)
        if _line_is_url_noise(l, cleaned):
            continue
        kept_lines.append(cleaned)
    # Join back; collapse multiple blank lines
    joined = '\n'.join(kept_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    # Normalize problematic spacing artifacts from PDFs/OCR before downstream indexing/embeddings
    joined = _normalize_pdf_spacing(joined)
    return joined.strip()

def _normalize_pdf_spacing(text: str) -> str:
    """Fix common PDF/OCR spacing artifacts that hurt BM25 and embeddings.

    Heuristics applied (order matters):
    - Replace non-breaking and unusual spaces with regular space
    - Remove zero-width spaces and soft hyphens
    - Join hyphenated line-breaks: 'exam-\nple' -> 'example' (letters on both sides)
    - Collapse letter-spaced words of length >= 4: 'p r a c t i c a l' -> 'practical'
    - Normalize repeated spaces and stray spaces before punctuation
    """
    if not text:
        return text

    # 1) Normalize space-like characters
    s = (text
         .replace('\u00A0', ' ')   # NBSP
         .replace('\u2007', ' ')   # Figure space
         .replace('\u202F', ' ')   # Narrow NBSP
    )
    # Remove zero-width characters and soft hyphen
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # zero-width space/joiners + BOM
    s = s.replace('\u00AD', '')  # soft hyphen

    # 2) Fix hyphenation at line breaks: letter-\n letter -> letterletter
    s = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", s)

    # 3) Collapse letter-spaced words (longer sequences first)
    LETTER_SPACED_LONG = re.compile(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b")
    def _collapse_letters_long(m: re.Match) -> str:
        return m.group(0).replace(' ', '')
    prev = None
    while prev != s:
        prev = s
        s = LETTER_SPACED_LONG.sub(_collapse_letters_long, s)

    # 3b) Collapse shorter spaced sequences (e.g. 'm i l d', 'c a r e') including optional trailing '’s' or "'s".
    # Guard: avoid collapsing ALL-CAPS acronyms of length <=4 (e.g. 'U S A').
    LETTER_SPACED_SHORT = re.compile(r"\b(?:[A-Za-z]\s+){1,6}[A-Za-z](?:['’]s)?\b")
    def _collapse_letters_short(m: re.Match) -> str:
        raw = m.group(0)
        collapsed = raw.replace(' ', '')
        # Skip if acronym-like (all uppercase and length <=4)
        if collapsed.isupper() and len(collapsed) <= 4:
            return raw
        return collapsed
    s = LETTER_SPACED_SHORT.sub(_collapse_letters_short, s)

    # 3c) Merge patterns where spaces appear between EVERY char including punctuation like “A l z h e i m e r ’ s”.
    # Normalize fancy apostrophes first then collapse again.
    s = s.replace('’', "'")
    ALZ_PATTERN = re.compile(r"A\s+l\s+z\s+h\s+e\s+i\s+m\s+e\s+r(?:\s+'\s+s)?", re.IGNORECASE)
    s = ALZ_PATTERN.sub(lambda m: m.group(0).replace(' ', ''), s)

    # 4) Remove stray space before punctuation (common PDF quirk)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    # 5) Collapse multiple spaces (but keep newlines)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

# Public helper to allow isolated testing/invocation
def fix_spacing_artifacts(text: str) -> str:
    """Convenience wrapper to apply PDF spacing normalization to arbitrary text.

    This can be used independently for unit tests or ad-hoc cleaning.
    """
    return _normalize_pdf_spacing(text)

def _suggest_question(text: str, source: str = "") -> str:
    """Heuristically suggest a simple question that this chunk could answer.
    Avoids external LLMs; uses keyword templates and filename hints.
    """
    t = text.strip().lower()
    # Quick templates based on common topics in this corpus
    if "personal hygiene" in t:
        return "How does dementia affect personal hygiene, and what support is available?"
    if "personal care" in t and "dementia" in t:
        return "How does dementia affect personal care and daily hygiene?"
    if "risk factor" in t or ("risk" in t and "dementia" in t):
        return "What are important risk factors for dementia?"
    if "communication" in t and ("intervention" in t or "study" in t):
        return "What communication outcomes are examined in cognitive intervention studies?"
    if "support" in t and ("helpline" in t or "nurse" in t or "contact" in t or "organisation" in t or "organization" in t):
        return "Which organisations provide support services for people living with dementia?"
    if "activity" in t or "exercise" in t:
        return "What activities or exercises are recommended for people with dementia?"
    if "sleep" in t:
        return "How does dementia affect sleep, and what strategies can help?"
    if "driving" in t and "dementia" in t:
        return "When should someone with dementia stop driving, and what are the alternatives?"
    if "medication" in t or "medicine" in t:
        return "What medication safety considerations apply to older adults with dementia?"
    # Fallback: derive from source filename as topic
    if source:
        base = source.replace('_', ' ').replace('-', ' ').replace('.pdf', '').replace('.docx', '').replace('.txt', '')
        base = base.strip().title()
        if base:
            return f"What does '{base}' cover?"
    return "What is the main idea presented in this chunk?"

def qualitative_sample(chunks: List[Document], n: int = 5, out_dir: Optional[str] = None, label: Optional[str] = None):
    """Print and optionally save a qualitative sample of chunks for manual inspection.

    Prompts for two checks:
      1) Can this chunk independently answer a simple question?
      2) Does this chunk contain too many unrelated topics?
    If out_dir is provided, saves a Markdown file with the sampled chunks to that directory.
    """
    if not chunks:
        print("❌ No chunks to sample.")
        return None
    n = min(n, len(chunks))
    sample = random.sample(chunks, n)

    lines = []
    lines.append("# Qualitative Chunk Sample (random sample)\n")
    lines.append("Please check each item:\n- Can it independently answer a simple question?\n- Does it include too many unrelated topics?\n")
    # Optionally source suggested questions from reviewed CSV
    reviewed_questions: Optional[List[str]] = None
    try:
        # If caller set RAG_QUAL_USE_REVIEWED=1, prefer selecting questions from CSV for each chunk
        use_reviewed = os.environ.get('RAG_QUAL_USE_REVIEWED', '1') == '1'
        if use_reviewed:
            gt_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ground_truth_answers.csv'))
            if os.path.exists(gt_csv):
                reviewed_questions = _load_reviewed_questions(gt_csv)
    except Exception:
        reviewed_questions = None

    for i, c in enumerate(sample, 1):
        meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
        src = meta.get('source_file', 'Unknown')
        text = getattr(c, 'page_content', '')
        toks = _approx_token_count(text)
        preview = textwrap.shorten(text.replace('\n', ' '), width=420, placeholder=' ...')
        lines.append(f"## [{i}] source={src} | chars={len(text)} | ~tokens={toks} | chunk_id={meta.get('chunk_id', '?')}")
        # Add suggested question line (prefer reviewed questions if available)
        if reviewed_questions:
            question = _select_best_question_for_text(text, reviewed_questions) or _suggest_question(text, src)
        else:
            question = _suggest_question(text, src)
        lines.append(f"Q: {question}")
        lines.append("")
        lines.append(preview)
        lines.append("")

    content = "\n".join(lines) + "\n"
    # Print to console
    print("\n🧪 Qualitative Chunk Sample (random sample):")
    print("Please check: 1) Can it answer a simple question by itself? 2) Does it include too many unrelated topics?\n")
    # Walk through blocks (each block adds 4 lines after header: Q, blank, preview, blank)
    idx = 0
    for i in range(1, n+1):
        header = lines[1 + idx + 1]  # first block header position
        qline = lines[1 + idx + 2]
        preview = lines[1 + idx + 4]
        print(header.replace('## ', ''))
        print(qline)
        print(f"     {preview}\n")
        idx += 4

    # Save to file if requested
    if out_dir:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        out_path = out_dir if os.path.isabs(out_dir) else os.path.join(base_dir, out_dir)
        os.makedirs(out_path, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"qualitative_sample_{label + '_' if label else ''}{ts}.md"
        fpath = os.path.join(out_path, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📝 Saved sample to file: {fpath}")
        return fpath
    return None

def _normalize_words(s: str) -> List[str]:
    t = (s or '').lower()
    tokens = [''.join(ch for ch in w if ch.isalnum()) for w in t.split()]
    return [w for w in tokens if len(w) > 2]

def _select_best_question_for_text(text: str, questions: List[str]) -> Optional[str]:
    """Select the best-matching question from a list for a given chunk text using simple word-overlap.
    Returns None if no overlap is found.
    """
    chunk_words = set(_normalize_words(text))
    if not chunk_words:
        return None
    best_q = None
    best_score = 0
    for q in questions:
        q_words = set(_normalize_words(q))
        if not q_words:
            continue
        score = len(chunk_words & q_words)
        if score > best_score:
            best_score = score
            best_q = q
    return best_q if best_score > 0 else None

def _load_reviewed_questions(csv_path: str) -> List[str]:
    """Load questions from a ground truth CSV where Answer Type == 'Answer - Reviewed'.
    Tries to be header-robust by normalizing names.
    """
    questions: List[str] = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Normalize fieldnames
            field_map = { (name or '').strip().lower(): name for name in (reader.fieldnames or []) }
            q_key = field_map.get('question') or field_map.get('questions') or 'Question'
            at_key = field_map.get('answer type') or field_map.get('answer_type') or 'Answer Type'
            for row in reader:
                at = (row.get(at_key) or '').strip()
                if at == 'Answer - Reviewed':
                    q = (row.get(q_key) or '').strip()
                    if q:
                        questions.append(q)
    except Exception as e:
        print(f"⚠️ Failed to load reviewed questions from {csv_path}: {e}")
    return questions

def qualitative_sample_from_questions(
    chunks: List[Document],
    keyword_index: Dict[str, List[int]],
    questions: List[str],
    n: int = 5,
    out_dir: Optional[str] = None,
    label: Optional[str] = None,
    top_k: int = 1,
):
    """Given a list of reviewed questions, retrieve top chunks and save a Markdown with Q->chunk preview pairs."""
    if not chunks or not questions:
        print("❌ No chunks or questions available for Q->Chunk sampling.")
        return None
    n = max(1, min(n, len(questions)))
    # Deterministic pick: take first N
    picked = questions[:n]

    lines = []
    lines.append("# Qualitative Q→Chunk Sample (Reviewed Questions)\n")
    lines.append("Source: ground_truth_answers.csv (Answer Type = Answer - Reviewed)\n")
    for i, q in enumerate(picked, 1):
        lines.append(f"## [{i}] Q: {q}")
        # simple keyword search using existing index
        try:
            # Reuse SimpleBrainCheckLoader.search_simple signature via a temporary instance
            tmp_loader = SimpleBrainCheckLoader()
            results = tmp_loader.search_simple(chunks, keyword_index, q, top_k=top_k)
        except Exception:
            results = []
        if not results:
            lines.append("No retrieved chunk.\n")
            continue
        # Use top-1 preview
        c = results[0]
        meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
        src = meta.get('source_file', 'Unknown')
        text = getattr(c, 'page_content', '')
        toks = _approx_token_count(text)
        preview = textwrap.shorten(text.replace('\n', ' '), width=600, placeholder=' ...')
        lines.append(f"- source={src} | chars={len(text)} | ~tokens={toks} | chunk_id={meta.get('chunk_id','?')} | score={meta.get('search_score','?')}")
        lines.append("")
        lines.append(preview)
        lines.append("")

    content = "\n".join(lines) + "\n"
    # Print concise console view
    print("\n🧪 Reviewed Q→Chunk Sample:")
    for i, q in enumerate(picked, 1):
        print(f"[{i}] Q: {q}")
    # Save to file
    if out_dir:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        out_path = out_dir if os.path.isabs(out_dir) else os.path.join(base_dir, out_dir)
        os.makedirs(out_path, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"qualitative_Q2chunk_{label + '_' if label else ''}{ts}.md"
        fpath = os.path.join(out_path, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📝 Saved Q→Chunk sample to file: {fpath}")
        return fpath
    return None

def persist_chunks_to_chroma(
    chunks: List[Any],
    chroma_dir: str = "./data/braincheck_vectordb",
    collection: str = "braincheck",
    model_name: str = None,
    batch_size: int = 512,
    replace: bool = None,
):
    """Persist chunks (with embeddings) to a local Chroma collection.

    Behavior:
    - If chunks already carry embeddings in chunk.metadata['embedding'], those are used.
    - Otherwise compute embeddings using sentence-transformers (default all-MiniLM-L6-v2 or env RAG_EMBED_MODEL).
    - By default replaces the existing collection (set RAG_CHROMA_REPLACE=0 to append/merge best-effort).

    Args:
        chunks: List of langchain-like Documents with page_content and metadata.
        chroma_dir: Persist directory for Chroma database.
        collection: Collection name (default 'braincheck').
        model_name: Embedding model to use when embeddings missing; overrides env RAG_EMBED_MODEL.
        batch_size: Add size per batch.
        replace: Whether to drop existing collection first. If None, read env RAG_CHROMA_REPLACE (default '1').
    """
    # Resolve flags and model
    if replace is None:
        replace = os.environ.get('RAG_CHROMA_REPLACE', '1') == '1'
    if model_name is None:
        model_name = os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2')

    # Soft-import chromadb and sentence-transformers
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        raise RuntimeError(f"chromadb is required to persist vectors: {e}")

    # Check if any embeddings are present
    have_any_emb = False
    for c in chunks:
        try:
            if isinstance(getattr(c, 'metadata', {}), dict) and c.metadata.get('embedding') is not None:
                have_any_emb = True
                break
        except Exception:
            continue

    # Compute embeddings if needed using sentence-transformers only
    if not have_any_emb:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔗 Computing embeddings with {model_name} (CPU)")
            texts = [getattr(c, 'page_content', '') for c in chunks]
            st = None
            try:
                st = SentenceTransformer(model_name, device='cpu')
            except TypeError:
                st = SentenceTransformer(model_name)
                try:
                    st.to('cpu')
                except Exception:
                    pass
            embs = st.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            for c, v in zip(chunks, embs):
                try:
                    if not hasattr(c, 'metadata') or not isinstance(c.metadata, dict):
                        c.metadata = {}
                    c.metadata['embedding'] = [float(x) for x in list(v)]
                except Exception:
                    try:
                        c.metadata['embedding'] = list(map(float, v))
                    except Exception:
                        pass
            have_any_emb = True
        except Exception as e:
            raise RuntimeError(
                "Failed to compute embeddings for Chroma persist using sentence-transformers. "
                f"Error: {e}. Ensure PyTorch and sentence-transformers are installed and compatible."
            )

    # Initialize Chroma
    abs_dir = os.path.abspath(chroma_dir)
    os.makedirs(abs_dir, exist_ok=True)
    metric = os.environ.get('RAG_CHROMA_METRIC', 'cosine')
    settings = Settings(persist_directory=abs_dir, is_persistent=True)
    try:
        client = chromadb.Client(settings=settings)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma at {abs_dir}: {e}")

    # Replace or get existing collection
    if replace:
        try:
            client.delete_collection(collection)
            print(f"🧹 Removed existing collection: {collection}")
        except Exception:
            pass
    # Create or get collection (no embedding function; we pass precomputed embeddings)
    try:
        coll = client.create_collection(collection, metadata={"distance_metric": metric})
    except Exception:
        try:
            coll = client.get_collection(collection)
        except Exception as e:
            raise RuntimeError(f"Failed to create/get Chroma collection '{collection}': {e}")

    # Add in batches
    ids, docs, metas, embs = [], [], [], []
    total = 0

    def flush():
        nonlocal ids, docs, metas, embs, total
        if not ids:
            return
        use_emb = (len(embs) == len(ids) and all(e is not None for e in embs))
        if use_emb:
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        else:
            coll.add(ids=ids, documents=docs, metadatas=metas)
        total += len(ids)
        print(f"   -> persisted {len(ids)} records (embeddings={'direct' if use_emb else 'none'})")
        ids, docs, metas, embs = [], [], [], []

    for i, c in enumerate(chunks):
        text = getattr(c, 'page_content', '')
        meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
        emb = None
        if isinstance(meta, dict):
            emb = meta.get('embedding')
            # Avoid storing large embedding twice in metadata
            md = {k: v for k, v in meta.items() if k != 'embedding'}
        else:
            md = {}

        cid = str(md.get('chunk_id', i))
        ids.append(cid)
        docs.append(text)
        metas.append(md)
        embs.append(emb)

        if len(ids) >= batch_size:
            flush()

    flush()
    print(f"✅ Chroma persist complete. Collection='{collection}', dir='{abs_dir}', total={total}")
    return {
        'collection': collection,
        'dir': abs_dir,
        'count': total,
        'metric': metric,
    }

def embed_chunks(chunks: List[Document], model_name: Optional[str] = None, batch_size: int = 16):
    """Embed a list of langchain Documents using a local sentence-transformers model only.

    Remote / Hugging Face Inference API fallbacks have been explicitly disabled.
    If the local model isn't available, a RuntimeError is raised instructing to
    install sentence-transformers or choose an available local model.

    Returns a list of embedding vectors (lists of floats) aligned with `chunks`.
    """
    texts = [getattr(c, 'page_content', '') for c in chunks]
    # Try sentence-transformers first (local, fast when model downloaded)
    try:
        from sentence_transformers import SentenceTransformer
        # Resolve model from env if not specified explicitly
        if not model_name:
            model_name = os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2')
        print(f"🔗 Using sentence-transformers locally: {model_name} (forcing CPU)")
        # Force CPU device to avoid MPS/GPU out-of-memory issues on some machines
        try:
            st = SentenceTransformer(model_name, device='cpu')
        except TypeError:
            # Older sentence-transformers versions may not accept device kwarg; fall back
            st = SentenceTransformer(model_name)
            try:
                st.to('cpu')
            except Exception:
                pass
        embs = st.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return [list(map(float, e)) for e in embs]
    except Exception as e:
        print(f"⚠️ sentence-transformers backend not available or failed: {e}")

    # All remote fallbacks removed intentionally.
    raise RuntimeError('Local sentence-transformers embedding failed and no remote fallback is enabled.')


def save_knowledge_base_with_embeddings(chunks: List[Document], keyword_index: Dict[str, List[int]],
                                        model_name: Optional[str] = None, save_path: str = "./braincheck_knowledge_base_with_emb.pkl"):
    """Create embeddings for chunks and save the KB including embeddings.

    Uses only local sentence-transformers. Remote Hugging Face Inference API fallbacks are disabled.
    """
    # Resolve model from env if not specified explicitly
    if not model_name:
        model_name = os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2')
    print(f"▶️ Embedding {len(chunks)} chunks with model {model_name} ...")
    try:
        embs = embed_chunks(chunks, model_name=model_name)
        # Attach embedding vector to each chunk metadata for convenience
        for c, v in zip(chunks, embs):
            c.metadata['embedding'] = v

        kb_path = os.path.abspath(save_path)
        knowledge_base = {
            'chunks': chunks,
            'keyword_index': keyword_index,
            'metadata': {
                'total_chunks': len(chunks),
                'total_keywords': len(keyword_index),
                'created_at': datetime.now().isoformat()
            }
        }
        with open(kb_path, 'wb') as f:
            pickle.dump(knowledge_base, f)
        print(f"💾 Knowledge base with embeddings saved to: {kb_path}")
        return kb_path
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return None

def main():
    """Main execution function"""
    print("🧠 BrainCheck Knowledge Base Loader (Simple Version)")
    print("=" * 55)
    
    # NEW BEHAVIOR OVERVIEW
    # 1. Default ALWAYS attempts to persist to Chroma (no need for RAG_PERSIST_TO_CHROMA).
    # 2. If a KB with embeddings already exists AND not forcing rebuild, we reuse it directly
    #    (skip document loading, chunking, keyword indexing, embedding).
    # 3. Set RAG_FORCE_REBUILD=1 to force a full pipeline rebuild.
    # 4. Set RAG_SKIP_EMBED=1 to skip pre-saving embeddings into the KB file; persistence will still
    #    compute embeddings if missing.
    # 5. Legacy RAG_PERSIST_ONLY still supported (explicit path-driven persist).

    force_rebuild = os.environ.get('RAG_FORCE_REBUILD', '0') == '1'
    skip_embed = os.environ.get('RAG_SKIP_EMBED', '0') == '1'

    # Fast path: persist-only mode to skip loading/splitting/embedding
    if os.environ.get('RAG_PERSIST_ONLY', '0') == '1':
        kb_env_path = os.environ.get('RAG_KB_PATH')
        # Try defaults if not provided
        default_dir = os.path.abspath(os.path.dirname(__file__))
        candidate_paths = [
            kb_env_path,
            os.path.join(default_dir, 'braincheck_knowledge_base_with_emb.pkl'),
            os.path.join(default_dir, 'braincheck_knowledge_base.pkl'),
        ]
        kb_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                kb_path = p
                break
        if not kb_path:
            print("❌ RAG_PERSIST_ONLY=1 but no KB found. Set RAG_KB_PATH to an existing .pkl file.")
            print("   Looked for: braincheck_knowledge_base_with_emb.pkl or braincheck_knowledge_base.pkl in the code folder.")
            return
        print(f"📦 Persist-only mode: loading KB from {kb_path}")
        kb = load_knowledge_base(kb_path)
        chunks = kb.get('chunks') if isinstance(kb, dict) else None
        if not chunks:
            print("❌ KB file didn't contain 'chunks'. Abort.")
            return
        print(f"📝 Chunks ready: {len(chunks)}")
        # Directly persist to Chroma
        try:
            persist_info = persist_chunks_to_chroma(
                chunks,
                chroma_dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'braincheck_vectordb'),
                collection=os.environ.get('RAG_CHROMA_COLLECTION', 'braincheck'),
                model_name=os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2'),
                batch_size=int(os.environ.get('RAG_CHROMA_BATCH', '512')),
                replace=None,
            )
            print(f"📌 Chroma persist summary: {persist_info}")
        except Exception as e:
            print(f"❌ Failed to persist to Chroma: {e}")
        return

    # Auto-reuse existing KB with embeddings unless forcing rebuild
    default_dir = os.path.abspath(os.path.dirname(__file__))
    existing_with_emb = os.path.join(default_dir, 'braincheck_knowledge_base_with_emb.pkl')
    existing_plain = os.path.join(default_dir, 'braincheck_knowledge_base.pkl')
    if not force_rebuild and os.path.exists(existing_with_emb):
        print(f"⚡ Reuse mode: Found existing KB with embeddings at {existing_with_emb}. Skipping rebuild.")
        kb = load_knowledge_base(existing_with_emb)
        chunks = kb.get('chunks') if isinstance(kb, dict) else []
        if not chunks:
            print("❌ Existing KB file did not contain chunks. Falling back to full rebuild.")
        else:
            try:
                persist_info = persist_chunks_to_chroma(
                    chunks,
                    chroma_dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'braincheck_vectordb'),
                    collection=os.environ.get('RAG_CHROMA_COLLECTION', 'braincheck'),
                    model_name=os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2'),
                    batch_size=int(os.environ.get('RAG_CHROMA_BATCH', '512')),
                    replace=None,
                )
                print(f"📌 Chroma persist summary: {persist_info}")
                print("✅ Done (reuse path). Use RAG_FORCE_REBUILD=1 to force regeneration.")
                return
            except Exception as e:
                print(f"❌ Failed to persist reused KB to Chroma: {e}. Will attempt full rebuild.")
    elif not force_rebuild and os.path.exists(existing_plain):
        print(f"⚡ Reuse mode: Found existing KB (no embeddings) at {existing_plain}. Will persist; embeddings may be computed during persist.")
        kb = load_knowledge_base(existing_plain)
        chunks = kb.get('chunks') if isinstance(kb, dict) else []
        if chunks:
            try:
                persist_info = persist_chunks_to_chroma(
                    chunks,
                    chroma_dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'braincheck_vectordb'),
                    collection=os.environ.get('RAG_CHROMA_COLLECTION', 'braincheck'),
                    model_name=os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2'),
                    batch_size=int(os.environ.get('RAG_CHROMA_BATCH', '512')),
                    replace=None,
                )
                print(f"📌 Chroma persist summary: {persist_info}")
                print("✅ Done (reuse plain KB path). Use RAG_FORCE_REBUILD=1 for regeneration.")
                return
            except Exception as e:
                print(f"❌ Failed to persist plain KB to Chroma: {e}. Will attempt full rebuild.")

    loader = SimpleBrainCheckLoader()
    
    # Load documents
    documents = loader.load_documents()
    
    if not documents:
        print("\n❌ No documents found. Please follow the download instructions above.")
        return
    
    # Pre-clean documents (remove references, headers/footers, URL-only noise)
    if os.environ.get('RAG_PRE_CLEAN', '1') == '1':
        documents = loader.preprocess_documents(documents)
    
    # Create chunks
    chunks = loader.create_chunks(documents)
    
    if not chunks:
        print("\n❌ No text chunks created.")
        return
    
    # Create keyword index (needed for question-driven sampling as well)
    print("🔧 Creating keyword index...")
    keyword_index = loader.create_keyword_index(chunks)
    print(f"📊 Indexed {len(keyword_index)} unique keywords")

    # Qualitative sampling
    try:
        sample_n = int(os.environ.get('RAG_QUAL_SAMPLE_N', '5'))
    except ValueError:
        sample_n = 5
    token_size = int(os.environ.get('RAG_CHUNK_TOKENS', '512'))
    token_overlap = int(os.environ.get('RAG_CHUNK_TOKENS_OVERLAP', '50'))
    out_dir = os.environ.get('RAG_QUAL_SAMPLE_OUTDIR', 'test')
    qual_mode = os.environ.get('RAG_QUAL_MODE', 'by-chunks')  # 'by-chunks' | 'q2chunk'

    gt_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ground_truth_answers.csv'))
    questions = _load_reviewed_questions(gt_csv) if os.path.exists(gt_csv) else []
    if qual_mode == 'q2chunk' and questions:
        qualitative_sample_from_questions(
            chunks,
            keyword_index,
            questions,
            n=sample_n,
            out_dir=out_dir,
            label=f"t{token_size}_o{token_overlap}_n{sample_n}",
            top_k=1,
        )
    else:
        qualitative_sample(
            chunks,
            n=sample_n,
            out_dir=out_dir,
            label=f"t{token_size}_o{token_overlap}_n{sample_n}"
        )

    # Save knowledge base
    save_path = loader.save_knowledge_base(chunks, keyword_index)
    
    print(f"\n🎉 SUCCESS! BrainCheck knowledge base created!")
    
    # Test searches
    test_queries = [
        "BrainCheck",
        "cognitive assessment", 
        "brain health",
        "monitoring",
        "digital biomarkers"
    ]
    
    for query in test_queries:
        loader.test_search(chunks, keyword_index, query)
    
    print(f"\n📁 Knowledge base saved to: {save_path}")
    print("🔄 You can now load this file for RAG applications!")
    
    # Embed chunks unless explicitly skipped (now default ON behavior)
    emb_path = None
    if skip_embed:
        print("⏭️  RAG_SKIP_EMBED=1 -> Skipping embedding before persistence (embeddings may be computed during persist).")
    else:
        emb_path = save_knowledge_base_with_embeddings(
            chunks,
            keyword_index,
            model_name=os.environ.get('RAG_EMBED_MODEL'),
            save_path=save_path.replace('.pkl', '_with_emb.pkl'),
        )
        if emb_path:
            print(f"📌 Embeddings saved to: {emb_path}")
        else:
            print("⚠️ Embedding step failed prior to persistence. Persist will attempt embedding if needed.")

    # ALWAYS persist chunks to Chroma now (default behavior)
    print('\n🗃️  Default: Persisting chunks to Chroma (disable by setting RAG_DISABLE_CHROMA=1)')
    if os.environ.get('RAG_DISABLE_CHROMA', '0') == '1':
        print('� RAG_DISABLE_CHROMA=1 -> Skipping Chroma persistence.')
    else:
        try:
            persist_info = persist_chunks_to_chroma(
                chunks,
                chroma_dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'braincheck_vectordb'),
                collection=os.environ.get('RAG_CHROMA_COLLECTION', 'braincheck'),
                model_name=os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2'),
                batch_size=int(os.environ.get('RAG_CHROMA_BATCH', '512')),
                replace=None,
            )
            print(f"📌 Chroma persist summary: {persist_info}")
        except Exception as e:
            print(f"❌ Failed to persist to Chroma: {e}")

    # Show summary
    print(f"\n📈 SUMMARY:")
    print(f"   • Documents processed: {len(documents)}")
    print(f"   • Text chunks created: {len(chunks)}")
    print(f"   • Keywords indexed: {len(keyword_index)}")
    print(f"   • Knowledge base file: {save_path}")

def load_knowledge_base(file_path: str = "./braincheck_knowledge_base.pkl"):
    """Load a saved knowledge base"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    main()
