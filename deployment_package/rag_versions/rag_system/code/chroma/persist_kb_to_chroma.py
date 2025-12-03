#!/usr/bin/env python3
"""Persist a saved BrainCheck KB (with embeddings) into a local Chroma collection.

This script expects a KB file created by `load_data.save_knowledge_base_with_embeddings`
or similar, where each chunk's metadata contains an 'embedding' key with a list[float].

It will create (or replace) a Chroma collection named 'braincheck' under data/braincheck_vectordb.
"""

import os
import pickle
import sys
from chromadb.config import Settings
import chromadb


def main(kb_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not kb_path:
        kb_path = os.path.join(base_dir, 'data', 'braincheck_knowledge_base_with_emb.pkl')

    if not os.path.exists(kb_path):
        print(f"KB file not found: {kb_path}")
        return 1

    print(f"Loading KB from: {kb_path}")
    try:
        with open(kb_path, 'rb') as f:
            kb = pickle.load(f)
    except Exception as e:
        print(f"Failed to load '{kb_path}': {e}")
        # fallback to non-embedded KB if available
        fallback = os.path.join(os.path.dirname(kb_path), 'braincheck_knowledge_base.pkl')
        if os.path.exists(fallback):
            print(f"Attempting to load fallback KB: {fallback}")
            with open(fallback, 'rb') as f:
                kb = pickle.load(f)
        else:
            raise

    chunks = kb.get('chunks', [])
    print(f"Loaded {len(chunks)} chunks from KB")

    # Ensure embeddings exist on chunks; if not, attempt to compute using sentence-transformers
    need_embed = False
    for c in chunks:
        meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
        if not (isinstance(meta, dict) and meta.get('embedding')):
            need_embed = True
            break

    if need_embed:
        print('Embeddings not present on chunks; attempting to compute embeddings with sentence-transformers...')
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2'), device='cpu')
            texts = [getattr(c, 'page_content', '') for c in chunks]
            embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            for c, v in zip(chunks, embs):
                if not hasattr(c, 'metadata'):
                    try:
                        c.metadata = {}
                    except Exception:
                        pass
                try:
                    c.metadata['embedding'] = list(map(float, v.tolist()))
                except Exception:
                    c.metadata['embedding'] = list(map(float, v))
            print('Computed embeddings for all chunks')
        except Exception as e:
            print(f'❌ Failed to compute embeddings: {e}')
            return 2

    chroma_dir = os.path.join(base_dir, 'data', 'braincheck_vectordb')
    os.makedirs(chroma_dir, exist_ok=True)
    settings = Settings(persist_directory=os.path.abspath(chroma_dir), is_persistent=True)
    try:
        client = chromadb.Client(settings=settings)
    except Exception as e:
        print(f"Warning: failed to open existing Chroma DB at {chroma_dir}: {e}")
        # attempt alternate fresh directory to avoid rust/sqlite panic from incompatible DB
        alt_dir = os.path.join(base_dir, 'data', f'braincheck_vectordb_kb_{int(os.times().system)}')
        os.makedirs(alt_dir, exist_ok=True)
        settings = Settings(persist_directory=os.path.abspath(alt_dir), is_persistent=True)
        print(f"Creating fresh Chroma DB at: {alt_dir}")
        client = chromadb.Client(settings=settings)
        chroma_dir = alt_dir

    # If a collection exists, try to delete it to replace with this KB
    try:
        client.delete_collection('braincheck')
        print('Removed existing collection `braincheck`')
    except Exception:
        pass

    print('Creating collection `braincheck`')
    try:
        collection = client.create_collection('braincheck')
    except Exception:
        # fallback to get_collection if create_collection signature differs
        collection = client.get_collection('braincheck')

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, c in enumerate(chunks):
        # chunk may be a langchain Document or similar object
        text = getattr(c, 'page_content', None) or str(c)
        meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
        emb = None
        if isinstance(meta, dict) and 'embedding' in meta:
            emb = meta.get('embedding')
            # remove embedding from metadata to avoid storing large vector twice
            md = {k: v for k, v in meta.items() if k != 'embedding'}
        else:
            md = meta if isinstance(meta, dict) else {}

        ids.append(str(md.get('chunk_id', i)))
        documents.append(text)
        metadatas.append(md)
        if emb is not None:
            embeddings.append(emb)

    if embeddings and len(embeddings) == len(documents):
        print(f"Adding {len(documents)} documents with embeddings to Chroma at {chroma_dir} ...")
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    else:
        print("Warning: embeddings not found or mismatch length; adding documents without embeddings")
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print('Completed persisting KB to Chroma.')
    print(f'Chroma dir: {chroma_dir}')
    return 0


if __name__ == '__main__':
    kb = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(kb))
