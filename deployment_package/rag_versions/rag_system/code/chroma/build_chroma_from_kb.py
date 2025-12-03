#!/usr/bin/env python3
"""
Build a persistent Chroma vector DB from an existing braincheck_knowledge_base.pkl.

Usage:
  python build_chroma_from_kb.py --kb /path/to/braincheck_knowledge_base.pkl --out ./data/braincheck_vectordb

This script uses sentence-transformers to compute embeddings (all-MiniLM-L6-v2) and chromadb
to persist a vector collection on disk. Both packages must be installed in your environment.
"""

import os
import argparse
import pickle
from typing import List

def build_chroma(kb_path: str, out_dir: str, collection_name: str = "braincheck"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required: pip install sentence-transformers") from e

    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        raise RuntimeError("chromadb is required: pip install chromadb") from e

    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"KB not found: {kb_path}")

    with open(kb_path, 'rb') as f:
        kb = pickle.load(f)

    chunks = kb.get('chunks', [])
    if not chunks:
        raise RuntimeError("No chunks found in KB")

    texts = [c.page_content for c in chunks]
    ids = [str(c.metadata.get('chunk_id', i)) for i, c in enumerate(chunks)]
    metadatas = [c.metadata for c in chunks]

    # Compute embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Prepare Chroma client with persistent directory
    os.makedirs(out_dir, exist_ok=True)
    # Use current chromadb Settings API: set persist_directory and is_persistent
    settings = Settings(persist_directory=os.path.abspath(out_dir), is_persistent=True)
    client = chromadb.Client(settings=settings)

    # Create or get collection
    try:
        collection = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' exists, will overwrite entries.")
    except Exception:
        collection = client.create_collection(name=collection_name)

    # Add documents
    print("Adding documents to Chroma collection in batches...")
    # Convert embeddings to lists for chromadb compatibility
    emb_lists = [e.tolist() for e in embeddings]
    # Add in batches to avoid backend max-batch-size errors
    batch_size = 4096
    for i in range(0, len(texts), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_emb = emb_lists[i:i+batch_size]
        print(f"  Adding batch {i}..{i+len(batch_texts)-1} ({len(batch_texts)} docs)")
        collection.add(ids=batch_ids, documents=batch_texts, metadatas=batch_meta, embeddings=batch_emb)

    # Persist
    try:
        client.persist()
    except Exception:
        # Some chromadb backends persist automatically
        pass

    print(f"✅ Chroma DB built at {out_dir} (collection: {collection_name})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kb', type=str, required=True, help='Path to braincheck_knowledge_base.pkl')
    p.add_argument('--out', type=str, default='../data/braincheck_vectordb', help='Output dir for chroma DB')
    p.add_argument('--name', type=str, default='braincheck', help='Collection name')
    args = p.parse_args()

    kb_path = os.path.abspath(args.kb)
    out_dir = os.path.abspath(args.out)

    build_chroma(kb_path, out_dir, collection_name=args.name)

if __name__ == '__main__':
    main()
