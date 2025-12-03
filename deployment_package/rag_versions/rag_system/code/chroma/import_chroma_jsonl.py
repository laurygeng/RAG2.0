#!/usr/bin/env python3
"""Import a JSONL file (produced by export_kb_for_chroma.py) into a Chroma collection.

Each line in the JSONL should be a JSON object with keys:
  - id: string
  - document: string
  - metadata: dict
  - embedding: list[float] or null

Usage:
  python3 code/import_chroma_jsonl.py /path/to/chroma_import_YYYYMMDD_HHMMSS.jsonl \
      --chroma-dir data/braincheck_vectordb --collection braincheck --batch-size 512

This script requires a working chromadb installation with compatible rust bindings.
If chromadb cannot start in this environment, run this script on a machine with a working chromadb.
"""

import os
import sys
import json
import argparse
from chromadb.config import Settings
import chromadb


def iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl', help='Path to JSONL file')
    p.add_argument('--chroma-dir', default=None, help='Chroma persist directory (default: ./data/braincheck_vectordb)')
    p.add_argument('--collection', default='braincheck', help='Collection name to create')
    p.add_argument('--batch-size', type=int, default=512, help='Batch size for adds')
    args = p.parse_args()

    jsonl_path = args.jsonl
    if not os.path.exists(jsonl_path):
        print(f'JSONL not found: {jsonl_path}', file=sys.stderr)
        return 2

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chroma_dir = args.chroma_dir or os.path.join(base_dir, 'data', 'braincheck_vectordb')
    os.makedirs(chroma_dir, exist_ok=True)

    settings = Settings(persist_directory=os.path.abspath(chroma_dir), is_persistent=True)
    print(f'Initializing Chroma client at: {chroma_dir}')
    client = chromadb.Client(settings=settings)

    # Try to remove existing collection if present
    try:
        client.delete_collection(args.collection)
        print(f'Removed existing collection: {args.collection}')
    except Exception:
        pass

    print(f'Creating collection: {args.collection}')
    try:
        collection = client.create_collection(name=args.collection)
    except Exception:
        collection = client.get_collection(name=args.collection)

    ids = []
    docs = []
    metas = []
    embs = []
    count = 0

    def flush_batch():
        nonlocal ids, docs, metas, embs, count
        if not ids:
            return
        # use embeddings only if present for all items in batch
        use_emb = all(e is not None for e in embs) and len(embs) == len(ids)
        if use_emb:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        else:
            collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f'  -> added batch of {len(ids)} (embeddings={use_emb})')
        ids, docs, metas, embs = [], [], [], []

    for rec in iter_jsonl(jsonl_path):
        _id = rec.get('id') or str(count)
        _doc = rec.get('document', '')
        _meta = rec.get('metadata', {}) or {}
        _emb = rec.get('embedding', None)

        ids.append(str(_id))
        docs.append(_doc)
        metas.append(_meta)
        embs.append(_emb)

        count += 1
        if len(ids) >= args.batch_size:
            flush_batch()

    # flush remaining
    flush_batch()

    print(f'Done. Total documents imported: {count}')
    print(f'Chroma directory: {chroma_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
