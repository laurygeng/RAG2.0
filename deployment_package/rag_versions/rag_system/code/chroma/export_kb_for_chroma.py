#!/usr/bin/env python3
"""Export KB chunks and embeddings to a JSONL file suitable for later import into Chroma.

This avoids creating a local Chroma client when the chromadb rust bindings are incompatible.
Output file: data/chroma_import_<ts>.jsonl
Each line is a JSON object: {"id":..., "document":..., "metadata":..., "embedding": [...]}
"""

import os
import sys
import json
import time
import pickle

def main(kb_path=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not kb_path:
        kb_path = os.path.join(base_dir, 'data', 'braincheck_knowledge_base.pkl')

    if not os.path.exists(kb_path):
        print(f"KB file not found: {kb_path}")
        return 1

    print(f"Loading KB from: {kb_path}")
    with open(kb_path, 'rb') as f:
        kb = pickle.load(f)

    chunks = kb.get('chunks', [])
    print(f"Loaded {len(chunks)} chunks")

    # compute embeddings if missing
    need_embed = False
    for c in chunks:
        meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
        if not (isinstance(meta, dict) and meta.get('embedding')):
            need_embed = True
            break

    embeddings = None
    if need_embed:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(os.environ.get('RAG_EMBED_MODEL', 'all-MiniLM-L6-v2'), device='cpu')
            texts = [getattr(c, 'page_content', '') for c in chunks]
            embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            embeddings = [list(map(float, e.tolist())) for e in embs]
            print('Computed embeddings')
        except Exception as e:
            print(f"Failed to compute embeddings: {e}")
            return 2
    else:
        embeddings = []
        for c in chunks:
            meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
            embeddings.append(meta.get('embedding'))

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'chroma_import_{ts}.jsonl')
    print(f'Writing to {out_path} ...')
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, c in enumerate(chunks):
            text = getattr(c, 'page_content', '')
            meta = getattr(c, 'metadata', {}) if hasattr(c, 'metadata') else {}
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            rec = {
                'id': str(meta.get('chunk_id', i)),
                'document': text,
                'metadata': {k: v for k, v in (meta.items() if isinstance(meta, dict) else []) if k != 'embedding'},
                'embedding': emb
            }
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print('Done')
    print('You can later import this JSONL into Chroma on a machine with compatible chromadb via a small script or the Chroma CLI.')
    return 0


if __name__ == '__main__':
    kb = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(kb))
