#!/usr/bin/env python3
"""
Augment latest results CSV with retrieved passages and scores using local Chroma DB.
Writes a new CSV with suffix _with_retrieval_<timestamp>.csv
"""
import os
import glob
import json
from datetime import datetime
import pandas as pd

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(base_dir, 'results')
pattern = os.path.join(results_dir, 'real_answers_*.csv')
files = glob.glob(pattern)
if not files:
    print('No results CSV found to augment.')
    raise SystemExit(1)

src = max(files, key=os.path.getmtime)
print(f'Using source file: {src}')

df = pd.read_csv(src)

# attempt to load chroma + sentence-transformers
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print('chromadb or sentence-transformers not available:', e)
    raise SystemExit(2)

chroma_dir = os.path.join(base_dir, 'data', 'braincheck_vectordb')
if not os.path.exists(chroma_dir):
    print('Chroma DB directory not found at', chroma_dir)
    raise SystemExit(3)

settings = Settings(persist_directory=os.path.abspath(chroma_dir), is_persistent=True)
client = chromadb.Client(settings=settings)
try:
    collection = client.get_collection('braincheck')
except Exception as e:
    print('Could not get chroma collection:', e)
    raise SystemExit(4)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

TOP_K = int(os.environ.get('RAG_RETRIEVAL_TOP_K', '10'))

retrieved_passages = []
retrieved_scores = []

for idx, row in df.iterrows():
    q = row.get('Question') or row.get('Question_Text') or row.get('Question')
    if pd.isna(q) or not q:
        retrieved_passages.append([])
        retrieved_scores.append([])
        continue
    q_emb = embed_model.encode(q).tolist()
    try:
        res = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=['documents', 'distances'])
    except Exception as e:
        print(f'Query error for row {idx}:', e)
        retrieved_passages.append([])
        retrieved_scores.append([])
        continue
    docs = []
    scores = []
    if res and 'documents' in res and res['documents']:
        for d in res['documents'][0]:
            docs.append(d)
    if res and 'distances' in res and res['distances']:
        for dist in res['distances'][0]:
            try:
                scores.append(1.0 - float(dist))
            except Exception:
                scores.append(None)
    retrieved_passages.append(docs)
    retrieved_scores.append(scores)

# add columns and write new CSV
df['Retrieved_Passages'] = [json.dumps(x, ensure_ascii=False) for x in retrieved_passages]
df['Retrieved_Scores'] = [json.dumps(x, ensure_ascii=False) for x in retrieved_scores]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out = os.path.join(results_dir, f'real_answers_with_retrieval_{timestamp}.csv')
df.to_csv(out, index=False)
print('Wrote augmented results to:', out)
print('Done')
