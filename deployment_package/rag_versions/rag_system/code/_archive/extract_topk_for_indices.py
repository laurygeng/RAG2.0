#!/usr/bin/env python3
import json
import sys
import os
import csv
import pandas as pd

CSV = 'deployment_package/rag_versions/rag_system/results/real_answers_20251105_201534.csv'
INDICES = [0,6]
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'extracted_topk')
# Allow overriding via CLI: python extract_topk_for_indices.py <csv_path> "0,5,7"
if len(sys.argv) > 1:
    CSV = sys.argv[1]
if len(sys.argv) > 2:
    try:
        INDICES = [int(x) for x in sys.argv[2].split(',') if x.strip() != '']
    except Exception:
        pass
os.makedirs(OUTDIR, exist_ok=True)

def load_passages(cell):
    if pd.isna(cell) or not cell:
        return []
    try:
        docs = json.loads(cell)
        if isinstance(docs, list):
            return docs
    except Exception:
        pass
    try:
        return str(cell).split('||')
    except Exception:
        return [str(cell)]


df = pd.read_csv(CSV)
for i in INDICES:
    if i < 0 or i >= len(df):
        print(f'Index {i} out of range')
        continue
    row = df.iloc[i]
    q = row.get('Question','')
    gt = row.get('Ground_Truth_Answer','')
    passages = load_passages(row.get('Retrieved_Passages',''))
    print('\n' + '='*80)
    print(f'Index {i}\nQuestion: {q}\nGround truth (truncated): {gt[:300]}')
    print(f'Num retrieved passages: {len(passages)}')
    for j,p in enumerate(passages):
        print('\n--- Passage %d ---' % (j+1))
        print(p)
    # also write to a file
    of = os.path.join(OUTDIR, f'topk_index_{i}.txt')
    with open(of,'w',encoding='utf-8') as f:
        f.write(f'Question: {q}\n\nGround Truth:\n{gt}\n\n')
        for j,p in enumerate(passages):
            f.write('\n--- Passage %d ---\n' % (j+1))
            f.write(p + '\n')
    print(f'Wrote extracted passages to {of}')
    # Also write CSV in the same format as topk_index_6.csv
    csv_path = os.path.join(OUTDIR, f'topk_index_{i}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Source_File','Question_Index','Question','Ground_Truth','Passage_Number','Passage_Text'])
        for j,p in enumerate(passages):
            writer.writerow([f'topk_index_{i}.txt', i, q, gt, j+1, p])
    print(f'Wrote extracted CSV to {csv_path}')

print('\nDone')
