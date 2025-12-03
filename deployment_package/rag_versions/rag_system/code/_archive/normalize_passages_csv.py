#!/usr/bin/env python3
import re
import os
import sys
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'extracted_topk')
FILES = ['topk_index_4_clean.csv', 'topk_index_6_clean.csv']

# Normalization heuristics:
# 1. Replace newlines with space
# 2. Fix hyphen line-breaks like 'exam- ple' or 'exam-\nple' -> 'example'
# 3. Collapse multiple spaces
# 4. Remove space before punctuation (,.!?;:)
# 5. Trim

HYPHEN_RE = re.compile(r'(-)\s+')
MULTI_WS = re.compile(r'\s+')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([,\.!?;:])')

for fn in FILES:
    path = os.path.join(BASE, fn)
    if not os.path.exists(path):
        print('File not found, skipping:', path)
        continue
    df = pd.read_csv(path, dtype=str)
    if 'Passage_Text' not in df.columns:
        print('No Passage_Text column in', path)
        continue
    cleaned = []
    for s in df['Passage_Text'].fillna(''):
        t = str(s)
        # replace newline with space
        t = t.replace('\n', ' ')
        # fix hyphen+space (common broken hyphenation)
        # also handle case like 'exam- ple' -> 'example'
        t = HYPHEN_RE.sub('', t)
        # collapse multiple whitespace
        t = MULTI_WS.sub(' ', t)
        # remove space before punctuation
        t = SPACE_BEFORE_PUNCT.sub(r'\1', t)
        t = t.strip()
        cleaned.append(t)
    df['Passage_Text'] = cleaned
    out_path = path
    df.to_csv(out_path, index=False)
    print('Normalized and overwrote:', out_path)

print('Done')
