#!/usr/bin/env python3
"""clean_extracted_topk.py

Parse the previously-extracted topk text files (topk_index_4.txt, topk_index_6.txt)
and write a cleaned CSV with columns: Source_File, Question_Index, Question, Ground_Truth,
Passage_Number, Passage_Text

Usage:
  python3 clean_extracted_topk.py

Outputs:
  deployment_package/rag_versions/rag_system/results/extracted_topk/topk_index_4_clean.csv
  deployment_package/rag_versions/rag_system/results/extracted_topk/topk_index_6_clean.csv

"""

import os
import re
import csv

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'extracted_topk')
FILES = ['topk_index_4.txt', 'topk_index_6.txt']
OUTDIR = os.path.join(BASE)
os.makedirs(OUTDIR, exist_ok=True)

PASSAGE_RE = re.compile(r'--- Passage\s*(\d+)\s*---')

def parse_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Extract Question and Ground Truth from top of file
    q_match = re.search(r'Question:\s*(.*?)\n\nGround Truth:', text, re.S)
    if not q_match:
        # try alternative
        q_match = re.search(r'Question:\s*(.*?)\n', text)
    question = q_match.group(1).strip() if q_match else ''

    gt_match = re.search(r'Ground Truth:\s*(.*?)\n\n--- Passage', text, re.S)
    if not gt_match:
        # try up to first passage marker
        gt_match = re.search(r'Ground Truth:\s*(.*?)\n--- Passage', text, re.S)
    ground_truth = gt_match.group(1).strip() if gt_match else ''

    # Split on passage markers, but keep passage numbers
    parts = PASSAGE_RE.split(text)
    # parts: [pre, num1, text1, num2, text2, ...]
    passages = []
    if len(parts) < 3:
        # Fallback: try splitting by lines that look like '--- Passage' without number
        raw_parts = re.split(r'-{3,}\s*Passage', text)
        for i, rp in enumerate(raw_parts[1:], start=1):
            passages.append((i, rp.strip()))
    else:
        # iterate pairs
        it = iter(parts)
        pre = next(it)  # text before first marker
        while True:
            try:
                num = next(it)
                ptext = next(it)
                try:
                    pnum = int(num.strip())
                except Exception:
                    pnum = None
                passages.append((pnum, ptext.strip()))
            except StopIteration:
                break

    return question, ground_truth, passages


def write_csv_for_file(fn):
    path = os.path.join(BASE, fn)
    if not os.path.exists(path):
        print('File not found:', path)
        return None
    q, gt, passages = parse_file(path)
    out_rows = []
    for pnum, ptext in passages:
        out_rows.append({
            'Source_File': fn,
            'Question_Index': fn.replace('topk_index_','').replace('.txt',''),
            'Question': q,
            'Ground_Truth': gt,
            'Passage_Number': pnum,
            'Passage_Text': ptext,
        })
    out_csv = os.path.join(OUTDIR, fn.replace('.txt','_clean.csv'))
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Source_File','Question_Index','Question','Ground_Truth','Passage_Number','Passage_Text'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    return out_csv


def main():
    results = []
    for fn in FILES:
        out = write_csv_for_file(fn)
        if out:
            print('Wrote', out)
            results.append(out)
    if not results:
        print('No files processed')

if __name__ == '__main__':
    main()
