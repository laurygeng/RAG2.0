#!/usr/bin/env python3
"""Wrapper to run generate_answers.py while temporarily hiding the on-disk Chroma folder.

This avoids initializing chromadb in environments where the rust bindings crash.
It renames the `data/braincheck_vectordb` folder (if present), runs the generator,
then restores the folder name. Logs are written to results/generate_no_chroma_<ts>.log.
"""
import os
import sys
import shutil
import subprocess
from datetime import datetime


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chroma_dir = os.path.join(base_dir, 'data', 'braincheck_vectordb')
    disabled_dir = None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'generate_no_chroma_{ts}.log')

    try:
        if os.path.exists(chroma_dir):
            disabled_dir = chroma_dir + f'.disabled_backup_{ts}'
            print(f'Renaming {chroma_dir} -> {disabled_dir} to avoid chromadb init')
            shutil.move(chroma_dir, disabled_dir)

        cmd = ['python3', 'code/generate_answers.py']
        print('Running:', ' '.join(cmd))
        # Use shell pipeline to tee output into log file for convenience
        with open(log_path, 'w') as lf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='')
                lf.write(line)
            proc.wait()
            rc = proc.returncode

        print(f'generate_answers.py exited with code {rc}; logs: {log_path}')
        return rc
    finally:
        # restore chroma dir if we moved it
        if disabled_dir and os.path.exists(disabled_dir) and not os.path.exists(chroma_dir):
            try:
                print(f'Restoring {disabled_dir} -> {chroma_dir}')
                shutil.move(disabled_dir, chroma_dir)
            except Exception as e:
                print(f'Warning: failed to restore chroma dir: {e}', file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
