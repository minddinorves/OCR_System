"""Driver for preprocess_pairs_search.py that runs each of the 28 technique
pairs in its OWN subprocess, instead of looping inside one long-lived Python
process.

Why: an earlier run that evaluated all 28 pairs in a single process leaked
GPU memory across pairs (VRAM climbed to 95% full and per-pair time grew
from ~5 minutes to ~16 minutes before being killed). Restarting the
interpreter for every pair forces the OS/CUDA driver to fully reclaim memory
between pairs, at the cost of a few extra seconds of PaddleOCR model
load time per pair.

Also resumable: skips any pair already present in results/pairs_search_log.csv,
so if this script itself is interrupted, rerunning it picks up where it left off.

Usage:
  python run_pairs_search.py
"""
import csv
import itertools
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "pairs_search_log.csv")
CANDIDATE_NAMES = sorted([
    "grayscale", "denoise", "sharpen",
    "upscale", "contrast_adaptive", "upscale800",
])
# 'deskew' pairs already covered by the earlier greedy-search round 2 (all worse
# than deskew alone). 'threshold' excluded because it scored catastrophically
# badly alone (CER 0.9025) and is very unlikely to become competitive paired
# with anything else. This narrows 28 pairs down to 15.


def already_done():
    done = set()
    if os.path.isfile(RESULTS_PATH):
        with open(RESULTS_PATH, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add(row["pair"])
    return done


def main():
    python_exe = sys.executable
    pairs = list(itertools.combinations(CANDIDATE_NAMES, 2))
    done = already_done()
    print(f"{len(pairs)} pairs total, {len(done)} already done, "
          f"{len(pairs) - len(done)} remaining")

    for i, pair in enumerate(pairs, 1):
        key = " + ".join(pair)
        if key in done:
            print(f"[{i}/{len(pairs)}] SKIP (already done): {key}")
            continue

        t0 = time.time()
        print(f"[{i}/{len(pairs)}] RUN: {key} ...", flush=True)
        result = subprocess.run(
            [python_exe, os.path.join(PROJECT_ROOT, "preprocess_pairs_search.py"),
             "--only", pair[0], pair[1]],
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - t0
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"[{i}/{len(pairs)}] DONE in {elapsed:.0f}s: {key} -> {status}", flush=True)


if __name__ == "__main__":
    main()
