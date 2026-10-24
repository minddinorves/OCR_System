"""Check whether the early preprocessing experiments (results/ocr_paddle_only.csv,
results/preprocess_*.csv, run 16 Jul 2026 before the dataset was expanded)
used the same images as the current dataset. Compares, per file name, the
reference (ground-truth) text stored in the early results with the current
reference text. Same file name + very different reference = a different
product saved under the same name.

Output: results/dataset_versions.txt

Usage:
  python compare_dataset_versions.py
"""
import csv
import os

from rapidfuzz import fuzz

import ocr_core as core


def refs(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["image_file"]: r["reference_text"] for r in csv.DictReader(fh)
                if r["dataset"] == "original" and r["status"] == "ok"}


def main():
    early = refs(os.path.join(core.RESULTS_DIR, "ocr_paddle_only.csv"))
    current = refs(os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_original_full_validation.csv"))
    common = sorted(set(early) & set(current))
    scores = [fuzz.ratio(early[f].lower(), current[f].lower()) for f in common]
    lines = [
        f"early-experiment images (original scenario, with label): {len(early)}",
        f"current images: {len(current)}",
        f"same file name in both: {len(common)}",
        f"  reference text identical: {sum(early[f] == current[f] for f in common)}",
        f"  near-identical (fuzz.ratio >= 90): {sum(s >= 90 for s in scores)}",
        f"  clearly different product (fuzz.ratio < 50): {sum(s < 50 for s in scores)}",
        f"  median similarity: {sorted(scores)[len(scores) // 2]:.1f}",
        f"only in the early experiment (file no longer exists): {len(set(early) - set(current))}",
        f"only in the current dataset: {len(set(current) - set(early))}",
    ]
    with open(os.path.join(core.RESULTS_DIR, "dataset_versions.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
