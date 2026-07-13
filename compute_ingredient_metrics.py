"""Compute a supplementary, format-insensitive metric on top of the existing
per-image OCR results: Ingredient Error Rate (IER) and set-based
Precision/Recall/F1 over INGREDIENT TOKENS instead of raw characters.

Why this exists: CER/WER compare the OCR output against a reference string
built by joining every ingredient name with ", " (see ocr_core.py:
load_reference_text). Real labels wrap ingredient lists across multiple
printed lines and sometimes hyphenate words at the line break (e.g.
"Hy-\\ndrogenated"), which the OCR output preserves but the comma-joined
reference does not. That formatting mismatch inflates CER/WER even when the
extracted ingredient names are correct. This script re-tokenizes BOTH the
OCR output and the reference into ingredient-name tokens (splitting on the
same commas used by the Fuzzy Matching step) before comparing, so line-break
and punctuation differences no longer count as errors:

  - IER (Ingredient Error Rate): Levenshtein distance between the hypothesis
    token list and the reference token list, divided by the number of
    reference tokens. Same idea as WER, but the "words" are ingredient names
    instead of whitespace-split words, and order still matters.
  - Set Precision/Recall/F1: treats both sides as sets of normalized tokens
    (order-independent). A hypothesis token counts as a match against a
    reference token if their rapidfuzz.fuzz.ratio score is >= 85 (the same
    threshold used by the Fuzzy Matching technique itself), matched greedily
    one-to-one by descending score. This intentionally credits near-correct
    spelling (e.g. a token corrected to its canonical INCI spelling that
    differs slightly from how the label transcriber wrote it) instead of
    requiring byte-for-byte equality, which would unfairly penalize Fuzzy
    Matching for "fixing" a token into a spelling that is correct but not
    identical to the reference label's own wording.

This does NOT require re-running OCR: it reads the ocr_text (or
ocr_text_fuzzy/ocr_text_raw) and reference_text columns already saved in
results/*.csv from earlier runs.

Usage:
  python compute_ingredient_metrics.py
"""
import csv
import glob
import os
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import ocr_core as core

_SPLIT_RE = re.compile(r",(?!\s*\d)")
SKIP_FILES = {"summary.csv", "summary_ingredient.csv", "combination_search_log.csv", "pairs_search_log.csv"}
MATCH_THRESHOLD = 85  # same score_cutoff used by the Fuzzy Matching technique itself


def tokenize(text):
    tokens = [t.strip(" .;:*-").lower() for t in _SPLIT_RE.split(text or "")]
    return [t for t in tokens if t]


def ier(hyp_tokens, ref_tokens):
    if not ref_tokens:
        return None
    return Levenshtein.distance(hyp_tokens, ref_tokens) / len(ref_tokens)


def set_prf1(hyp_tokens, ref_tokens):
    """Greedy one-to-one fuzzy matching between the two token lists (score >= 85),
    matched in descending score order, so a token can only be used once."""
    candidates = []
    for hi, h in enumerate(hyp_tokens):
        for ri, r in enumerate(ref_tokens):
            score = fuzz.ratio(h, r)
            if score >= MATCH_THRESHOLD:
                candidates.append((score, hi, ri))
    candidates.sort(reverse=True)

    used_hyp, used_ref = set(), set()
    tp = 0
    for score, hi, ri in candidates:
        if hi in used_hyp or ri in used_ref:
            continue
        used_hyp.add(hi)
        used_ref.add(ri)
        tp += 1

    precision = tp / len(hyp_tokens) if hyp_tokens else 0.0
    recall = tp / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def process_file(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], []

    fieldnames = list(rows[0].keys())
    text_field = (
        "ocr_text_fuzzy" if "ocr_text_fuzzy" in fieldnames
        else "ocr_text" if "ocr_text" in fieldnames
        else "ocr_text_raw"
    )

    out_rows = []
    for row in rows:
        ref = row.get("reference_text", "")
        hyp = row.get(text_field, "")
        if row.get("status") != "ok" or not ref:
            out_rows.append({**row, "ier": "", "set_precision": "", "set_recall": "", "set_f1": ""})
            continue
        hyp_tokens = tokenize(hyp)
        ref_tokens = tokenize(ref)
        i = ier(hyp_tokens, ref_tokens)
        p, r, f1 = set_prf1(hyp_tokens, ref_tokens)
        out_rows.append({
            **row,
            "ier": round(i, 4) if i is not None else "",
            "set_precision": round(p, 4),
            "set_recall": round(r, 4),
            "set_f1": round(f1, 4),
        })

    by_dataset = {}
    for row in out_rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summaries = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        def _mean(field):
            values = [float(r[field]) for r in dataset_rows if r.get(field) not in ("", None)]
            return round(sum(values) / len(values), 4) if values else ""
        summaries.append({
            "source_file": os.path.basename(path),
            "dataset": dataset,
            "n_images": len(dataset_rows),
            "mean_ier": _mean("ier"),
            "mean_set_precision": _mean("set_precision"),
            "mean_set_recall": _mean("set_recall"),
            "mean_set_f1": _mean("set_f1"),
        })
    return out_rows, summaries


def main():
    csv_paths = sorted(
        p for p in glob.glob(os.path.join(core.RESULTS_DIR, "*.csv"))
        if os.path.basename(p) not in SKIP_FILES and not os.path.basename(p).endswith("_ingredient.csv")
    )
    if not csv_paths:
        print(f"No result CSVs found in {core.RESULTS_DIR}.")
        return

    all_summaries = []
    for path in csv_paths:
        out_rows, summaries = process_file(path)
        if not out_rows:
            continue
        annotated_path = path.replace(".csv", "_ingredient.csv")
        with open(annotated_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        all_summaries.extend(summaries)

    out_path = os.path.join(core.RESULTS_DIR, "summary_ingredient.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["source_file", "dataset", "n_images", "mean_ier",
                        "mean_set_precision", "mean_set_recall", "mean_set_f1"],
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"Wrote {len(all_summaries)} summary rows to {out_path}\n")
    header = f"{'source_file':<32} {'dataset':<12} {'n':>4} {'IER':>7} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(f"{s['source_file']:<32} {s['dataset']:<12} {s['n_images']:>4} "
              f"{s['mean_ier']!s:>7} {s['mean_set_precision']!s:>7} "
              f"{s['mean_set_recall']!s:>7} {s['mean_set_f1']!s:>7}")


if __name__ == "__main__":
    main()
