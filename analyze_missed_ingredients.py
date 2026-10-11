"""Reproducible replacement for the unsourced "62.7% of missed ingredients are
detector misses" claim in ABSTRACT_FINAL.md section 4.3.

That number was traced back to a comment saying only "qualitative analysis" --
no script, no saved intermediate data, and no definition of "miss" survives
in this repo or its git history (checked via `git log --all -S"62.7"` and
`-S"detector"`: no hits). This script defines a reproducible proxy and
recomputes it from data that is already saved (results/*_ingredient.csv and
the underlying full-validation CSV) -- no OCR re-run needed.

Method (since we have no bounding-box ground truth to check detector output
directly):
  1. For each image, take the reference ingredient tokens and the FINAL
     (fuzzy-corrected) hypothesis tokens, and run the exact same greedy
     score>=85 matching used for Set Precision/Recall/F1 (see
     compute_ingredient_metrics.set_prf1). Reference tokens left unmatched
     are "missed ingredients" (False Negatives) -- same set the 4.3 claim is
     about.
  2. For each missed ingredient, search the RAW OCR text (ocr_text_raw, i.e.
     before fuzzy correction) for any trace of it: tokenize the raw text the
     same way, and take the best fuzz.partial_ratio score between the missed
     ingredient name and any raw token.
       - If the raw text contains nothing resembling the ingredient at all
         (best score below a threshold), the simplest explanation is that
         the text detector never produced a text region for it in the first
         place -- classified as "no trace in raw OCR" (detector-miss proxy).
       - If some trace exists (best score >= threshold), the detector likely
         found *something* there, but recognition misread it badly enough,
         or fuzzy matching failed to correct it -- classified as
         "trace found in raw OCR" (recognition/matching-miss proxy).
  3. Because the classification boundary is an arbitrary threshold, results
     are reported across several thresholds (40/50/60/70) instead of a single
     number, so the sensitivity of the claim to that choice is visible.

This is a proxy, not a ground-truth measurement (no bounding boxes are
available in this dataset) -- report it as such.

Usage:
  python analyze_missed_ingredients.py
"""
import csv

from rapidfuzz import fuzz

from old.compute_ingredient_metrics import tokenize, MATCH_THRESHOLD

SOURCE_CSV = "results/ocr_paddle_fuzzy_FINAL_original_full_validation.csv"
OUT_CSV = "results/missed_ingredient_analysis.csv"
TRACE_THRESHOLDS = [40, 50, 60, 70]


def find_missed(hyp_tokens, ref_tokens):
    """Same greedy score>=85 matching as compute_ingredient_metrics.set_prf1,
    but returns the list of unmatched (missed) reference tokens instead of
    just precision/recall/f1."""
    candidates = []
    for hi, h in enumerate(hyp_tokens):
        for ri, r in enumerate(ref_tokens):
            score = fuzz.ratio(h, r)
            if score >= MATCH_THRESHOLD:
                candidates.append((score, hi, ri))
    candidates.sort(reverse=True)

    used_hyp, used_ref = set(), set()
    for score, hi, ri in candidates:
        if hi in used_hyp or ri in used_ref:
            continue
        used_hyp.add(hi)
        used_ref.add(ri)

    return [ref_tokens[ri] for ri in range(len(ref_tokens)) if ri not in used_ref]


def best_trace_score(missed_token, raw_tokens):
    if not raw_tokens:
        return 0.0
    return max(fuzz.partial_ratio(missed_token, rt) for rt in raw_tokens)


def main():
    with open(SOURCE_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

    out_rows = []
    for row in rows:
        ref_tokens = tokenize(row["reference_text"])
        hyp_tokens = tokenize(row["ocr_text_fuzzy"])
        raw_tokens = tokenize(row["ocr_text_raw"])

        missed = find_missed(hyp_tokens, ref_tokens)
        for m in missed:
            score = best_trace_score(m, raw_tokens)
            out_rows.append({
                "image_file": row["image_file"],
                "missed_ingredient_token": m,
                "best_partial_ratio_vs_raw_ocr": round(score, 1),
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n_missed = len(out_rows)
    print(f"Total missed ingredients (False Negatives) across n={len(rows)} images: {n_missed}")
    print(f"Wrote per-token detail to {OUT_CSV}\n")

    print(f"{'threshold':>10} {'no trace in raw OCR (proxy: detector miss)':>45} {'trace found (proxy: recognition/matching miss)':>50}")
    print("-" * 110)
    for t in TRACE_THRESHOLDS:
        no_trace = sum(1 for r in out_rows if r["best_partial_ratio_vs_raw_ocr"] < t)
        has_trace = n_missed - no_trace
        pct_no_trace = 100.0 * no_trace / n_missed
        pct_has_trace = 100.0 * has_trace / n_missed
        print(f"{t:>10} {no_trace:>6} images ({pct_no_trace:5.1f}%){'':>20} {has_trace:>6} images ({pct_has_trace:5.1f}%)")

    print(
        "\nFor reference, the unsourced claim in ABSTRACT_FINAL.md 4.3 was "
        "62.7% classified as detector misses. Compare against the row above "
        "whose threshold best matches your intended definition of 'no trace'."
    )


if __name__ == "__main__":
    main()
