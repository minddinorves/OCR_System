"""TP/FP/FN transition analysis for the Fuzzy Matching correction step.

For every token PaddleOCR read (before correction), we check whether it
already fuzzy-matches some ground-truth ingredient (score>=85, same rule
set_prf1 uses to score TP) BEFORE correction, and again AFTER correction is
applied to that same token. This directly tests the over-correction
hypothesis raised while investigating why Fuzzy Matching used to make Set F1
worse than raw OCR: if RAW-TP -> FUZZY-FP is common, the vocabulary
correction step is actively breaking tokens that were already correct.

Four transition categories, computed only over tokens that the correction
step actually changed (original != corrected -- untouched tokens can't
change TP/FP status from this step):
  TP->TP : already matched a reference ingredient, correction kept it matching
           (e.g. corrected to the same canonical spelling)
  TP->FP : already matched a reference ingredient, correction broke the match
           -- this is the over-correction signature the user asked to isolate
  FN->TP : did not match anything, correction fixed it into a real match
           -- the intended benefit of Fuzzy Matching
  FN->FN : did not match before, still doesn't after -- wasted/neutral correction

Run twice: once against the OLD (buggy, pre missing-comma-fix) correction
logic, once against the NEW (fixed) logic, to see whether over-correction
was a real independent contributor to the Set F1 regression, or whether it
was entirely explained by the missing-comma bug already fixed.

Usage:
  python analyze_fuzzy_transitions.py
"""
import csv

from rapidfuzz import fuzz, process

import ocr_core as core
from old.compute_ingredient_metrics import tokenize
from ocr_paddle_fuzzy import (
    _strip_header, _SPLIT_RE, _is_reliable_match, _split_match_glued_names,
    MIN_TOKEN_LEN,
)

SOURCE_CSV = "results/ocr_paddle_fuzzy_FINAL_original_full_validation.csv"
THRESHOLD = 90.0
MATCH_THRESHOLD = 85  # same as compute_ingredient_metrics.MATCH_THRESHOLD


def trace_correction(raw_text, vocabulary, threshold, use_split_fallback):
    """Mirror ocr_paddle_fuzzy.correct_with_vocabulary but return
    (original_token, [corrected_token(s)]) pairs instead of just the final
    joined string, so we can track what each individual raw token became."""
    line = _strip_header(raw_text)
    tokens = [t.strip(" .;:*-") for t in _SPLIT_RE.split(line)]

    trace = []
    for token in tokens:
        if not token:
            continue
        if len(token) < MIN_TOKEN_LEN:
            trace.append((token, [token]))
            continue
        match = process.extractOne(token, vocabulary, scorer=fuzz.ratio, score_cutoff=threshold)
        if match and _is_reliable_match(token, match[0]):
            trace.append((token, [match[0]]))
            continue
        if use_split_fallback:
            split = _split_match_glued_names(token, vocabulary, threshold)
            if split:
                trace.append((token, split))
                continue
        trace.append((token, [token]))
    return trace


def best_match_score(token, ref_tokens):
    if not ref_tokens:
        return 0.0
    return max(fuzz.ratio(token, r) for r in ref_tokens)


def run(use_split_fallback, label):
    with open(SOURCE_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

    vocabulary = core.load_inci_vocabulary()

    counts = {"TP->TP": 0, "TP->FP": 0, "FN->TP": 0, "FN->FN": 0}
    examples = {"TP->FP": [], "FN->TP": []}

    n_substituted = 0
    for row in rows:
        ref_tokens = tokenize(row["reference_text"])
        trace = trace_correction(row["ocr_text_raw"], vocabulary, THRESHOLD, use_split_fallback)

        for original, corrected_list in trace:
            original_norm = original.strip().lower()
            if len(corrected_list) == 1 and corrected_list[0] == original:
                continue  # untouched token, not a substitution
            n_substituted += 1

            before_score = best_match_score(original_norm, ref_tokens)
            after_score = max(best_match_score(c.lower(), ref_tokens) for c in corrected_list)

            before_tp = before_score >= MATCH_THRESHOLD
            after_tp = after_score >= MATCH_THRESHOLD

            if before_tp and after_tp:
                counts["TP->TP"] += 1
            elif before_tp and not after_tp:
                counts["TP->FP"] += 1
                if len(examples["TP->FP"]) < 8:
                    examples["TP->FP"].append((row["image_file"], original, corrected_list))
            elif not before_tp and after_tp:
                counts["FN->TP"] += 1
                if len(examples["FN->TP"]) < 8:
                    examples["FN->TP"].append((row["image_file"], original, corrected_list))
            else:
                counts["FN->FN"] += 1

    print("=" * 78)
    print(f"{label}  (n_images={len(rows)}, n_substituted_tokens={n_substituted})")
    print("=" * 78)
    for k, v in counts.items():
        pct = 100.0 * v / n_substituted if n_substituted else 0.0
        print(f"  {k:8s} : {v:5d}  ({pct:5.1f}% of substituted tokens)")

    net = counts["FN->TP"] - counts["TP->FP"]
    print(f"\n  Net effect on Set-level TP count from corrections: {net:+d}  "
          f"(FN->TP gained {counts['FN->TP']}, TP->FP lost {counts['TP->FP']})")

    print("\n  Example TP->FP (over-correction, raw was already right):")
    for img, orig, corr in examples["TP->FP"]:
        print(f"    [{img}] {orig!r} -> {corr!r}")
    print("\n  Example FN->TP (correction helped, as intended):")
    for img, orig, corr in examples["FN->TP"]:
        print(f"    [{img}] {orig!r} -> {corr!r}")
    print()
    return counts, n_substituted


if __name__ == "__main__":
    run(use_split_fallback=False, label="OLD correction logic (pre missing-comma-fix, threshold=90)")
    run(use_split_fallback=True, label="NEW correction logic (with missing-comma fix, threshold=90)")
