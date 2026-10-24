"""Re-validate the missing-comma post-processing fix (ocr_paddle_fuzzy.py:
_split_match_glued_names) against the existing full "original" validation
run, WITHOUT re-running OCR.

Why this is valid without re-running OCR: the fix only changes the
post-OCR correction step (correct_with_vocabulary), not OCR itself. The raw
OCR text is already saved in
results/ocr_paddle_fuzzy_FINAL_original_full_validation.csv (ocr_text_raw
column), so re-applying the fixed correction function to that saved text and
recomputing metrics reproduces exactly what a full OCR re-run would produce,
in a fraction of the time.

Usage:
  python revalidate_missing_comma_fix.py
"""
import csv

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import tokenize, ier, set_prf1

SOURCE_CSV = "results/ocr_paddle_fuzzy_FINAL_original_full_validation.csv"
OUT_CSV = "results/ocr_paddle_fuzzy_FIXED_missing_comma_full_validation.csv"
THRESHOLD = 90.0


def main():
    vocabulary = core.load_inci_vocabulary()

    with open(SOURCE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out_rows = []
    for row in rows:
        if row.get("status") != "ok":
            out_rows.append(row)
            continue

        raw_text = row["ocr_text_raw"]
        new_fuzzy_text = correct_with_vocabulary([raw_text], vocabulary, THRESHOLD)
        ref = row["reference_text"]

        new_row = dict(row)
        new_row["ocr_text_fuzzy"] = new_fuzzy_text
        new_row["cer_fuzzy"] = round(core.compute_cer(new_fuzzy_text, ref), 4)
        new_row["wer_fuzzy"] = round(core.compute_wer(new_fuzzy_text, ref), 4)
        out_rows.append(new_row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT_CSV}")

    # Recompute ingredient-level metrics (same logic as compute_ingredient_metrics.py)
    ok_rows = [r for r in out_rows if r.get("status") == "ok"]
    ier_vals, p_vals, r_vals, f1_vals, cer_vals = [], [], [], [], []
    for row in ok_rows:
        hyp_tokens = tokenize(row["ocr_text_fuzzy"])
        ref_tokens = tokenize(row["reference_text"])
        i = ier(hyp_tokens, ref_tokens)
        p, r, f1 = set_prf1(hyp_tokens, ref_tokens)
        if i is not None:
            ier_vals.append(i)
        p_vals.append(p)
        r_vals.append(r)
        f1_vals.append(f1)
        cer_vals.append(float(row["cer_fuzzy"]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nn = {len(ok_rows)}")
    print(f"mean CER (fuzzy)     : {mean(cer_vals):.4f}   (baseline was 0.2644)")
    print(f"mean IER             : {mean(ier_vals):.4f}   (baseline was 0.6757)")
    print(f"mean Set Precision   : {mean(p_vals):.4f}   (baseline was 0.5487)")
    print(f"mean Set Recall      : {mean(r_vals):.4f}   (baseline was 0.6127)")
    print(f"mean Set F1          : {mean(f1_vals):.4f}   (baseline was 0.5715)")


if __name__ == "__main__":
    main()
