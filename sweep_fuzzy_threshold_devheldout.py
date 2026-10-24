"""Proper dev/held-out split for selecting the Fuzzy Matching score_cutoff,
mirroring the box_thresh methodology already used in section 3.9/4.2 (same
40-image "tuning" dev set, same 176-image held-out set -- see
core.iter_dataset_entries(datasets=["original"], limit=40) and section 4.2.4).

Rule enforced here: the threshold is selected using ONLY the 40-image dev
set. The winning value is then evaluated on the 176-image held-out set,
which never influenced the choice, and THAT held-out number is the one to
report as final performance -- not the number the winning threshold happens
to score on the same images used to pick it.

This runs entirely on already-saved raw OCR text (ocr_text_raw in
results/ocr_paddle_fuzzy_FINAL_original_full_validation.csv) through the
fixed correct_with_vocabulary (missing-comma bug fix applied) -- no OCR
re-run needed, box_thresh/border-padding/doc-unwarping/textline-orientation
are therefore identical to the locked 0.6307 checkpoint.

Usage:
  python sweep_fuzzy_threshold_devheldout.py
"""
import csv

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import tokenize, set_prf1

SOURCE_CSV = "results/ocr_paddle_fuzzy_FINAL_original_full_validation.csv"
SWEEP_VALUES = [80, 82, 84, 85, 86, 88, 90, 92, 95, 98]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def score_at_threshold(rows, vocabulary, threshold):
    cers, ps, rs, f1s = [], [], [], []
    for row in rows:
        fuzzy_text = correct_with_vocabulary([row["ocr_text_raw"]], vocabulary, threshold)
        ref = row["reference_text"]
        cers.append(core.compute_cer(fuzzy_text, ref))
        p, r, f1 = set_prf1(tokenize(fuzzy_text), tokenize(ref))
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    return {
        "n": len(rows),
        "mean_cer": mean(cers),
        "mean_precision": mean(ps),
        "mean_recall": mean(rs),
        "mean_set_f1": mean(f1s),
    }


def main():
    dev_entries = list(core.iter_dataset_entries(datasets=["original"], limit=40))
    dev_files = set(e["image_file"] for e in dev_entries)

    with open(SOURCE_CSV, newline="", encoding="utf-8") as fh:
        all_rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

    dev_rows = [r for r in all_rows if r["image_file"] in dev_files]
    heldout_rows = [r for r in all_rows if r["image_file"] not in dev_files]
    print(f"dev set (tuning): n={len(dev_rows)}   held-out set: n={len(heldout_rows)}   "
          f"total: n={len(all_rows)}")
    assert len(dev_rows) == 40 and len(dev_rows) + len(heldout_rows) == len(all_rows)

    vocabulary = core.load_inci_vocabulary()

    print("\n=== Step 1: sweep threshold on DEV SET ONLY (n=40) ===")
    header = "%10s %6s %10s %10s %8s %8s" % ("threshold", "n", "mean CER", "Precision", "Recall", "Set F1")
    print(header)
    dev_results = {}
    for t in SWEEP_VALUES:
        res = score_at_threshold(dev_rows, vocabulary, t)
        dev_results[t] = res
        print("%10d %6d %10.4f %10.4f %8.4f %8.4f" % (
            t, res["n"], res["mean_cer"], res["mean_precision"], res["mean_recall"], res["mean_set_f1"]))

    best_t = max(dev_results, key=lambda t: dev_results[t]["mean_set_f1"])
    print(f"\nBest threshold on DEV SET by Set F1: {best_t}  "
          f"(dev Set F1 = {dev_results[best_t]['mean_set_f1']:.4f} -- "
          f"this number must NOT be reported as final performance)")

    print(f"\n=== Step 2: lock threshold={best_t}, evaluate on HELD-OUT SET (n={len(heldout_rows)}, never used for selection) ===")
    heldout_res = score_at_threshold(heldout_rows, vocabulary, best_t)
    print(f"  n            = {heldout_res['n']}")
    print(f"  mean CER     = {heldout_res['mean_cer']:.4f}")
    print(f"  Precision    = {heldout_res['mean_precision']:.4f}")
    print(f"  Recall       = {heldout_res['mean_recall']:.4f}")
    print(f"  Set F1       = {heldout_res['mean_set_f1']:.4f}   <-- FINAL, held-out performance at threshold={best_t}")

    print(f"\n=== For reference only (NOT the final number, same pooled 216 shown before): ===")
    all_res = score_at_threshold(all_rows, vocabulary, best_t)
    print(f"  all 216 pooled at threshold={best_t}: Set F1 = {all_res['mean_set_f1']:.4f}, "
          f"CER = {all_res['mean_cer']:.4f}")

    print(f"\n=== Comparison: current locked checkpoint (threshold=90) on the SAME held-out set ===")
    heldout_at_90 = score_at_threshold(heldout_rows, vocabulary, 90)
    print(f"  held-out Set F1 @ threshold=90 : {heldout_at_90['mean_set_f1']:.4f}")
    print(f"  held-out Set F1 @ threshold={best_t} : {heldout_res['mean_set_f1']:.4f}")


if __name__ == "__main__":
    main()
