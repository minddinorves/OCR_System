"""5-fold cross-validation check of the box_thresh=0.30 vs 0.45 selection,
on the FIXED (missing-comma bug fixed) pipeline.

This replaces an unsourced script of the same name referenced in section
4.2.5 / Appendix A of the thesis (full.pdf), which does not exist anywhere
in this repo or its git history (`git log --all -S"kfold"` has zero hits) --
so its result can't be verified as belonging to the current implementation.
Rebuilt here from the methodology description in 4.2.5: split the 40-image
tuning dev set into 5 folds of 8 images each; for each round, use the other
4 folds (32 images) to decide whether box_thresh=0.45 or 0.30 gives the
higher mean Set F1, then score the WINNING value on the held-out fold (8
images, no part in the decision); repeat for all 5 folds.

Uses the already-saved raw OCR text for both box_thresh values (no re-run of
OCR needed) run through the FIXED correct_with_vocabulary, so this reflects
the current implementation, not the orphaned original run.

Usage:
  python compute_kfold_check.py
"""
import csv

import ocr_core as core
from compute_ingredient_metrics import tokenize, set_prf1

FILE_045 = "results/ocr_paddle_fuzzy_FIXED_missing_comma_boxthresh045_full_validation.csv"
FILE_030 = "results/ocr_paddle_fuzzy_FIXED_missing_comma_full_validation.csv"
N_FOLDS = 5
FOLD_SIZE = 8


def load_f1_by_file(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    out = {}
    for r in rows:
        p, rec, f1 = set_prf1(tokenize(r["ocr_text_fuzzy"]), tokenize(r["reference_text"]))
        out[r["image_file"]] = f1
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    dev_entries = list(core.iter_dataset_entries(datasets=["original"], limit=40))
    dev_files = [e["image_file"] for e in dev_entries]
    assert len(dev_files) == 40

    f1_045 = load_f1_by_file(FILE_045)
    f1_030 = load_f1_by_file(FILE_030)

    folds = [dev_files[i * FOLD_SIZE:(i + 1) * FOLD_SIZE] for i in range(N_FOLDS)]
    assert all(len(f) == FOLD_SIZE for f in folds)

    print(f"{'round':>6} {'selected on 32 (0.45 vs 0.30)':>32} {'held-out F1 (8 imgs)':>22}")
    print("-" * 64)
    winners = []
    heldout_f1s = []
    for i in range(N_FOLDS):
        heldout = folds[i]
        selection_set = [f for j, fold in enumerate(folds) if j != i for f in fold]

        sel_045 = mean([f1_045[f] for f in selection_set])
        sel_030 = mean([f1_030[f] for f in selection_set])
        winner = "0.30" if sel_030 >= sel_045 else "0.45"
        winners.append(winner)

        held_scores = f1_030 if winner == "0.30" else f1_045
        held_f1 = mean([held_scores[f] for f in heldout])
        heldout_f1s.append(held_f1)

        print(f"{i + 1:>6} {'0.45=' + format(sel_045, '.4f') + '  0.30=' + format(sel_030, '.4f') + '  -> ' + winner:>32} {held_f1:>22.4f}")

    n_030_wins = sum(1 for w in winners if w == "0.30")
    print(f"\nbox_thresh=0.30 selected in {n_030_wins}/{N_FOLDS} rounds")
    print(f"mean held-out Set F1 across all 5 rounds = {mean(heldout_f1s):.4f}")

    import statistics
    if len(heldout_f1s) > 1:
        print(f"std dev across rounds = {statistics.stdev(heldout_f1s):.4f}")


if __name__ == "__main__":
    main()
