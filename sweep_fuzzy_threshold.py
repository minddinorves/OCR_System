"""Fuzzy-matching threshold sweep with a dev / held-out split.

  dev      = first 40 "original" images by filename (the set every earlier
             parameter sweep was tuned on)
  held-out = the remaining 176 images (never used to pick any parameter)

All runs use the current OCR settings (box_thresh=0.30, 25% padding); only
the fuzzy correction step changes. Two correction variants are scored:

  current  -- ocr_paddle_fuzzy.correct_with_vocabulary exactly as the
              pipeline runs it, fed the per-line OCR output from
              results/sweep_cache/pad_0.25.json (written by sweep_sensitivity.py).
  fixed    -- an inline re-implementation of the fix (lines joined,
              case-insensitive) on the joined raw text saved by the FINAL run.

The fix has since been applied to ocr_paddle_fuzzy.py, so both variants now
give the same numbers. Results from before the fix (when "current" was the
per-line, case-sensitive version) are kept in results/_before_fuzzy_fix/.

Outputs:
  results/sweep_fuzzy_threshold_per_image.csv  -- IER, Set P/R/F1, CER per image
  results/sweep_fuzzy_threshold_summary.csv    -- means, 95% CI, paired delta vs 90

Usage:
  python sweep_fuzzy_threshold.py                       # existing 199-name patch (built from all labels)
  python sweep_fuzzy_threshold.py --patch dev_min1      # patch built from dev labels only (no leakage;
                                                        # dev images scored with 5-fold cross-fitted patches)
"""
import argparse
import csv
import json
import os

import numpy as np

import ocr_core as core
import ocr_paddle_fuzzy as fz
from build_vocabulary_patch import build_patch

from compute_bootstrap_ci import ci_mean, ci_paired_delta
from compute_ingredient_metrics import ier, set_prf1, tokenize

THRESHOLDS = [None, 75, 80, 82, 84, 85, 86, 88, 90, 92, 95]  # None = no fuzzy correction
REFERENCE_THRESHOLD = 90
N_DEV = 40
FINAL_CSV = os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_original_full_validation.csv")
CACHE_PATH = os.path.join(core.RESULTS_DIR, "sweep_cache", "pad_0.25.json")
METRICS = ("ier", "set_precision", "set_recall", "set_f1", "cer")
DEV_PATCH_PATH = os.path.join(core.RESULTS_DIR, "leakage_check", "patch_dev_min1.csv")
N_DEV_FOLDS = 5


def label(threshold):
    return "no_fuzzy" if threshold is None else str(threshold)


def score(hyp, ref):
    hyp_t, ref_t = tokenize(hyp), tokenize(ref)
    p, r, f1 = set_prf1(hyp_t, ref_t)
    return {"ier": ier(hyp_t, ref_t), "set_precision": p, "set_recall": r, "set_f1": f1,
            "cer": core.compute_cer(hyp, ref)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", choices=["leaked", "dev_min1"], default="leaked",
                        help="leaked = common_ingredients_patch.csv; dev_min1 = "
                             "results/leakage_check/patch_dev_min1.csv (run leakage_check_vocab_patch.py first)")
    args = parser.parse_args()
    suffix = "" if args.patch == "leaked" else f"_{args.patch}"
    with open(CACHE_PATH, encoding="utf-8") as fh:
        cache = json.load(fh)
    with open(FINAL_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    dev_entries = list(core.iter_dataset_entries(["original"], limit=N_DEV))
    dev = {e["image_file"] for e in dev_entries}

    def make_vocab(patch):
        vocab = list({n.lower(): n for n in core._load_names(core.INCI_DB_PATH) + patch}.values())
        return vocab, list(dict.fromkeys(v.lower() for v in vocab))

    if args.patch == "leaked":
        shared = make_vocab(core._load_names(core.LEGACY_PATCH_PATH))
        vocab_for = {r["image_file"]: shared for r in rows}
    else:
        # Held-out images use the patch built from all 40 dev labels. Each dev
        # image is scored with a patch built from the OTHER dev folds only:
        # scoring dev images against a vocabulary containing their own answers
        # rewards low thresholds (every token snaps to the answer key) and would
        # leak into the threshold choice.
        heldout_vocab = make_vocab(core._load_names(DEV_PATCH_PATH))
        vocab_for = {r["image_file"]: heldout_vocab for r in rows}
        for k in range(N_DEV_FOLDS):
            fold = dev_entries[k::N_DEV_FOLDS]
            others = [e["label_path"] for e in dev_entries if e not in fold]
            fold_vocab = make_vocab(build_patch(1, others))
            for e in fold:
                vocab_for[e["image_file"]] = fold_vocab

    def correct(variant, threshold, row):
        raw = row["ocr_text_raw"]
        if threshold is None:
            return raw
        vocab, vocab_lower = vocab_for[row["image_file"]]
        if variant == "current":
            return fz.correct_with_vocabulary(cache[row["image_file"]]["lines"], vocab, threshold)
        return fz.correct_with_vocabulary([raw.lower()], vocab_lower, threshold)

    per_image = []
    scores = {}  # (variant, split, threshold) -> {image_file: metrics}
    for variant in ("current", "fixed"):
        for row in rows:
            if variant == "current" and row["image_file"] not in cache:
                continue
            split = "dev" if row["image_file"] in dev else "heldout"
            for t in THRESHOLDS:
                m = score(correct(variant, t, row), row["reference_text"])
                scores.setdefault((variant, split, t), {})[row["image_file"]] = m
                per_image.append({"variant": variant, "split": split, "threshold": label(t),
                                  "image_file": row["image_file"],
                                  **{k: round(v, 4) for k, v in m.items()}})

    summary = []
    for variant in ("current", "fixed"):
        dev_f1 = {t: np.mean([m["set_f1"] for m in scores[(variant, "dev", t)].values()]) for t in THRESHOLDS}
        best_dev = max(THRESHOLDS, key=lambda t: (dev_f1[t], t or 0))
        for split in ("dev", "heldout"):
            if (variant, split, REFERENCE_THRESHOLD) not in scores:
                continue
            ref = scores[(variant, split, REFERENCE_THRESHOLD)]
            files = sorted(ref)
            print(f"\n=== {variant} / {split} (n={len(files)}) ===")
            print(f"{'threshold':>9} {'IER':>7} {'P':>7} {'R':>7} {'F1':>7} {'F1 95% CI':>18} {'CER':>7} "
                  f"{'dF1 vs 90 [95% CI]':>28}")
            for t in THRESHOLDS:
                cur = scores[(variant, split, t)]
                out = {"variant": variant, "split": split, "n_images": len(files), "threshold": label(t),
                       "selected_on_dev": t == best_dev}
                for metric in METRICS:
                    mean, lo, hi = ci_mean([cur[f][metric] for f in files])
                    out[metric] = round(mean, 4)
                    out[f"{metric}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
                if t == REFERENCE_THRESHOLD:
                    delta_txt = "(reference)"
                else:
                    d, lo, hi = ci_paired_delta([ref[f]["set_f1"] for f in files], [cur[f]["set_f1"] for f in files])
                    out["set_f1_delta_vs_90"] = round(d, 4)
                    out["set_f1_delta_ci95"] = f"[{lo:+.4f}, {hi:+.4f}]"
                    out["set_f1_delta_significant"] = lo > 0 or hi < 0
                    delta_txt = f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]" + ("*" if lo > 0 or hi < 0 else " ")
                summary.append(out)
                mark = " <- best on dev" if t == best_dev else ""
                print(f"{label(t):>9} {out['ier']:7.4f} {out['set_precision']:7.4f} {out['set_recall']:7.4f} "
                      f"{out['set_f1']:7.4f} {out['set_f1_ci95']:>18} {out['cer']:7.4f} {delta_txt:>28}{mark}")

    core.write_csv(os.path.join(core.RESULTS_DIR, f"sweep_fuzzy_threshold{suffix}_per_image.csv"), per_image,
                   fieldnames=["variant", "split", "threshold", "image_file", *METRICS])
    fields = ["variant", "split", "n_images", "threshold", "selected_on_dev"]
    for metric in METRICS:
        fields += [metric, f"{metric}_ci95"]
    fields += ["set_f1_delta_vs_90", "set_f1_delta_ci95", "set_f1_delta_significant"]
    core.write_csv(os.path.join(core.RESULTS_DIR, f"sweep_fuzzy_threshold{suffix}_summary.csv"), summary,
                   fieldnames=fields)
    print("\n* = paired-bootstrap 95% CI of the difference vs threshold 90 excludes 0")


if __name__ == "__main__":
    main()
