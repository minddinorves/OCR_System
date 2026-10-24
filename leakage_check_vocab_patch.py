"""Measure how much the vocabulary patch's label leakage inflates results.

common_ingredients_patch.csv was built from the labels of all 216 products,
which are the same labels every scenario is scored against. Here the patch is
rebuilt from the 40 dev products' labels only (the first 40 "original"
images, the set every parameter sweep was tuned on) and every vocabulary is
scored on the 176 held-out products only, in all 5 scenarios. The split is by
product (label file), so augmented copies of a dev image never count as
held-out.

Vocabularies compared (all = ingredient_master_dataset_fixed.csv + patch):
  no_patch          main database only
  leaked_patch      the existing 199-name patch (built from all 216 labels)
  dev_patch_min1    names in >= 1 dev label, missing from the main database
                    (5 of 216 products scaled to 40 products rounds to 1)
  dev_patch_min2    names in >= 2 dev labels (sensitivity check)

Correction step:
  current  ocr_paddle_fuzzy.correct_with_vocabulary per OCR line,
           case-sensitive, as the pipeline runs today. "original" only, and
           only images whose per-line OCR is cached by sweep_sensitivity.py.
  fixed    lines joined, matched case-insensitively. All 5 scenarios, from the
           raw OCR text already saved by the FINAL validation runs.

Nothing existing is overwritten. Outputs:
  results/leakage_check/patch_dev_min1.csv, patch_dev_min2.csv
  results/leakage_check/per_image.csv
  results/leakage_check/summary.csv

Usage:
  python leakage_check_vocab_patch.py [--threshold 90]
"""
import argparse
import csv
import json
import os
import sys

import ocr_core as core
import ocr_paddle_fuzzy as fz
from build_vocabulary_patch import build_patch, write_patch

sys.path.insert(0, os.path.join(core.PROJECT_ROOT, "old"))
from compute_bootstrap_ci import ci_mean, ci_paired_delta  # noqa: E402
from compute_ingredient_metrics import ier, set_prf1, tokenize  # noqa: E402

OUT_DIR = os.path.join(core.RESULTS_DIR, "leakage_check")
FINAL_CSVS = [
    os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_original_full_validation.csv"),
    os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_augmented_full_validation.csv"),
]
CACHE_PATH = os.path.join(core.RESULTS_DIR, "sweep_cache", "pad_0.25.json")
SCENARIOS = ("original", "indoor", "low_light", "blur", "night")
METRICS = ("ier", "set_precision", "set_recall", "set_f1", "cer")
N_DEV = 40


def score(hyp, ref):
    hyp_t, ref_t = tokenize(hyp), tokenize(ref)
    p, r, f1 = set_prf1(hyp_t, ref_t)
    return {"ier": ier(hyp_t, ref_t), "set_precision": p, "set_recall": r, "set_f1": f1,
            "cer": core.compute_cer(hyp, ref)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    dev_entries = list(core.iter_dataset_entries(["original"], limit=N_DEV))
    dev_labels = {e["label_file"] for e in dev_entries}
    dev_paths = [e["label_path"] for e in dev_entries]

    main_db = core._load_names(core.INCI_DB_PATH)
    patches = {
        "no_patch": [],
        "leaked_patch": core._load_names(core.INCI_PATCH_PATH),
        "dev_patch_min1": build_patch(1, dev_paths),
        "dev_patch_min2": build_patch(2, dev_paths),
    }
    for name in ("dev_patch_min1", "dev_patch_min2"):
        write_patch(os.path.join(OUT_DIR, f"patch_{name.replace('dev_patch_', 'dev_')}.csv"), patches[name])
    leaked_l = {n.lower() for n in patches["leaked_patch"]}
    for name, names in patches.items():
        print(f"{name:15s} {len(names):4d} names, overlap with leaked 199: {len({n.lower() for n in names} & leaked_l)}")

    vocabs = {}
    for name, names in patches.items():
        vocab = list({n.lower(): n for n in main_db + names}.values())
        vocabs[name] = (vocab, list(dict.fromkeys(v.lower() for v in vocab)))

    with open(CACHE_PATH, encoding="utf-8") as fh:
        cache = json.load(fh)
    rows = []
    for path in FINAL_CSVS:
        with open(path, newline="", encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(fh) if r["status"] == "ok" and r["label_file"] not in dev_labels]

    per_image, scores = [], {}
    for row in rows:
        ref = row["reference_text"]
        for vname, (vocab, vocab_lower) in vocabs.items():
            outputs = {"fixed": fz.correct_with_vocabulary([row["ocr_text_raw"].lower()], vocab_lower, args.threshold)}
            if row["dataset"] == "original" and row["image_file"] in cache:
                outputs["current"] = fz.correct_with_vocabulary(cache[row["image_file"]]["lines"], vocab,
                                                                args.threshold)
            for variant, hyp in outputs.items():
                m = score(hyp, ref)
                scores.setdefault((variant, row["dataset"], vname), {})[row["label_file"]] = m
                per_image.append({"variant": variant, "dataset": row["dataset"], "vocabulary": vname,
                                  "image_file": row["image_file"], "label_file": row["label_file"],
                                  **{k: round(v, 4) for k, v in m.items()}})

    summary = []
    print(f"\nheld-out products only, threshold={args.threshold:g}")
    print(f"{'variant':8} {'dataset':10} {'vocabulary':15} {'n':>4} {'IER':>7} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'CER':>7} {'dF1 vs leaked [95% CI]':>28}")
    for variant in ("current", "fixed"):
        for dataset in SCENARIOS:
            if (variant, dataset, "leaked_patch") not in scores:
                continue
            leaked = scores[(variant, dataset, "leaked_patch")]
            files = sorted(leaked)
            for vname in vocabs:
                cur = scores[(variant, dataset, vname)]
                out = {"variant": variant, "dataset": dataset, "vocabulary": vname, "n_images": len(files),
                       "threshold": args.threshold}
                for metric in METRICS:
                    mean, lo, hi = ci_mean([cur[f][metric] for f in files])
                    out[metric] = round(mean, 4)
                    out[f"{metric}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
                if vname == "leaked_patch":
                    delta_txt = "(reference)"
                else:
                    d, lo, hi = ci_paired_delta([leaked[f]["set_f1"] for f in files], [cur[f]["set_f1"] for f in files])
                    out["set_f1_delta_vs_leaked"] = round(d, 4)
                    out["set_f1_delta_ci95"] = f"[{lo:+.4f}, {hi:+.4f}]"
                    out["set_f1_delta_significant"] = lo > 0 or hi < 0
                    delta_txt = f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]" + ("*" if lo > 0 or hi < 0 else " ")
                summary.append(out)
                print(f"{variant:8} {dataset:10} {vname:15} {len(files):>4} {out['ier']:7.4f} "
                      f"{out['set_precision']:7.4f} {out['set_recall']:7.4f} {out['set_f1']:7.4f} "
                      f"{out['cer']:7.4f} {delta_txt:>28}")

    core.write_csv(os.path.join(OUT_DIR, "per_image.csv"), per_image,
                   fieldnames=["variant", "dataset", "vocabulary", "image_file", "label_file", *METRICS])
    fields = ["variant", "dataset", "vocabulary", "n_images", "threshold"]
    for metric in METRICS:
        fields += [metric, f"{metric}_ci95"]
    fields += ["set_f1_delta_vs_leaked", "set_f1_delta_ci95", "set_f1_delta_significant"]
    core.write_csv(os.path.join(OUT_DIR, "summary.csv"), summary, fieldnames=fields)
    print("\n* = paired-bootstrap 95% CI of the difference vs leaked_patch excludes 0")


if __name__ == "__main__":
    main()
