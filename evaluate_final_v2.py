"""Regenerate the FINAL validation results after the fuzzy-matching fix
(join OCR lines before splitting, case-insensitive matching, threshold 85),
from the raw OCR text already saved by the FINAL runs -- no OCR re-run
(the fixed correct_with_vocabulary gives identical output on the joined raw
text as on the per-line OCR output; verified 216/216).

The pre-fix FINAL files are left untouched (copies also kept in
results/_before_fuzzy_fix/). Configurations compared, all 5 scenarios:

  before_fix  ocr_text_fuzzy as stored in the pre-fix FINAL files
  no_fuzzy    raw OCR text, no correction
  legacy_patch  fixed code, threshold 85, the earlier 199-name patch
              (common_ingredients_patch.csv, ocr_core.LEGACY_PATCH_PATH) built
              from all 216 labels, so dev and held-out are both exposed to
              label leakage. Comparison only.
  leak_free   the pipeline as shipped: fixed code, threshold 85, patch built
              from the 40 dev labels only (vocabulary_patch_dev.csv,
              ocr_core.INCI_PATCH_PATH). Reported on held-out products only:
              the unbiased estimate, since the threshold and patch were both
              chosen without them.

Splits: all 216 / dev 40 / held-out 176, by product (label file), so
augmented copies of dev images are never counted as held-out.

Outputs (results/final_v2/):
  ocr_paddle_fuzzy_FINAL_v2_original_full_validation.csv    per image, leak_free config,
  ocr_paddle_fuzzy_FINAL_v2_augmented_full_validation.csv   same columns as the pre-fix
                                                            *_ingredient.csv files
  per_image_all_configs.csv
  summary.csv            means + 95% bootstrap CI
  deltas.csv             paired-bootstrap differences between configs

Usage:
  python evaluate_final_v2.py
"""
import csv
import os

import ocr_core as core
import ocr_paddle_fuzzy as fz

from compute_bootstrap_ci import ci_mean, ci_paired_delta
from compute_ingredient_metrics import ier, set_prf1, tokenize

THRESHOLD = 85.0
N_DEV = 40
OUT_DIR = os.path.join(core.RESULTS_DIR, "final_v2")
FINAL = {
    "original": os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_original_full_validation.csv"),
    "augmented": os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_augmented_full_validation.csv"),
}
SCENARIOS = ("original", "indoor", "low_light", "blur", "night")
CONFIGS = ("before_fix", "no_fuzzy", "legacy_patch", "leak_free")
METRICS = ("set_f1", "set_precision", "set_recall", "ier", "cer", "wer")
INGREDIENT_FIELDS = ["ier", "set_precision", "set_recall", "set_f1"]


def metrics(hyp, ref):
    hyp_t, ref_t = tokenize(hyp), tokenize(ref)
    p, r, f1 = set_prf1(hyp_t, ref_t)
    return {"set_f1": f1, "set_precision": p, "set_recall": r, "ier": ier(hyp_t, ref_t),
            "cer": core.compute_cer(hyp, ref), "wer": core.compute_wer(hyp, ref)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    legacy_vocab, seen = [], set()  # same de-duplication as core.load_inci_vocabulary
    for name in core._load_names(core.INCI_DB_PATH) + core._load_names(core.LEGACY_PATCH_PATH):
        if name.lower() not in seen:
            seen.add(name.lower())
            legacy_vocab.append(name)
    leak_free_vocab = core.load_inci_vocabulary()  # main DB + vocabulary_patch_dev.csv, as the pipeline runs
    dev_labels = {e["label_file"] for e in core.iter_dataset_entries(["original"], limit=N_DEV)}

    per_image, scores = [], {}  # scores[(config, scenario)][label_file] = metrics
    for which, path in FINAL.items():
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        v2_rows = []
        for row in rows:
            entry = {"dataset": row["dataset"], "image_file": row["image_file"], "label_file": row["label_file"],
                     "status": row["status"], "reference_text": row["reference_text"]}
            if row["status"] != "ok":
                v2_rows.append(fz.build_row(entry, "", "", 0.0, row["error"] or None))
                continue
            raw, ref = row["ocr_text_raw"], row["reference_text"]
            outputs = {
                "before_fix": row["ocr_text_fuzzy"],
                "no_fuzzy": raw,
                "legacy_patch": fz.correct_with_vocabulary([raw], legacy_vocab, THRESHOLD),
                "leak_free": fz.correct_with_vocabulary([raw], leak_free_vocab, THRESHOLD),
            }
            split = "dev" if row["label_file"] in dev_labels else "heldout"
            for config, hyp in outputs.items():
                m = metrics(hyp, ref)
                scores.setdefault((config, row["dataset"]), {})[row["label_file"]] = (split, m)
                per_image.append({"config": config, "dataset": row["dataset"], "split": split,
                                  "image_file": row["image_file"], **{k: round(v, 4) for k, v in m.items()}})
            v2 = fz.build_row(entry, raw, outputs["leak_free"], float(row["ocr_time_sec"] or 0))
            m = metrics(outputs["leak_free"], ref)
            v2.update({k: round(m[k], 4) for k in INGREDIENT_FIELDS})
            v2_rows.append(v2)
        core.write_csv(os.path.join(OUT_DIR, f"ocr_paddle_fuzzy_FINAL_v2_{which}_full_validation.csv"), v2_rows,
                       fieldnames=fz.FIELDNAMES + INGREDIENT_FIELDS)

    summary, deltas = [], []
    for scenario in SCENARIOS:
        for split in ("all", "dev", "heldout"):
            for config in CONFIGS:
                if config == "leak_free" and split != "heldout":
                    continue  # dev images are inside the leak-free patch's source
                sel = {lf: m for lf, (s, m) in scores[(config, scenario)].items() if split in ("all", s)}
                out = {"dataset": scenario, "split": split, "config": config, "n_images": len(sel),
                       "threshold": "" if config in ("before_fix", "no_fuzzy") else THRESHOLD}
                for metric in METRICS:
                    mean, lo, hi = ci_mean([m[metric] for m in sel.values()])
                    out[metric] = round(mean, 4)
                    out[f"{metric}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
                summary.append(out)
        for a, b in (("before_fix", "legacy_patch"), ("before_fix", "leak_free"), ("no_fuzzy", "leak_free"),
                     ("legacy_patch", "leak_free")):
            held = {lf: m for lf, (s, m) in scores[(b, scenario)].items() if s == "heldout"}
            files = sorted(held)
            for metric in ("set_f1", "cer"):
                d, lo, hi = ci_paired_delta([scores[(a, scenario)][f][1][metric] for f in files],
                                            [held[f][metric] for f in files])
                deltas.append({"dataset": scenario, "split": "heldout", "n_images": len(files),
                               "comparison": f"{b} - {a}", "metric": metric, "delta": round(d, 4),
                               "ci95": f"[{lo:+.4f}, {hi:+.4f}]", "significant": lo > 0 or hi < 0})

    core.write_csv(os.path.join(OUT_DIR, "per_image_all_configs.csv"), per_image,
                   fieldnames=["config", "dataset", "split", "image_file", *METRICS])
    fields = ["dataset", "split", "config", "n_images", "threshold"]
    for metric in METRICS:
        fields += [metric, f"{metric}_ci95"]
    core.write_csv(os.path.join(OUT_DIR, "summary.csv"), summary, fieldnames=fields)
    core.write_csv(os.path.join(OUT_DIR, "deltas.csv"), deltas,
                   fieldnames=["dataset", "split", "n_images", "comparison", "metric", "delta", "ci95", "significant"])

    print(f"{'dataset':10} {'split':8} {'config':11} {'n':>4} {'Set F1 [95% CI]':>26} {'CER [95% CI]':>26}")
    for s in summary:
        print(f"{s['dataset']:10} {s['split']:8} {s['config']:11} {s['n_images']:>4} "
              f"{s['set_f1']:.4f} {s['set_f1_ci95']:>18} {s['cer']:.4f} {s['cer_ci95']:>18}")
    print()
    for d in deltas:
        print(f"{d['dataset']:10} held-out n={d['n_images']} {d['comparison']:24} {d['metric']:6} "
              f"{d['delta']:+.4f} {d['ci95']}{' *' if d['significant'] else ''}")


if __name__ == "__main__":
    main()
