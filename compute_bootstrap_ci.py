"""Bootstrap 95% confidence intervals: ci_mean (percentile bootstrap of a
mean) and ci_paired_delta (paired bootstrap of a difference). These two
functions are used by every current evaluation script; the CIs of the
reported results come from evaluate_final_v2.py / export_heldout_results.py.

NOTE: running this file directly (main below) reproduces the CIs of the
PRE-FIX results in old/ABSTRACT_FINAL.md (Set F1 0.5715 etc.), kept for
history only -- they are not the numbers reported in the thesis now.

main() reports:
  1. Main result (box_thresh=0.30, n=216 "original"): mean Set F1 / CER with CI.
  2. Paired bootstrap on the box_thresh 0.45 -> 0.30 delta (same 216 images
     both sides) to check whether the improvement is likely not noise.
  3. Per-scenario means (original + 4 augmented environments, n=216 each)
     with CI, for the table in section 4.5.

Usage:
  python compute_bootstrap_ci.py
"""
import csv

import numpy as np

N_BOOT = 10000
RNG = np.random.default_rng(0)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") == "ok"]


def ci_mean(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return values.mean(), lo, hi


def ci_paired_delta(values_a, values_b, n_boot=N_BOOT):
    """values_b - values_a, resampling image indices jointly (paired)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = len(a)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_delta = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(boot_delta, [2.5, 97.5])
    return (b.mean() - a.mean()), lo, hi


def main():
    print("NOTE: pre-fix results (history only). Reported results: results/final_v2/ "
          "(evaluate_final_v2.py, export_heldout_results.py)\n")
    print("=" * 70)
    print("1. Main result: box_thresh=0.30, n=216 'original' (current pipeline)")
    print("=" * 70)
    rows = load_rows("results/ocr_paddle_fuzzy_FINAL_original_full_validation_ingredient.csv")
    for metric in ("set_f1", "set_precision", "set_recall", "cer_fuzzy"):
        vals = [r[metric] for r in rows]
        mean, lo, hi = ci_mean(vals)
        print(f"  {metric:15s} mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    print()
    print("=" * 70)
    print("2. Paired delta: box_thresh 0.45 -> 0.30 (same 216 images)")
    print("=" * 70)
    rows_045 = load_rows("results/_backup_before_rerun/ocr_paddle_fuzzy_FINAL_original_full_validation_ingredient.csv")
    rows_030 = rows  # already loaded above
    by_file_045 = {r["image_file"]: r for r in rows_045}
    by_file_030 = {r["image_file"]: r for r in rows_030}
    common = sorted(set(by_file_045) & set(by_file_030))
    print(f"  matched images: {len(common)}")
    for metric in ("set_f1", "cer_fuzzy"):
        a = [float(by_file_045[f][metric]) for f in common]
        b = [float(by_file_030[f][metric]) for f in common]
        delta, lo, hi = ci_paired_delta(a, b)
        sig = "excludes 0 (likely real)" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "includes 0 (not significant at 95%)"
        print(f"  delta {metric:15s} = {delta:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {sig}")

    print()
    print("=" * 70)
    print("3. Per-scenario means with 95% CI (section 4.5 table)")
    print("=" * 70)
    scenario_files = {"original": "results/ocr_paddle_fuzzy_FINAL_original_full_validation_ingredient.csv"}
    aug_rows = load_rows("results/ocr_paddle_fuzzy_FINAL_augmented_full_validation_ingredient.csv")
    by_scenario = {}
    for r in aug_rows:
        by_scenario.setdefault(r["dataset"], []).append(r)

    all_scenarios = {"original": rows, **by_scenario}
    for name in ("original", "indoor", "low_light", "blur", "night"):
        srows = all_scenarios[name]
        f1_vals = [r["set_f1"] for r in srows]
        cer_vals = [r["cer_fuzzy"] for r in srows]
        f1_mean, f1_lo, f1_hi = ci_mean(f1_vals)
        cer_mean, cer_lo, cer_hi = ci_mean(cer_vals)
        print(f"  {name:12s} n={len(srows):4d}  Set F1={f1_mean:.4f} [{f1_lo:.4f},{f1_hi:.4f}]   CER={cer_mean:.4f} [{cer_lo:.4f},{cer_hi:.4f}]")


if __name__ == "__main__":
    main()
