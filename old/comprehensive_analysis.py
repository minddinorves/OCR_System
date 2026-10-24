"""Supplementary statistical analysis on top of the existing saved results,
covering 4 things that were missing for thesis-level completeness (none of
this requires re-running OCR -- it all reads already-saved results/*.csv):

  1. Wilcoxon signed-rank test: is "Deskew+Fuzzy beats Baseline" (and a few
     other key pairs) actually statistically significant, or could the ~1-2%
     CER gap be sampling noise?
  2. CER distribution per technique: mean, median, std, and % of images with
     catastrophic failure (CER > 0.9), since a low mean can hide a heavy
     failure tail (as found for GOT-OCR2.0 in the earlier experiment).
  3. Set Precision vs Recall shown separately (not just F1), to see whether
     a technique trades one for the other.
  4. Correlation between original image resolution (height in px) and CER,
     to quantify (not just describe) how much image size drives errors.

Usage:
  python comprehensive_analysis.py
"""
import csv
import os

import cv2
import numpy as np
from scipy.stats import wilcoxon, pearsonr

import ocr_core as core

RESULTS_DIR = core.RESULTS_DIR

KEY_FILES = {
    "Baseline": "ocr_paddle_only.csv",
    "Deskew": "preprocess_deskew.csv",
    "Fuzzy": "ocr_paddle_fuzzy.csv",
    "Deskew+Fuzzy": "preprocess_best_plus_fuzzy.csv",
}

ALL_TECHNIQUE_FILES = {
    "Baseline": ("ocr_paddle_only.csv", "cer"),
    "Fuzzy": ("ocr_paddle_fuzzy.csv", "cer_fuzzy"),
    "Grayscale": ("preprocess_grayscale.csv", "cer"),
    "Threshold": ("preprocess_threshold.csv", "cer"),
    "Denoise": ("preprocess_denoise.csv", "cer"),
    "Sharpen": ("preprocess_sharpen.csv", "cer"),
    "Upscale": ("preprocess_upscale.csv", "cer"),
    "Deskew": ("preprocess_deskew.csv", "cer"),
    "Contrast Adaptive": ("preprocess_contrast_adaptive.csv", "cer"),
    "Upscale800": ("preprocess_upscale800.csv", "cer"),
    "Tiled OCR": ("preprocess_tiled_ocr.csv", "cer"),
    "Multi-Tier Fallback": ("preprocess_multitier_fallback.csv", "cer"),
    "Deskew+Fuzzy": ("preprocess_best_plus_fuzzy.csv", "cer_fuzzy"),
}


def load_cer_by_key(filename, cer_field):
    """Return {(dataset, image_file): cer} for scored rows in a results CSV."""
    path = os.path.join(RESULTS_DIR, filename)
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            val = row.get(cer_field)
            if val in ("", None):
                continue
            out[(row["dataset"], row["image_file"])] = float(val)
    return out


def paired_values(cer_a, cer_b):
    keys = sorted(set(cer_a) & set(cer_b))
    a = [cer_a[k] for k in keys]
    b = [cer_b[k] for k in keys]
    return a, b, keys


# ---------------------------------------------------------------------------
print("=" * 70)
print("1. WILCOXON SIGNED-RANK TEST (paired, per-image CER)")
print("=" * 70)

cer_baseline = load_cer_by_key(*("ocr_paddle_only.csv", "cer"))
cer_deskew = load_cer_by_key(*("preprocess_deskew.csv", "cer"))
cer_fuzzy = load_cer_by_key(*("ocr_paddle_fuzzy.csv", "cer_fuzzy"))
cer_deskew_fuzzy = load_cer_by_key(*("preprocess_best_plus_fuzzy.csv", "cer_fuzzy"))

pairs_to_test = [
    ("Deskew+Fuzzy", cer_deskew_fuzzy, "Baseline", cer_baseline),
    ("Deskew", cer_deskew, "Baseline", cer_baseline),
    ("Fuzzy", cer_fuzzy, "Baseline", cer_baseline),
    ("Deskew+Fuzzy", cer_deskew_fuzzy, "Deskew", cer_deskew),
]

for name_a, dict_a, name_b, dict_b in pairs_to_test:
    a, b, keys = paired_values(dict_a, dict_b)
    diff = np.array(a) - np.array(b)
    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        print(f"{name_a} vs {name_b}: n={len(keys)}, all differences are zero, cannot test")
        continue
    stat, p = wilcoxon(a, b)
    direction = "LOWER (better)" if np.median(diff) < 0 else "HIGHER (worse)"
    sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant (p>=0.05)"
    print(f"{name_a} vs {name_b}: n={len(keys)}, mean_diff={diff.mean():+.4f}, "
          f"median_diff={np.median(diff):+.4f} [{direction}], W={stat:.1f}, p={p:.4g} -> {sig}")

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("2. CER DISTRIBUTION PER TECHNIQUE (mean / median / std / %CER>0.9)")
print("=" * 70)
print(f"{'Technique':<22} {'mean':>7} {'median':>7} {'std':>7} {'%CER>0.9':>9} {'n':>5}")
print("-" * 62)

dist_rows = []
for name, (fname, field) in ALL_TECHNIQUE_FILES.items():
    cers = list(load_cer_by_key(fname, field).values())
    if not cers:
        continue
    arr = np.array(cers)
    pct_bad = 100.0 * np.mean(arr > 0.9)
    print(f"{name:<22} {arr.mean():>7.4f} {np.median(arr):>7.4f} {arr.std():>7.4f} "
          f"{pct_bad:>8.1f}% {len(arr):>5}")
    dist_rows.append({
        "technique": name, "mean_cer": round(arr.mean(), 4), "median_cer": round(float(np.median(arr)), 4),
        "std_cer": round(arr.std(), 4), "pct_cer_gt_0.9": round(pct_bad, 2), "n": len(arr),
    })

with open(os.path.join(RESULTS_DIR, "cer_distribution.csv"), "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["technique", "mean_cer", "median_cer", "std_cer", "pct_cer_gt_0.9", "n"])
    writer.writeheader()
    writer.writerows(dist_rows)
print(f"\nWrote results/cer_distribution.csv")

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("3. SET PRECISION vs RECALL (from results/summary_ingredient.csv)")
print("=" * 70)

summary_ing_path = os.path.join(RESULTS_DIR, "summary_ingredient.csv")
if os.path.isfile(summary_ing_path):
    with open(summary_ing_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_file = {}
    for r in rows:
        by_file.setdefault(r["source_file"], []).append(r)

    print(f"{'Technique':<32} {'mean P':>8} {'mean R':>8} {'mean F1':>8}")
    print("-" * 58)
    for fname in ["ocr_paddle_only.csv", "ocr_paddle_fuzzy.csv", "preprocess_deskew.csv",
                  "preprocess_best_plus_fuzzy.csv"]:
        if fname not in by_file:
            continue
        rs = by_file[fname]
        p = np.mean([float(r["mean_set_precision"]) for r in rs if r["mean_set_precision"] not in ("", None)])
        r_ = np.mean([float(r["mean_set_recall"]) for r in rs if r["mean_set_recall"] not in ("", None)])
        f1 = np.mean([float(r["mean_set_f1"]) for r in rs if r["mean_set_f1"] not in ("", None)])
        print(f"{fname:<32} {p:>8.4f} {r_:>8.4f} {f1:>8.4f}")
else:
    print("results/summary_ingredient.csv not found -- run compute_ingredient_metrics.py first")

# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("4. CORRELATION: original image height (px) vs CER")
print("=" * 70)


_HEIGHT_CACHE = {}


def image_height(dataset, image_file):
    key = (dataset, image_file)
    if key in _HEIGHT_CACHE:
        return _HEIGHT_CACHE[key]
    image_dir, _ = core.DATASETS[dataset]
    path = os.path.join(image_dir, image_file)
    img = cv2.imread(path)
    h = img.shape[0] if img is not None else None
    _HEIGHT_CACHE[key] = h
    return h


targets = [("Baseline", ("ocr_paddle_only.csv", "cer")),
           ("Deskew", ("preprocess_deskew.csv", "cer")),
           ("Deskew+Fuzzy", ("preprocess_best_plus_fuzzy.csv", "cer_fuzzy"))]

# Pre-warm the height cache once over the union of all keys needed, so each
# unique (dataset, image_file) is read from disk exactly once total instead
# of once per technique.
all_keys = set()
for _, (fname, field) in targets:
    all_keys.update(load_cer_by_key(fname, field).keys())
print(f"Reading dimensions for {len(all_keys)} unique images (once each)...", flush=True)
for dataset, image_file in all_keys:
    image_height(dataset, image_file)
print("Done reading dimensions.", flush=True)

for name, (fname, field) in targets:
    cer_map = load_cer_by_key(fname, field)
    heights, cers = [], []
    for (dataset, image_file), cer in cer_map.items():
        h = image_height(dataset, image_file)
        if h is None:
            continue
        heights.append(h)
        cers.append(cer)
    r, p = pearsonr(heights, cers)
    sig = "significant" if p < 0.05 else "not significant"
    print(f"{name:<15} Pearson r = {r:+.4f} (p={p:.4g}, {sig}), n={len(heights)}  "
          f"[negative r = taller image -> lower CER, as expected]")

print("\nDone.")
