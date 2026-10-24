import csv, glob, re
from collections import Counter

import numpy as np

# ---------- 1) INCI coverage gap (used to justify the 58.6% figure) ----------
master = set()
with open("ingredient_master_dataset_fixed.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        master.add(row["inci_name"].strip().lower())

patch = set()
with open("common_ingredients_patch.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        patch.add(row["inci_name"].strip().lower())

# count occurrences of each ingredient name across the 216 ground-truth label files
counter = Counter()
files = sorted(glob.glob("test_SkinSafe/label/*.txt"))
for fp in files:
    with open(fp, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                counter[name.lower()] += 1

total_unique = len(counter)
missing = {name: c for name, c in counter.items() if name not in master}
print(f"label files: {len(files)}")
print(f"unique ingredient names in ground truth: {total_unique}")
print(f"unique names NOT in master db: {len(missing)}  ({len(missing)/total_unique*100:.1f}% of unique names)")

# occurrence-weighted version (counts every mention, not just unique names)
total_occurrences = sum(counter.values())
missing_occurrences = sum(missing.values())
print(f"total ingredient mentions: {total_occurrences}")
print(f"mentions missing from master db: {missing_occurrences} ({missing_occurrences/total_occurrences*100:.1f}%)")

# how many of the missing names appear >=5 times (candidate patch criterion)
freq5 = {n: c for n, c in missing.items() if c >= 5}
print(f"missing names appearing in >=5 label files: {len(freq5)}")
print(f"of those, how many are actually in the patch file: {sum(1 for n in freq5 if n in patch)}")
print(f"patch file size: {len(patch)}")

# occurrence-weighted missing % after patch is added
still_missing_after_patch = {n: c for n, c in missing.items() if n not in patch}
print(f"unique names still missing after patch: {len(still_missing_after_patch)} of {total_unique} ({len(still_missing_after_patch)/total_unique*100:.1f}%)")
occ_still_missing = sum(still_missing_after_patch.values())
print(f"mentions still missing after patch: {occ_still_missing} ({occ_still_missing/total_occurrences*100:.1f}%)")

print()
print("top 15 missing-by-frequency names NOT in patch (candidates advisor may ask about):")
not_in_patch_missing = {n: c for n, c in missing.items() if n not in patch}
for n, c in sorted(not_in_patch_missing.items(), key=lambda x: -x[1])[:15]:
    print(f"  {c:3d}  {n}")

print()
print("=" * 70)
print("2) paired bootstrap: original vs each augmented scenario (joined on label_file)")
print("=" * 70)

N_BOOT = 10000
RNG = np.random.default_rng(0)

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") == "ok"]

def ci_paired_delta(a, b, n_boot=N_BOOT):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = len(a)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_delta = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(boot_delta, [2.5, 97.5])
    return b.mean() - a.mean(), lo, hi

orig_rows = load_rows("results/ocr_paddle_fuzzy_FINAL_original_full_validation_ingredient.csv")
aug_rows = load_rows("results/ocr_paddle_fuzzy_FINAL_augmented_full_validation_ingredient.csv")

orig_by_label = {r["label_file"]: r for r in orig_rows}
by_scenario = {}
for r in aug_rows:
    by_scenario.setdefault(r["dataset"], {})[r["label_file"]] = r

for scen in ("indoor", "low_light", "blur", "night"):
    common = sorted(set(orig_by_label) & set(by_scenario[scen]))
    for metric in ("set_f1",):
        a = [float(orig_by_label[k][metric]) for k in common]
        b = [float(by_scenario[scen][k][metric]) for k in common]
        delta, lo, hi = ci_paired_delta(a, b)
        print(f"  {scen:10s} n={len(common):3d}  delta Set F1 (scenario - original) = {delta:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
