import csv
import numpy as np

RNG = np.random.default_rng(0)

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image_file"]: float(r["set_f1"]) for r in csv.DictReader(f) if r.get("status") == "ok"}

dev_files = sorted(load("results/sweep_det_baseline_ingredient.csv").keys())
base_045 = load("results/sweep_det_baseline_ingredient.csv")
low_030 = load("results/sweep_det_low_box_thresh_ingredient.csv")  # n=216, need subset

# restrict to dev-40 subset
low_030_dev = {k: v for k, v in low_030.items() if k in base_045}
print(f"dev images: {len(dev_files)}, baseline(0.45): {len(base_045)}, box_thresh=0.30 subset: {len(low_030_dev)}")

# 5-fold CV
folds = np.array_split(np.array(dev_files), 5)
fold_results = []
for i, test_fold in enumerate(folds):
    train_files = [f for f in dev_files if f not in set(test_fold)]
    mean_045_train = np.mean([base_045[f] for f in train_files])
    mean_030_train = np.mean([low_030_dev[f] for f in train_files])
    chosen = "0.30" if mean_030_train > mean_045_train else "0.45"
    test_045 = np.mean([base_045[f] for f in test_fold])
    test_030 = np.mean([low_030_dev[f] for f in test_fold])
    chosen_test_score = test_030 if chosen == "0.30" else test_045
    fold_results.append((i+1, len(test_fold), mean_045_train, mean_030_train, chosen, test_045, test_030, chosen_test_score))
    print(f"fold {i+1} (test n={len(test_fold)}): train-mean F1 0.45={mean_045_train:.4f} 0.30={mean_030_train:.4f} -> choose {chosen} | test F1 0.45={test_045:.4f} 0.30={test_030:.4f} chosen={chosen_test_score:.4f}")

chosen_scores = [r[7] for r in fold_results]
print(f"\nmean test F1 of the per-fold *selected* config: {np.mean(chosen_scores):.4f} (std {np.std(chosen_scores):.4f})")
print(f"0.30 chosen in {sum(1 for r in fold_results if r[4]=='0.30')}/5 folds")
