"""Offline simulation of a PaddleOCR -> Google Vision fallback hybrid.

Policy: run PaddleOCR on every image; if a label-free "this read looks bad"
signal computed from PaddleOCR's own output is below a threshold, send the
image to Google Vision instead. Both engines go through the identical
correct_with_vocabulary post-processing and the same CER / Set P-R-F1
evaluation used everywhere else in this project.

No OCR is run here: it reads the saved PaddleOCR raw text (results/
ocr_paddle_fuzzy_FIXED_missing_comma_*.csv) and the Google Vision raw text
(results/ocr_google_vision.csv, produced by ocr_google_vision.py), so any
number of routing policies can be tried without paying for extra API calls.

Routing signals (computed from PaddleOCR output only -- no ground truth):
  inci_ratio : fraction of output tokens that are exact INCI vocabulary names
               after fuzzy correction (low = OCR produced unrecognizable text)
  n_chars    : length of the raw PaddleOCR text (low = read almost nothing)
PaddleOCR rec_scores are not used: the saved runs don't contain them, and
re-running PaddleOCR on CPU (~30 s/image) for 1,080 images would take ~9 h.

Selection is done on a DEV split and reported on a HELD-OUT split, same rule
as sweep_fuzzy_threshold_devheldout.py: dev = the first 40 "original" products
(and their 4 augmented copies, so one product never sits on both sides), held
out = the remaining 176 products x 5 scenarios.

Usage:
  python hybrid_paddle_google.py
"""
import csv
import os
from collections import defaultdict

import numpy as np

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import set_prf1, tokenize

PADDLE_CSVS = [
    "results/ocr_paddle_fuzzy_FIXED_missing_comma_full_validation.csv",
    "results/ocr_paddle_fuzzy_FIXED_missing_comma_augmented_full_validation.csv",
]
GOOGLE_CSV = "results/ocr_google_vision.csv"
FUZZY_THRESHOLD = 90
DEV_LIMIT = 40
SIGNALS = ["inci_ratio", "n_chars"]
# Max fraction of images allowed to go to Google when choosing a policy on dev.
BUDGETS = [0.1, 0.2, 0.3, 0.5, 1.0]
N_BOOTSTRAP = 2000
SEED = 0

OUT_PER_IMAGE = "results/hybrid_paddle_google_per_image.csv"
OUT_CURVE = "results/hybrid_paddle_google_curve_heldout.csv"
OUT_SUMMARY = "results/hybrid_paddle_google_summary.csv"


def product_key(dataset, image_file):
    prefix = core.DATASETS[dataset][1]
    return image_file[len(prefix):] if prefix and image_file.startswith(prefix) else image_file


def load_rows(paths):
    rows = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["status"] == "ok":
                    rows[(row["dataset"], row["image_file"])] = row
    return rows


def score(raw_text, reference, vocabulary, vocab_lower):
    fuzzy_text = correct_with_vocabulary([raw_text], vocabulary, FUZZY_THRESHOLD)
    p, r, f1 = set_prf1(tokenize(fuzzy_text), tokenize(reference))
    out_tokens = [t for t in fuzzy_text.split(", ") if t]
    inci_ratio = sum(t.lower() in vocab_lower for t in out_tokens) / len(out_tokens) if out_tokens else 0.0
    return {
        "cer": core.compute_cer(fuzzy_text, reference),
        "p": p, "r": r, "f1": f1,
        "inci_ratio": inci_ratio,
        "n_chars": len(core.normalize_text(raw_text)),
    }


def build_table():
    paddle = load_rows(PADDLE_CSVS)
    google = load_rows([GOOGLE_CSV])
    missing = set(paddle) - set(google)
    if missing:
        raise SystemExit(f"{len(missing)} images have PaddleOCR results but no Google Vision result "
                         f"(e.g. {sorted(missing)[:3]}); run ocr_google_vision.py on all datasets first")

    dev_products = {e["image_file"] for e in core.iter_dataset_entries(datasets=["original"], limit=DEV_LIMIT)}
    vocabulary = core.load_inci_vocabulary()
    vocab_lower = {v.lower() for v in vocabulary}

    table = []
    for key in sorted(paddle):
        dataset, image_file = key
        ref = paddle[key]["reference_text"]
        ps = score(paddle[key]["ocr_text_raw"], ref, vocabulary, vocab_lower)
        gs = score(google[key]["ocr_text_raw"], ref, vocabulary, vocab_lower)
        product = product_key(dataset, image_file)
        table.append({
            "dataset": dataset,
            "image_file": image_file,
            "product": product,
            "split": "dev" if product in dev_products else "heldout",
            "paddle_inci_ratio": ps["inci_ratio"],
            "paddle_n_chars": ps["n_chars"],
            "google_inci_ratio": gs["inci_ratio"],
            **{f"paddle_{m}": ps[m] for m in ("cer", "p", "r", "f1")},
            **{f"google_{m}": gs[m] for m in ("cer", "p", "r", "f1")},
        })
    return table


def apply_policy(rows, route_mask):
    """route_mask[i] True -> image i uses Google's result."""
    out = defaultdict(list)
    for row, use_google in zip(rows, route_mask):
        src = "google" if use_google else "paddle"
        for m in ("cer", "p", "r", "f1"):
            out[m].append(row[f"{src}_{m}"])
    return {m: float(np.mean(v)) for m, v in out.items()} | {
        "route_frac": float(np.mean(route_mask)) if len(route_mask) else 0.0,
        "per_image_f1": np.array(out["f1"]),
    }


def route_mask(rows, signal, threshold):
    return np.array([row[f"paddle_{signal}"] < threshold for row in rows])


def candidate_thresholds(rows, signal):
    # Every distinct observed value plus +inf (= route everything) -- the
    # routing rule "signal < t" only changes at observed values.
    values = sorted({row[f"paddle_{signal}"] for row in rows})
    return values + [float("inf")]


def cluster_bootstrap_diff(rows, f1_a, f1_b, n=N_BOOTSTRAP, seed=SEED):
    """95% CI of mean(f1_a - f1_b), resampling whole products (all scenarios
    of a product together), since the 5 scenario copies are not independent."""
    rng = np.random.default_rng(seed)
    by_product = defaultdict(list)
    for i, row in enumerate(rows):
        by_product[row["product"]].append(i)
    groups = list(by_product.values())
    diff = f1_a - f1_b
    stats = []
    for _ in range(n):
        picks = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in picks])
        stats.append(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    table = build_table()
    core.write_csv(OUT_PER_IMAGE, table, fieldnames=list(table[0].keys()))
    dev = [r for r in table if r["split"] == "dev"]
    held = [r for r in table if r["split"] == "heldout"]
    print(f"dev n={len(dev)} ({len(dev) // 5} products x 5 scenarios)   held-out n={len(held)}")

    # Held-out curve (descriptive only -- not used for any selection).
    curve = []
    for signal in SIGNALS:
        for t in candidate_thresholds(dev, signal):
            res = apply_policy(held, route_mask(held, signal, t))
            curve.append({"signal": signal, "threshold": t, "route_frac": round(res["route_frac"], 4),
                          "set_f1": round(res["f1"], 4), "cer": round(res["cer"], 4)})
    core.write_csv(OUT_CURVE, curve, fieldnames=list(curve[0].keys()))

    paddle_only = apply_policy(held, np.zeros(len(held), dtype=bool))
    google_only = apply_policy(held, np.ones(len(held), dtype=bool))
    # Run both, keep the output whose own INCI-match ratio is higher (label-free, 100% API cost).
    select_mask = np.array([r["google_inci_ratio"] > r["paddle_inci_ratio"] for r in held])
    select_both = apply_policy(held, select_mask)
    # Per-image oracle: uses the ground truth to pick the better engine. NOT a
    # deployable system -- only the ceiling any routing rule could reach.
    oracle = apply_policy(held, np.array([r["google_f1"] > r["paddle_f1"] for r in held]))

    summary = []

    def add(name, res, chosen_on_dev=""):
        lo_hi = ""
        if name != "Paddle only":
            d, lo, hi = cluster_bootstrap_diff(held, res["per_image_f1"], paddle_only["per_image_f1"])
            lo_hi = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        summary.append({"policy": name, "chosen_on_dev": chosen_on_dev,
                        "route_frac_heldout": round(res["route_frac"], 4),
                        "precision": round(res["p"], 4), "recall": round(res["r"], 4),
                        "set_f1": round(res["f1"], 4), "cer": round(res["cer"], 4),
                        "delta_f1_vs_paddle_95ci": lo_hi})

    add("Paddle only", paddle_only)
    add("Google only", google_only)
    for budget in BUDGETS:
        best = None
        for signal in SIGNALS:
            for t in candidate_thresholds(dev, signal):
                dres = apply_policy(dev, route_mask(dev, signal, t))
                if dres["route_frac"] <= budget and (best is None or dres["f1"] > best[0]):
                    best = (dres["f1"], signal, t)
        _, signal, t = best
        add(f"Hybrid fallback (budget <= {int(budget * 100)}%)",
            apply_policy(held, route_mask(held, signal, t)), f"{signal} < {t:g}")
    add("Run both, pick higher inci_ratio", select_both)
    add("Oracle (uses labels, ceiling only)", oracle)

    core.write_csv(OUT_SUMMARY, summary, fieldnames=list(summary[0].keys()))

    print(f"\n=== HELD-OUT results (n={len(held)}) ===")
    print(f"{'policy':<38} {'rule (chosen on dev)':<24} {'->Google':>8} {'P':>7} {'R':>7} {'F1':>7} {'CER':>7}  dF1 vs Paddle [95% CI]")
    for s in summary:
        print(f"{s['policy']:<38} {s['chosen_on_dev']:<24} {s['route_frac_heldout']:>8.1%} "
              f"{s['precision']:>7.4f} {s['recall']:>7.4f} {s['set_f1']:>7.4f} {s['cer']:>7.4f}  "
              f"{s['delta_f1_vs_paddle_95ci']}")

    print("\n=== Held-out Set F1 by scenario ===")
    print(f"{'scenario':<10} {'Paddle':>8} {'Google':>8}")
    for ds in core.DATASETS:
        rs = [r for r in held if r["dataset"] == ds]
        if rs:
            print(f"{ds:<10} {np.mean([r['paddle_f1'] for r in rs]):>8.4f} {np.mean([r['google_f1'] for r in rs]):>8.4f}")

    print(f"\nWrote {OUT_SUMMARY}, {OUT_CURVE}, {OUT_PER_IMAGE}")


if __name__ == "__main__":
    main()
