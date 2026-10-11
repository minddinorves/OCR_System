"""Token-level fusion of Google Vision + PaddleOCR, both after the same fuzzy
INCI correction (correct_with_vocabulary, threshold 90).

hybrid_paddle_google.py only ever *chooses* one engine's output per image.
This script instead *merges* the two ingredient lists, to test whether
PaddleOCR can add ingredients that Google Vision missed:

  google_only          Google + fuzzy (reference point)
  paddle_only          PaddleOCR + fuzzy (current pipeline)
  google_then_paddle   Google + fuzzy; PaddleOCR + fuzzy only when Google returned no text
  union                Google tokens + every PaddleOCR token not already in Google's list
  google_plus_paddle_inci
                       Google tokens + only those PaddleOCR tokens that are exact INCI
                       vocabulary names (i.e. the fuzzy step recognized them) and not
                       already in Google's list -- a precision guard on the union

"Already in the list" uses fuzz.ratio >= 85 on lowercased tokens, the same
rule the Set P/R/F1 metric uses to count a match. None of the variants has a
tunable parameter, so no dev-set selection is needed; results are still
reported on the same held-out split as hybrid_paddle_google.py (and on all
1,080 images for reference).

CER is reported for completeness but is not meaningful for the merged
variants: appended PaddleOCR tokens are out of label order by construction.
Set P/R/F1 (order-independent) is the metric to compare them on.

Usage:
  python fusion_google_paddle.py
"""
import numpy as np
from rapidfuzz import fuzz

import ocr_core as core
from old.hybrid_paddle_google import (
    FUZZY_THRESHOLD, GOOGLE_CSV, PADDLE_CSVS, DEV_LIMIT,
    cluster_bootstrap_diff, load_rows, product_key,
)
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import set_prf1, tokenize

DUP_THRESHOLD = 85
VARIANTS = ["paddle_only", "google_only", "google_then_paddle", "union", "google_plus_paddle_inci"]
OUT_PER_IMAGE = "results/fusion_google_paddle_per_image.csv"
OUT_SUMMARY = "results/fusion_google_paddle_summary.csv"


def fuzzy_tokens(raw_text, vocabulary):
    text = correct_with_vocabulary([raw_text], vocabulary, FUZZY_THRESHOLD)
    return [t for t in text.split(", ") if t]


def _already_in(token, tokens):
    low = token.lower()
    return any(fuzz.ratio(low, t.lower()) >= DUP_THRESHOLD for t in tokens)


def fuse(google_tokens, paddle_tokens, vocab_lower):
    extra = [t for t in paddle_tokens if not _already_in(t, google_tokens)]
    return {
        "paddle_only": paddle_tokens,
        "google_only": google_tokens,
        "google_then_paddle": google_tokens if google_tokens else paddle_tokens,
        "union": google_tokens + extra,
        "google_plus_paddle_inci": google_tokens + [t for t in extra if t.lower() in vocab_lower],
    }


def main():
    paddle = load_rows(PADDLE_CSVS)
    google = load_rows([GOOGLE_CSV])
    missing = set(paddle) - set(google)
    if missing:
        raise SystemExit(f"{len(missing)} images lack a Google Vision result; run ocr_google_vision.py first")

    dev_products = {e["image_file"] for e in core.iter_dataset_entries(datasets=["original"], limit=DEV_LIMIT)}
    vocabulary = core.load_inci_vocabulary()
    vocab_lower = {v.lower() for v in vocabulary}

    table = []
    for key in sorted(paddle):
        dataset, image_file = key
        ref = paddle[key]["reference_text"]
        ref_tokens = tokenize(ref)
        merged = fuse(fuzzy_tokens(google[key]["ocr_text_raw"], vocabulary),
                      fuzzy_tokens(paddle[key]["ocr_text_raw"], vocabulary), vocab_lower)
        product = product_key(dataset, image_file)
        row = {"dataset": dataset, "image_file": image_file, "product": product,
               "split": "dev" if product in dev_products else "heldout"}
        for name, tokens in merged.items():
            text = ", ".join(tokens)
            p, r, f1 = set_prf1(tokenize(text), ref_tokens)
            row.update({f"{name}_p": p, f"{name}_r": r, f"{name}_f1": f1,
                        f"{name}_cer": core.compute_cer(text, ref), f"{name}_n": len(tokens)})
        table.append(row)
    core.write_csv(OUT_PER_IMAGE, table, fieldnames=list(table[0].keys()))

    summary = []
    for split_name, rows in (("heldout", [r for r in table if r["split"] == "heldout"]), ("all", table)):
        f1 = {v: np.array([r[f"{v}_f1"] for r in rows]) for v in VARIANTS}
        print(f"\n=== {split_name} (n={len(rows)}) ===")
        print(f"{'variant':<26} {'P':>7} {'R':>7} {'F1':>7} {'CER':>7} {'#tok':>6}  "
              f"{'dF1 vs Paddle [95% CI]':<28} dF1 vs Google [95% CI]")
        for v in VARIANTS:
            entry = {"split": split_name, "variant": v, "n": len(rows)}
            for m in ("p", "r", "f1", "cer", "n"):
                entry[m] = round(float(np.mean([r[f"{v}_{m}"] for r in rows])), 4)
            for base in ("paddle_only", "google_only"):
                if v == base:
                    entry[f"delta_f1_vs_{base}"] = ""
                    continue
                d, lo, hi = cluster_bootstrap_diff(rows, f1[v], f1[base])
                entry[f"delta_f1_vs_{base}"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            summary.append(entry)
            print(f"{v:<26} {entry['p']:>7.4f} {entry['r']:>7.4f} {entry['f1']:>7.4f} {entry['cer']:>7.4f} "
                  f"{entry['n']:>6.1f}  {entry['delta_f1_vs_paddle_only']:<28} {entry['delta_f1_vs_google_only']}")

        if split_name == "heldout":
            print("\nSet F1 by scenario (held-out):")
            print(f"{'scenario':<10} " + " ".join(f"{v[:22]:>23}" for v in VARIANTS))
            for ds in core.DATASETS:
                rs = [r for r in rows if r["dataset"] == ds]
                print(f"{ds:<10} " + " ".join(f"{np.mean([r[f'{v}_f1'] for r in rs]):>23.4f}" for v in VARIANTS))

    core.write_csv(OUT_SUMMARY, summary, fieldnames=list(summary[0].keys()))
    print(f"\nWrote {OUT_SUMMARY}, {OUT_PER_IMAGE}")


if __name__ == "__main__":
    main()
