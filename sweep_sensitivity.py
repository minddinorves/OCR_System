"""Sensitivity sweeps for the three pipeline parameters that were chosen with
little or no recorded evidence: border padding fraction, fuzzy-matching
score cutoff, and the fuzzy-match length-ratio guard (MAX_LEN_RATIO).

Everything else is held at the current pipeline defaults (box_thresh=0.30,
doc unwarping + textline orientation on, see ocr_core.get_ocr). Each sweep
varies ONE parameter and is compared against the current default
(pad=0.25, threshold=90, ratio=1.6) on the same images with a paired
bootstrap, so a reported difference is not just sample noise.

Two phases:
  1. OCR (slow, ~16 s/image on CPU): for each padding fraction, OCR every
     image once and cache the raw recognized lines to
     results/sweep_cache/pad_<frac>.json. Resumable -- images already in the
     cache are skipped, so the run can be interrupted and restarted.
  2. Analysis (fast, no OCR): re-run fuzzy correction on the cached lines for
     every config, score with Set P/R/F1 + CER, and write
       results/sweep_sensitivity_per_image.csv
       results/sweep_sensitivity_summary.csv

The threshold and ratio sweeps only need the pad=0.25 cache, which is OCR'd
first, so they are available after ~1 hour even if the full padding sweep
(~6 hours for 6 pad values x 216 images) is still running.

Usage:
  python sweep_sensitivity.py                      # everything, all 216 images
  python sweep_sensitivity.py --pads 0.25          # only what fuzzy sweeps need
  python sweep_sensitivity.py --limit 40           # quick check
  python sweep_sensitivity.py --skip-ocr           # re-analyse existing cache
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
from tqdm import tqdm

import ocr_core as core
import ocr_paddle_fuzzy as fuzzy

# Metric + bootstrap helpers live in old/ (moved there with the thesis
# drafts); reuse them so these numbers are computed exactly like the
# reported ones.
sys.path.insert(0, os.path.join(core.PROJECT_ROOT, "old"))
from compute_bootstrap_ci import ci_mean, ci_paired_delta  # noqa: E402
from compute_ingredient_metrics import set_prf1, tokenize  # noqa: E402

CACHE_DIR = os.path.join(core.RESULTS_DIR, "sweep_cache")

DEFAULT_PAD = 0.25
DEFAULT_THRESHOLD = 90.0
DEFAULT_RATIO = 1.6

PADS = [0.0, 0.10, 0.15, 0.25, 0.35, 0.50]
THRESHOLDS = [None, 80.0, 85.0, 88.0, 90.0, 92.0, 95.0]  # None = no fuzzy correction
RATIOS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, float("inf")]  # inf = guard disabled

METRICS = ("set_f1", "set_precision", "set_recall", "cer")


def _cache_path(pad):
    return os.path.join(CACHE_DIR, f"pad_{pad:.2f}.json")


def _load_cache(pad):
    path = _cache_path(pad)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_cache(pad, cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(pad)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    os.replace(tmp, path)


def ocr_with_pad(entries, pad):
    """Phase 1: OCR every entry at the given padding fraction, resuming from cache."""
    cache = _load_cache(pad)
    todo = [e for e in entries if e["image_file"] not in cache]
    if not todo:
        print(f"[pad={pad:.2f}] all {len(entries)} images already cached")
        return
    ocr = core.get_ocr()
    for entry in tqdm(todo, desc=f"OCR pad={pad:.2f}"):
        image = cv2.imread(entry["image_path"])
        t0 = time.time()
        if image is None:
            cache[entry["image_file"]] = {"lines": [], "elapsed": 0.0, "error": "failed to read image"}
        else:
            try:
                lines = []
                for page in ocr.predict(core._add_border(image, frac=pad)):
                    lines.extend(page.get("rec_texts", []))
                cache[entry["image_file"]] = {"lines": lines, "elapsed": time.time() - t0, "error": None}
            except Exception as exc:  # noqa: BLE001 - record any OCR failure per-image
                cache[entry["image_file"]] = {"lines": [], "elapsed": time.time() - t0, "error": str(exc)}
        _save_cache(pad, cache)


def correct(lines, vocabulary, threshold, ratio):
    if threshold is None:
        return " ".join(lines)
    # _is_reliable_match reads the module global at call time.
    saved = fuzzy.MAX_LEN_RATIO
    fuzzy.MAX_LEN_RATIO = ratio
    try:
        return fuzzy.correct_with_vocabulary(lines, vocabulary, threshold)
    finally:
        fuzzy.MAX_LEN_RATIO = saved


def score(hyp, ref):
    p, r, f1 = set_prf1(tokenize(hyp), tokenize(ref))
    return {"set_precision": p, "set_recall": r, "set_f1": f1, "cer": core.compute_cer(hyp, ref)}


def build_configs(pads_available):
    """(sweep, label, pad, threshold, ratio) for every config to evaluate."""
    configs = []
    for pad in PADS:
        if pad in pads_available:
            configs.append(("padding", f"pad={pad:.2f}", pad, DEFAULT_THRESHOLD, DEFAULT_RATIO))
    if DEFAULT_PAD in pads_available:
        for thr in THRESHOLDS:
            label = "no_fuzzy" if thr is None else f"threshold={thr:g}"
            configs.append(("threshold", label, DEFAULT_PAD, thr, DEFAULT_RATIO))
        for ratio in RATIOS:
            label = "ratio=off" if ratio == float("inf") else f"ratio={ratio:g}"
            configs.append(("length_ratio", label, DEFAULT_PAD, DEFAULT_THRESHOLD, ratio))
    return configs


def analyse(entries, pads_available):
    """Phase 2: score every config from cache, with bootstrap CIs vs the default."""
    vocabulary = core.load_inci_vocabulary()
    caches = {pad: _load_cache(pad) for pad in pads_available}
    configs = build_configs(pads_available)

    per_image = []
    scores = {}  # label -> {image_file: metrics}
    for sweep, label, pad, thr, ratio in tqdm(configs, desc="analyse"):
        scores[label] = {}
        for entry in entries:
            cached = caches[pad].get(entry["image_file"])
            if cached is None or cached["error"]:
                continue
            m = score(correct(cached["lines"], vocabulary, thr, ratio), entry["reference_text"])
            scores[label][entry["image_file"]] = m
            per_image.append({"sweep": sweep, "config": label, "image_file": entry["image_file"],
                              **{k: round(v, 4) for k, v in m.items()}})

    default_label = f"pad={DEFAULT_PAD:.2f}"
    default = scores.get(default_label, {})
    summary = []
    for sweep, label, *_ in configs:
        rows = scores[label]
        common = sorted(set(rows) & set(default))
        out = {"sweep": sweep, "config": label, "n": len(rows),
               "is_default": label in (default_label, f"threshold={DEFAULT_THRESHOLD:g}",
                                       f"ratio={DEFAULT_RATIO:g}")}
        for metric in METRICS:
            mean, lo, hi = ci_mean([rows[f][metric] for f in rows])
            out[f"{metric}_mean"] = round(mean, 4)
            out[f"{metric}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
            if common and not out["is_default"]:
                delta, dlo, dhi = ci_paired_delta([default[f][metric] for f in common],
                                                  [rows[f][metric] for f in common])
                out[f"{metric}_delta_vs_default"] = round(delta, 4)
                out[f"{metric}_delta_ci"] = f"[{dlo:+.4f}, {dhi:+.4f}]"
                out[f"{metric}_significant"] = dlo > 0 or dhi < 0
        summary.append(out)

    core.write_csv(os.path.join(core.RESULTS_DIR, "sweep_sensitivity_per_image.csv"), per_image,
                   fieldnames=["sweep", "config", "image_file", *METRICS])
    fieldnames = ["sweep", "config", "n", "is_default"]
    for metric in METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_ci", f"{metric}_delta_vs_default",
                       f"{metric}_delta_ci", f"{metric}_significant"]
    core.write_csv(os.path.join(core.RESULTS_DIR, "sweep_sensitivity_summary.csv"), summary,
                   fieldnames=fieldnames)

    print(f"\n{'sweep':<13} {'config':<15} {'n':>4} {'Set F1':>7} {'dF1 vs default [95% CI]':>28} {'CER':>7}")
    print("-" * 80)
    for s in summary:
        delta = "(default)" if s["is_default"] else (
            f"{s.get('set_f1_delta_vs_default', 0):+.4f} {s.get('set_f1_delta_ci', '')}"
            + (" *" if s.get("set_f1_significant") else ""))
        print(f"{s['sweep']:<13} {s['config']:<15} {s['n']:>4} {s['set_f1_mean']:>7.4f} "
              f"{delta:>28} {s['cer_mean']:>7.4f}")
    print("\n* = paired-bootstrap 95% CI of the difference excludes 0")


def sanity_check(entries):
    """pad=0.25 cache should reproduce the reported FINAL validation run exactly."""
    final_path = os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_original_full_validation.csv")
    cache = _load_cache(DEFAULT_PAD)
    if not cache or not os.path.isfile(final_path):
        return
    with open(final_path, newline="", encoding="utf-8") as fh:
        final = {r["image_file"]: r["ocr_text_raw"] for r in csv.DictReader(fh)}
    common = [e["image_file"] for e in entries if e["image_file"] in cache and e["image_file"] in final]
    same = sum(" ".join(cache[f]["lines"]) == final[f] for f in common)
    print(f"Sanity check: pad=0.25 raw OCR identical to FINAL validation on {same}/{len(common)} images")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Max images (default: all 216)")
    parser.add_argument("--pads", type=float, nargs="+", default=PADS,
                        help="Padding fractions to OCR (default: %(default)s)")
    parser.add_argument("--skip-ocr", action="store_true", help="Only analyse what is already cached")
    args = parser.parse_args()

    entries = [e for e in core.iter_dataset_entries(datasets=["original"], limit=args.limit)
               if e["status"] == "ok"]
    # Default pad first so the threshold/ratio sweeps are ready early.
    pads = sorted(set(args.pads), key=lambda p: (p != DEFAULT_PAD, p))

    if not args.skip_ocr:
        for pad in pads:
            ocr_with_pad(entries, pad)

    sanity_check(entries)
    available = [p for p in PADS if _load_cache(p)]
    analyse(entries, available)


if __name__ == "__main__":
    main()
