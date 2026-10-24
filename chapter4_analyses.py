"""Recompute the Dev-split parameter checks and the missed-ingredient analysis
for thesis Chapter 4 under the final evaluation protocol: fixed fuzzy code
(ocr_paddle_fuzzy.py), threshold 85, and a vocabulary patch that never
contains the scored image's own labels (5-fold cross-fitting inside Dev;
the 40-Dev-label patch for Held-out). No OCR re-run: uses saved OCR text.

  1. box_thresh / detection settings on Dev   <- results/sweep_det_*.csv
  2. border padding on Dev                     <- results/sweep_cache/pad_*.json
  3. length-ratio guard on Dev                 <- results/sweep_cache/pad_0.25.json
  4. missed ingredients on Held-out: is there any trace of each missed name in
     the raw OCR text? (proxy for detector miss vs recognition/matching miss;
     no bounding-box ground truth exists)     <- results/final_v2/heldout_per_image.csv

Outputs: results/chapter4/*.csv, printed tables.

Usage:
  python chapter4_analyses.py
"""
import csv
import json
import os

from rapidfuzz import fuzz

import ocr_core as core
import ocr_paddle_fuzzy as fz
from build_vocabulary_patch import build_patch

from compute_bootstrap_ci import ci_mean, ci_paired_delta
from compute_ingredient_metrics import MATCH_THRESHOLD, set_prf1, tokenize

THRESHOLD = 85.0
N_DEV, N_FOLDS = 40, 5
OUT_DIR = os.path.join(core.RESULTS_DIR, "chapter4")
DET_CONFIGS = [("baseline", "box_thresh 0.45 (model default)"), ("low_box_thresh", "box_thresh 0.30"),
               ("box_thresh_025", "box_thresh 0.25"), ("box_thresh_02", "box_thresh 0.20"),
               ("box_thresh_015", "box_thresh 0.15"), ("high_unclip", "unclip_ratio 1.8"),
               ("low_box_high_unclip", "box_thresh 0.30 + unclip_ratio 1.8"),
               ("bigger_limit", "limit_side_len 1536")]
PADS = [0.0, 0.10, 0.15, 0.25, 0.35, 0.50]
RATIOS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, float("inf")]
TRACE_THRESHOLDS = [40, 50, 60, 70]


def dev_vocabs():
    """image_file -> vocabulary built from the OTHER Dev folds' labels."""
    dev = list(core.iter_dataset_entries(["original"], limit=N_DEV))
    main_db = core._load_names(core.INCI_DB_PATH)
    vocab_for = {}
    for k in range(N_FOLDS):
        fold = dev[k::N_FOLDS]
        others = [e["label_path"] for e in dev if e not in fold]
        vocab = list({n.lower(): n for n in main_db + build_patch(1, others)}.values())
        for e in fold:
            vocab_for[e["image_file"]] = vocab
    return dev, vocab_for


def score(hyp, ref):
    p, r, f1 = set_prf1(tokenize(hyp), tokenize(ref))
    return {"set_f1": f1, "set_precision": p, "set_recall": r, "cer": core.compute_cer(hyp, ref)}


def table(name, results, reference_label):
    """results: list of (label, {image_file: metrics}); writes + prints with paired delta vs reference."""
    ref = dict(results)[reference_label]
    files = sorted(ref)
    rows = []
    print(f"\n=== {name} (Dev, n={len(files)}) ===")
    for label, per_img in results:
        out = {"setting": label, "n_images": len(files)}
        for m in ("set_f1", "set_precision", "set_recall", "cer"):
            mean, lo, hi = ci_mean([per_img[f][m] for f in files])
            out[m] = round(mean, 4)
            out[f"{m}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
        if label == reference_label:
            delta = "(reference)"
        else:
            d, lo, hi = ci_paired_delta([ref[f]["set_f1"] for f in files], [per_img[f]["set_f1"] for f in files])
            out.update(set_f1_delta_vs_reference=round(d, 4), set_f1_delta_ci95=f"[{lo:+.4f}, {hi:+.4f}]",
                       set_f1_delta_significant=lo > 0 or hi < 0)
            delta = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" *" if lo > 0 or hi < 0 else "")
        rows.append(out)
        print(f"  {label:40s} F1={out['set_f1']:.4f} P={out['set_precision']:.4f} R={out['set_recall']:.4f} "
              f"CER={out['cer']:.4f}  dF1={delta}")
    fields = ["setting", "n_images", "set_f1", "set_f1_ci95", "set_precision", "set_precision_ci95", "set_recall",
              "set_recall_ci95", "cer", "cer_ci95", "set_f1_delta_vs_reference", "set_f1_delta_ci95",
              "set_f1_delta_significant"]
    core.write_csv(os.path.join(OUT_DIR, f"{name}.csv"), rows, fieldnames=fields)


def missed_ingredients():
    with open(os.path.join(core.RESULTS_DIR, "final_v2", "heldout_per_image.csv"), newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out, summary = [], []
    for row in rows:
        ref_t, hyp_t, raw_t = (tokenize(row[k]) for k in ("reference_text", "text_leak_free", "ocr_text_raw"))
        cands = sorted(((fuzz.ratio(h, r), hi, ri) for hi, h in enumerate(hyp_t) for ri, r in enumerate(ref_t)
                        if fuzz.ratio(h, r) >= MATCH_THRESHOLD), reverse=True)
        used_h, used_r = set(), set()
        for _, hi, ri in cands:
            if hi not in used_h and ri not in used_r:
                used_h.add(hi)
                used_r.add(ri)
        for ri, tok in enumerate(ref_t):
            if ri in used_r:
                continue
            trace = max((fuzz.partial_ratio(tok, t) for t in raw_t), default=0.0)
            out.append({"dataset": row["dataset"], "image_file": row["image_file"], "missed_name": tok,
                        "raw_ocr_empty": not raw_t, "best_partial_ratio_vs_raw_ocr": round(trace, 1)})
    core.write_csv(os.path.join(OUT_DIR, "missed_ingredients_heldout.csv"), out,
                   fieldnames=["dataset", "image_file", "missed_name", "raw_ocr_empty", "best_partial_ratio_vs_raw_ocr"])
    print("\n=== missed ingredients (Held-out, leak-free system): share with NO trace in raw OCR ===")
    for scenario in ("original", "indoor", "low_light", "blur", "night"):
        sel = [o for o in out if o["dataset"] == scenario]
        n_ref = sum(len(tokenize(r["reference_text"])) for r in rows if r["dataset"] == scenario)
        s = {"dataset": scenario, "reference_names": n_ref, "missed_names": len(sel),
             "missed_from_empty_ocr_images": sum(o["raw_ocr_empty"] for o in sel)}
        for t in TRACE_THRESHOLDS:
            s[f"no_trace_pct_at_{t}"] = round(100 * sum(o["best_partial_ratio_vs_raw_ocr"] < t for o in sel) / len(sel), 1)
        summary.append(s)
        print(f"  {scenario:10s} missed {len(sel):5d}/{n_ref:5d}  " +
              "  ".join(f"<{t}: {s[f'no_trace_pct_at_{t}']:5.1f}%" for t in TRACE_THRESHOLDS))
    core.write_csv(os.path.join(OUT_DIR, "missed_ingredients_heldout_summary.csv"), summary,
                   fieldnames=list(summary[0].keys()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev, vocab_for = dev_vocabs()
    dev_files = {e["image_file"] for e in dev}
    ref_of = {e["image_file"]: e["reference_text"] for e in dev}

    det = []
    for cfg, label in DET_CONFIGS:
        with open(os.path.join(core.RESULTS_DIR, f"sweep_det_{cfg}.csv"), newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["status"] == "ok" and r["image_file"] in dev_files]
        det.append((label, {r["image_file"]: score(fz.correct_with_vocabulary(
            [r["ocr_text_raw"]], vocab_for[r["image_file"]], THRESHOLD), r["reference_text"]) for r in rows}))
    table("dev_detection_settings", det, "box_thresh 0.30")

    pads = []
    for pad in PADS:
        with open(os.path.join(core.RESULTS_DIR, "sweep_cache", f"pad_{pad:.2f}.json"), encoding="utf-8") as fh:
            cache = json.load(fh)
        pads.append((f"padding {int(pad * 100)}%", {f: score(fz.correct_with_vocabulary(
            cache[f]["lines"], vocab_for[f], THRESHOLD), ref_of[f]) for f in dev_files}))
    table("dev_padding", pads, "padding 25%")

    with open(os.path.join(core.RESULTS_DIR, "sweep_cache", "pad_0.25.json"), encoding="utf-8") as fh:
        cache = json.load(fh)
    ratios, saved = [], fz.MAX_LEN_RATIO
    try:
        for ratio in RATIOS:
            fz.MAX_LEN_RATIO = ratio
            label = "length ratio off" if ratio == float("inf") else f"length ratio <= {ratio:g}"
            ratios.append((label, {f: score(fz.correct_with_vocabulary(cache[f]["lines"], vocab_for[f], THRESHOLD),
                                            ref_of[f]) for f in dev_files}))
    finally:
        fz.MAX_LEN_RATIO = saved
    table("dev_length_ratio", ratios, "length ratio <= 1.6")

    missed_ingredients()


if __name__ == "__main__":
    main()
