"""Rerun table 4.1's 3 configs that need fresh OCR (unclip_ratio / limit_side_len
variants -- no raw text was ever saved for these), on the FINAL implementation
(missing-comma bug fix applied), same 40-image dev set used throughout
parameter tuning.

Configs 1 (baseline box_thresh=0.45) and 2 (box_thresh=0.30) are NOT rerun
here -- they're scored directly from already-saved raw OCR text in a
companion check, since OCR itself is unaffected by the correction-step fix.

Every config here keeps the rest of the pipeline identical to production:
lang=en, use_doc_orientation_classify=False, use_doc_unwarping=True,
use_textline_orientation=True, device=cpu, enable_mkldnn=False, 25% border
padding -- only the text-detection parameter under test changes, matching
how table 4.1 was originally described.

Usage:
  python rerun_table41_configs.py
"""
import csv

import cv2

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import tokenize, set_prf1

FUZZY_THRESHOLD = 90.0

CONFIGS = [
    {"label": "box_thresh=0.30 + unclip_ratio=1.8", "text_det_box_thresh": 0.30, "text_det_unclip_ratio": 1.8},
    {"label": "unclip_ratio=1.8 (alone)", "text_det_box_thresh": 0.45, "text_det_unclip_ratio": 1.8},
    {"label": "limit_side_len=1536", "text_det_box_thresh": 0.45, "text_det_limit_side_len": 1536},
]


def build_ocr(cfg):
    from paddleocr import PaddleOCR

    kwargs = dict(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device="cpu",
        enable_mkldnn=False,
    )
    if "text_det_box_thresh" in cfg:
        kwargs["text_det_box_thresh"] = cfg["text_det_box_thresh"]
    if "text_det_unclip_ratio" in cfg:
        kwargs["text_det_unclip_ratio"] = cfg["text_det_unclip_ratio"]
    if "text_det_limit_side_len" in cfg:
        kwargs["text_det_limit_side_len"] = cfg["text_det_limit_side_len"]
    return PaddleOCR(**kwargs)


def run_one(ocr, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "cv2.imread failed"
    image = core._add_border(image)
    result = ocr.predict(image)
    lines = []
    for page in result:
        lines.extend(page.get("rec_texts", []))
    return lines, None


def main():
    vocabulary = core.load_inci_vocabulary()
    entries = [e for e in core.iter_dataset_entries(datasets=["original"], limit=40) if e["status"] == "ok"]
    print(f"Running {len(CONFIGS)} configs x {len(entries)} images = {len(CONFIGS) * len(entries)} OCR calls...")

    all_rows = []
    summaries = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']} ===", flush=True)
        ocr = build_ocr(cfg)

        rows = []
        for entry in entries:
            lines, error = run_one(ocr, entry["image_path"])
            if error:
                rows.append({"config": cfg["label"], "image_file": entry["image_file"], "error": error,
                              "set_precision": "", "set_recall": "", "set_f1": ""})
                continue
            raw_text = " ".join(lines)
            fuzzy_text = correct_with_vocabulary([raw_text], vocabulary, FUZZY_THRESHOLD)
            ref = entry["reference_text"]
            p, r, f1 = set_prf1(tokenize(fuzzy_text), tokenize(ref))
            rows.append({
                "config": cfg["label"], "image_file": entry["image_file"], "error": "",
                "set_precision": round(p, 4), "set_recall": round(r, 4), "set_f1": round(f1, 4),
            })

        all_rows.extend(rows)
        ok_rows = [r for r in rows if not r["error"]]
        n = len(ok_rows)
        mean_p = sum(r["set_precision"] for r in ok_rows) / n
        mean_r = sum(r["set_recall"] for r in ok_rows) / n
        mean_f1 = sum(r["set_f1"] for r in ok_rows) / n
        summaries.append({"config": cfg["label"], "n": n, "mean_precision": round(mean_p, 4),
                           "mean_recall": round(mean_r, 4), "mean_set_f1": round(mean_f1, 4)})
        print(f"n={n}  Precision={mean_p:.4f}  Recall={mean_r:.4f}  Set F1={mean_f1:.4f}", flush=True)

    out_path = "results/table4_1_rerun_final_implementation.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 78)
    print(f"{'config':<38} {'n':>4} {'Precision':>10} {'Recall':>8} {'Set F1':>8}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['config']:<38} {s['n']:>4} {s['mean_precision']:>10.4f} {s['mean_recall']:>8.4f} {s['mean_set_f1']:>8.4f}")


if __name__ == "__main__":
    main()
