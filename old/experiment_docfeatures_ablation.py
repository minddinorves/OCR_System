"""Isolated ablation: does use_doc_unwarping / use_textline_orientation actually
help, on TOP of the current best pipeline (border padding 25%, box_thresh=0.30,
fuzzy matching threshold=90)?

Why this script exists: ocr_core.py's get_ocr() comment claims "Both ON:
empirically cuts mean CER from 0.748 to 0.654 (60-image config search, see
experiment_paddle_config.py)" -- but experiment_paddle_config.py does not exist
in this repo and was never committed (confirmed via `git log --all`), so that
number is not reproducible or verifiable anymore. This reruns the same idea
as a real, checked-in experiment: all 4 combinations of the two flags, on the
same 40-image dev subset used throughout parameter tuning (see ABSTRACT_FINAL.md
4.2.4), with every other setting held at the current pipeline's values, so the
result isolates the effect of just these two flags instead of being bundled
with the mkldnn/vocab-patch/box_thresh fixes like the headline 0.39->0.55
number in section 4.1 is.

Usage:
  python experiment_docfeatures_ablation.py
"""
import csv
import itertools

import cv2

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import tokenize, ier, set_prf1

N_IMAGES = 40
FUZZY_THRESHOLD = 90.0

CONFIGS = [
    {"use_doc_unwarping": False, "use_textline_orientation": False, "label": "OFF-OFF (original default)"},
    {"use_doc_unwarping": True, "use_textline_orientation": False, "label": "ON-OFF (unwarping only)"},
    {"use_doc_unwarping": False, "use_textline_orientation": True, "label": "OFF-ON (textline only)"},
    {"use_doc_unwarping": True, "use_textline_orientation": True, "label": "ON-ON (current pipeline)"},
]


def build_ocr(use_doc_unwarping, use_textline_orientation):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        device="cpu",
        enable_mkldnn=False,
        text_det_box_thresh=0.3,
    )


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
    entries = [e for e in core.iter_dataset_entries(datasets=["original"], limit=N_IMAGES) if e["status"] == "ok"]
    print(f"Running {len(CONFIGS)} configs x {len(entries)} images = {len(CONFIGS) * len(entries)} OCR calls...")

    all_rows = []
    summaries = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']} ===")
        ocr = build_ocr(cfg["use_doc_unwarping"], cfg["use_textline_orientation"])

        rows = []
        for entry in entries:
            lines, error = run_one(ocr, entry["image_path"])
            if error:
                rows.append({
                    "config": cfg["label"], "image_file": entry["image_file"],
                    "error": error, "cer_fuzzy": "", "ier": "",
                    "set_precision": "", "set_recall": "", "set_f1": "",
                })
                continue

            fuzzy_text = correct_with_vocabulary(lines, vocabulary, FUZZY_THRESHOLD)
            ref = entry["reference_text"]
            hyp_tokens = tokenize(fuzzy_text)
            ref_tokens = tokenize(ref)
            i = ier(hyp_tokens, ref_tokens)
            p, r, f1 = set_prf1(hyp_tokens, ref_tokens)

            rows.append({
                "config": cfg["label"],
                "image_file": entry["image_file"],
                "error": "",
                "cer_fuzzy": round(core.compute_cer(fuzzy_text, ref), 4),
                "ier": round(i, 4) if i is not None else "",
                "set_precision": round(p, 4),
                "set_recall": round(r, 4),
                "set_f1": round(f1, 4),
            })

        all_rows.extend(rows)
        ok_rows = [r for r in rows if not r["error"]]
        n = len(ok_rows)
        mean_cer = sum(r["cer_fuzzy"] for r in ok_rows) / n
        mean_p = sum(r["set_precision"] for r in ok_rows) / n
        mean_r = sum(r["set_recall"] for r in ok_rows) / n
        mean_f1 = sum(r["set_f1"] for r in ok_rows) / n
        summaries.append({
            "config": cfg["label"], "n": n,
            "mean_cer_fuzzy": round(mean_cer, 4),
            "mean_set_precision": round(mean_p, 4),
            "mean_set_recall": round(mean_r, 4),
            "mean_set_f1": round(mean_f1, 4),
        })
        print(f"n={n}  mean CER={mean_cer:.4f}  Precision={mean_p:.4f}  Recall={mean_r:.4f}  Set F1={mean_f1:.4f}")

    out_path = "results/experiment_docfeatures_ablation.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = "results/experiment_docfeatures_ablation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nWrote {out_path} and {summary_path}")
    print("\n" + "=" * 78)
    print(f"{'config':<28} {'n':>4} {'mean CER':>10} {'Precision':>10} {'Recall':>8} {'Set F1':>8}")
    print("-" * 78)
    for s in summaries:
        print(f"{s['config']:<28} {s['n']:>4} {s['mean_cer_fuzzy']:>10.4f} "
              f"{s['mean_set_precision']:>10.4f} {s['mean_set_recall']:>8.4f} {s['mean_set_f1']:>8.4f}")


if __name__ == "__main__":
    main()
