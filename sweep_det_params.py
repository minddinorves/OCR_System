"""Sweep PaddleOCR text-detection inference params (no fine-tuning, no new
data) to see if recall improves. Current defaults, read from the bundled
PP-OCRv6_medium_det/inference.yml: box_thresh=0.45, unclip_ratio=1.4,
thresh=0.2. This only overrides them at inference time via the paddleocr
3.7.0 PaddleOCR(...) kwargs -- no training involved.

Runs the full current best pipeline (doc unwarping + textline orientation +
25% border padding from ocr_core, then fuzzy-vocabulary correction from
ocr_paddle_fuzzy) for each param combo, on the "original" dataset only.

Usage:
  python sweep_det_params.py --limit 40
  python sweep_det_params.py --limit 40 --configs baseline low_box_thresh
"""
import argparse
import time

import cv2
from paddleocr import PaddleOCR
from tqdm import tqdm

import ocr_core as core
from ocr_paddle_fuzzy import FIELDNAMES, build_row, correct_with_vocabulary

CONFIGS = {
    "baseline": {},
    "low_box_thresh": {"text_det_box_thresh": 0.3},
    "high_unclip": {"text_det_unclip_ratio": 1.8},
    "low_box_high_unclip": {"text_det_box_thresh": 0.3, "text_det_unclip_ratio": 1.8},
    "bigger_limit": {"text_det_limit_side_len": 1536},
    "box_thresh_025": {"text_det_box_thresh": 0.25},
    "box_thresh_02": {"text_det_box_thresh": 0.2},
    "box_thresh_015": {"text_det_box_thresh": 0.15},
}


def build_ocr(overrides):
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device="cpu",
        enable_mkldnn=False,
        **overrides,
    )


def run_one_config(config_name, overrides, entries, vocabulary, threshold):
    ocr = build_ocr(overrides)
    rows = []
    for entry in tqdm(entries, desc=config_name):
        if entry["status"] != "ok":
            rows.append(build_row(entry, "", "", 0.0))
            continue

        image = cv2.imread(entry["image_path"])
        if image is None:
            rows.append(build_row(entry, "", "", 0.0, error="failed to read image"))
            continue

        image = core._add_border(image)
        t0 = time.time()
        try:
            result = ocr.predict(image)
            lines = []
            for page in result:
                lines.extend(page.get("rec_texts", []))
            elapsed = time.time() - t0
            error = None
        except Exception as exc:  # noqa: BLE001
            lines, elapsed, error = [], time.time() - t0, str(exc)

        if error:
            rows.append(build_row(entry, "", "", elapsed, error))
            continue

        raw_text = " ".join(lines)
        fuzzy_text = correct_with_vocabulary(lines, vocabulary, threshold)
        rows.append(build_row(entry, raw_text, fuzzy_text, elapsed))

    out_path = f"{core.RESULTS_DIR}/sweep_det_{config_name}.csv"
    core.write_csv(out_path, rows, fieldnames=FIELDNAMES)
    print(f"[{config_name}] wrote {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    args = parser.parse_args()

    vocabulary = core.load_inci_vocabulary()
    print(f"Loaded {len(vocabulary)} unique INCI names from {core.INCI_DB_PATH} + patch")
    entries = list(core.iter_dataset_entries(datasets=["original"], limit=args.limit))

    for config_name in args.configs:
        run_one_config(config_name, CONFIGS[config_name], entries, vocabulary, args.threshold)


if __name__ == "__main__":
    main()
