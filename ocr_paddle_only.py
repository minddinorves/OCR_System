"""Baseline OCR evaluation: raw PaddleOCR output vs. ground-truth labels, no preprocessing.

Usage:
  python ocr_paddle_only.py
  python ocr_paddle_only.py --datasets original blur --limit 10
"""
import os

from tqdm import tqdm

import ocr_core as core


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()

    entries = list(core.iter_dataset_entries(datasets=args.datasets, limit=args.limit))
    rows = []
    for entry in tqdm(entries, desc="ocr_paddle_only"):
        if entry["status"] != "ok":
            rows.append(core.evaluate_entry(entry, "", 0.0))
            continue
        ocr_text, elapsed, error = core.run_ocr(entry["image_path"], lang=args.lang)
        rows.append(core.evaluate_entry(entry, ocr_text, elapsed, error))

    out_path = os.path.join(core.RESULTS_DIR, "ocr_paddle_only.csv")
    core.write_csv(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
