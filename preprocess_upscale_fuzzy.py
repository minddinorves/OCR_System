"""Upscale-before-OCR combined with fuzzy-vocabulary correction.

preprocess_upscale.py tests upscaling but skips the fuzzy-correction step, so
its F1 numbers aren't comparable to the current best pipeline (ocr_core.py
padding + ocr_paddle_fuzzy.py correction, validated at Set F1 0.5515 on
"original"). This script adds the same upscale-if-small preprocessing on top
of that full pipeline, to measure upscale's incremental value fairly.

Usage:
  python preprocess_upscale_fuzzy.py
  python preprocess_upscale_fuzzy.py --datasets original --limit 20 --threshold 90
"""
import cv2
from tqdm import tqdm

import ocr_core as core
from ocr_paddle_fuzzy import FIELDNAMES, build_row, correct_with_vocabulary

TARGET_HEIGHT = 480
MAX_SCALE = 6.0


def upscale_if_small(image_bgr):
    height = image_bgr.shape[0]
    if height >= TARGET_HEIGHT:
        return image_bgr
    scale = min(TARGET_HEIGHT / height, MAX_SCALE)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    parser = core.build_arg_parser(__doc__)
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()

    vocabulary = core.load_inci_vocabulary()
    print(f"Loaded {len(vocabulary)} unique INCI names from {core.INCI_DB_PATH} + patch")

    entries = list(core.iter_dataset_entries(datasets=args.datasets, limit=args.limit))
    rows = []
    for entry in tqdm(entries, desc="preprocess_upscale_fuzzy"):
        if entry["status"] != "ok":
            rows.append(build_row(entry, "", "", 0.0))
            continue

        image = cv2.imread(entry["image_path"])
        if image is None:
            rows.append(build_row(entry, "", "", 0.0, error="failed to read image"))
            continue

        upscaled = upscale_if_small(image)
        lines, elapsed, error = core.run_ocr_lines(upscaled, lang=args.lang)
        if error:
            rows.append(build_row(entry, "", "", elapsed, error))
            continue

        raw_text = " ".join(lines)
        fuzzy_text = correct_with_vocabulary(lines, vocabulary, args.threshold)
        rows.append(build_row(entry, raw_text, fuzzy_text, elapsed))

    out_path = core.RESULTS_DIR + "/preprocess_upscale_fuzzy.csv"
    core.write_csv(out_path, rows, fieldnames=FIELDNAMES)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
