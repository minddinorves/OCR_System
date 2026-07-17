"""Best-performing preprocessing technique (deskew) combined with fuzzy matching.

Of the 10 preprocessing techniques benchmarked (see results/summary.csv), deskew had
the lowest mean CER averaged across all 5 scenarios (0.6086), edging out doing no
preprocessing at all (0.6191) and plain fuzzy-only (0.6177). This script chains the
two best individual ideas: deskew the image first, run PaddleOCR, then fuzzy-match
the output against the INCI ingredient database (ingredient_master_dataset_fixed.csv),
to see whether the improvements stack.

NOTE: that ranking was measured with PaddleOCR's use_doc_unwarping/use_textline_orientation
forced OFF (see ocr_core.py history). With those now ON by default (ocr_core.get_ocr),
deskew actively HURTS instead of helping -- it conflicts with the model-based textline
orientation correction. This script is kept for reference/comparison only; don't treat
its "best" claim as current.

Usage:
  python preprocess_best_plus_fuzzy.py
  python preprocess_best_plus_fuzzy.py --datasets night --limit 10 --threshold 90
"""
import os

import cv2

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from preprocess_deskew import preprocess as deskew

TECHNIQUE = "best_plus_fuzzy"

FIELDNAMES = [
    "dataset",
    "image_file",
    "label_file",
    "status",
    "reference_text",
    "ocr_text_raw",
    "ocr_text_fuzzy",
    "ref_chars",
    "ref_words",
    "cer_raw",
    "wer_raw",
    "cer_fuzzy",
    "wer_fuzzy",
    "ocr_time_sec",
    "error",
]


def build_row(entry, raw_text, fuzzy_text, elapsed, error=None):
    row = {field: "" for field in FIELDNAMES}
    row.update(
        dataset=entry["dataset"],
        image_file=entry["image_file"],
        label_file=entry.get("label_file") or "",
        status=entry["status"],
        reference_text=entry.get("reference_text", ""),
        ocr_text_raw=raw_text,
        ocr_text_fuzzy=fuzzy_text,
        ocr_time_sec=round(elapsed, 4),
        error=error or "",
    )
    if entry["status"] == "ok" and error is None:
        ref = entry["reference_text"]
        row["ref_chars"] = len(core.normalize_text(ref))
        row["ref_words"] = len(core.normalize_text(ref).split())
        row["cer_raw"] = round(core.compute_cer(raw_text, ref), 4)
        row["wer_raw"] = round(core.compute_wer(raw_text, ref), 4)
        row["cer_fuzzy"] = round(core.compute_cer(fuzzy_text, ref), 4)
        row["wer_fuzzy"] = round(core.compute_wer(fuzzy_text, ref), 4)
    elif error is not None:
        row["status"] = "ocr_error"
    return row


def main():
    parser = core.build_arg_parser(__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Minimum fuzzy similarity score (0-100) required to accept a vocabulary match",
    )
    args = parser.parse_args()

    vocabulary = core.load_inci_vocabulary()
    print(f"Loaded {len(vocabulary)} unique INCI names from {core.INCI_DB_PATH}")

    entries = list(core.iter_dataset_entries(datasets=args.datasets, limit=args.limit))
    rows = []
    from tqdm import tqdm

    for entry in tqdm(entries, desc=f"preprocess_{TECHNIQUE}"):
        if entry["status"] != "ok":
            rows.append(build_row(entry, "", "", 0.0))
            continue

        image = cv2.imread(entry["image_path"])
        if image is None:
            rows.append(build_row(entry, "", "", 0.0, error="failed to read image"))
            continue

        try:
            processed = deskew(image)
        except Exception as exc:  # noqa: BLE001
            rows.append(build_row(entry, "", "", 0.0, error=f"preprocess error: {exc}"))
            continue

        if getattr(args, "save_images", True):
            save_dir = os.path.join(core.PREPROCESSED_IMAGES_DIR, TECHNIQUE, entry["dataset"])
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, entry["image_file"]), processed)

        lines, elapsed, error = core.run_ocr_lines(processed, lang=args.lang)
        if error:
            rows.append(build_row(entry, "", "", elapsed, error))
            continue

        raw_text = " ".join(lines)
        fuzzy_text = correct_with_vocabulary(lines, vocabulary, args.threshold)
        rows.append(build_row(entry, raw_text, fuzzy_text, elapsed))

    out_path = os.path.join(core.RESULTS_DIR, f"preprocess_{TECHNIQUE}.csv")
    core.write_csv(out_path, rows, fieldnames=FIELDNAMES)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
