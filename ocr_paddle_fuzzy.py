"""PaddleOCR + fuzzy-matching against a known INCI ingredient database.

Raw PaddleOCR output is noisy (broken words, hyphenation, stray characters).
This script splits each detected text line into comma-separated tokens, then
snaps every token to its closest match in ingredient_master_dataset_fixed.csv
(the INCI ingredient database) when the fuzzy similarity is high enough,
under the assumption that the true ingredient name is almost always one of
the entries in that "known" database. Both the raw and the corrected output
are scored against the ground-truth label so the improvement is visible.

Usage:
  python ocr_paddle_fuzzy.py
  python ocr_paddle_fuzzy.py --datasets original blur --limit 10 --threshold 85
"""
import os
import re

from rapidfuzz import fuzz, process
from tqdm import tqdm

import ocr_core as core

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

_HEADER_RE = re.compile(r"^[^:：]*ingredient[^:：]*[:：]\s*", re.IGNORECASE)
# Split on commas that separate ingredients, but not commas inside numbers
# like "1,2-Hexanediol" (comma followed by a digit stays attached).
_SPLIT_RE = re.compile(r",(?!\s*\d)")
MIN_TOKEN_LEN = 3
MAX_LEN_RATIO = 1.6


def _strip_header(line):
    return _HEADER_RE.sub("", line)


def _is_reliable_match(token, match_name):
    # Guard against fuzz.ratio/WRatio matching a short or partial token to an
    # unrelated, much longer (or shorter) vocabulary entry.
    longer, shorter = sorted((len(token), len(match_name)), reverse=True)
    return shorter > 0 and longer / shorter <= MAX_LEN_RATIO


def correct_with_vocabulary(lines, vocabulary, threshold):
    tokens = []
    for line in lines:
        line = _strip_header(line)
        tokens.extend(t.strip(" .;:*-") for t in _SPLIT_RE.split(line))

    corrected = []
    for token in tokens:
        if not token:
            continue
        if len(token) < MIN_TOKEN_LEN:
            corrected.append(token)
            continue
        match = process.extractOne(token, vocabulary, scorer=fuzz.ratio, score_cutoff=threshold)
        if match and _is_reliable_match(token, match[0]):
            corrected.append(match[0])
        else:
            corrected.append(token)
    return ", ".join(corrected)


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
        default=85.0,
        help="Minimum fuzzy similarity score (0-100) required to accept a vocabulary match",
    )
    args = parser.parse_args()

    vocabulary = core.load_inci_vocabulary()
    print(f"Loaded {len(vocabulary)} unique INCI names from {core.INCI_DB_PATH}")

    entries = list(core.iter_dataset_entries(datasets=args.datasets, limit=args.limit))
    rows = []
    for entry in tqdm(entries, desc="ocr_paddle_fuzzy"):
        if entry["status"] != "ok":
            rows.append(build_row(entry, "", "", 0.0))
            continue

        lines, elapsed, error = core.run_ocr_lines(entry["image_path"], lang=args.lang)
        if error:
            rows.append(build_row(entry, "", "", elapsed, error))
            continue

        raw_text = " ".join(lines)
        fuzzy_text = correct_with_vocabulary(lines, vocabulary, args.threshold)
        rows.append(build_row(entry, raw_text, fuzzy_text, elapsed))

    out_path = os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy.csv")
    core.write_csv(out_path, rows, fieldnames=FIELDNAMES)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
