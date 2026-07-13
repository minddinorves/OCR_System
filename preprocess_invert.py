"""OCR performance with adaptive dark-background inversion.

Motivated by manual inspection of near-total OCR failures (CER > 0.9 on baseline):
several labels use light/white text on a dark colored background (e.g. red, green,
purple product packaging) -- PaddleOCR's recognition model, trained mostly on dark
text on a light background, fails almost completely on these ("x40.jpg", "x59.jpg",
"x69.jpg" all scored CER = 1.0 on baseline).

A simple global-brightness or saturation threshold turned out to be unreliable: it
misclassifies ordinary photos with dark surroundings (e.g. "A9.jpg", mean brightness
80.9, high JPEG-artifact saturation) as "inverted-scheme" labels even though they are
normal dark-text-on-light-background labels. Instead this technique runs OCR on both
the original and the color-inverted image and keeps whichever reads more text (same
"sufficient text" heuristic used by the multi-tier fallback technique), so the
decision is made by the OCR model itself rather than a brittle pixel-level rule.

Usage:
  python preprocess_invert.py
  python preprocess_invert.py --datasets night low_light --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "invert"
MIN_CHARS = 20
MIN_BOXES = 3


def _is_sufficient(lines):
    return len(lines) >= MIN_BOXES and sum(len(t) for t in lines) >= MIN_CHARS


def run_one(entry, image_bgr, args):
    lines, elapsed, error = core.run_ocr_lines(image_bgr, lang=args.lang)
    total_time = elapsed
    if error is None and _is_sufficient(lines):
        return " ".join(lines), total_time, None, image_bgr

    inverted = cv2.bitwise_not(image_bgr)
    inv_lines, inv_elapsed, inv_error = core.run_ocr_lines(inverted, lang=args.lang)
    total_time += inv_elapsed

    inv_chars = sum(len(t) for t in inv_lines) if inv_error is None else -1
    orig_chars = sum(len(t) for t in lines) if error is None else -1

    if inv_chars > orig_chars:
        return " ".join(inv_lines), total_time, inv_error, inverted
    return " ".join(lines), total_time, error, image_bgr


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_custom_experiment(TECHNIQUE, run_one, args)


if __name__ == "__main__":
    main()
