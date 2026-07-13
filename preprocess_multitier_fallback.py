"""OCR performance with a 4-tier fallback pipeline (escalates only when needed).

Mirrors the SkinSafe BOT pipeline doc step 5 (Four-Tier Fallback Pipeline):
  Tier 1: brightness-adaptive contrast (CLAHE if bright, else Histogram Equalization)
          + bicubic upscale to >=800px shortest side.
  Tier 2: if Tier 1 read too little text, swap to the *other* contrast method and retry.
  Tier 3: if still too little, replace the bicubic upscale with EDSR super-resolution
          (cv2.dnn_superres, EDSR_x4 model) before OCR.
  Tier 4: last resort - adaptive Gaussian threshold, auto-inverting colors first if the
          image looks like light text on a dark background. Used unconditionally as
          the final answer if Tiers 1-3 all read too little.

"Read too little" (not specified in the source doc) is defined here as: fewer than
MIN_CHARS recognized characters, or fewer than MIN_BOXES recognized text boxes.

Usage:
  python preprocess_multitier_fallback.py
  python preprocess_multitier_fallback.py --datasets night --limit 10
"""
import os

import cv2

import ocr_core as core

TECHNIQUE = "multitier_fallback"
MIN_CHARS = 20
MIN_BOXES = 3
TARGET_SHORT_SIDE = 800
MAX_BICUBIC_SCALE = 12.0
EDSR_MAX_INPUT_SHORT_SIDE = 250  # skip EDSR (too slow/memory-heavy) above this size
EDSR_MODEL_PATH = os.path.join(core.PROJECT_ROOT, "models", "EDSR_x4.pb")
EDSR_SCALE = 4

_edsr_model = None


def _get_edsr():
    global _edsr_model
    if _edsr_model is None:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(EDSR_MODEL_PATH)
        sr.setModel("edsr", EDSR_SCALE)
        _edsr_model = sr
    return _edsr_model


def _is_sufficient(lines):
    return len(lines) >= MIN_BOXES and sum(len(t) for t in lines) >= MIN_CHARS


def _contrast(gray, method):
    if method == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.equalizeHist(gray)


def _pick_contrast_method(gray):
    return "clahe" if gray.mean() >= 127 else "histeq"


def _bicubic_to_target(image_bgr):
    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if short_side >= TARGET_SHORT_SIDE:
        return image_bgr
    scale = min(TARGET_SHORT_SIDE / short_side, MAX_BICUBIC_SCALE)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _tier_contrast_then_upscale(image_bgr, method):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.cvtColor(_contrast(gray, method), cv2.COLOR_GRAY2BGR)
    return _bicubic_to_target(enhanced)


def _tier_contrast_then_edsr(image_bgr, method):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.cvtColor(_contrast(gray, method), cv2.COLOR_GRAY2BGR)
    h, w = enhanced.shape[:2]
    if min(h, w) > EDSR_MAX_INPUT_SHORT_SIDE:
        return _bicubic_to_target(enhanced)
    upscaled = _get_edsr().upsample(enhanced)
    return _bicubic_to_target(upscaled)


def _tier_adaptive_threshold(image_bgr):
    upscaled = _bicubic_to_target(image_bgr)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 100:
        gray = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=15,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def run_one(entry, image_bgr, args):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    primary_method = _pick_contrast_method(gray)
    secondary_method = "histeq" if primary_method == "clahe" else "clahe"
    total_time = 0.0

    for method in (primary_method, secondary_method):
        processed = _tier_contrast_then_upscale(image_bgr, method)
        lines, elapsed, error = core.run_ocr_lines(processed, lang=args.lang)
        total_time += elapsed
        if error is None and _is_sufficient(lines):
            return " ".join(lines), total_time, None, processed

    processed = _tier_contrast_then_edsr(image_bgr, primary_method)
    lines, elapsed, error = core.run_ocr_lines(processed, lang=args.lang)
    total_time += elapsed
    if error is None and _is_sufficient(lines):
        return " ".join(lines), total_time, None, processed

    processed = _tier_adaptive_threshold(image_bgr)
    lines, elapsed, error = core.run_ocr_lines(processed, lang=args.lang)
    total_time += elapsed
    return " ".join(lines), total_time, error, processed


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_custom_experiment(TECHNIQUE, run_one, args)


if __name__ == "__main__":
    main()
