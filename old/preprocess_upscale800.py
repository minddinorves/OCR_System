"""OCR performance with bicubic upscaling so the shortest side is >= 800px.

Mirrors the SkinSafe BOT pipeline doc step 3 (Bicubic Upscaling), which targets the
shortest side rather than the height specifically. Images already >= 800px on their
shortest side are left untouched.

Usage:
  python preprocess_upscale800.py
  python preprocess_upscale800.py --datasets original --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "upscale800"
TARGET_SHORT_SIDE = 800
MAX_SCALE = 12.0


def preprocess(image_bgr):
    h, w = image_bgr.shape[:2]
    short_side = min(h, w)
    if short_side >= TARGET_SHORT_SIDE:
        return image_bgr
    scale = min(TARGET_SHORT_SIDE / short_side, MAX_SCALE)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
