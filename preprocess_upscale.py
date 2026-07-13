"""OCR performance when small images are upscaled before recognition.

Many photos in this dataset are very low resolution (some under 100px tall),
which is likely a major source of OCR error. This script upscales any image
shorter than TARGET_HEIGHT (cubic interpolation) and leaves larger images untouched.

Usage:
  python preprocess_upscale.py
  python preprocess_upscale.py --datasets original --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "upscale"
TARGET_HEIGHT = 480
MAX_SCALE = 6.0


def preprocess(image_bgr):
    height = image_bgr.shape[0]
    if height >= TARGET_HEIGHT:
        return image_bgr
    scale = min(TARGET_HEIGHT / height, MAX_SCALE)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
