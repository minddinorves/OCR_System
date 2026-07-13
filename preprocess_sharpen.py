"""OCR performance with an unsharp-mask sharpening filter before recognition.

Usage:
  python preprocess_sharpen.py
  python preprocess_sharpen.py --datasets blur --limit 10
"""
import cv2
import numpy as np

import ocr_core as core

TECHNIQUE = "sharpen"


def preprocess(image_bgr):
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image_bgr, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
