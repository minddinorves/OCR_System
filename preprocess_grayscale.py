"""OCR performance with grayscale conversion as the only preprocessing step.

Usage:
  python preprocess_grayscale.py
  python preprocess_grayscale.py --datasets original --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "grayscale"


def preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # PaddleOCR expects a 3-channel image, so convert back without adding real color info.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
