"""OCR performance with grayscale + Otsu adaptive thresholding (binarization).

Usage:
  python preprocess_threshold.py
  python preprocess_threshold.py --datasets blur --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "threshold"


def preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # PaddleOCR expects a 3-channel image, so convert back without adding real color info.
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
