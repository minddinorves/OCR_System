
"""OCR performance with brightness-adaptive contrast enhancement.

Mirrors the SkinSafe BOT pipeline doc step 2: pick CLAHE for bright images and plain
Histogram Equalization for dark images, based on mean grayscale brightness.

Usage:
  python preprocess_contrast_adaptive.py
  python preprocess_contrast_adaptive.py --datasets night low_light --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "contrast_adaptive"
BRIGHTNESS_THRESHOLD = 127


def preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= BRIGHTNESS_THRESHOLD:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    else:
        enhanced = cv2.equalizeHist(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
