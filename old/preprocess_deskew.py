"""OCR performance with deskew (rotation correction) as the only preprocessing step.

Based on the "Projection Profile Variance Maximization" technique described in the
SkinSafe BOT pipeline doc: try rotation angles in [-5, +5] degrees, binarize, sum each
row (horizontal projection), and keep the angle whose projection has the highest
variance (text lines aligned horizontally produce sharp peaks/troughs -> high variance).

Usage:
  python preprocess_deskew.py
  python preprocess_deskew.py --datasets blur --limit 10
"""
import cv2
import numpy as np

import ocr_core as core

TECHNIQUE = "deskew"
ANGLE_RANGE_DEG = 5.0
ANGLE_STEP_DEG = 0.5


def _rotate(image, angle, border_value):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def _best_angle(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle, best_variance = 0.0, -1.0
    angle = -ANGLE_RANGE_DEG
    while angle <= ANGLE_RANGE_DEG + 1e-9:
        rotated = _rotate(binary, angle, border_value=0)
        row_sums = rotated.sum(axis=1).astype(np.float64)
        variance = row_sums.var()
        if variance > best_variance:
            best_variance, best_angle = variance, angle
        angle += ANGLE_STEP_DEG
    return best_angle


def preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    angle = _best_angle(gray)
    if angle == 0.0:
        return image_bgr
    return _rotate(image_bgr, angle, border_value=(255, 255, 255))


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
