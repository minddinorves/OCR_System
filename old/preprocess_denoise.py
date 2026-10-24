"""OCR performance with fast non-local-means denoising before recognition.

Usage:
  python preprocess_denoise.py
  python preprocess_denoise.py --datasets night low_light --limit 10
"""
import cv2

import ocr_core as core

TECHNIQUE = "denoise"


def preprocess(image_bgr):
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_preprocessing_experiment(TECHNIQUE, preprocess, args)


if __name__ == "__main__":
    main()
