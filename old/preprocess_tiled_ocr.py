"""OCR performance with tiled (horizontal-strip) OCR.

Mirrors the SkinSafe BOT pipeline doc step 4: split the image into overlapping
horizontal strips, run OCR on each strip separately, then merge the per-strip
detections back into one result, dropping duplicates from the overlap regions
using Intersection-over-Union (IoU) on the recognized text boxes.

Strip height is tuned to this dataset's image sizes (median height ~163px), not the
800px figure implied by the source doc, so tiling actually exercises on most images
instead of only the handful that are unusually tall.

Usage:
  python preprocess_tiled_ocr.py
  python preprocess_tiled_ocr.py --datasets original --limit 10
"""
import time

import ocr_core as core

TECHNIQUE = "tiled_ocr"
STRIP_HEIGHT = 150
OVERLAP = 40
IOU_DEDUPE_THRESHOLD = 0.5


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe(boxes):
    """boxes: list of [x1, y1, x2, y2, text, score]. Keep highest-score box per overlap cluster."""
    ordered = sorted(boxes, key=lambda b: -b[5])
    kept = []
    for box in ordered:
        if all(_iou(box, k) < IOU_DEDUPE_THRESHOLD for k in kept):
            kept.append(box)
    return kept


def run_one(entry, image_bgr, args):
    h = image_bgr.shape[0]
    if h <= STRIP_HEIGHT:
        lines, elapsed, error = core.run_ocr_lines(image_bgr, lang=args.lang)
        return " ".join(lines), elapsed, error, None

    stride = STRIP_HEIGHT - OVERLAP
    boxes_all = []
    total_time = 0.0
    ocr = core.get_ocr(args.lang)
    y = 0
    while True:
        y2 = min(y + STRIP_HEIGHT, h)
        strip = image_bgr[y:y2, :]
        t0 = time.time()
        result = ocr.predict(strip)
        total_time += time.time() - t0
        for page in result:
            texts = page.get("rec_texts", [])
            boxes = page.get("rec_boxes", [])
            scores = page.get("rec_scores", [])
            for text, box, score in zip(texts, boxes, scores):
                x1, by1, x2, by2 = box
                boxes_all.append([float(x1), float(by1) + y, float(x2), float(by2) + y, text, float(score)])
        if y2 >= h:
            break
        y += stride

    deduped = _dedupe(boxes_all)
    deduped.sort(key=lambda b: (b[1], b[0]))
    ocr_text = " ".join(b[4] for b in deduped)
    return ocr_text, total_time, None, None


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()
    core.run_custom_experiment(TECHNIQUE, run_one, args)


if __name__ == "__main__":
    main()
