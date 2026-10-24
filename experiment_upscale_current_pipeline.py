"""Quick test: does upscaling small source images help recall on TOP of the
current best pipeline (doc unwarping/textline orientation on, border padding
25%, box_thresh=0.30, fuzzy matching)?

The earlier upscale experiments (preprocess_upscale.py / preprocess_upscale800.py,
2026-07-16) predate the engine-config fixes (mkldnn/doc-unwarping/box_thresh,
2026-07-19), so their results aren't a valid comparison against the current
pipeline. This reruns the same upscale idea (cubic interpolation to 480px
target height, applied BEFORE border padding) with the current best config,
on the same 40-image dev subset used throughout parameter tuning, so it's
directly comparable to the existing Tuning-40 baseline (Set F1=0.5920,
CER=0.2616 -- see ABSTRACT_FINAL.md 4.2.4).

Usage:
  python experiment_upscale_current_pipeline.py
"""
import csv

import cv2

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from old.compute_ingredient_metrics import tokenize, ier, set_prf1

TARGET_HEIGHT = 480
MAX_SCALE = 6.0


def upscale(image_bgr):
    height = image_bgr.shape[0]
    if height >= TARGET_HEIGHT:
        return image_bgr
    scale = min(TARGET_HEIGHT / height, MAX_SCALE)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    vocabulary = core.load_inci_vocabulary()
    entries = list(core.iter_dataset_entries(datasets=["original"], limit=40))

    rows = []
    for entry in entries:
        if entry["status"] != "ok":
            continue
        image = cv2.imread(entry["image_path"])
        if image is None:
            continue
        orig_h, orig_w = image.shape[:2]
        image = upscale(image)
        lines, elapsed, error = core.run_ocr_lines(image)
        if error:
            rows.append({"image_file": entry["image_file"], "error": error})
            continue

        raw_text = " ".join(lines)
        fuzzy_text = correct_with_vocabulary(lines, vocabulary, 90.0)
        ref = entry["reference_text"]

        hyp_tokens = tokenize(fuzzy_text)
        ref_tokens = tokenize(ref)
        i = ier(hyp_tokens, ref_tokens)
        p, r, f1 = set_prf1(hyp_tokens, ref_tokens)

        rows.append({
            "image_file": entry["image_file"],
            "orig_size": f"{orig_w}x{orig_h}",
            "upscaled": orig_h < TARGET_HEIGHT,
            "cer_fuzzy": round(core.compute_cer(fuzzy_text, ref), 4),
            "ier": round(i, 4) if i is not None else "",
            "set_precision": round(p, 4),
            "set_recall": round(r, 4),
            "set_f1": round(f1, 4),
        })

    out_path = "results/experiment_upscale_current_pipeline.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if "error" not in r]
    n = len(ok_rows)
    mean_cer = sum(r["cer_fuzzy"] for r in ok_rows) / n
    mean_p = sum(r["set_precision"] for r in ok_rows) / n
    mean_r = sum(r["set_recall"] for r in ok_rows) / n
    mean_f1 = sum(r["set_f1"] for r in ok_rows) / n
    n_upscaled = sum(1 for r in ok_rows if r["upscaled"])

    print(f"n={n} ({n_upscaled} images were actually upscaled, i.e. shorter than {TARGET_HEIGHT}px)")
    print(f"mean CER={mean_cer:.4f}  Precision={mean_p:.4f}  Recall={mean_r:.4f}  Set F1={mean_f1:.4f}")
    print()
    print("Baseline (current pipeline, no upscale, same 40 images): CER=0.2616  Set F1=0.5920")


if __name__ == "__main__":
    main()
