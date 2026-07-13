"""GOT-OCR2.0 evaluation: same eval harness as ocr_paddle_only.py, different OCR engine.

GOT-OCR2.0 is an end-to-end (no separate detector) OCR model, chosen to test whether a
model built for dense/small text handles this dataset's very low-resolution label crops
better than PaddleOCR's detect+recognize pipeline.

Usage:
  python ocr_got.py
  python ocr_got.py --datasets original blur --limit 10
"""
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

import ocr_core as core

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _MODEL = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32
        ).to(_DEVICE).eval()
    return _MODEL, _PROCESSOR


def run_got_ocr(image_path):
    """Run GOT-OCR2.0 on an image file path.

    Returns (recognized_text, elapsed_seconds, error_message_or_None).
    """
    import time

    model, processor = get_model()
    t0 = time.time()
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=1024,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
            )
        text = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text.strip(), time.time() - t0, None
    except Exception as exc:  # noqa: BLE001 - record any OCR failure per-image
        return "", time.time() - t0, str(exc)


def main():
    parser = core.build_arg_parser(__doc__)
    args = parser.parse_args()

    entries = list(core.iter_dataset_entries(datasets=args.datasets, limit=args.limit))
    rows = []
    for entry in tqdm(entries, desc="ocr_got"):
        if entry["status"] != "ok":
            rows.append(core.evaluate_entry(entry, "", 0.0))
            continue
        ocr_text, elapsed, error = run_got_ocr(entry["image_path"])
        rows.append(core.evaluate_entry(entry, ocr_text, elapsed, error))

    out_path = os.path.join(core.RESULTS_DIR, "ocr_got.csv")
    core.write_csv(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
