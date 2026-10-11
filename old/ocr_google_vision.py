"""Google Cloud Vision OCR + the same fuzzy-matching post-processing as
ocr_paddle_fuzzy.py, so the two engines are compared under identical
post-processing and evaluation (only the OCR engine differs).

Every API response is cached as JSON under results/google_vision_cache/, and
an image that already has a cached response is never sent again -- re-running
this script (or hybrid_paddle_google.py, which reads the output CSV) costs
nothing after the first full run. 1 image = 1 billable unit; the first 1,000
units/month are free, so the full 1,080-image run is ~1,080 units.

Authentication: set an API key (Google Cloud Console -> APIs & Services ->
Credentials, with Cloud Vision API enabled on a project that has billing):
  PowerShell:  $env:GOOGLE_VISION_API_KEY = "AIza..."
  bash:        export GOOGLE_VISION_API_KEY=AIza...

Unlike PaddleOCR, the image is sent as-is (no white border): the border is a
fix for PaddleOCR's own text detector (see ocr_core._add_border), not a
property of the data.

Usage:
  python ocr_google_vision.py --datasets original --limit 3   # smoke test (3 units)
  python ocr_google_vision.py                                 # all 5 scenarios
"""
import base64
import json
import os
import time

import requests
from tqdm import tqdm

import ocr_core as core
from ocr_paddle_fuzzy import FIELDNAMES, build_row, correct_with_vocabulary

API_URL = "https://vision.googleapis.com/v1/images:annotate"
CACHE_DIR = os.path.join(core.RESULTS_DIR, "google_vision_cache")
MAX_RETRIES = 3


def _cache_path(feature, entry):
    return os.path.join(CACHE_DIR, feature, entry["dataset"], entry["image_file"] + ".json")


def _call_api(image_path, feature, api_key, language_hints):
    with open(image_path, "rb") as fh:
        content = base64.b64encode(fh.read()).decode("ascii")
    body = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": feature}],
            "imageContext": {"languageHints": language_hints},
        }]
    }
    for attempt in range(MAX_RETRIES):
        resp = requests.post(API_URL, params={"key": api_key}, json=body, timeout=60)
        # 429/5xx are transient; anything else (bad key, API disabled, no
        # billing) will not fix itself, so fail immediately.
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()["responses"][0]
    resp.raise_for_status()
    return resp.json()["responses"][0]


def get_response(entry, feature, api_key, language_hints):
    """Return (response_dict, elapsed_seconds, from_cache)."""
    path = _cache_path(feature, entry)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        return cached["response"], cached["elapsed_sec"], True

    t0 = time.time()
    response = _call_api(entry["image_path"], feature, api_key, language_hints)
    elapsed = time.time() - t0
    # Only successful responses are cached, so a per-image API error is
    # retried on the next run instead of being frozen into the cache.
    if "error" not in response:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"response": response, "elapsed_sec": elapsed}, fh, ensure_ascii=False)
    return response, elapsed, False


def response_lines(response):
    text = (response.get("fullTextAnnotation") or {}).get("text")
    if text is None:
        # TEXT_DETECTION may omit fullTextAnnotation; the first
        # textAnnotation is then the whole detected text block.
        annotations = response.get("textAnnotations") or []
        text = annotations[0]["description"] if annotations else ""
    return [line for line in text.split("\n") if line.strip()]


def main():
    parser = core.build_arg_parser(__doc__)
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="Fuzzy score_cutoff, same default as ocr_paddle_fuzzy.py")
    parser.add_argument("--feature", default="DOCUMENT_TEXT_DETECTION",
                        choices=["DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"])
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_VISION_API_KEY")
    vocabulary = core.load_inci_vocabulary()
    entries = list(core.iter_dataset_entries(datasets=args.datasets, limit=args.limit))

    n_uncached = sum(
        1 for e in entries
        if e["status"] == "ok" and not os.path.isfile(_cache_path(args.feature, e))
    )
    print(f"{len(entries)} images, {n_uncached} not yet cached (= billable API calls this run)")
    if n_uncached and not api_key:
        raise SystemExit("GOOGLE_VISION_API_KEY is not set (see the usage notes at the top of this file)")

    rows = []
    for entry in tqdm(entries, desc="ocr_google_vision"):
        if entry["status"] != "ok":
            rows.append(build_row(entry, "", "", 0.0))
            continue
        try:
            response, elapsed, _ = get_response(entry, args.feature, api_key, [args.lang])
        except requests.RequestException as exc:
            rows.append(build_row(entry, "", "", 0.0, error=str(exc)))
            continue
        if "error" in response:
            rows.append(build_row(entry, "", "", elapsed, error=response["error"].get("message", "api error")))
            continue

        lines = response_lines(response)
        # Joined into one line exactly like the saved PaddleOCR runs
        # (ocr_text_raw = " ".join(lines)), so both engines go through
        # correct_with_vocabulary in the same form.
        raw_text = " ".join(lines)
        fuzzy_text = correct_with_vocabulary([raw_text], vocabulary, args.threshold)
        rows.append(build_row(entry, raw_text, fuzzy_text, elapsed))

    out_path = args.out or os.path.join(core.RESULTS_DIR, "ocr_google_vision.csv")
    core.write_csv(out_path, rows, fieldnames=FIELDNAMES)
    n_err = sum(1 for r in rows if r["error"])
    print(f"Wrote {len(rows)} rows to {out_path} ({n_err} errors)")


if __name__ == "__main__":
    main()
