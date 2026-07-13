"""Exhaustive search over all 2-technique combinations of the 8 chainable
preprocessing techniques, run on the FULL dataset (all 5 scenarios, ~685
images) for every pair.

This exists because the greedy forward-selection search in
preprocess_combination_search.py can miss a winning pair where neither
technique looks good alone: greedy only ever extends the best path found so
far, so a technique that loses to deskew as a first step is discarded and
never gets tried as a *second* step alongside something else. This script
checks that blind spot directly: all C(8,2) = 28 unordered pairs, each
applied in a fixed order (alphabetical technique name) and evaluated on the
full dataset, so the result is directly comparable to results/summary.csv
and the greedy search's chain-only numbers (no fuzzy matching applied here --
this measures the image-preprocessing step in isolation, matching how the
other single-technique CSVs were produced).

Usage:
  python preprocess_pairs_search.py
  python preprocess_pairs_search.py --limit 10   # smoke test
"""
import argparse
import itertools

import cv2
from tqdm import tqdm

import ocr_core as core
from preprocess_combination_search import CANDIDATES, apply_chain

RESULTS_PATH = core.RESULTS_DIR + "/pairs_search_log.csv"


def mean_cer_for_pair(entries, pair):
    total_cer, n = 0.0, 0
    for entry in entries:
        if entry["status"] != "ok":
            continue
        image = cv2.imread(entry["image_path"])
        if image is None:
            continue
        try:
            processed = apply_chain(image, pair)
        except Exception:  # noqa: BLE001
            continue
        lines, _, error = core.run_ocr_lines(processed, lang="en")
        if error:
            continue
        text = " ".join(lines)
        cer = core.compute_cer(text, entry["reference_text"])
        if cer is None:
            continue
        total_cer += cer
        n += 1
    return total_cer / n if n else float("inf")


def _append_result(pair, cer):
    """Append one pair's result to the CSV immediately (create with header if new)."""
    import csv
    import os

    file_exists = os.path.isfile(RESULTS_PATH)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pair", "mean_cer"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"pair": " + ".join(pair), "mean_cer": round(cer, 4)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Images per dataset scenario (default: full dataset)")
    parser.add_argument("--only", nargs=2, metavar=("TECH_A", "TECH_B"), default=None,
                         help="Evaluate a single pair only, print+append the result, and exit "
                              "(used by run_pairs_search.py to run each pair in its own process)")
    args = parser.parse_args()

    entries = list(core.iter_dataset_entries(limit=args.limit))

    if args.only:
        pair = tuple(sorted(args.only))
        cer = mean_cer_for_pair(entries, pair)
        print(f"{' + '.join(pair)} mean CER = {cer:.4f}", flush=True)
        _append_result(pair, cer)
        return

    names = sorted(CANDIDATES.keys())
    pairs = list(itertools.combinations(names, 2))
    print(f"Testing {len(pairs)} pairs on {len(entries)} images each "
          f"(~{len(pairs)} full-dataset OCR passes)")

    for pair in tqdm(pairs, desc="pairs_search"):
        cer = mean_cer_for_pair(entries, pair)
        print(f"  {' + '.join(pair):<35} mean CER = {cer:.4f}", flush=True)
        _append_result(pair, cer)

    print(f"\nWrote results to {RESULTS_PATH}")
    print("\nFor reference: Deskew alone = 0.6084, Deskew+Fuzzy = 0.6076 "
          "(from the earlier single-technique and greedy-search runs).")


if __name__ == "__main__":
    main()
