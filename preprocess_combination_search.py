"""Greedy forward-selection search over combinations of preprocessing techniques
+ fuzzy matching, to find the best-performing *pipeline* rather than just the
best single technique.

Why greedy instead of exhaustive: there are 8 chainable image-preprocessing
techniques (grayscale, threshold, denoise, sharpen, upscale, deskew,
contrast_adaptive, upscale800). Testing every subset (2^8 = 256, x2 for
fuzzy on/off = 512 pipelines) on the full 685-image dataset would take on the
order of a day of GPU OCR calls. Instead this script does sequential forward
selection: start from no preprocessing, repeatedly try adding each
not-yet-selected technique to the current chain, keep whichever addition
improves mean CER the most, and stop once no candidate improves further.
Finally it tests appending fuzzy-matching post-processing on top of the
winning chain. (Note: this is a greedy heuristic, not a proof of global
optimality -- a technique combination that only helps once some other
technique is already applied could in principle be missed.)

Two phases:
  1. Search phase: evaluate each candidate step on a smaller stratified
     sample (--search-limit images per scenario, default 15) to keep runtime
     tractable (roughly 15-20 minutes on a single GPU for the default
     settings).
  2. Validation phase: re-run the winning pipeline on the FULL dataset (all
     5 scenarios, ~685 images) to get a trustworthy final number comparable
     to results/summary.csv.

Usage:
  python preprocess_combination_search.py
  python preprocess_combination_search.py --search-limit 10
  python preprocess_combination_search.py --skip-validation
"""
import argparse
import time

import cv2
from tqdm import tqdm

import ocr_core as core
from ocr_paddle_fuzzy import correct_with_vocabulary
from preprocess_grayscale import preprocess as grayscale
from preprocess_threshold import preprocess as threshold
from preprocess_denoise import preprocess as denoise
from preprocess_sharpen import preprocess as sharpen
from preprocess_upscale import preprocess as upscale
from preprocess_deskew import preprocess as deskew
from preprocess_contrast_adaptive import preprocess as contrast_adaptive
from preprocess_upscale800 import preprocess as upscale800

CANDIDATES = {
    "grayscale": grayscale,
    "threshold": threshold,
    "denoise": denoise,
    "sharpen": sharpen,
    "upscale": upscale,
    "deskew": deskew,
    "contrast_adaptive": contrast_adaptive,
    "upscale800": upscale800,
}

SEARCH_LOG_PATH = core.RESULTS_DIR + "/combination_search_log.csv"
FINAL_RESULTS_PATH = core.RESULTS_DIR + "/preprocess_best_combination.csv"


def apply_chain(image, chain):
    for name in chain:
        image = CANDIDATES[name](image)
    return image


def mean_cer_for_chain(entries, chain, use_fuzzy=False, vocabulary=None, threshold_score=85.0):
    """Run OCR (optionally + fuzzy matching) over `entries` with `chain` applied.

    Skips images that fail to load/process/OCR rather than counting them as
    zero error, so the mean is only over images that actually produced text.
    """
    total_cer, n = 0.0, 0
    for entry in entries:
        if entry["status"] != "ok":
            continue
        image = cv2.imread(entry["image_path"])
        if image is None:
            continue
        try:
            processed = apply_chain(image, chain)
        except Exception:  # noqa: BLE001 - a broken chain just scores as a skip
            continue
        lines, _, error = core.run_ocr_lines(processed, lang="en")
        if error:
            continue
        text = " ".join(lines)
        if use_fuzzy:
            text = correct_with_vocabulary(lines, vocabulary, threshold_score)
        cer = core.compute_cer(text, entry["reference_text"])
        if cer is None:
            continue
        total_cer += cer
        n += 1
    return total_cer / n if n else float("inf")


def _append_log_row(row):
    """Append one row to the search log CSV immediately, so results survive a crash
    or an interrupted run instead of only being written at the very end."""
    file_exists = os.path.isfile(SEARCH_LOG_PATH)
    os.makedirs(os.path.dirname(SEARCH_LOG_PATH), exist_ok=True)
    with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
        import csv
        writer = csv.DictWriter(fh, fieldnames=["step", "chain", "mean_cer"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def greedy_search(entries, log_rows):
    remaining = list(CANDIDATES.keys())
    chain = []
    best_cer = mean_cer_for_chain(entries, chain)
    row = {"step": 0, "chain": "baseline", "mean_cer": round(best_cer, 4)}
    log_rows.append(row)
    _append_log_row(row)
    print(f"[baseline] mean CER = {best_cer:.4f}  (n={len(entries)} search images)")

    step = 1
    while remaining:
        scored = []
        for name in remaining:
            trial_chain = chain + [name]
            cer = mean_cer_for_chain(entries, trial_chain)
            scored.append((cer, name))
            print(f"  try {' -> '.join(trial_chain):<45} mean CER = {cer:.4f}")
            row = {
                "step": step,
                "chain": " -> ".join(trial_chain),
                "mean_cer": round(cer, 4),
            }
            log_rows.append(row)
            _append_log_row(row)
        scored.sort(key=lambda t: t[0])
        best_trial_cer, best_trial_name = scored[0]
        if best_trial_cer < best_cer:
            chain.append(best_trial_name)
            remaining.remove(best_trial_name)
            best_cer = best_trial_cer
            print(f"[step {step}] ADD '{best_trial_name}' -> chain={chain}, mean CER={best_cer:.4f}")
            step += 1
        else:
            print(f"[stop] no remaining technique improves on {best_cer:.4f}")
            break

    return chain, best_cer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-limit", type=int, default=15,
                         help="Images per dataset scenario used during the greedy search phase")
    parser.add_argument("--fuzzy-threshold", type=float, default=85.0)
    parser.add_argument("--skip-validation", action="store_true",
                         help="Skip the full-dataset validation run at the end")
    args = parser.parse_args()

    search_entries = list(core.iter_dataset_entries(limit=args.search_limit))
    print(f"Search phase: {len(search_entries)} images "
          f"({args.search_limit}/scenario x {len(core.DATASETS)} scenarios)")

    log_rows = []
    chain, chain_cer = greedy_search(search_entries, log_rows)

    vocabulary = core.load_inci_vocabulary()
    fuzzy_cer = mean_cer_for_chain(
        search_entries, chain, use_fuzzy=True,
        vocabulary=vocabulary, threshold_score=args.fuzzy_threshold,
    )
    log_rows.append({
        "step": "final+fuzzy",
        "chain": (" -> ".join(chain) + " -> fuzzy") if chain else "fuzzy",
        "mean_cer": round(fuzzy_cer, 4),
    })
    print(f"[+fuzzy] mean CER = {fuzzy_cer:.4f}  (chain-only was {chain_cer:.4f})")

    use_fuzzy_final = fuzzy_cer < chain_cer
    print("\n=== Best pipeline found (search phase) ===")
    print(f"Chain: {chain if chain else '(none / baseline)'}")
    print(f"Use fuzzy matching: {use_fuzzy_final}")
    print(f"Search-phase mean CER: {min(chain_cer, fuzzy_cer):.4f}  (n={len(search_entries)})")

    core.write_csv(SEARCH_LOG_PATH, log_rows, fieldnames=["step", "chain", "mean_cer"])
    print(f"\nWrote search log to {SEARCH_LOG_PATH}")

    if args.skip_validation:
        return

    print("\nValidation phase: running winning pipeline on the FULL dataset...")
    full_entries = list(core.iter_dataset_entries())
    rows = []
    for entry in tqdm(full_entries, desc="validate_best_combination"):
        if entry["status"] != "ok":
            rows.append(core.evaluate_entry(entry, "", 0.0))
            continue
        image = cv2.imread(entry["image_path"])
        if image is None:
            rows.append(core.evaluate_entry(entry, "", 0.0, error="failed to read image"))
            continue

        t0 = time.time()
        try:
            processed = apply_chain(image, chain)
            lines, _, error = core.run_ocr_lines(processed, lang="en")
            if error:
                rows.append(core.evaluate_entry(entry, "", time.time() - t0, error))
                continue
            text = " ".join(lines)
            if use_fuzzy_final:
                text = correct_with_vocabulary(lines, vocabulary, args.fuzzy_threshold)
        except Exception as exc:  # noqa: BLE001 - record any pipeline failure per-image
            rows.append(core.evaluate_entry(entry, "", time.time() - t0, error=str(exc)))
            continue

        rows.append(core.evaluate_entry(entry, text, time.time() - t0))

    core.write_csv(FINAL_RESULTS_PATH, rows)
    print(f"Wrote {len(rows)} rows to {FINAL_RESULTS_PATH}")

    cers = [r["cer"] for r in rows if r["cer"] not in ("", None)]
    if cers:
        print(f"\nFull-dataset mean CER for best combination "
              f"({' -> '.join(chain) if chain else 'baseline'}"
              f"{' -> fuzzy' if use_fuzzy_final else ''}): {sum(cers) / len(cers):.4f}")


if __name__ == "__main__":
    main()
