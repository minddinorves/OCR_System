"""Aggregate every results/*.csv into one summary table (mean CER/WER/time per dataset).

Usage:
  python summarize_results.py
"""
import csv
import glob
import os

import ocr_core as core

SUMMARY_FIELDNAMES = [
    "source_file",
    "dataset",
    "n_images",
    "n_scored",
    "mean_cer",
    "mean_wer",
    "mean_ocr_time_sec",
]


def _mean(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
    return sum(values) / len(values) if values else None


def summarize_file(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    # ocr_paddle_fuzzy.csv reports cer_raw/cer_fuzzy instead of a plain "cer";
    # prefer the fuzzy-corrected score as the headline number when both exist.
    cer_field = "cer_fuzzy" if "cer_fuzzy" in fieldnames else "cer"
    wer_field = "wer_fuzzy" if "wer_fuzzy" in fieldnames else "wer"

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summaries = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        n_scored = sum(1 for r in dataset_rows if r.get(cer_field) not in ("", None))
        mean_cer = _mean(dataset_rows, cer_field)
        mean_wer = _mean(dataset_rows, wer_field)
        mean_time = _mean(dataset_rows, "ocr_time_sec")
        summaries.append(
            {
                "source_file": os.path.basename(path),
                "dataset": dataset,
                "n_images": len(dataset_rows),
                "n_scored": n_scored,
                "mean_cer": round(mean_cer, 4) if mean_cer is not None else "",
                "mean_wer": round(mean_wer, 4) if mean_wer is not None else "",
                "mean_ocr_time_sec": round(mean_time, 4) if mean_time is not None else "",
            }
        )
    return summaries


def main():
    csv_paths = sorted(
        p for p in glob.glob(os.path.join(core.RESULTS_DIR, "*.csv")) if os.path.basename(p) != "summary.csv"
    )
    if not csv_paths:
        print(f"No result CSVs found in {core.RESULTS_DIR}. Run an OCR script first.")
        return

    all_summaries = []
    for path in csv_paths:
        all_summaries.extend(summarize_file(path))

    out_path = os.path.join(core.RESULTS_DIR, "summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"Wrote {len(all_summaries)} summary rows to {out_path}\n")
    header = f"{'source_file':<28} {'dataset':<12} {'n':>5} {'mean_cer':>9} {'mean_wer':>9} {'avg_sec':>8}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['source_file']:<28} {s['dataset']:<12} {s['n_images']:>5} "
            f"{s['mean_cer']!s:>9} {s['mean_wer']!s:>9} {s['mean_ocr_time_sec']!s:>8}"
        )


if __name__ == "__main__":
    main()
