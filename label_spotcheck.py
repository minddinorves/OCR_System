"""Manual spot-check of the ground-truth labels against their images.

The labels were transcribed by Claude (an AI assistant) from these same
images and have not been reviewed by a person. This draws a reproducible
random sample of products for a person to check by eye, then scores the
filled-in result.

  python label_spotcheck.py make [--n 20] [--seed 0]
      Creates results/label_spotcheck/:
        review.html   open in a browser: each sampled image (click to enlarge)
                      next to its numbered label names
        names.csv     one row per label name -> fill in `verdict`
                      (ok / wrong / extra / unreadable) and, for `wrong`,
                      `correct_text`
        products.csv  one row per product -> fill in `missing_names` (how many
                      names are on the image but absent from the label) and
                      `reviewer`
      Refuses to overwrite a sample that already has verdicts filled in.

  python label_spotcheck.py score
      Reads the filled-in CSVs and prints/writes label accuracy:
        accuracy = ok / (names in label + missing names)
      with a 95% bootstrap CI over products, plus counts per error type.

verdict meanings:
  ok          name matches the image (ignore letter case)
  wrong       a name that is on the image but transcribed with an error
  extra       not on the image at all
  unreadable  the image is too blurry/small for a person to verify this name
"""
import argparse
import csv
import html
import os
import random
import sys

import ocr_core as core

sys.path.insert(0, os.path.join(core.PROJECT_ROOT, "old"))
from compute_bootstrap_ci import ci_mean  # noqa: E402

OUT_DIR = os.path.join(core.RESULTS_DIR, "label_spotcheck")
NAMES_CSV = os.path.join(OUT_DIR, "names.csv")
PRODUCTS_CSV = os.path.join(OUT_DIR, "products.csv")
VERDICTS = ("ok", "wrong", "extra", "unreadable")


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _write(path, rows, fieldnames):
    # utf-8-sig so Excel opens the file with the right encoding
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make(n, seed):
    if os.path.isfile(NAMES_CSV) and any(r["verdict"].strip() for r in _read(NAMES_CSV)):
        sys.exit(f"{NAMES_CSV} already has verdicts filled in -- not overwriting. "
                 "Move it away first if you really want a new sample.")
    entries = [e for e in core.iter_dataset_entries(["original"]) if e["status"] == "ok"]
    sample = sorted(random.Random(seed).sample(entries, n), key=lambda e: e["image_file"])
    dev = {e["image_file"] for e in core.iter_dataset_entries(["original"], limit=40)}
    os.makedirs(OUT_DIR, exist_ok=True)

    names, products, cards = [], [], []
    for e in sample:
        import cv2
        h, w = cv2.imread(e["image_path"]).shape[:2]
        products.append({"label_file": e["label_file"], "image_file": e["image_file"],
                         "split": "dev" if e["image_file"] in dev else "heldout",
                         "image_size": f"{w}x{h}", "n_names": len(e["ingredients"]),
                         "missing_names": "", "reviewer": "", "notes": ""})
        for i, name in enumerate(e["ingredients"], 1):
            names.append({"label_file": e["label_file"], "image_file": e["image_file"], "position": i,
                          "name": name, "verdict": "", "correct_text": "", "notes": ""})
        rel = os.path.relpath(e["image_path"], OUT_DIR).replace(os.sep, "/")
        items = "".join(f"<li>{html.escape(x)}</li>" for x in e["ingredients"])
        cards.append(
            f'<section><h2>{html.escape(e["image_file"])} <small>{w}×{h}px · '
            f'{len(e["ingredients"])} names · {html.escape(e["label_file"])}</small></h2>'
            f'<a href="{rel}" target="_blank"><img src="{rel}" alt="{html.escape(e["image_file"])}"></a>'
            f"<ol>{items}</ol></section>")

    _write(NAMES_CSV, names, ["label_file", "image_file", "position", "name", "verdict", "correct_text", "notes"])
    _write(PRODUCTS_CSV, products, ["label_file", "image_file", "split", "image_size", "n_names",
                                    "missing_names", "reviewer", "notes"])
    page = f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Label Spot-check</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a1a; --fg:#eee; --muted:#aaa; --line:#444; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
section {{ border-top:1px solid var(--line); padding:16px 0; }}
h2 {{ font-size:17px; margin:0 0 8px; }} small {{ color:var(--muted); font-weight:normal; }}
img {{ max-width:100%; image-rendering:auto; background:#fff; border:1px solid var(--line); }}
ol {{ columns:2; column-gap:32px; }} @media (max-width:640px) {{ ol {{ columns:1; }} }}
</style></head><body>
<h1>Label spot-check ({n} products, seed {seed})</h1>
<p>Compare each image with its numbered names, and fill in <code>names.csv</code> (verdict: ok / wrong / extra / unreadable)
and <code>products.csv</code> (missing_names). Click an image to open it full size.</p>
{''.join(cards)}
</body></html>"""
    with open(os.path.join(OUT_DIR, "review.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Sampled {n} products ({sum(p['split'] == 'dev' for p in products)} dev, "
          f"{sum(p['split'] == 'heldout' for p in products)} held-out), {len(names)} names to check.")
    print(f"Open {os.path.join(OUT_DIR, 'review.html')} and fill in names.csv / products.csv")


def score():
    names, products = _read(NAMES_CSV), _read(PRODUCTS_CSV)
    blank = [r for r in names if r["verdict"].strip().lower() not in VERDICTS]
    if blank:
        sys.exit(f"{len(blank)} rows in names.csv have no valid verdict yet "
                 f"(first: {blank[0]['label_file']} #{blank[0]['position']}). Use one of {VERDICTS}.")
    missing = {}
    for p in products:
        if not p["missing_names"].strip().isdigit():
            sys.exit(f"products.csv: missing_names for {p['label_file']} is not filled in (use 0 if none).")
        missing[p["label_file"]] = int(p["missing_names"])

    per_product, rows = [], []
    totals = {v: 0 for v in VERDICTS}
    for p in products:
        lf = p["label_file"]
        mine = [r["verdict"].strip().lower() for r in names if r["label_file"] == lf]
        counts = {v: mine.count(v) for v in VERDICTS}
        for v in VERDICTS:
            totals[v] += counts[v]
        denom = len(mine) + missing[lf]
        acc = counts["ok"] / denom if denom else 0.0
        per_product.append(acc)
        rows.append({"label_file": lf, "image_size": p["image_size"], "n_names": len(mine),
                     **counts, "missing": missing[lf], "accuracy": round(acc, 4)})
    total_missing = sum(missing.values())
    denom = sum(totals.values()) + total_missing
    mean, lo, hi = ci_mean(per_product)
    _write(os.path.join(OUT_DIR, "score.csv"), rows,
           ["label_file", "image_size", "n_names", *VERDICTS, "missing", "accuracy"])
    print(f"products: {len(products)}   names in labels: {sum(totals.values())}   missing names: {total_missing}")
    for v in VERDICTS:
        print(f"  {v:10s} {totals[v]:5d}  ({totals[v] / denom:.1%} of label names + missing)")
    print(f"name-level accuracy (pooled): {totals['ok'] / denom:.1%}")
    print(f"mean per-product accuracy: {mean:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"wrote {os.path.join(OUT_DIR, 'score.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    args = parser.parse_args()
    make(args.n, args.seed) if args.cmd == "make" else score()


if __name__ == "__main__":
    main()
