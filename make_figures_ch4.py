"""Figures 4.7 (OCR result) and 4.8 (fuzzy matching result) for thesis
Chapter 4, built from a real tested image and the saved outputs of the
shipped pipeline -- nothing is typed in by hand, so every value in the
figures traces back to files in results/.

Example image selection (fixed rule, decided before looking at outputs):
  - Held-out product, "original" scenario (never used to tune anything)
  - Set F1 of the shipped system within 0.10 of the Held-out median, so the
    example is typical rather than a best case
  - at least 2 tokens corrected by fuzzy matching with a score below 100
    (so Figure 4.8 shows real corrections, not only exact matches)
  - at most 30 tokens (so Figure 4.8 fits on one page)
  among those, the one closest to the median Set F1 (ties: file name).

Sources:
  image        test_SkinSafe/image/<file>
  OCR lines    results/sweep_cache/pad_0.25.json (per-line output; identical
               to ocr_text_raw in the FINAL run, checked 216/216)
  fuzzy step   ocr_paddle_fuzzy.correct_with_vocabulary logic, threshold 85,
               vocabulary = core.load_inci_vocabulary() (shipped Dev patch)

Outputs (results/figures/):
  fig4_7_ocr_example.png, fig4_8_fuzzy_example.png
  fig4_7_4_8_data.csv   per-token data shown in Figure 4.8
  fig4_7_4_8_info.txt   chosen image, why, and its metrics

Usage:
  python make_figures_ch4.py
"""
import csv
import json
import os
import statistics
import textwrap

from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import fuzz, process

import ocr_core as core
import ocr_paddle_fuzzy as fz
from compute_ingredient_metrics import set_prf1, tokenize

THRESHOLD = 85.0
N_DEV = 40
OUT_DIR = os.path.join(core.RESULTS_DIR, "figures")
FONT_DIR = r"C:\Windows\Fonts"
WIDTH = 1600
MARGIN = 40
INK, MUTED, LINE = (20, 20, 20), (90, 90, 90), (200, 200, 200)
GOOD, WARN = (220, 245, 225), (255, 238, 210)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def split_with_scores(token, vocab):
    """Same search as fz._split_match_glued_names, but also returns both halves' scores."""
    words, best = token.split(" "), None
    for i in range(1, len(words)):
        left, right = " ".join(words[:i]), " ".join(words[i:])
        if len(left) < fz.MIN_TOKEN_LEN or len(right) < fz.MIN_TOKEN_LEN:
            continue
        ml, mr = fz._extract_one(left, vocab, THRESHOLD), fz._extract_one(right, vocab, THRESHOLD)
        if not ml or not mr or not (fz._is_reliable_match(left, ml[0]) and fz._is_reliable_match(right, mr[0])):
            continue
        if best is None or ml[1] + mr[1] > best[0]:
            best = (ml[1] + mr[1], ml, mr)
    return None if best is None else (best[1], best[2])


def token_details(lines, vocab):
    """Re-trace correct_with_vocabulary token by token, keeping the scores."""
    text = fz._strip_header(" ".join(lines))
    rows = []
    for tok in (t.strip(" .;:*-") for t in fz._SPLIT_RE.split(text)):
        if not tok:
            continue
        best = process.extractOne(tok, vocab, scorer=fuzz.ratio, processor=str.lower)
        row = {"ocr_token": tok, "closest_inci": best[0], "score": f"{best[1]:.1f}", "output": tok}
        if len(tok) < fz.MIN_TOKEN_LEN:
            row["decision"] = "kept (too short)"
        elif best[1] >= THRESHOLD and fz._is_reliable_match(tok, best[0]):
            row["decision"] = "matched, unchanged" if best[0].lower() == tok.lower() else "replaced"
            row["output"] = best[0]
        else:
            split = split_with_scores(tok, vocab)
            assert (split is None) == (fz._split_match_glued_names(tok, vocab, THRESHOLD) is None)
            if split:
                (nl, sl), (nr, sr) = split[0][:2], split[1][:2]
                row.update(decision="split into 2 names", closest_inci=f"{nl} | {nr}",
                           score=f"{sl:.1f} | {sr:.1f}", output=f"{nl} | {nr}")
            else:
                row["decision"] = "kept (score < 85)"
        rows.append(row)
    return rows


def pick_example(rows, cache, vocab):
    dev = {e["label_file"] for e in core.iter_dataset_entries(["original"], limit=N_DEV)}
    cands = []
    for r in rows:
        if r["label_file"] in dev:
            continue
        lines = cache[r["image_file"]]["lines"]
        out = fz.correct_with_vocabulary(lines, vocab, THRESHOLD)
        f1 = set_prf1(tokenize(out), tokenize(r["reference_text"]))[2]
        cands.append((r, lines, out, f1))
    median = statistics.median(c[3] for c in cands)
    ok = []
    for r, lines, out, f1 in cands:
        if abs(f1 - median) > 0.10:
            continue
        det = token_details(lines, vocab)
        fixes = sum(d["decision"] == "replaced" and float(d["score"]) < 100 for d in det)
        if fixes >= 2 and len(det) <= 30:
            ok.append((abs(f1 - median), r["image_file"], r, lines, out, f1, det))
    ok.sort(key=lambda x: (x[0], x[1]))
    return median, len(cands), ok[0][2:]


def draw_fig_4_7(img_path, lines, out_path, title):
    img = Image.open(img_path).convert("RGB")
    scale = min((WIDTH - 2 * MARGIN) / img.width, 2.5)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    f_title, f_head, f_mono = font("arialbd.ttf", 30), font("arialbd.ttf", 24), font("consola.ttf", 22)
    wrapped = []
    for i, ln in enumerate(lines, 1):
        parts = textwrap.wrap(ln, 105) or [""]
        wrapped += [f"{i:>2}  {parts[0]}"] + [f"    {p}" for p in parts[1:]]
    h = MARGIN + 50 + 40 + img.height + 40 + 40 + 32 * len(wrapped) + MARGIN
    canvas = Image.new("RGB", (WIDTH, h), "white")
    d = ImageDraw.Draw(canvas)
    y = MARGIN
    d.text((MARGIN, y), title, font=f_title, fill=INK)
    y += 50
    d.text((MARGIN, y), "(a) Input label image (original scenario)", font=f_head, fill=MUTED)
    y += 40
    canvas.paste(img, (MARGIN, y))
    d.rectangle([MARGIN - 1, y - 1, MARGIN + img.width, y + img.height], outline=LINE)
    y += img.height + 40
    d.text((MARGIN, y), f"(b) PaddleOCR output: {len(lines)} recognized text lines "
                        "(25% padding, box_thresh 0.30)", font=f_head, fill=MUTED)
    y += 40
    for w in wrapped:
        d.text((MARGIN, y), w, font=f_mono, fill=INK)
        y += 32
    canvas.save(out_path)


def draw_fig_4_8(details, out_path, title):
    f_title, f_head, f_cell = font("arialbd.ttf", 30), font("arialbd.ttf", 20), font("consola.ttf", 19)
    cols = [("#", 44), ("OCR token", 470), ("Closest INCI name in vocabulary", 470),
            ("Score", 150), ("Decision", 236), ("Correct?", 150)]
    char_w, line_h, pad, x0 = 10.5, 25, 7, MARGIN

    def cells_of(i, r):
        gt = "yes" if r["output_in_ground_truth"] else "no"
        if r["decision"].startswith("kept"):
            gt = "(kept)"
        return [str(i), r["ocr_token"], r["closest_inci"], r["score"], r["decision"], gt]

    def wrap(s, w):
        return textwrap.wrap(s, max(4, int((w - 2 * pad) / char_w)), break_on_hyphens=False) or [""]

    rows = [[wrap(v, w) for (name, w), v in zip(cols, cells_of(i, r))] for i, r in enumerate(details, 1)]
    heights = [max(len(c) for c in row) * line_h + 2 * pad for row in rows]
    head_h = line_h + 2 * pad
    h = MARGIN + 50 + 36 + head_h + sum(heights) + 110 + MARGIN
    canvas = Image.new("RGB", (WIDTH, h), "white")
    d = ImageDraw.Draw(canvas)
    y = MARGIN
    d.text((MARGIN, y), title, font=f_title, fill=INK)
    y += 50
    d.text((MARGIN, y), "RapidFuzz fuzz.ratio, case-insensitive; a token is replaced when score >= 85. "
                        "A token with no match is split at a space if both halves score >= 85.",
           font=f_head, fill=MUTED)
    y += 36
    x = x0
    for name, w in cols:
        d.rectangle([x, y, x + w, y + head_h], fill=(235, 235, 235), outline=LINE)
        d.text((x + pad, y + pad), name, font=f_head, fill=INK)
        x += w
    y += head_h
    for r, row, rh in zip(details, rows, heights):
        fill = WARN if r["decision"].startswith("kept") else GOOD
        x = x0
        for (name, w), lines_ in zip(cols, row):
            cell_fill = fill
            if name == "Correct?" and lines_[0] == "no":
                cell_fill = (250, 215, 215)
            d.rectangle([x, y, x + w, y + rh], fill=cell_fill, outline=LINE)
            for k, ln in enumerate(lines_):
                d.text((x + pad, y + pad + k * line_h), ln, font=f_cell, fill=INK)
            x += w
        y += rh
    y += 20
    legend = [(GOOD, "output is the vocabulary name (score >= 85)"),
              (WARN, "token kept unchanged (score < 85; closest name shown for reference)"),
              ((250, 215, 215), "wrong correction: output not in this image's ground truth, "
                                "or an already-correct token was changed")]
    for k, (col, txt) in enumerate(legend):
        yy = y + k * 30
        d.rectangle([x0, yy + 4, x0 + 24, yy + 24], fill=col, outline=LINE)
        d.text((x0 + 34, yy), txt, font=f_head, fill=INK)
    canvas.save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vocab = core.load_inci_vocabulary()
    with open(os.path.join(core.RESULTS_DIR, "sweep_cache", "pad_0.25.json"), encoding="utf-8") as fh:
        cache = json.load(fh)
    with open(os.path.join(core.RESULTS_DIR, "ocr_paddle_fuzzy_FINAL_original_full_validation.csv"),
              newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]

    median, n_cands, (r, lines, out, f1, details) = pick_example(rows, cache, vocab)
    assert " ".join(lines) == r["ocr_text_raw"], "cached OCR lines differ from the FINAL run"
    raw_p, raw_r, raw_f1 = set_prf1(tokenize(r["ocr_text_raw"]), tokenize(r["reference_text"]))
    p, rec, _ = set_prf1(tokenize(out), tokenize(r["reference_text"]))
    ref_names = {n.lower() for n in tokenize(r["reference_text"])}
    for dct in details:
        ok = all(o.strip().lower() in ref_names for o in dct["output"].split(" | "))
        # a token that was already a correct name but got changed is a wrong correction, even if
        # the new name happens to appear elsewhere in this label (e.g. one CI number -> another)
        if dct["decision"] == "replaced" and dct["ocr_token"].lower() in ref_names:
            ok = False
        dct["output_in_ground_truth"] = ok

    img_path = os.path.join(core.IMAGE_DIR, r["image_file"])
    draw_fig_4_7(img_path, lines, os.path.join(OUT_DIR, "fig4_7_ocr_example.png"),
                 f"Figure 4.7  OCR result for test image {r['image_file']}")
    draw_fig_4_8(details, os.path.join(OUT_DIR, "fig4_8_fuzzy_example.png"),
                 f"Figure 4.8  Fuzzy matching result for test image {r['image_file']}")
    core.write_csv(os.path.join(OUT_DIR, "fig4_7_4_8_data.csv"),
                   [{"image_file": r["image_file"], "token_no": i, **dct} for i, dct in enumerate(details, 1)],
                   fieldnames=["image_file", "token_no", "ocr_token", "closest_inci", "score", "decision", "output",
                               "output_in_ground_truth"])
    counts = {k: sum(dd["decision"] == k for dd in details) for k in
              ("matched, unchanged", "replaced", "split into 2 names", "kept (score < 85)", "kept (too short)")}
    changed = [dd for dd in details if dd["decision"] in ("replaced", "split into 2 names")]
    info_wrong = sum(not dd["output_in_ground_truth"] for dd in changed)
    info = [
        f"image: {r['image_file']}  (label {r['label_file']}, Held-out, original scenario)",
        f"selection: closest to Held-out median Set F1 ({median:.4f}, n={n_cands}) among images with "
        f">=2 fuzzy corrections (score<100) and <=30 tokens",
        f"OCR lines: {len(lines)}   tokens: {len(details)}   reference names: {len(tokenize(r['reference_text']))}",
        f"token decisions: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        f"tokens whose output matches a ground-truth name exactly: "
        f"{sum(dd['output_in_ground_truth'] for dd in details)}/{len(details)}",
        f"replaced/split tokens whose output is NOT in the ground truth (wrong corrections): "
        f"{info_wrong}/{len(changed)}",
        f"OCR only     : Set P={raw_p:.4f} R={raw_r:.4f} F1={raw_f1:.4f}  CER={core.compute_cer(r['ocr_text_raw'], r['reference_text']):.4f}",
        f"OCR + fuzzy  : Set P={p:.4f} R={rec:.4f} F1={f1:.4f}  CER={core.compute_cer(out, r['reference_text']):.4f}",
    ]
    with open(os.path.join(OUT_DIR, "fig4_7_4_8_info.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(info) + "\n")
    print("\n".join(info))


if __name__ == "__main__":
    main()
