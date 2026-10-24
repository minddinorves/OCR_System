"""Figures 4.11 and 4.12, from a fresh run on the current dataset with the
current OCR settings (padding 25%, box_thresh 0.30, doc unwarping on unless
stated). Metrics are OCR-only (no fuzzy matching), so the figures show what
each setting does to the recognized text itself.

  4.11  one label image under 4 preprocessing techniques (original,
        grayscale, bicubic upscale to 480 px, Otsu thresholding) -- the
        experiment's own preprocess() functions from old/preprocess_*.py.
        Image: "original" scenario, shorter than 480 px (so upscaling really
        changes it), OCR-only Set F1 closest to the median of such images.
  4.12  one Dev image with document unwarping off vs on. Every Dev image is
        OCR'd with unwarping off (on = saved cache); the image with the
        largest Set F1 gain from unwarping is shown, as a clear example.
        Also shows the unwarped image PaddleOCR actually fed to recognition.

Outputs (results/figures/): fig4_11_preprocessing_example.png,
fig4_12_unwarping_example.png, fig4_11_4_12_info.txt,
fig4_12_unwarping_off_dev.csv (per-image off/on metrics for all 40 Dev images)

Usage:
  python make_figures_4_11_4_12.py
"""
import csv
import importlib.util
import json
import os
import statistics
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ocr_core as core
from compute_ingredient_metrics import set_prf1, tokenize

OUT_DIR = os.path.join(core.RESULTS_DIR, "figures")
FONT_DIR = r"C:\Windows\Fonts"
WIDTH, MARGIN = 1600, 40
INK, MUTED, LINE = (20, 20, 20), (90, 90, 90), (200, 200, 200)
TECHNIQUES = [("Original (no preprocessing)", None), ("Grayscale", "preprocess_grayscale"),
              ("Bicubic upscale to 480 px height", "preprocess_upscale"),
              ("Otsu thresholding (binarization)", "preprocess_threshold")]


def fnt(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def metrics(text, ref):
    return set_prf1(tokenize(text), tokenize(ref))[2], core.compute_cer(text, ref)


def load_preprocess(module):
    spec = importlib.util.spec_from_file_location(module, os.path.join(core.PROJECT_ROOT, "old", f"{module}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preprocess


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def draw_rows(title, rows, out_path, img_w=760, wrap=68, max_lines=14):
    """rows: list of (heading, pil_image, text)."""
    f_title, f_head, f_mono = fnt("arialbd.ttf", 30), fnt("arialbd.ttf", 23), fnt("consola.ttf", 20)
    blocks = []
    for heading, img, text in rows:
        img = img.resize((img_w, max(1, int(img.height * img_w / img.width))), Image.LANCZOS)
        lines = textwrap.wrap(text, wrap) or ["(no text recognized)"]
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + [lines[max_lines - 1][: wrap - 4] + " ..."]
        blocks.append((heading, img, lines))
    h = MARGIN + 60 + sum(40 + max(b[1].height, 28 * len(b[2])) + 30 for b in blocks) + MARGIN
    canvas = Image.new("RGB", (WIDTH, h), "white")
    d = ImageDraw.Draw(canvas)
    y = MARGIN
    d.text((MARGIN, y), title, font=f_title, fill=INK)
    y += 60
    for heading, img, lines in blocks:
        d.text((MARGIN, y), heading, font=f_head, fill=MUTED)
        y += 40
        canvas.paste(img, (MARGIN, y))
        d.rectangle([MARGIN - 1, y - 1, MARGIN + img.width, y + img.height], outline=LINE)
        for k, ln in enumerate(lines):
            d.text((MARGIN + img_w + 30, y + 28 * k), ln, font=f_mono, fill=INK)
        y += max(img.height, 28 * len(lines)) + 30
        d.line([MARGIN, y - 15, WIDTH - MARGIN, y - 15], fill=LINE)
    canvas.save(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(core.RESULTS_DIR, "sweep_cache", "pad_0.25.json"), encoding="utf-8") as fh:
        cache = json.load(fh)
    entries = {e["image_file"]: e for e in core.iter_dataset_entries(["original"]) if e["status"] == "ok"}
    info = []

    # ---------------- Figure 4.11 ----------------
    cands = []
    for f, e in entries.items():
        img = cv2.imread(e["image_path"])
        if img.shape[0] < 480:
            cands.append((f, metrics(" ".join(cache[f]["lines"]), e["reference_text"])[0]))
    med = statistics.median(c[1] for c in cands)
    f411 = min(cands, key=lambda c: (abs(c[1] - med), c[0]))[0]
    e = entries[f411]
    original = cv2.imread(e["image_path"])
    rows = []
    info.append(f"Figure 4.11 image: {f411} (OCR-only Set F1 closest to median {med:.4f} of {len(cands)} "
                f"images < 480 px tall); fresh OCR run, current settings")
    for label, module in TECHNIQUES:
        img = original if module is None else load_preprocess(module)(original)
        lines, _, err = core.run_ocr_lines(img)
        text = " ".join(lines)
        f1, cer = metrics(text, e["reference_text"])
        rows.append((f"{label}   [{img.shape[1]}x{img.shape[0]} px, Set F1 = {f1:.3f}, CER = {cer:.3f}]",
                     to_pil(img), text))
        info.append(f"  {label:36s} Set F1={f1:.4f} CER={cer:.4f} chars={len(text)}")
    draw_rows(f"Figure 4.11  OCR results under different preprocessing techniques ({f411})", rows,
              os.path.join(OUT_DIR, "fig4_11_preprocessing_example.png"))

    # ---------------- Figure 4.12 ----------------
    from paddleocr import PaddleOCR
    ocr_off = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=True, device="cpu", enable_mkldnn=False,
                        text_det_box_thresh=0.3)
    dev = [e for e in core.iter_dataset_entries(["original"], limit=40) if e["status"] == "ok"]
    table = []
    for e in dev:
        padded = core._add_border(cv2.imread(e["image_path"]))
        off_text = " ".join(t for p in ocr_off.predict(padded) for t in p.get("rec_texts", []))
        on_text = " ".join(cache[e["image_file"]]["lines"])
        (f_off, c_off), (f_on, c_on) = metrics(off_text, e["reference_text"]), metrics(on_text, e["reference_text"])
        table.append({"image_file": e["image_file"], "set_f1_off": round(f_off, 4), "set_f1_on": round(f_on, 4),
                      "cer_off": round(c_off, 4), "cer_on": round(c_on, 4), "off_text": off_text, "on_text": on_text})
    core.write_csv(os.path.join(OUT_DIR, "fig4_12_unwarping_off_dev.csv"), table, fieldnames=list(table[0].keys()))
    best = max(table, key=lambda t: (t["set_f1_on"] - t["set_f1_off"], t["image_file"]))
    e = next(x for x in dev if x["image_file"] == best["image_file"])
    padded = core._add_border(cv2.imread(e["image_path"]))
    res = list(core.get_ocr().predict(padded))[0]
    unwarped = res["doc_preprocessor_res"]["output_img"]
    on_text = " ".join(res.get("rec_texts", []))
    assert on_text == best["on_text"], "fresh unwarping-on OCR differs from cache"
    unwarped_bgr = unwarped if unwarped.ndim == 3 else cv2.cvtColor(unwarped, cv2.COLOR_GRAY2BGR)
    rows = [
        (f"(a) Document unwarping OFF: input image as given to OCR   "
         f"[Set F1 = {best['set_f1_off']:.3f}, CER = {best['cer_off']:.3f}]", to_pil(padded), best["off_text"]),
        (f"(b) Document unwarping ON: image after unwarping   "
         f"[Set F1 = {best['set_f1_on']:.3f}, CER = {best['cer_on']:.3f}]", to_pil(np.ascontiguousarray(unwarped_bgr)),
         best["on_text"]),
    ]
    draw_rows(f"Figure 4.12  OCR results with document unwarping off vs on ({best['image_file']})", rows,
              os.path.join(OUT_DIR, "fig4_12_unwarping_example.png"))
    gains = [t["set_f1_on"] - t["set_f1_off"] for t in table]
    info += [f"Figure 4.12 image: {best['image_file']} (largest Set F1 gain from unwarping among {len(table)} Dev images)",
             f"  off: Set F1={best['set_f1_off']:.4f} CER={best['cer_off']:.4f}   "
             f"on: Set F1={best['set_f1_on']:.4f} CER={best['cer_on']:.4f}",
             f"  all 40 Dev (OCR only): mean Set F1 off={np.mean([t['set_f1_off'] for t in table]):.4f} "
             f"on={np.mean([t['set_f1_on'] for t in table]):.4f}; unwarping better on "
             f"{sum(g > 0 for g in gains)}, worse on {sum(g < 0 for g in gains)}, same on {sum(g == 0 for g in gains)}"]
    with open(os.path.join(OUT_DIR, "fig4_11_4_12_info.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(info) + "\n")
    print("\n".join(info))


if __name__ == "__main__":
    main()
