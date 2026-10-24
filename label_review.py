"""Full human review of the ground-truth labels, by two independent reviewers.

The labels were transcribed by Claude (an AI assistant) from these same images.
label_spotcheck.py already checked a random 20 products; this covers every
other product (216 - 20 = 196), Held-out first. Each reviewer works in their
own folder without seeing the other's answers, then `agree` measures how well
they agree and lists every disagreement to settle together, and `apply`
writes corrected label files to a NEW folder (the originals are never touched).
Labelling rules for the reviewers: LABELING_GUIDELINES.md.

  python label_review.py make [--reviewers A B]
      Creates results/label_review/reviewer_<X>/ for each reviewer:
        review.html   the review page: each image next to its numbered names,
                      verdicts are clicked in place and kept in the browser;
                      "download" saves them as label_review_<X>.json
        names.csv     one row per label name -> `verdict` (ok / wrong / extra /
                      unreadable), and for `wrong` the `correct_text`
        missing.csv   one row per name that is on the image but not in the
                      label: label_file, after_position, name
        products.csv  one row per product -> `checked` = yes when finished
                      (blank verdicts of a checked product count as ok),
                      `reviewer`, `notes`
      The CSVs are filled by `import` (or by hand in Excel instead of the page).
      Refuses to overwrite a reviewer folder that already has work in it.

  python label_review.py import <X> <label_review_X.json>
      Writes the results downloaded from review.html into reviewer_<X>'s CSVs.
      Re-import a newer download any time; it replaces the previous import.

  python label_review.py status
      Progress of each reviewer.

  python label_review.py agree
      Over products BOTH reviewers have checked: Cohen's kappa, Gwet's AC1 and
      raw agreement on the name verdicts, agreement on missing names, and
      results/label_review/adjudication.csv listing every disagreement with
      empty `final_verdict` / `final_text` columns to fill in together.

  python label_review.py spotfix
      The 20 spot-checked products recorded only HOW MANY names were missing,
      not which. Creates results/label_review/spotcheck_fix/ with missing.csv
      (one empty row per missing name, the spot-check's notes as hints) and
      review.html, for the reviewers to fill in together.

  python label_review.py apply
      Needs all 196 products checked by both reviewers, adjudication.csv
      complete and spotcheck_fix/missing.csv filled in. Writes
      test_SkinSafe/label_corrected/ (all 216 label files: the 196 from this
      review, the 20 spot-checked ones with the spot-check's extra / wrong
      verdicts and the spotfix missing names applied; footnote asterisks and
      percentages stripped from every name, guidelines rule 3) and
      results/label_review/changes.csv.

verdict meanings (names.csv):
  ok          name matches the image (differences in letter case only are ok)
  wrong       the name is on the image but transcribed with an error ->
              write the name exactly as printed in `correct_text`
  extra       not on the image at all -> will be removed
  unreadable  the image is too blurry/small to verify this name -> kept as is
"""
import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
from collections import Counter

from rapidfuzz import fuzz

import ocr_core as core

OUT_DIR = os.path.join(core.RESULTS_DIR, "label_review")
SPOTCHECK_PRODUCTS = os.path.join(core.RESULTS_DIR, "label_spotcheck", "products.csv")
SPOTCHECK_NAMES = os.path.join(core.RESULTS_DIR, "label_spotcheck", "names.csv")
SPOTFIX_DIR = os.path.join(OUT_DIR, "spotcheck_fix")
ADJUDICATION_CSV = os.path.join(OUT_DIR, "adjudication.csv")
CORRECTED_DIR = os.path.join(core.PROJECT_ROOT, "test_SkinSafe", "label_corrected")
GUIDELINES = os.path.join(core.PROJECT_ROOT, "LABELING_GUIDELINES.md")
TEMPLATE = os.path.join(core.PROJECT_ROOT, "label_review_template.html")
VERDICTS = ("ok", "wrong", "extra", "unreadable")
N_DEV = 40
MISSING_MATCH = 85  # fuzz.ratio for two reviewers' missing names to count as the same name

NAMES_FIELDS = ["split", "label_file", "image_file", "position", "name", "verdict", "correct_text", "notes"]
MISSING_FIELDS = ["label_file", "after_position", "name", "notes"]
PRODUCTS_FIELDS = ["order", "split", "label_file", "image_file", "image_size", "n_names",
                   "checked", "reviewer", "notes"]


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _write(path, rows, fieldnames):
    # utf-8-sig so Excel opens the file with the right encoding
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reviewer_dirs():
    if not os.path.isdir(OUT_DIR):
        sys.exit(f"{OUT_DIR} does not exist -- run `python label_review.py make` first.")
    dirs = sorted(d for d in os.listdir(OUT_DIR) if d.startswith("reviewer_")
                  and os.path.isdir(os.path.join(OUT_DIR, d)))
    if not dirs:
        sys.exit(f"no reviewer_* folders in {OUT_DIR}")
    return [(d[len("reviewer_"):], os.path.join(OUT_DIR, d)) for d in dirs]


def review_entries():
    """All products except the 20 already spot-checked, Held-out first."""
    if not os.path.isfile(SPOTCHECK_PRODUCTS):
        sys.exit(f"{SPOTCHECK_PRODUCTS} not found -- needed to exclude the spot-checked products.")
    done = {r["label_file"] for r in _read(SPOTCHECK_PRODUCTS)}
    dev = {e["image_file"] for e in core.iter_dataset_entries(["original"], limit=N_DEV)}
    entries = [e for e in core.iter_dataset_entries(["original"])
               if e["status"] == "ok" and e["label_file"] not in done]
    for e in entries:
        e["split"] = "dev" if e["image_file"] in dev else "heldout"
    return sorted(entries, key=lambda e: e["split"] != "heldout"), done


def _has_work(folder):
    names = os.path.join(folder, "names.csv")
    missing = os.path.join(folder, "missing.csv")
    products = os.path.join(folder, "products.csv")
    return ((os.path.isfile(names) and any(r["verdict"].strip() for r in _read(names)))
            or (os.path.isfile(missing) and _read(missing))
            or (os.path.isfile(products) and any(r["checked"].strip() for r in _read(products))))


def make(reviewers):
    entries, done = review_entries()
    import cv2

    products, names, cards = [], [], []
    for i, e in enumerate(entries, 1):
        h, w = cv2.imread(e["image_path"]).shape[:2]
        products.append({"order": i, "split": e["split"], "label_file": e["label_file"],
                         "image_file": e["image_file"], "image_size": f"{w}x{h}",
                         "n_names": len(e["ingredients"]), "checked": "", "reviewer": "", "notes": ""})
        for pos, name in enumerate(e["ingredients"], 1):
            names.append({"split": e["split"], "label_file": e["label_file"], "image_file": e["image_file"],
                          "position": pos, "name": name, "verdict": "", "correct_text": "", "notes": ""})
        cards.append((i, e, w, h))

    for r in reviewers:
        folder = os.path.join(OUT_DIR, f"reviewer_{r}")
        if os.path.isdir(folder) and _has_work(folder):
            sys.exit(f"{folder} already has work in it -- not overwriting. "
                     "Move it away first if you really want to start over.")
    data_hash = _data_hash(entries)
    for r in reviewers:
        folder = os.path.join(OUT_DIR, f"reviewer_{r}")
        os.makedirs(folder, exist_ok=True)
        _write(os.path.join(folder, "names.csv"), names, NAMES_FIELDS)
        _write(os.path.join(folder, "missing.csv"), [], MISSING_FIELDS)
        _write(os.path.join(folder, "products.csv"), products, PRODUCTS_FIELDS)
        for stale in ("imported_from.txt",):
            if os.path.isfile(os.path.join(folder, stale)):
                os.remove(os.path.join(folder, stale))
        with open(os.path.join(folder, "review_meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"reviewer": r, "data_hash": data_hash, "n_products": len(entries)}, fh)
        with open(os.path.join(folder, "review.html"), "w", encoding="utf-8") as fh:
            fh.write(_interactive_page(r, folder, cards, data_hash))

    n_ho = sum(p["split"] == "heldout" for p in products)
    print(f"Excluded {len(done)} spot-checked products. {len(products)} products to review "
          f"({n_ho} held-out first, then {len(products) - n_ho} dev), {len(names)} names.")
    for r in reviewers:
        print(f"  reviewer {r}: {os.path.join(OUT_DIR, f'reviewer_{r}', 'review.html')}")


def _href(path, folder):
    try:
        return os.path.relpath(path, folder).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return "file:///" + os.path.abspath(path).replace(os.sep, "/")


def _data_hash(entries):
    """Fingerprint of the products + names a page was built from (same for every reviewer)."""
    blob = json.dumps([[e["label_file"], e["ingredients"]] for e in entries], ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _interactive_page(reviewer, folder, cards, data_hash):
    """review.html where verdicts are clicked in place; results download as one JSON file for `import`."""
    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read()
    products = [{"i": i, "lf": e["label_file"], "img": e["image_file"], "split": e["split"],
                 "w": w, "h": h, "href": _href(e["image_path"], folder), "names": e["ingredients"]}
                for i, e, w, h in cards]
    data = json.dumps({"reviewer": reviewer, "hash": data_hash, "products": products}, ensure_ascii=False)
    title = f"Label review · ผู้ตรวจ {reviewer} ({len(cards)} ผลิตภัณฑ์)"
    return (page.replace("__TITLE__", html.escape(title))
                .replace("__GUIDE__", html.escape(_href(GUIDELINES, folder)))
                .replace("__DATA__", data.replace("</", "<\\/")))


SPOTFIX_INTRO = """<p>อ่านกฎ: <a href="{guide}">LABELING_GUIDELINES.md</a> (ข้อ 8) · ผลิตภัณฑ์จากการสุ่มตรวจรอบก่อนที่บันทึกไว้ว่ามีชื่อขาด
แต่ไม่ได้บันทึกว่าเป็นชื่ออะไร กรอกชื่อที่ขาดใน <code>missing.csv</code> แถวละหนึ่งชื่อ (ตรวจร่วมกันทั้งสองคน)</p>"""


def _page(title, intro, folder, cards):
    guide = _href(GUIDELINES, folder)
    sections = []
    for i, e, w, h in cards:
        rel = _href(e["image_path"], folder)
        items = "".join(f"<li>{html.escape(x)}</li>" for x in e["ingredients"])
        split = "Held-out" if e.get("split") == "heldout" else "Dev"
        sections.append(
            f'<section id="p{i}"><h2>#{i} · {html.escape(e["image_file"])} '
            f'<small>{split} · {w}×{h}px · {len(e["ingredients"])} names · {html.escape(e["label_file"])}</small></h2>'
            f'<a href="{rel}" target="_blank"><img loading="lazy" src="{rel}" alt="{html.escape(e["image_file"])}"></a>'
            f"<ol>{items}</ol></section>")
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Label Review</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a1a; --fg:#eee; --muted:#aaa; --line:#444; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
section {{ border-top:1px solid var(--line); padding:16px 0; }}
h2 {{ font-size:17px; margin:0 0 8px; }} small {{ color:var(--muted); font-weight:normal; }}
img {{ max-width:100%; background:#fff; border:1px solid var(--line); }}
ol {{ columns:2; column-gap:32px; }} @media (max-width:640px) {{ ol {{ columns:1; }} }}
</style></head><body>
<h1>{html.escape(title)}</h1>
{intro.format(guide=guide)}
{''.join(sections)}
</body></html>"""


def load_reviewer(folder):
    """-> (verdicts {(label_file, pos): (verdict, correct_text)}, missing {label_file: [(after, name)]},
           checked set of label_files, problems list)"""
    names = _read(os.path.join(folder, "names.csv"))
    products = _read(os.path.join(folder, "products.csv"))
    missing_rows = _read(os.path.join(folder, "missing.csv"))
    checked = {p["label_file"] for p in products if p["checked"].strip().lower() in ("yes", "y", "1")}
    known = {p["label_file"]: int(p["n_names"]) for p in products}
    problems, verdicts, missing = [], {}, {}
    for r in names:
        lf, pos = r["label_file"], int(r["position"])
        v = r["verdict"].strip().lower()
        text = r["correct_text"].strip()
        if not v and lf in checked:
            v = "ok"
        if v and v not in VERDICTS:
            problems.append(f"names.csv {lf} #{pos}: verdict '{r['verdict']}' is not one of {VERDICTS}")
        elif v == "wrong" and not text:
            problems.append(f"names.csv {lf} #{pos}: verdict wrong but correct_text is empty")
        if v:
            verdicts[(lf, pos)] = (v, text)
    for r in missing_rows:
        lf, name = r["label_file"].strip(), r["name"].strip()
        if not lf and not name:
            continue
        if lf not in known:
            problems.append(f"missing.csv: label_file '{lf}' is not in this review")
            continue
        try:
            after = float(r["after_position"])
        except ValueError:
            problems.append(f"missing.csv {lf}: after_position '{r['after_position']}' is not a number")
            continue
        if not name or not 0 <= after <= known[lf]:
            problems.append(f"missing.csv {lf}: needs a name and after_position between 0 and {known[lf]}")
            continue
        missing.setdefault(lf, []).append((after, name))
    return verdicts, missing, checked, problems


def import_results(reviewer, path):
    """Turn the JSON downloaded from review.html into this reviewer's names/missing/products CSVs."""
    folder = os.path.join(OUT_DIR, f"reviewer_{reviewer}")
    meta_path = os.path.join(folder, "review_meta.json")
    if not os.path.isfile(meta_path):
        sys.exit(f"{meta_path} not found -- run `python label_review.py make` first.")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(path, encoding="utf-8-sig") as fh:
        s = json.load(fh)
    if s.get("format") != "label_review/1":
        sys.exit(f"{path} is not a label_review results file.")
    if s.get("reviewer_id") != reviewer:
        sys.exit(f"{path} belongs to reviewer {s.get('reviewer_id')}, not {reviewer}.")
    if s.get("data_hash") != meta["data_hash"]:
        sys.exit(f"{path} was made from a different review.html (labels changed?) -- not importing.")
    marker = os.path.join(folder, "imported_from.txt")
    if _has_work(folder) and not os.path.isfile(marker):
        sys.exit(f"{folder} has results typed into the CSVs by hand -- importing would overwrite them. "
                 "Use either review.html or the CSVs, not both.")

    names = _read(os.path.join(folder, "names.csv"))
    products = _read(os.path.join(folder, "products.csv"))
    n_names = {p["label_file"]: int(p["n_names"]) for p in products}
    state = s.get("products", {})
    unknown = set(state) - set(n_names)
    if unknown:
        sys.exit(f"{path} has products not in this review: {sorted(unknown)[:5]}")
    for r in names:
        x = state.get(r["label_file"], {}).get("verdicts", {}).get(r["position"])
        v = x["v"] if x else ""
        if v and v not in VERDICTS:
            sys.exit(f"{path}: {r['label_file']} #{r['position']} has unknown verdict '{v}'")
        r["verdict"] = v
        r["correct_text"] = x.get("text", "").strip() if v == "wrong" else ""
        r["notes"] = x.get("note", "").strip() if x else ""
    missing = []
    for lf, st in state.items():
        for m in st.get("missing", []):
            name = m.get("name", "").strip()
            if not name:
                if st.get("checked"):
                    sys.exit(f"{path}: {lf} is marked done but has a missing-name row without a name")
                continue
            missing.append({"label_file": lf, "after_position": int(m["after"]), "name": name,
                            "notes": m.get("note", "").strip()})
    reviewer_name = s.get("reviewer_name", "").strip()
    for p in products:
        st = state.get(p["label_file"], {})
        p["checked"] = "yes" if st.get("checked") else ""
        p["reviewer"] = reviewer_name if st.get("checked") else ""
        p["notes"] = st.get("notes", "").strip()
    _write(os.path.join(folder, "names.csv"), names, NAMES_FIELDS)
    _write(os.path.join(folder, "missing.csv"), missing, MISSING_FIELDS)
    _write(os.path.join(folder, "products.csv"), products, PRODUCTS_FIELDS)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(f"{os.path.abspath(path)}\nexported_at {s.get('exported_at')}\n")
    print(f"imported {path} into {folder}")
    status()


def status():
    for r, folder in _reviewer_dirs():
        verdicts, missing, checked, problems = load_reviewer(folder)
        products = _read(os.path.join(folder, "products.csv"))
        by_split = Counter(p["split"] for p in products if p["label_file"] in checked)
        total = Counter(p["split"] for p in products)
        errors = Counter(v for (lf, _), (v, _) in verdicts.items() if lf in checked and v != "ok")
        print(f"reviewer {r}: {len(checked)}/{len(products)} products checked "
              f"(held-out {by_split['heldout']}/{total['heldout']}, dev {by_split['dev']}/{total['dev']}); "
              f"non-ok verdicts {dict(errors)}; missing names {sum(len(v) for v in missing.values())}")
        for p in problems[:10]:
            print(f"  ! {p}")
        if len(problems) > 10:
            print(f"  ! ... {len(problems) - 10} more")


def cohen_kappa(pairs, categories):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in categories) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def gwet_ac1(pairs, categories):
    # less affected than kappa when almost every item is in one category (here: "ok")
    n, q = len(pairs), len(categories)
    po = sum(a == b for a, b in pairs) / n
    pi = {c: (sum(a == c for a, _ in pairs) + sum(b == c for _, b in pairs)) / (2 * n) for c in categories}
    pe = sum(p * (1 - p) for p in pi.values()) / (q - 1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def match_missing(a_list, b_list):
    """Greedy one-to-one match of two reviewers' missing names -> (pairs, only_a, only_b)."""
    scored = sorted(((fuzz.ratio(x[1].lower(), y[1].lower()), i, j)
                     for i, x in enumerate(a_list) for j, y in enumerate(b_list)), reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for s, i, j in scored:
        if s < MISSING_MATCH or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((a_list[i], b_list[j]))
    return (pairs, [x for i, x in enumerate(a_list) if i not in used_a],
            [y for j, y in enumerate(b_list) if j not in used_b])


def _two_reviewers():
    revs = _reviewer_dirs()
    if len(revs) != 2:
        sys.exit(f"agreement needs exactly 2 reviewer folders, found {[r for r, _ in revs]}")
    loaded = []
    for r, folder in revs:
        verdicts, missing, checked, problems = load_reviewer(folder)
        if problems:
            sys.exit(f"reviewer {r} has {len(problems)} problems to fix first "
                     f"(run `python label_review.py status`). First: {problems[0]}")
        loaded.append((r, verdicts, missing, checked))
    return loaded


def compare():
    """Both reviewers' answers over products both checked -> (rows for adjudication, stats)."""
    (ra, va, ma, ca), (rb, vb, mb, cb) = _two_reviewers()
    both = ca & cb
    pairs, rows = [], []
    for key in sorted(k for k in va if k[0] in both):
        (a, ta), (b, tb) = va[key], vb[key]
        pairs.append((a, b))
        if a != b or (a == "wrong" and ta.lower() != tb.lower()):
            rows.append({"type": "name", "label_file": key[0], "position": key[1],
                         "a_verdict": a, "a_text": ta, "b_verdict": b, "b_text": tb})
    miss_both = miss_a = miss_b = 0
    for lf in sorted(both):
        matched, only_a, only_b = match_missing(ma.get(lf, []), mb.get(lf, []))
        miss_both += len(matched)
        miss_a += len(only_a)
        miss_b += len(only_b)
        for (aa, an), (ba, bn) in matched:
            if an.lower() != bn.lower() or aa != ba:
                rows.append({"type": "missing", "label_file": lf, "position": aa,
                             "a_verdict": "missing", "a_text": an, "b_verdict": "missing", "b_text": bn,
                             "b_after": ba})
        for after, name in only_a:
            rows.append({"type": "missing", "label_file": lf, "position": after,
                         "a_verdict": "missing", "a_text": name, "b_verdict": "", "b_text": ""})
        for after, name in only_b:
            rows.append({"type": "missing", "label_file": lf, "position": after,
                         "a_verdict": "", "a_text": "", "b_verdict": "missing", "b_text": name})
    return (ra, rb), both, pairs, rows, (miss_both, miss_a, miss_b), (va, ma, ca), (vb, mb, cb)


ADJ_FIELDS = ["type", "label_file", "position", "b_after", "current_name",
              "a_verdict", "a_text", "b_verdict", "b_text", "final_verdict", "final_text", "notes"]


def agree():
    (ra, rb), both, pairs, rows, (m_both, m_a, m_b), *_ = compare()
    if not pairs:
        sys.exit("no product has been checked by both reviewers yet.")
    labels = {e["label_file"]: e["ingredients"] for e in review_entries()[0]}
    for row in rows:
        if row["type"] == "name":
            row["current_name"] = labels[row["label_file"]][int(row["position"]) - 1]

    # keep decisions already typed into an earlier adjudication.csv
    if os.path.isfile(ADJUDICATION_CSV):
        old = {(r["type"], r["label_file"], r["position"], r["a_text"], r["b_text"]): r
               for r in _read(ADJUDICATION_CSV)}
        for row in rows:
            prev = old.get((row["type"], row["label_file"], str(row["position"]), row["a_text"], row["b_text"]))
            if prev:
                for f in ("final_verdict", "final_text", "notes"):
                    row[f] = prev[f]
    _write(ADJUDICATION_CSV, rows, ADJ_FIELDS)

    po, kappa = cohen_kappa(pairs, VERDICTS)
    ac1 = gwet_ac1(pairs, VERDICTS)
    table = Counter(pairs)
    err_a = sum(a != "ok" for a, _ in pairs)
    err_b = sum(b != "ok" for _, b in pairs)
    err_both = sum(a != "ok" and b != "ok" for a, b in pairs)
    pos_agree = 2 * err_both / (err_a + err_b) if err_a + err_b else float("nan")
    m_union = m_both + m_a + m_b
    lines = [
        f"products checked by both reviewers: {len(both)}   names compared: {len(pairs)}",
        f"raw agreement on verdicts: {po:.2%}",
        f"Cohen's kappa: {kappa:.3f}",
        f"Gwet's AC1: {ac1:.3f}",
        f"positive agreement on 'not ok' (found the same problem names): {pos_agree:.2%} "
        f"({ra} flagged {err_a}, {rb} flagged {err_b}, both {err_both})",
        f"missing names: found by both {m_both}, only {ra} {m_a}, only {rb} {m_b}"
        + (f" -> agreement {m_both / m_union:.2%}" if m_union else ""),
        "",
        f"verdict table (rows = {ra}, columns = {rb}):",
        "            " + "".join(f"{c:>11s}" for c in VERDICTS),
    ]
    lines += [f"{a:>11s} " + "".join(f"{table[(a, b)]:>11d}" for b in VERDICTS) for a in VERDICTS]
    lines += ["", f"disagreements to settle: {len(rows)} -> {ADJUDICATION_CSV}",
              "fill final_verdict (names: ok / wrong / extra / unreadable; missing: add / skip) "
              "and final_text (the correct name, for wrong or add)"]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "agreement.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def apply():
    (ra, rb), both, _, rows, _, (va, ma, ca), (vb, mb, cb) = compare()
    entries, _ = review_entries()
    todo = {e["label_file"] for e in entries}
    if both != todo:
        sys.exit(f"{len(todo - both)} products are not checked by both reviewers yet "
                 "(run `python label_review.py status`).")
    if not os.path.isfile(ADJUDICATION_CSV):
        sys.exit("adjudication.csv not found -- run `python label_review.py agree` first.")
    adj = {(r["type"], r["label_file"], r["position"], r["a_text"], r["b_text"]): r
           for r in _read(ADJUDICATION_CSV)}
    decided = {}
    for row in rows:
        key = (row["type"], row["label_file"], str(row["position"]), row["a_text"], row["b_text"])
        a = adj.get(key)
        if a is None:
            sys.exit("adjudication.csv is out of date -- run `python label_review.py agree` again.")
        fv, ft = a["final_verdict"].strip().lower(), a["final_text"].strip()
        allowed = VERDICTS if row["type"] == "name" else ("add", "skip")
        if fv not in allowed or (fv in ("wrong", "add") and not ft):
            sys.exit(f"adjudication.csv {row['label_file']} #{row['position']} ({row['type']}): "
                     f"final_verdict must be one of {allowed}" + (" with final_text" if fv in ("wrong", "add") else ""))
        decided[key] = (fv, ft)

    spot_verdicts, spot_adds, spot_entries = load_spotfix()

    disputed_names = {(r["label_file"], int(r["position"])): r for r in rows if r["type"] == "name"}
    changes = []
    if os.path.isdir(CORRECTED_DIR) and os.listdir(CORRECTED_DIR):
        sys.exit(f"{CORRECTED_DIR} is not empty -- move it away first.")
    os.makedirs(CORRECTED_DIR, exist_ok=True)
    for e in entries:
        lf = e["label_file"]

        def verdict(pos):
            row = disputed_names.get((lf, pos))
            if row:
                return decided[(row["type"], lf, str(pos), row["a_text"], row["b_text"])]
            return va[(lf, pos)]  # both agree

        adds = []
        matched, only_a, only_b = match_missing(ma.get(lf, []), mb.get(lf, []))
        for (aa, an), (ba, bn) in matched:
            key = ("missing", lf, str(aa), an, bn)
            if key in decided:
                v, t = decided[key]
                if v == "add":
                    adds.append((aa, t))
            else:
                adds.append((aa, an))
        for after, name in only_a:
            v, t = decided[("missing", lf, str(after), name, "")]
            if v == "add":
                adds.append((after, t))
        for after, name in only_b:
            v, t = decided[("missing", lf, str(after), "", name)]
            if v == "add":
                adds.append((after, t))
        changes += _write_corrected(lf, e["ingredients"], verdict, adds, "review")
    for e in spot_entries:
        lf = e["label_file"]
        changes += _write_corrected(lf, e["ingredients"], lambda pos: spot_verdicts.get((lf, pos), ("ok", "")),
                                    spot_adds.get(lf, []), "spotcheck")
    copied = 0
    done = todo | {e["label_file"] for e in spot_entries}
    for f in os.listdir(core.LABEL_DIR):
        if f.startswith("label") and f.endswith(".txt") and f not in done:
            shutil.copy2(os.path.join(core.LABEL_DIR, f), os.path.join(CORRECTED_DIR, f))
            copied += 1
    _write(os.path.join(OUT_DIR, "changes.csv"), changes,
           ["source", "label_file", "position", "change", "old", "new"])
    print(f"rule: removed footnote asterisks / percentages from "
          f"{sum(ch['change'] == 'strip_marks' for ch in changes)} names")
    for src in ("review", "spotcheck"):
        c = Counter(ch["change"] for ch in changes if ch["source"] == src)
        print(f"{src}: fix {c['fix']}, remove {c['remove']}, add {c['add']}")
    print(f"wrote {len(entries)} reviewed + {len(spot_entries)} spot-checked"
          + (f" + {copied} unreviewed" if copied else "") + f" label files to {CORRECTED_DIR}")
    print(f"changes -> {os.path.join(OUT_DIR, 'changes.csv')}")


_PERCENT_RE = re.compile(r"\s*\(\s*\d+(?:[.,]\d+)?\s*%\s*\)|\s*\b\d+(?:[.,]\d+)?\s*%")


def strip_marks(name):
    """Guidelines rule 3: footnote asterisks and percentages printed with a name are not part of it."""
    return re.sub(r"\s{2,}", " ", _PERCENT_RE.sub("", name.replace("*", ""))).strip()


def _write_corrected(lf, ingredients, verdict, adds, source):
    """Write one corrected label file. verdict(pos) -> (verdict, correct_text); adds = [(after, name)]."""
    lines, changes = [], []  # lines: (sort position, name)
    for pos, name in enumerate(ingredients, 1):
        v, t = verdict(pos)
        if v == "extra":
            changes.append({"source": source, "label_file": lf, "position": pos, "change": "remove",
                            "old": name, "new": ""})
            continue
        if v == "wrong" and t:
            changes.append({"source": source, "label_file": lf, "position": pos, "change": "fix",
                            "old": name, "new": t})
            name = t
        lines.append((float(pos), name))
    for k, (after, name) in enumerate(sorted(adds)):
        lines.append((after + 0.5 + k * 1e-3, name))
        changes.append({"source": source, "label_file": lf, "position": after, "change": "add",
                        "old": "", "new": name})
    out = []
    for key, name in sorted(lines):
        clean = strip_marks(name)
        if clean != name:
            changes.append({"source": "rule", "label_file": lf, "position": key, "change": "strip_marks",
                            "old": name, "new": clean})
        out.append(clean)
    with open(os.path.join(CORRECTED_DIR, lf), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return changes


def _spot_entries():
    """The 20 spot-checked products, with the spot-check's per-product records."""
    products = {p["label_file"]: p for p in _read(SPOTCHECK_PRODUCTS)}
    dev = {e["image_file"] for e in core.iter_dataset_entries(["original"], limit=N_DEV)}
    entries = [e for e in core.iter_dataset_entries(["original"])
               if e["status"] == "ok" and e["label_file"] in products]
    for e in entries:
        e["split"] = "dev" if e["image_file"] in dev else "heldout"
        e["n_missing"] = int(products[e["label_file"]]["missing_names"] or 0)
    return entries


def spotfix():
    path = os.path.join(SPOTFIX_DIR, "missing.csv")
    if os.path.isfile(path) and any(r["name"].strip() for r in _read(path)):
        sys.exit(f"{path} already has names filled in -- not overwriting.")
    names = _read(SPOTCHECK_NAMES)
    entries = [e for e in _spot_entries() if e["n_missing"]]
    import cv2

    rows, cards = [], []
    for i, e in enumerate(entries, 1):
        lf = e["label_file"]
        hints = [r["notes"].strip() for r in names if r["label_file"] == lf and r["notes"].strip()]
        hint = f"ต้องกรอก {e['n_missing']} ชื่อ"
        if hints:
            hint += " · โน้ตจากการสุ่มตรวจ (ตำแหน่งไม่แน่นอน ใช้ประกอบเท่านั้น): " + "; ".join(hints)
        for k in range(e["n_missing"]):
            rows.append({"label_file": lf, "after_position": "", "name": "", "notes": hint if k == 0 else ""})
        h, w = cv2.imread(e["image_path"]).shape[:2]
        cards.append((i, e, w, h))
    os.makedirs(SPOTFIX_DIR, exist_ok=True)
    _write(path, rows, MISSING_FIELDS)
    with open(os.path.join(SPOTFIX_DIR, "review.html"), "w", encoding="utf-8") as fh:
        fh.write(_page(f"Spot-check fix ({len(entries)} products)", SPOTFIX_INTRO, SPOTFIX_DIR, cards))
    extras = sum(r["verdict"] in ("extra", "wrong") for r in names)
    print(f"{len(entries)} products, {len(rows)} missing names to fill in -> {path}")
    print(f"open {os.path.join(SPOTFIX_DIR, 'review.html')}")
    print(f"({extras} extra/wrong names from the spot-check are applied automatically by `apply`)")


def load_spotfix():
    """-> (verdicts {(label_file, pos): (verdict, correct_text)}, adds {label_file: [(after, name)]}, entries)"""
    path = os.path.join(SPOTFIX_DIR, "missing.csv")
    entries = _spot_entries()
    if any(e["n_missing"] for e in entries) and not os.path.isfile(path):
        sys.exit("spotcheck_fix/missing.csv not found -- run `python label_review.py spotfix` and fill it in.")
    verdicts = {}
    for r in _read(SPOTCHECK_NAMES):
        v = r["verdict"].strip().lower()
        if v == "wrong" and not r["correct_text"].strip():
            print(f"note: spot-check {r['label_file']} #{r['position']} is 'wrong' without correct_text "
                  "-- kept unchanged")
        verdicts[(r["label_file"], int(r["position"]))] = (v, r["correct_text"].strip())
    n_names = {e["label_file"]: len(e["ingredients"]) for e in entries}
    adds = {}
    for r in _read(path) if os.path.isfile(path) else []:
        lf, name = r["label_file"].strip(), r["name"].strip()
        if not name:
            sys.exit(f"spotcheck_fix/missing.csv: a row for {lf} has no name yet. Fill it in, or delete the row "
                     "and write why in `notes` of another row if the spot-check count was wrong.")
        try:
            after = float(r["after_position"])
        except ValueError:
            sys.exit(f"spotcheck_fix/missing.csv {lf} '{name}': after_position is not a number")
        if lf not in n_names or not 0 <= after <= n_names[lf]:
            sys.exit(f"spotcheck_fix/missing.csv {lf} '{name}': unknown label_file or after_position out of range")
        adds.setdefault(lf, []).append((after, name))
    for e in entries:
        got = len(adds.get(e["label_file"], []))
        if got != e["n_missing"]:
            print(f"note: {e['label_file']} spot-check counted {e['n_missing']} missing names, "
                  f"spotcheck_fix has {got}")
    return verdicts, adds, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--reviewers", nargs="+", default=["A", "B"])
    imp = sub.add_parser("import")
    imp.add_argument("reviewer")
    imp.add_argument("json_file")
    for cmd in ("status", "agree", "spotfix", "apply"):
        sub.add_parser(cmd)
    args = parser.parse_args()
    {"make": lambda: make(args.reviewers), "import": lambda: import_results(args.reviewer, args.json_file),
     "status": status, "agree": agree, "spotfix": spotfix, "apply": apply}[args.cmd]()


if __name__ == "__main__":
    main()
