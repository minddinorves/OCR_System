"""Automated quality audit of the ground-truth label files
(test_SkinSafe/label/label*.txt) -- supports the description of the answer
key in thesis section 3.3.1. Checks formatting problems and spelling
inconsistencies; it cannot check labels against the images themselves.

Outputs results/label_audit.csv (one row per flagged item) and prints a summary.

Usage:
  python audit_labels.py
"""
import glob
import os
import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

import ocr_core as core

STRAY_RE = re.compile(r"[*+†‡#^~]|\(\d\)|\s{2,}")


def main():
    main_db = {n.lower() for n in core._load_names(core.INCI_DB_PATH)}
    paths = sorted(glob.glob(os.path.join(core.LABEL_DIR, "label*.txt")))
    flags = []
    numbered_files, occurrences = 0, 0
    spellings = defaultdict(Counter)  # lower -> exact spellings
    for path in paths:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            raw_lines = [l.rstrip("\n") for l in fh]
        if any("\t" in l and l.split("\t", 1)[0].strip().isdigit() for l in raw_lines):
            numbered_files += 1
        ingredients, _ = core.load_reference_text(path)
        occurrences += len(ingredients)
        seen = Counter(i.lower() for i in ingredients)
        for ing, c in seen.items():
            if c > 1:
                flags.append({"label_file": name, "issue": "duplicate_in_label", "text": ing, "detail": f"x{c}"})
        for ing in ingredients:
            spellings[ing.strip().lower()][ing.strip()] += 1
            if STRAY_RE.search(ing):
                flags.append({"label_file": name, "issue": "stray_symbol", "text": ing, "detail": ""})
            if ing.count("(") != ing.count(")"):
                flags.append({"label_file": name, "issue": "unbalanced_parenthesis", "text": ing, "detail": ""})
            if len(ing) > 80 or ("," in ing and not re.search(r"\d,\d", ing)):
                flags.append({"label_file": name, "issue": "possibly_merged_names", "text": ing, "detail": ""})

    # near-identical spellings of what is probably one ingredient, across labels
    names = sorted(spellings)
    variant_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(len(a) - len(b)) > 3:
                continue
            score = fuzz.ratio(a, b)
            if score >= 95 and re.sub(r"[^a-z0-9]", "", a) == re.sub(r"[^a-z0-9]", "", b):
                variant_pairs.append((a, b, score))
                flags.append({"label_file": "(across labels)", "issue": "spelling_variant",
                              "text": f"{a}  |  {b}", "detail": "punctuation/spacing only"})

    in_db = sum(c for n, sp in spellings.items() for c in [sum(sp.values())] if n in main_db)
    core.write_csv(os.path.join(core.RESULTS_DIR, "label_audit.csv"), flags,
                   fieldnames=["label_file", "issue", "text", "detail"])

    by_issue = Counter(f["issue"] for f in flags)
    print(f"label files: {len(paths)}  ingredient occurrences: {occurrences}  unique (case-insensitive): {len(spellings)}")
    print(f"files still using legacy 'N<TAB>name' numbering: {numbered_files}")
    print(f"occurrences found in main INCI DB: {in_db}/{occurrences} ({in_db / occurrences:.1%})")
    for issue, n in by_issue.most_common():
        print(f"  {issue:24s} {n}")
    for issue in ("stray_symbol", "unbalanced_parenthesis", "possibly_merged_names", "duplicate_in_label"):
        ex = [f for f in flags if f["issue"] == issue][:5]
        for f in ex:
            print(f"    e.g. {issue}: {f['label_file']}: {f['text']!r} {f['detail']}")
    for a, b, s in variant_pairs[:8]:
        print(f"    e.g. spelling_variant: {a!r} vs {b!r}")


if __name__ == "__main__":
    main()
