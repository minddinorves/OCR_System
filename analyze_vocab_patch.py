"""Full analysis of the vocabulary patch for thesis sections 3.6 and 4.9.
Uses saved OCR text only (no OCR re-run); fuzzy matching as shipped
(ocr_paddle_fuzzy.correct_with_vocabulary, threshold 85).

  1. coverage    how much of the label vocabulary the main INCI database
                 covers, and how much each patch adds (overall, Dev, Held-out)
  2. composition the shipped Dev patch: product-count distribution, overlap
                 with the legacy 199-name patch, most common names
  3. dev_cutoff  cutoff k chosen the protocol-valid way: Dev only, every Dev
                 image scored with a patch built from the OTHER 4 Dev folds
  4. heldout     report-only: patches built from the 40 Dev labels with
                 k = 1/2/3, plus no patch and the legacy patch, scored on the
                 176 Held-out products in all 5 scenarios
  5. cv216       sensitivity only (uses Held-out labels, so NOT used for any
                 choice): 5-fold CV over all 216 products, k = 1..20
  6. legacy_code the same patches under the pre-fix fuzzy code (per line,
                 case-sensitive) on Held-out "original", showing why the patch
                 had no effect before the fix

Outputs: results/vocab_patch/*.csv

Usage:
  python analyze_vocab_patch.py
"""
import csv
import glob
import json
import os
from collections import Counter

import numpy as np
from rapidfuzz import fuzz, process

import ocr_core as core
import ocr_paddle_fuzzy as fz
from build_vocabulary_patch import build_patch

from compute_bootstrap_ci import ci_mean, ci_paired_delta
from compute_ingredient_metrics import set_prf1, tokenize

THRESHOLD = 85.0
N_DEV, N_FOLDS = 40, 5
OUT_DIR = os.path.join(core.RESULTS_DIR, "vocab_patch")
SCENARIOS = ("original", "indoor", "low_light", "blur", "night")
FINAL = [os.path.join(core.RESULTS_DIR, f"ocr_paddle_fuzzy_FINAL_{w}_full_validation.csv")
         for w in ("original", "augmented")]


def vocab_of(patch):
    """main DB + patch, de-duplicated exactly like core.load_inci_vocabulary."""
    out, seen = [], set()
    for name in core._load_names(core.INCI_DB_PATH) + list(patch):
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def score(hyp, ref):
    p, r, f1 = set_prf1(tokenize(hyp), tokenize(ref))
    return {"set_f1": f1, "set_precision": p, "set_recall": r, "cer": core.compute_cer(hyp, ref)}


def summarize(results, reference, files):
    """results: list of (label, {file: metrics}). Returns rows with CI and paired delta vs reference."""
    ref = dict(results)[reference]
    rows = []
    for label, per in results:
        out = {"setting": label, "n_images": len(files)}
        for m in ("set_f1", "set_precision", "set_recall", "cer"):
            mean, lo, hi = ci_mean([per[f][m] for f in files])
            out[m], out[f"{m}_ci95"] = round(mean, 4), f"[{lo:.4f}, {hi:.4f}]"
        if label != reference:
            for m in ("set_f1", "cer"):
                d, lo, hi = ci_paired_delta([ref[f][m] for f in files], [per[f][m] for f in files])
                out[f"{m}_delta_vs_{reference}"] = round(d, 4)
                out[f"{m}_delta_ci95"] = f"[{lo:+.4f}, {hi:+.4f}]"
                out[f"{m}_delta_significant"] = lo > 0 or hi < 0
        rows.append(out)
    return rows


def write(name, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    core.write_csv(os.path.join(OUT_DIR, f"{name}.csv"), rows, fieldnames=fields)


def show(title, rows):
    print(f"\n=== {title} ===")
    for r in rows:
        d = [k for k in r if k.startswith("set_f1_delta_vs")]
        delta = (f"dF1 {r[d[0]]:+.4f} {r['set_f1_delta_ci95']}{' *' if r['set_f1_delta_significant'] else ''}"
                 if d else "(reference)")
        print(f"  {r.get('dataset', ''):10} {r['setting']:28} n={r['n_images']:3d} F1={r['set_f1']:.4f} "
              f"{r['set_f1_ci95']} CER={r['cer']:.4f}  {delta}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    main_l = {n.lower() for n in core._load_names(core.INCI_DB_PATH)}
    dev = list(core.iter_dataset_entries(["original"], limit=N_DEV))
    dev_labels = {e["label_file"] for e in dev}
    all_paths = sorted(glob.glob(os.path.join(core.LABEL_DIR, "label*.txt")))
    names_of = {os.path.basename(p): [n.strip() for n in core.load_reference_text(p)[0]] for p in all_paths}
    shipped = core._load_names(core.INCI_PATCH_PATH)
    legacy = core._load_names(core.LEGACY_PATCH_PATH)

    # ---- 1. coverage ----------------------------------------------------
    cov = []
    for group, labels in (("all 216", list(names_of)), ("dev 40", sorted(dev_labels)),
                          ("heldout 176", sorted(set(names_of) - dev_labels))):
        occ = [n.lower() for lf in labels for n in names_of[lf]]
        uniq = set(occ)
        for vname, patch in (("main DB only", []), ("main DB + shipped Dev patch (399)", shipped),
                             ("main DB + legacy patch (199)", legacy)):
            v = main_l | {n.lower() for n in patch}
            cov.append({"labels": group, "vocabulary": vname, "name_occurrences": len(occ),
                        "unique_names": len(uniq),
                        "occurrences_exactly_in_vocab_pct": round(100 * sum(n in v for n in occ) / len(occ), 1),
                        "unique_names_exactly_in_vocab_pct": round(100 * sum(n in v for n in uniq) / len(uniq), 1)})
    write("coverage", cov)
    print("=== 1. coverage (exact, case-insensitive) ===")
    for c in cov:
        print(f"  {c['labels']:12} {c['vocabulary']:34} occurrences {c['occurrences_exactly_in_vocab_pct']:5.1f}%"
              f"  unique {c['unique_names_exactly_in_vocab_pct']:5.1f}%  (n_occ={c['name_occurrences']}, "
              f"n_unique={c['unique_names']})")

    # ---- 2. composition --------------------------------------------------
    dev_count = Counter(n.lower() for lf in dev_labels for n in {x.lower() for x in names_of[lf]})
    hist = Counter(dev_count[n.lower()] for n in shipped)
    held_names = {n.lower() for lf in set(names_of) - dev_labels for n in names_of[lf]}
    comp = [{"item": f"names in exactly {k} Dev product(s)", "value": hist[k]} for k in sorted(hist)]
    comp += [{"item": "total names in shipped patch", "value": len(shipped)},
             {"item": "also in legacy 199-name patch", "value": len({n.lower() for n in shipped} & {n.lower() for n in legacy})},
             {"item": "legacy names missing from shipped patch", "value": len({n.lower() for n in legacy} - {n.lower() for n in shipped})},
             {"item": "shipped names that also occur in >=1 Held-out label", "value": sum(n.lower() in held_names for n in shipped)}]
    comp += [{"item": f"top name #{i + 1}", "value": f"{n} ({dev_count[n.lower()]} Dev products)"} for i, n in enumerate(shipped[:15])]
    write("composition", comp)
    print("\n=== 2. shipped patch composition ===")
    for c in comp:
        print(f"  {c['item']:52} {c['value']}")

    # saved OCR text
    rows = []
    for path in FINAL:
        with open(path, newline="", encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    orig = {r["label_file"]: r for r in rows if r["dataset"] == "original"}

    # ---- 3. Dev cutoff with cross-fitting --------------------------------
    dev_ks = [None, 1, 2, 3, 4, 5]
    per = {k: {} for k in dev_ks}
    for i in range(N_FOLDS):
        fold = dev[i::N_FOLDS]
        others = [e["label_path"] for e in dev if e not in fold]
        for k in dev_ks:
            vocab = vocab_of([] if k is None else build_patch(k, others))
            for e in fold:
                r = orig[e["label_file"]]
                per[k][e["label_file"]] = score(fz.correct_with_vocabulary([r["ocr_text_raw"]], vocab, THRESHOLD),
                                                r["reference_text"])
    lab = lambda k: "no patch" if k is None else f"k>={k} (of 32 Dev labels)"  # noqa: E731
    res = summarize([(lab(k), per[k]) for k in dev_ks], lab(1), sorted(dev_labels))
    write("dev_cutoff_crossfit", res)
    show("3. cutoff k on Dev, 5-fold cross-fitted (protocol-valid)", res)

    # ---- 4. Held-out report-only ----------------------------------------
    dev_paths = [e["label_path"] for e in dev]
    patches = {"no patch": [], "dev k>=1 (shipped)": build_patch(1, dev_paths),
               "dev k>=2": build_patch(2, dev_paths), "dev k>=3": build_patch(3, dev_paths),
               "legacy k>=5 of all 216 (leaks)": legacy}
    assert [n.lower() for n in patches["dev k>=1 (shipped)"]] == [n.lower() for n in shipped]
    vocabs = {name: vocab_of(p) for name, p in patches.items()}
    held_rows = []
    for scenario in SCENARIOS:
        sel = [r for r in rows if r["dataset"] == scenario and r["label_file"] not in dev_labels]
        per = {name: {r["label_file"]: score(fz.correct_with_vocabulary([r["ocr_text_raw"]], v, THRESHOLD),
                                             r["reference_text"]) for r in sel} for name, v in vocabs.items()}
        out = summarize([(f"{name} [{len(patches[name])}]", per[name]) for name in patches],
                        f"dev k>=1 (shipped) [{len(shipped)}]", sorted(per["no patch"]))
        for o in out:
            o["dataset"] = scenario
        held_rows += out
    write("heldout_patches", held_rows)
    show("4. patches on Held-out (report only, never used to choose)", held_rows)

    # ---- 5. CV over all 216 (sensitivity) --------------------------------
    cv_ks = [None, 1, 2, 3, 4, 5, 7, 10, 15, 20]
    files = sorted(orig)
    order = np.random.default_rng(0).permutation(len(files))
    folds = [[files[i] for i in idx] for idx in np.array_split(order, 5)]
    per = {k: {} for k in cv_ks}
    sizes = {k: [] for k in cv_ks}
    for test in folds:
        train_paths = [os.path.join(core.LABEL_DIR, lf) for lf in files if lf not in set(test)]
        for k in cv_ks:
            kk = None if k is None else max(1, round(k * len(train_paths) / len(files)))
            patch = [] if k is None else build_patch(kk, train_paths)
            sizes[k].append(len(patch))
            vocab = vocab_of(patch)
            for lf in test:
                r = orig[lf]
                per[k][lf] = score(fz.correct_with_vocabulary([r["ocr_text_raw"]], vocab, THRESHOLD), r["reference_text"])
    lab = lambda k: "no patch" if k is None else f"k>={k} of 216 (scaled)"  # noqa: E731
    res = summarize([(lab(k), per[k]) for k in cv_ks], lab(5), files)
    for k, r in zip(cv_ks, res):
        r["patch_size_full_216"] = 0 if k is None else len(build_patch(k))
        r["patch_size_mean_per_fold"] = round(float(np.mean(sizes[k])), 1)
        r["effective_k_per_fold"] = "" if k is None else max(1, round(k * 173 / 216))
    write("cv216_cutoff", res)
    show("5. cutoff k, 5-fold CV over all 216 (sensitivity only)", res)

    # ---- 6. pre-fix fuzzy code ------------------------------------------
    with open(os.path.join(core.RESULTS_DIR, "sweep_cache", "pad_0.25.json"), encoding="utf-8") as fh:
        cache = json.load(fh)
    with open(os.path.join(core.RESULTS_DIR, "_before_fuzzy_fix", "ocr_paddle_fuzzy.py"), encoding="utf-8") as fh:
        old_ns = {"__name__": "old_fuzzy"}
        exec(compile(fh.read(), "old_fuzzy", "exec"), old_ns)  # noqa: S102 - our own pre-fix file, for comparison
    sel = [r for lf, r in orig.items() if lf not in dev_labels]
    per = {name: {r["label_file"]: score(old_ns["correct_with_vocabulary"](cache[r["image_file"]]["lines"], v, 90.0),
                                         r["reference_text"]) for r in sel}
           for name, v in vocabs.items() if name in ("no patch", "dev k>=1 (shipped)", "legacy k>=5 of all 216 (leaks)")}
    res = summarize(list(per.items()), "dev k>=1 (shipped)", sorted(per["no patch"]))
    write("prefix_code_heldout_original", res)
    show("6. pre-fix fuzzy code (per line, case-sensitive, threshold 90), Held-out original", res)


if __name__ == "__main__":
    main()
