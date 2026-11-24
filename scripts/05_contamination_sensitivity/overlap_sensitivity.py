"""
overlap_sensitivity.py
======================
Recompute the calibration results of Experiment A (manuscript Table 8, Fig. 4)
on subsets that cannot be affected by train/evaluation overlap:

  all                   every aligned question (point estimates must equal Table 8)
  excluding_flagged     all questions minus those flagged by overlap_check.py
  in_dist_clean         MedMCQA, MedQA, PubMedQA minus the flagged questions
  mmlu_only             the six MMLU subsets (not in the fine-tuning corpus)

It reuses the loader and metrics of calibration_cascade_pilot.py, so the
definitions (15-bin ECE, AUROC with ties = 1/2, AURC, item bootstrap) are
identical to the manuscript.

Usage (from the repository root)
--------------------------------
    python scripts/05_overlap_check/overlap_sensitivity.py --logs results/lm_eval_logs \
        --flagged overlap_out/flagged_items.csv --out overlap_out

Outputs: sensitivity_by_model.csv, sensitivity_paired.csv, SENSITIVITY_REPORT.txt
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# works in the repository layout (scripts/01_calibration_cascade/) and when all
# scripts sit in one folder next to calibration_cascade_pilot.py
for _p in (os.path.join(HERE, "..", "01_calibration_cascade"), HERE, os.getcwd()):
    sys.path.insert(0, _p)
from calibration_cascade_pilot import load_model, latest_files, ece, auroc, aurc, boot_ci  # noqa: E402

IN_DIST = ("medmcqa", "medqa", "pubmedqa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="results/lm_eval_logs")
    ap.add_argument("--models", nargs="*", default=["medilite_q4", "medilite", "base", "openbiollm8b"])
    ap.add_argument("--flagged", required=True, help="flagged_items.csv (overlap_check.py) or contaminated_items.json / not_clean_items.json (contamination_check.py)")
    ap.add_argument("--pairs", nargs="*", default=["medilite:base", "medilite_q4:medilite",
                                                     "medilite:openbiollm8b", "medilite_q4:openbiollm8b"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="overlap_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    report = []
    P = lambda s="": (report.append(s), print(s))

    flagged = set()
    if a.flagged.lower().endswith(".json"):
        # contamination_check.py output: {task: [doc_id, ...]}
        with open(a.flagged) as fh:
            for task, ids in json.load(fh).items():
                flagged.update((task, int(i)) for i in ids)
    else:
        with open(a.flagged) as fh:
            for r in csv.DictReader(fh):
                if r["flagged"] == "1":
                    flagged.add((r["task"], int(r["doc_id"])))

    raw = {}
    for m in a.models:
        d = os.path.join(a.logs, m)
        if not os.path.isdir(d) or not latest_files(d):
            P(f"[skip] {m}: no logs under {d}")
            continue
        raw[m], _ = load_model(d)
    tasks = sorted(set.intersection(*[set(raw[m]) for m in raw]))
    keys = []
    for t in tasks:
        ids = sorted(set.intersection(*[set(raw[m][t]) for m in raw]))
        keys += [(t, i) for i in ids]
    D = {m: np.array([raw[m][t][i][:2] for t, i in keys], dtype=float) for m in raw}
    task_arr = np.array([t for t, _ in keys])
    is_flag = np.array([(t, int(i)) in flagged for t, i in keys])
    is_in = np.array([any(b in t.lower() for b in IN_DIST) for t in task_arr])
    is_mmlu = np.array([t.lower().startswith("mmlu") for t in task_arr])
    unmatched = len(flagged) - int(is_flag.sum())
    P(f"Aligned questions: {len(keys)}; flagged: {int(is_flag.sum())}"
      + (f" ({unmatched} flagged keys not found among aligned questions; check task names)" if unmatched else ""))

    subsets = {"all": np.ones(len(keys), bool), "excluding_flagged": ~is_flag,
               "in_dist_clean": is_in & ~is_flag, "mmlu_only": is_mmlu}

    rows, prow = [], []
    P(f"\n{'model':<14}{'subset':<19}{'n':>6}{'acc':>7}{'conf':>7}{'ECE [95% CI]':>22}{'AUROC [95% CI]':>22}{'AURC':>7}")
    for name, sel in subsets.items():
        idx_all = np.where(sel)[0]
        n = len(idx_all)
        if n == 0:
            continue
        for m, arr in D.items():
            c, y = arr[idx_all, 0], arr[idx_all, 1]
            e_lo, e_hi = boot_ci(lambda i: ece(c[i], y[i]), n, a.bootstrap, rng)
            u_lo, u_hi = boot_ci(lambda i: auroc(c[i], y[i]), n, a.bootstrap, rng)
            r_lo, r_hi = boot_ci(lambda i: aurc(c[i], y[i]), n, a.bootstrap, rng)
            row = [m, name, n, y.mean(), c.mean(), ece(c, y), e_lo, e_hi, auroc(c, y), u_lo, u_hi, aurc(c, y), r_lo, r_hi]
            rows.append(row)
            P(f"{m:<14}{name:<19}{n:>6}{row[3]:>7.3f}{row[4]:>7.3f}"
              f"{f'{row[5]:.3f} [{e_lo:.3f},{e_hi:.3f}]':>22}{f'{row[8]:.3f} [{u_lo:.3f},{u_hi:.3f}]':>22}{row[11]:>7.3f}")
        P()
    with open(os.path.join(a.out, "sensitivity_by_model.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "n", "accuracy", "mean_conf", "ECE", "ECE_lo", "ECE_hi",
                    "AUROC", "AUROC_lo", "AUROC_hi", "AURC", "AURC_lo", "AURC_hi"])
        w.writerows(rows)

    P("=== Paired differences (a - b), item bootstrap 95% CI ===")
    for name, sel in subsets.items():
        idx_all = np.where(sel)[0]
        n = len(idx_all)
        if n == 0:
            continue
        for p in a.pairs:
            x, z = p.split(":")
            if x not in D or z not in D:
                continue
            ca, ya = D[x][idx_all, 0], D[x][idx_all, 1]
            cb, yb = D[z][idx_all, 0], D[z][idx_all, 1]
            for label, fn in (("acc", lambda i: ya[i].mean() - yb[i].mean()),
                              ("ECE", lambda i: ece(ca[i], ya[i]) - ece(cb[i], yb[i])),
                              ("AUROC", lambda i: auroc(ca[i], ya[i]) - auroc(cb[i], yb[i])),
                              ("AURC", lambda i: aurc(ca[i], ya[i]) - aurc(cb[i], yb[i]))):
                pt = fn(np.arange(n)); lo, hi = boot_ci(fn, n, a.bootstrap, rng)
                prow.append([name, x, z, label, n, pt, lo, hi])
                P(f"  {name:<19}{x} - {z}  {label:<6}{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    with open(os.path.join(a.out, "sensitivity_paired.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "a", "b", "metric", "n", "difference", "ci_lo", "ci_hi"])
        w.writerows(prow)

    P("\nCheck: with the default --models order, --bootstrap 2000 and --seed 1234, the 'all' rows must "
      "reproduce manuscript Table 8 exactly, intervals included (e.g. medilite_q4 ECE 0.092 [0.082, 0.102], "
      "AUROC 0.758 [0.746, 0.769]). If they do not, the logs or settings differ from the paper's run.")
    open(os.path.join(a.out, "SENSITIVITY_REPORT.txt"), "w").write("\n".join(report) + "\n")
    print(f"\nwritten to {a.out}/")


if __name__ == "__main__":
    main()
