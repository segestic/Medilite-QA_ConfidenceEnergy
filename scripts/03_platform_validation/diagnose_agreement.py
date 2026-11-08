"""
diagnose_agreement.py
=====================
Why do the laptop (CPU) option-scoring predictions differ from the logged
Q4 predictions on some items?

The scoring code in energy_per_item.py follows the Q4 scorer in
run_all_one_session.py (same lm-eval tokenisation, no BOS, same logit read).
The remaining difference is the hardware: the logged Q4 results were scored on a
T4 GPU (CUDA kernels, n_ctx 4096), the laptop runs use CPU kernels (n_ctx 2048).
Different kernels give slightly different log-likelihoods (the Kaggle reference
check allowed up to 0.5 nats), which can flip items whose top two options are
nearly tied. If that is the cause, the items that changed should be near-ties in
the logged scores, and accuracy should not move systematically.

Rule fixed before looking at the result: the differences are attributed to
numerical near-ties if every changed item has a logged top-2 margin below
--margin (default 0.5 nats, the bound used by the Kaggle reference check) and
the exact McNemar test of logged vs laptop correctness is not significant
(p >= 0.05). Otherwise the scoring needs investigating.

Usage
-----
    python diagnose_agreement.py --logs /path/to/lm_eval_logs --items items.jsonl \
        --csv energy_items.csv --out agreement_out
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy.stats import binomtest

DEFAULT_MAP = ["medilite_q4=medilite_q4", "base_q4=base_q4", "openbiollm8b_q4=openbiollm8b"]


def task_of(path):
    name = os.path.basename(path)[len("samples_"):-len(".jsonl")]
    return re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", name)


def latest_files(d):
    best = {}
    for f in glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True):
        t = task_of(f)
        if t not in best or os.path.basename(f) > os.path.basename(best[t]):
            best[t] = f
    return best


def first_number(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def load_lls(folder):
    out = {}
    for task, f in latest_files(folder).items():
        for line in open(f):
            try:
                r = json.loads(line)
                out[(task, str(r["doc_id"]))] = np.array([first_number(x) for x in r["resps"]])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--csv", default="energy_items.csv")
    ap.add_argument("--map", nargs="+", default=DEFAULT_MAP, help="label=log_folder pairs")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", default="agreement_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = []
    P = lambda s="": (report.append(s), print(s))

    gold = {(it["task"], str(it["doc_id"])): it["gold"] for it in map(json.loads, open(args.items))}
    new = {}
    for r in csv.DictReader(open(args.csv)):
        if r["mode"] == "loglik" and r["rep"] == "1":
            new[(r["model"], r["task"], str(r["doc_id"]))] = (int(r["pred"]), float(r["conf"]))

    changed_rows = []
    for pair in args.map:
        label, folder = pair.split("=")
        lls = load_lls(os.path.join(args.logs, folder))
        keys = [k for k in gold if (label,) + k in new and k in lls]
        if not keys:
            P(f"{label}: no overlapping items with '{folder}', skipped")
            continue
        g = np.array([gold[k] for k in keys])
        p_new = np.array([new[(label,) + k][0] for k in keys])
        conf_new = np.array([new[(label,) + k][1] for k in keys])
        p_log = np.array([int(np.argmax(lls[k])) for k in keys])
        marg = np.array([np.diff(np.sort(lls[k])[-2:])[0] for k in keys])
        diff = p_new != p_log
        y_new, y_log = (p_new == g), (p_log == g)
        n01, n10 = int((~y_log & y_new).sum()), int((y_log & ~y_new).sum())
        pval = binomtest(n10, n01 + n10).pvalue if n01 + n10 else 1.0

        P(f"=== {label}  vs logged '{folder}' ===")
        P(f"  items {len(keys)}; same prediction {int((~diff).sum())} ({100 * (~diff).mean():.1f}%); changed {int(diff.sum())}")
        P(f"  accuracy: logged {y_log.mean():.3f}, laptop {y_new.mean():.3f} "
          f"(laptop-only correct {n01}, logged-only correct {n10}, exact McNemar p={pval:.3g})")
        P(f"  logged top-2 margin (nats), median: unchanged items {np.median(marg[~diff]):.3f}, "
          f"changed items {np.median(marg[diff]) if diff.any() else float('nan'):.3f}")
        if diff.any():
            P(f"  changed items: max logged margin {marg[diff].max():.3f}; "
              f"{int((marg[diff] < args.margin).sum())}/{int(diff.sum())} below {args.margin} nats")
        P(f"  share of all items with logged margin < {args.margin} nats: {100 * (marg < args.margin).mean():.1f}%")
        if "openbiollm8b" in folder and not folder.endswith("_q4"):
            P("  note: reference is fp16, so changes mix quantization with CPU/GPU numerics")
        else:
            ok = (not diff.any() or marg[diff].max() < args.margin) and pval >= 0.05
            P(f"  VERDICT: {'consistent with numerical near-ties (CPU vs GPU kernels)' if ok else 'NOT explained by near-ties -- investigate the scoring'}")
        P()
        for k, d, m, pn, pl, c in zip(keys, diff, marg, p_new, p_log, conf_new):
            if d:
                changed_rows.append([label, folder, k[0], k[1], gold[k], pl, pn, f"{m:.4f}", f"{c:.4f}"])

    with open(os.path.join(args.out, "changed_items.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ref_folder", "task", "doc_id", "gold", "logged_pred", "laptop_pred",
                    "logged_margin_nats", "laptop_conf"])
        w.writerows(changed_rows)
    open(os.path.join(args.out, "REPORT_agreement.txt"), "w").write("\n".join(report) + "\n")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
