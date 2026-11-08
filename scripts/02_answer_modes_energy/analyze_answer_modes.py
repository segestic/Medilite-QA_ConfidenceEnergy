"""
analyze_answer_modes.py
=======================
Summarise the per-item rows written by energy_per_item.py.

For each (model, mode): accuracy (item-pooled and macro over tasks), parse
failures, mean prompt/generated tokens, seconds and joules per item (gross and
idle-subtracted), and accuracy per kJ, with item-bootstrap 95% CIs. Energy and
time are averaged over repetitions per item first; correctness is taken from
repetition 1 (greedy decoding is deterministic; disagreements between
repetitions are counted and reported).

Also:
  * paired comparisons between modes of the same model (exact McNemar test and
    bootstrap CI of the accuracy difference);
  * confidence-gated chain-of-thought (secondary analysis): answer by option
    log-likelihood when its confidence >= tau, otherwise use the CoT answer;
  * the measured per-item energies to pass to calibration_cascade_pilot.py.

Usage
-----
    python analyze_answer_modes.py --csv energy_items.csv --small medilite_q4 --large openbiollm8b_q4 --out modes_out
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def boot(fn, n, B, rng):
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="energy_items.csv")
    ap.add_argument("--small", default="medilite_q4")
    ap.add_argument("--large", default="openbiollm8b_q4")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="modes_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = []
    P = lambda s="": (report.append(s), print(s))

    rows = list(csv.DictReader(open(args.csv)))
    # (model, mode) -> item -> list of rows
    G = defaultdict(lambda: defaultdict(list))
    for r in rows:
        G[(r["model"], r["mode"])][(r["task"], r["doc_id"])].append(r)

    per = {}  # (model, mode) -> dict of arrays aligned to sorted item keys
    for key, items in G.items():
        ks = sorted(items)
        def avg(f):
            return np.array([np.mean([float(x[f]) for x in items[k]]) for k in ks])
        r1 = [min(items[k], key=lambda x: int(x["rep"])) for k in ks]
        flips = sum(len({x["pred"] for x in items[k]}) > 1 for k in ks)
        per[key] = dict(keys=ks, task=np.array([k[0] for k in ks]),
                        correct=np.array([float(x["correct"]) for x in r1]),
                        parse=np.array([float(x["parse_ok"]) for x in r1]),
                        conf=np.array([float(x["conf"]) for x in r1]),
                        J=avg("joules"), Jn=avg("joules_net"), sec=avg("seconds"),
                        ptok=avg("prompt_tokens"), gtok=avg("gen_tokens"), flips=flips,
                        reps=int(np.mean([len(items[k]) for k in ks])))

    P("=== Accuracy, time and energy per item ===")
    hdr = ["model", "mode", "n", "reps", "acc", "acc_lo", "acc_hi", "acc_macro", "parse_fail_%", "rep_disagree",
           "prompt_tok", "gen_tok", "sec", "J", "J_lo", "J_hi", "J_net", "J_net_lo", "J_net_hi", "acc_per_kJ"]
    table = []
    for (m, md), d in sorted(per.items()):
        n = len(d["keys"])
        y, J, Jn = d["correct"], d["J"], d["Jn"]
        macro = np.mean([y[d["task"] == t].mean() for t in np.unique(d["task"])])
        a_lo, a_hi = boot(lambda i: y[i].mean(), n, args.bootstrap, rng)
        j_lo, j_hi = boot(lambda i: J[i].mean(), n, args.bootstrap, rng)
        n_lo, n_hi = boot(lambda i: Jn[i].mean(), n, args.bootstrap, rng)
        table.append([m, md, n, d["reps"], y.mean(), a_lo, a_hi, macro, 100 * (1 - d["parse"].mean()), d["flips"],
                      d["ptok"].mean(), d["gtok"].mean(), d["sec"].mean(), J.mean(), j_lo, j_hi,
                      Jn.mean(), n_lo, n_hi, 1000 * y.mean() / J.mean()])
    with open(os.path.join(args.out, "modes_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(table)
    P(f"{'model':<16}{'mode':<8}{'n':>5}{'acc [95% CI]':>22}{'macro':>7}{'parsefail%':>8}{'p_tok':>7}{'g_tok':>7}"
      f"{'s/item':>8}{'J/item [95% CI]':>24}{'J_net':>8}{'acc/kJ':>8}")
    for t in table:
        P(f"{t[0]:<16}{t[1]:<8}{t[2]:>5}{t[4]:>7.3f} [{t[5]:.3f},{t[6]:.3f}]{t[7]:>7.3f}{t[8]:>8.1f}{t[10]:>7.0f}"
          f"{t[11]:>7.0f}{t[12]:>8.2f}{t[13]:>9.1f} [{t[14]:.1f},{t[15]:.1f}]{t[16]:>8.1f}{t[19]:>8.2f}")
        if t[9]:
            P(f"   note: {t[9]} items changed prediction between repetitions")

    # paired comparisons within a model
    P("\n=== Paired comparisons between modes (same items; exact McNemar) ===")
    prow = []
    models = sorted({m for m, _ in per})
    for m in models:
        modes = sorted(md for mm, md in per if mm == m)
        for i, a in enumerate(modes):
            for b in modes[i + 1:]:
                A, B = per[(m, a)], per[(m, b)]
                common = sorted(set(A["keys"]) & set(B["keys"]))
                ia = [A["keys"].index(k) for k in common]; ib = [B["keys"].index(k) for k in common]
                ya, yb = A["correct"][ia], B["correct"][ib]
                n01, n10 = int(((ya == 0) & (yb == 1)).sum()), int(((ya == 1) & (yb == 0)).sum())
                p = binomtest(n10, n01 + n10).pvalue if n01 + n10 else 1.0
                diff = 100 * (ya.mean() - yb.mean())
                lo, hi = boot(lambda j: 100 * (ya[j].mean() - yb[j].mean()), len(common), args.bootstrap, rng)
                ratio = A["J"][ia].mean() / B["J"][ib].mean()
                prow.append([m, a, b, len(common), diff, lo, hi, n10, n01, p, ratio])
                P(f"  {m}: {a} - {b}  acc {diff:+.1f} pp [{lo:+.1f}, {hi:+.1f}]  "
                  f"discordant {n10}/{n01}  McNemar p={p:.3g}  energy ratio {a}/{b} = {ratio:.2f}")
    with open(os.path.join(args.out, "paired_modes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode_a", "mode_b", "n", "acc_diff_pp", "ci_lo", "ci_hi", "a_only_correct",
                    "b_only_correct", "mcnemar_p", "energy_ratio_a_over_b"])
        w.writerows(prow)

    # confidence-gated CoT (secondary)
    P("\n=== Confidence-gated chain-of-thought (secondary analysis) ===")
    gated = {}
    for m in models:
        if (m, "loglik") not in per or (m, "cot") not in per:
            continue
        L, C = per[(m, "loglik")], per[(m, "cot")]
        common = sorted(set(L["keys"]) & set(C["keys"]))
        il = [L["keys"].index(k) for k in common]; ic = [C["keys"].index(k) for k in common]
        conf, yl, yc = L["conf"][il], L["correct"][il], C["correct"][ic]
        El, Ec = L["J"][il], C["J"][ic]
        P(f"{m}: n={len(common)}; always loglik acc {yl.mean():.3f} at {El.mean():.1f} J; "
          f"always CoT acc {yc.mean():.3f} at {Ec.mean():.1f} J; gated cost = loglik J + CoT J for escalated items")
        P(f"  {'CoT share':>10}{'acc':>8}{'J/item':>9}")
        res = []
        for share in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            k = int(round(share * len(common)))
            use_cot = np.zeros(len(common), bool)
            use_cot[np.argsort(conf, kind="mergesort")[:k]] = True
            acc = np.where(use_cot, yc, yl).mean()
            en = (El + use_cot * Ec).mean()
            res.append(dict(cot_share=share, accuracy=float(acc), joules=float(en)))
            P(f"  {share:>10.0%}{acc:>8.3f}{en:>9.1f}")
        P("  (thresholds here are fixed shares of the lowest-confidence items on the same items, so this is descriptive, not a held-out test)")
        gated[m] = res

    # energies for the cascade pilot
    out = {"gated_cot": gated}
    if (args.small, "loglik") in per and (args.large, "loglik") in per:
        S, Lg = per[(args.small, "loglik")], per[(args.large, "loglik")]
        out["cascade_energy"] = {"small_J": float(S["J"].mean()), "large_J": float(Lg["J"].mean()),
                                 "small_J_net": float(S["Jn"].mean()), "large_J_net": float(Lg["Jn"].mean())}
        P("\n=== Measured energies for calibration_cascade_pilot.py (loglik mode) ===")
        P(f"  gross: --energy-small {S['J'].mean():.3f} --energy-large {Lg['J'].mean():.3f}  "
          f"(break-even coverage {S['J'].mean() / Lg['J'].mean():.3f})")
        P(f"  net:   --energy-small {S['Jn'].mean():.3f} --energy-large {Lg['Jn'].mean():.3f}  "
          f"(break-even coverage {S['Jn'].mean() / Lg['Jn'].mean():.3f})")
        P("  Use the convention (gross or idle-subtracted) that matches the MediLITE-QA paper, and state it.")
    json.dump(out, open(os.path.join(args.out, "modes_extra.json"), "w"), indent=2)
    open(os.path.join(args.out, "REPORT_modes.txt"), "w").write("\n".join(report) + "\n")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
