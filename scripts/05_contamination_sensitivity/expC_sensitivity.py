"""
expC_sensitivity.py
===================
Sensitivity analyses for Experiment C (answer modes on the laptop processor).
Recomputes the existing measurements; no model is run.

Part 1. Contamination sensitivity
    Accuracy, tokens, time and gross/net energy per question for every model and
    answer mode (manuscript Table 10), and the paired comparisons between the three
    answer modes of MediLITE-QA (Table 11), on three question sets:
        all            the 198 measured questions
        excl_dirty     without the questions in contaminated_items.json
        excl_notclean  without the questions in not_clean_items.json
    The 'all' rows must reproduce Tables 10 and 11.

Part 2. Selective reasoning
    (a) Confidence-gated reasoning, as in Fig. 5: the least confident fraction of
        questions under option scoring is answered again with step-by-step reasoning;
        the reasoning answer replaces the scored one and both passes are paid for.
    (b) Paired accuracy of option scoring and reasoning within each option-scoring
        confidence quartile, with an exact McNemar test per quartile.
    (c) Oracle upper bound: accuracy if reasoning were used exactly on the questions
        where it is correct and option scoring is not.
    These analyses use the same questions on which any threshold would be chosen,
    so they are descriptive.

Conventions (as in the manuscript): correctness from repetition 1; energy is the
mean over repetitions of each question; item-level percentile bootstrap,
2,000 resamples, seed 1234; exact two-sided McNemar tests, Holm correction
across the three mode comparisons.

Usage
-----
    python expC_sensitivity.py --csv energy_items_final.csv \
        --dirty contamination_out/contaminated_items.json \
        --notclean contamination_out/not_clean_items.json --out expC_sensitivity_out

Outputs (in --out): REPORT_expC_sensitivity.txt, modes_by_subset.csv,
paired_by_subset.csv, gated_reasoning.csv, reasoning_by_confidence_quartile.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from math import comb

import numpy as np

SMALL = "medilite_q4"
MODES = ("loglik", "letter", "cot")
MODE_NAME = {"loglik": "Option scoring", "letter": "Letter generation", "cot": "Step-by-step reasoning"}
PAIRS = (("letter", "loglik"), ("cot", "loglik"), ("cot", "letter"))
GATE_FRACTIONS = (0.0, 0.10, 0.20, 0.30, 0.50, 1.0)


# ------------------------------------------------------------------ statistics
def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci(stat, n, B, seed):
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ------------------------------------------------------------------ data
def load_mask(path):
    if not path:
        return set()
    with open(path) as fh:
        return {(task, str(i)) for task, ids in json.load(fh).items() for i in ids}


def load_csv(path):
    """-> {(model, mode): {(task, doc_id): record}} with rep-1 correctness and
    repetition-averaged energy and time."""
    reps = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            reps[(r["model"], r["mode"])][(r["task"], r["doc_id"])].append(r)
    data = {}
    for key, items in reps.items():
        out = {}
        for q, rows in items.items():
            first = min(rows, key=lambda r: int(r["rep"]))
            out[q] = dict(
                correct=int(first["correct"]), parse_ok=int(first["parse_ok"]),
                conf=float(first["conf"]) if first["conf"] not in ("", None) else float("nan"),
                prompt_tokens=float(np.mean([float(r["prompt_tokens"]) for r in rows])),
                gen_tokens=float(np.mean([float(r["gen_tokens"]) for r in rows])),
                seconds=float(np.mean([float(r["seconds"]) for r in rows])),
                joules=float(np.mean([float(r["joules"]) for r in rows])),
                joules_net=float(np.mean([float(r["joules_net"]) for r in rows])),
                n_reps=len(rows))
        data[key] = out
    return data


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="energy_items_final.csv")
    ap.add_argument("--dirty", default="contamination_out/contaminated_items.json")
    ap.add_argument("--notclean", default="contamination_out/not_clean_items.json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="expC_sensitivity_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = []

    def P(s=""):
        report.append(s)
        print(s)

    data = load_csv(a.csv)
    if not all((SMALL, m) in data for m in MODES):
        raise SystemExit(f"{a.csv} must contain {SMALL} rows for modes {MODES}")
    common = sorted(set.intersection(*[set(v) for v in data.values()]))
    dirty, notclean = load_mask(a.dirty), load_mask(a.notclean)
    subsets = {
        "all": common,
        "excl_dirty": [q for q in common if q not in dirty],
        "excl_notclean": [q for q in common if q not in notclean],
    }
    P("Experiment C sensitivity analyses")
    P(f"input: {a.csv}; questions measured in every condition: {len(common)}")
    P(f"flagged among them: dirty {len(set(common) & dirty)}, not clean {len(set(common) & notclean)}")
    P(f"bootstrap {a.bootstrap} resamples, seed {a.seed}\n")

    # ---------------- Part 1a: per model and mode (Table 10)
    rows_modes = []
    P("=== Part 1a. Accuracy and cost per question (Table 10) ===")
    P(f"{'subset':<15}{'model':<17}{'mode':<9}{'n':>5}{'acc [95% CI]':>24}{'parse fail':>12}"
      f"{'tok p/g':>12}{'s':>7}{'gross J [95% CI]':>26}{'net J':>8}{'corr/kJ':>9}")
    for sname, qs in subsets.items():
        n = len(qs)
        for (model, mode) in sorted(data, key=lambda k: (k[0] != SMALL, k[0], MODES.index(k[1]) if k[1] in MODES else 9)):
            d = data[(model, mode)]
            y = np.array([d[q]["correct"] for q in qs], float)
            e = np.array([d[q]["joules"] for q in qs])
            acc_lo, acc_hi = boot_ci(lambda i: y[i].mean(), n, a.bootstrap, a.seed)
            e_lo, e_hi = boot_ci(lambda i: e[i].mean(), n, a.bootstrap, a.seed)
            row = dict(subset=sname, model=model, mode=mode, n=n,
                       accuracy=y.mean(), acc_lo=acc_lo, acc_hi=acc_hi,
                       parse_fail=1 - np.mean([d[q]["parse_ok"] for q in qs]),
                       prompt_tokens=np.mean([d[q]["prompt_tokens"] for q in qs]),
                       gen_tokens=np.mean([d[q]["gen_tokens"] for q in qs]),
                       seconds=np.mean([d[q]["seconds"] for q in qs]),
                       gross_J=e.mean(), gross_lo=e_lo, gross_hi=e_hi,
                       net_J=np.mean([d[q]["joules_net"] for q in qs]),
                       correct_per_kJ=1000 * y.sum() / e.sum())
            rows_modes.append(row)
            acc_s = "%.3f [%.3f, %.3f]" % (row["accuracy"], acc_lo, acc_hi)
            tok_s = "%.0f/%.0f" % (row["prompt_tokens"], row["gen_tokens"])
            e_s = "%.1f [%.1f, %.1f]" % (row["gross_J"], e_lo, e_hi)
            P(f"{sname:<15}{model:<17}{mode:<9}{n:>5}{acc_s:>24}{100 * row['parse_fail']:>11.1f}%"
              f"{tok_s:>12}{row['seconds']:>7.2f}{e_s:>26}{row['net_J']:>8.1f}{row['correct_per_kJ']:>9.2f}")
        P()
    with open(os.path.join(a.out, "modes_by_subset.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_modes[0]))
        w.writeheader()
        w.writerows(rows_modes)

    # ---------------- Part 1b: paired comparisons (Table 11)
    rows_pairs = []
    P("=== Part 1b. Paired comparisons between answer modes, MediLITE-QA 4-bit (Table 11) ===")
    for sname, qs in subsets.items():
        n = len(qs)
        block = []
        for first, second in PAIRS:
            d1, d2 = data[(SMALL, first)], data[(SMALL, second)]
            y1 = np.array([d1[q]["correct"] for q in qs], float)
            y2 = np.array([d2[q]["correct"] for q in qs], float)
            e1 = np.array([d1[q]["joules"] for q in qs])
            e2 = np.array([d2[q]["joules"] for q in qs])
            b, c = int(((y1 == 1) & (y2 == 0)).sum()), int(((y1 == 0) & (y2 == 1)).sum())
            lo, hi = boot_ci(lambda i: 100 * (y1[i].mean() - y2[i].mean()), n, a.bootstrap, a.seed)
            block.append(dict(subset=sname, comparison=f"{first} - {second}", n=n,
                              diff_points=100 * (y1.mean() - y2.mean()), ci_lo=lo, ci_hi=hi,
                              only_first=b, only_second=c, p_mcnemar=mcnemar_exact(b, c),
                              energy_ratio=e1.mean() / e2.mean()))
        for r, p_adj in zip(block, holm([r["p_mcnemar"] for r in block])):
            r["p_holm"] = p_adj
            rows_pairs.append(r)
            P(f"  {sname:<15}{r['comparison']:<17} n={n:<4}{r['diff_points']:+6.1f} pts "
              f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  {r['only_first']:>2}/{r['only_second']:<2} "
              f"p={r['p_mcnemar']:.4f}  p_Holm={p_adj:.4f}  energy ratio {r['energy_ratio']:.2f}")
        P()
    with open(os.path.join(a.out, "paired_by_subset.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_pairs[0]))
        w.writeheader()
        w.writerows(rows_pairs)

    # ---------------- Part 2: selective reasoning (all measured questions)
    qs = common
    n = len(qs)
    S, R = data[(SMALL, "loglik")], data[(SMALL, "cot")]
    conf = np.array([S[q]["conf"] for q in qs])
    yS = np.array([S[q]["correct"] for q in qs], float)
    yR = np.array([R[q]["correct"] for q in qs], float)
    eS = np.array([S[q]["joules"] for q in qs])
    eR = np.array([R[q]["joules"] for q in qs])
    order = np.argsort(conf, kind="stable")          # least confident first

    P("=== Part 2a. Confidence-gated reasoning (Fig. 5; descriptive) ===")
    rows_gate = []
    for f in GATE_FRACTIONS:
        k = int(round(f * n))
        gated = np.zeros(n, bool)
        gated[order[:k]] = True
        y = np.where(gated, yR, yS)
        energy = eS + np.where(gated, eR, 0.0)
        rows_gate.append(dict(fraction_gated=f, n_gated=k, accuracy=y.mean(), energy_J=energy.mean(),
                              gated_reasoning_correct=int(yR[gated].sum()), gated_scoring_correct=int(yS[gated].sum())))
        P(f"  re-answer least confident {100 * f:>5.0f}% ({k:>3} q): accuracy {y.mean():.3f}, "
          f"energy {energy.mean():.1f} J; on gated questions reasoning {int(yR[gated].sum())} vs scoring "
          f"{int(yS[gated].sum())} correct")
    with open(os.path.join(a.out, "gated_reasoning.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_gate[0]))
        w.writeheader()
        w.writerows(rows_gate)

    P("\n=== Part 2b. Option scoring vs reasoning within option-scoring confidence quartiles ===")
    rows_q = []
    for qi, idx in enumerate(np.array_split(order, 4), 1):
        b = int(((yR[idx] == 1) & (yS[idx] == 0)).sum())
        c = int(((yR[idx] == 0) & (yS[idx] == 1)).sum())
        r = dict(quartile=qi, n=len(idx), conf_min=conf[idx].min(), conf_max=conf[idx].max(),
                 acc_scoring=yS[idx].mean(), acc_reasoning=yR[idx].mean(),
                 reasoning_only=b, scoring_only=c, p_mcnemar=mcnemar_exact(b, c))
        rows_q.append(r)
        P(f"  Q{qi} (conf {r['conf_min']:.2f}-{r['conf_max']:.2f}, n={len(idx)}): scoring {r['acc_scoring']:.3f}, "
          f"reasoning {r['acc_reasoning']:.3f}; correct only with reasoning {b}, only with scoring {c}, "
          f"exact McNemar p={r['p_mcnemar']:.3f} (unadjusted)")
    with open(os.path.join(a.out, "reasoning_by_confidence_quartile.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_q[0]))
        w.writeheader()
        w.writerows(rows_q)

    gain = int(((yR == 1) & (yS == 0)).sum())
    P("\n=== Part 2c. Oracle upper bound for selective reasoning ===")
    P(f"  questions correct only with reasoning: {gain} of {n}")
    P(f"  option scoring {yS.mean():.3f}; an oracle that sent exactly these questions to reasoning "
      f"would reach {(yS.sum() + gain) / n:.3f} (+{100 * gain / n:.1f} points)")
    P("  Reaching that bound requires a gate that identifies these questions; Part 2a shows how "
      "the confidence gate actually performs.")
    P("\nCheck: the 'all' rows must reproduce manuscript Tables 10 and 11 (e.g. option scoring "
      "0.717 at 126.2 J, reasoning 0.631 at 201.2 J, reasoning - scoring p = 0.0015).")
    with open(os.path.join(a.out, "REPORT_expC_sensitivity.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"\nwritten to {a.out}/")


if __name__ == "__main__":
    main()
