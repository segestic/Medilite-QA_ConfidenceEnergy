"""
calibration_cascade_pilot.py
============================
Pilot analysis: calibration, selective prediction, and confidence-gated cascading
of small medical language models, computed from existing lm-evaluation-harness
per-item logs (no GPU and no new model runs).

Question
--------
Can a small model (default: MediLITE-QA Q4_K_M) answer the questions it is
confident about and escalate the rest to a larger model (default: OpenBioLLM-8B),
saving energy at equal accuracy?

Inputs
------
<logs>/<model>/**/samples_<task>_<timestamp>.jsonl   (lm-eval --log_samples output;
the Q4_K_M logs written by run_all_one_session.py use the same layout)
Each record needs: doc_id, resps (one log-likelihood per answer option), and
either `acc` or a `target` from which correctness can be derived.

Method
------
1. Confidence = maximum softmax probability over the answer options' log-likelihoods.
2. Calibration: ECE (15 equal-width bins), top-label Brier score, NLL of the gold
   option, AUROC of confidence for predicting correctness, and AURC
   (area under the risk-coverage curve); item-bootstrap 95% CIs.
3. Selective prediction: accuracy at fixed coverage levels.
4. Cascade: the small model answers when its confidence >= tau, otherwise the item
   is escalated to the large model. tau is chosen on a development split
   (random half of --dev-task) as the smallest threshold whose cascade accuracy is
   at least the large model's accuracy minus --delta-pp (--tau-rule match, default),
   or as the threshold that maximises dev cascade accuracy among thresholds whose
   dev coverage is at least the break-even coverage plus the --go-saving target
   (--tau-rule maxacc); the selected tau is then applied unchanged to
   (a) the remaining half of the development task and (b) all other tasks.
   Expected energy per item = E_small + (1 - coverage) * E_large.

Energy inputs
-------------
Pass measured joules per item (--energy-small, --energy-large). If omitted, the
script uses a PROXY (E_small = 1, E_large = --energy-ratio) and says so in the
report; proxy results are only for deciding whether a measured run is worthwhile.

Decision rule (set BEFORE running; do not adjust after seeing results)
---------------------------------------------------------------------
GO if (small-model AUROC >= --go-auroc) and (energy saving on the held-out tasks
>= --go-saving percent) and (cascade accuracy on those tasks is within
--tol-pp percentage points of the large model) and (cascade accuracy exceeds
always-small by at least --min-gain-pp, so that the gating itself adds value
rather than the small model simply matching the large one).

Usage
-----
    python calibration_cascade_pilot.py --logs lm_eval_logs \\
        --models medilite_q4 medilite base openbiollm8b \\
        --small medilite_q4 --large openbiollm8b --out pilot_out

Outputs (in --out): REPORT.txt, calibration_by_model.csv, calibration_by_task.csv,
paired_differences.csv, risk_coverage.csv, cascade_curve.csv, cascade_summary.json,
risk_coverage.png, cascade_frontier.png (plots need matplotlib).
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy.stats import rankdata

N_BINS = 15
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)


# ------------------------------------------------------------------ loading
def task_of(path):
    name = os.path.basename(path)[len("samples_"):-len(".jsonl")]
    return re.sub(r"_\d{4}-\d{2}-\d{2}T.*$", "", name)


def latest_files(model_dir):
    best = {}
    for f in glob.glob(os.path.join(model_dir, "**", "samples_*.jsonl"), recursive=True):
        t = task_of(f)
        if t not in best or os.path.basename(f) > os.path.basename(best[t]):
            best[t] = f
    return best


def first_number(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def get_requests(rec):
    args = rec.get("arguments")
    if isinstance(args, dict):
        keys = sorted(args, key=lambda k: int(re.findall(r"\d+", k)[-1]))
        return [(args[k]["arg_0"], args[k]["arg_1"]) for k in keys]
    return [(a[0], a[1]) for a in args] if args else []


def gold_index(rec):
    """Index of the correct option, or None if it cannot be determined."""
    t = rec.get("target")
    if isinstance(t, list) and t:
        t = t[0]
    try:
        return int(t)
    except (TypeError, ValueError):
        pass
    try:
        conts = [c.strip() for _, c in get_requests(rec)]
        return conts.index(str(t).strip())
    except (ValueError, TypeError):
        return None


def softmax(ll):
    z = ll - np.max(ll)
    e = np.exp(z)
    return e / e.sum()


def load_model(model_dir):
    """{task: {doc_id: (conf, correct, p_gold, n_options)}}"""
    out, mismatches, checked = {}, 0, 0
    for task, f in latest_files(model_dir).items():
        d = {}
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                ll = np.array([first_number(x) for x in r["resps"]], dtype=float)
                p = softmax(ll)
                pred = int(np.argmax(p))
                gold = gold_index(r)
                if "acc" in r:
                    correct = float(r["acc"])
                elif gold is not None:
                    correct = float(pred == gold)
                else:
                    continue
                if gold is not None and "acc" in r:
                    checked += 1
                    mismatches += int((pred == gold) != bool(round(correct)))
                d[r["doc_id"]] = (float(p[pred]), correct,
                                  float(p[gold]) if gold is not None else float("nan"), len(ll))
        out[task] = d
    rate = 100 * (1 - mismatches / checked) if checked else float("nan")
    return out, rate


# ------------------------------------------------------------------ metrics
def ece(conf, correct, bins=N_BINS):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    n = len(conf)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return total


def auroc(conf, correct):
    pos = correct > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(conf)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def risk_coverage(conf, correct):
    order = np.argsort(-conf, kind="mergesort")
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    sel_acc = np.cumsum(c) / k
    return k / len(c), sel_acc


def aurc(conf, correct):
    _, sel_acc = risk_coverage(conf, correct)
    return float(np.mean(1.0 - sel_acc))


def acc_at_coverage(conf, correct, cov):
    cov_grid, sel_acc = risk_coverage(conf, correct)
    i = min(int(np.ceil(cov * len(conf))) - 1, len(conf) - 1)
    return float(sel_acc[max(i, 0)])


def boot_ci(fn, n, B, rng):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if len(vals) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="lm_eval_logs")
    ap.add_argument("--models", nargs="*", default=None, help="folder names under --logs (default: all)")
    ap.add_argument("--small", default="medilite_q4")
    ap.add_argument("--large", default="openbiollm8b")
    ap.add_argument("--energy-small", type=float, default=None, help="measured joules per item, small model")
    ap.add_argument("--energy-large", type=float, default=None, help="measured joules per item, large model")
    ap.add_argument("--energy-ratio", type=float, default=1.68, help="PROXY E_large/E_small if energies not given")
    ap.add_argument("--dev-task", default="medmcqa")
    ap.add_argument("--dev-frac", type=float, default=0.5)
    ap.add_argument("--delta-pp", type=float, default=0.0, help="accuracy slack vs the large model on the dev split")
    ap.add_argument("--tau-rule", choices=("match", "maxacc"), default="match",
                    help="match: smallest tau with dev accuracy >= large - delta; "
                         "maxacc: highest dev accuracy subject to the coverage needed for --go-saving")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--go-auroc", type=float, default=0.70)
    ap.add_argument("--go-saving", type=float, default=20.0, help="percent energy saving required")
    ap.add_argument("--tol-pp", type=float, default=1.0, help="max accuracy deficit vs large (pp) on held-out tasks")
    ap.add_argument("--min-gain-pp", type=float, default=2.0, help="required cascade gain over always-small (pp) on held-out tasks")
    ap.add_argument("--pairs", nargs="*", default=None, help="model pairs as a:b (paired differences a-b)")
    ap.add_argument("--out", default="pilot_out")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = []
    P = lambda s="": (report.append(s), print(s))

    names = args.models or sorted(d for d in os.listdir(args.logs)
                                  if os.path.isdir(os.path.join(args.logs, d)) and not d.endswith("_smoke"))
    for need in (args.small, args.large):
        if need and need not in names and os.path.isdir(os.path.join(args.logs, need)):
            names.append(need)
    raw, parse_rate = {}, {}
    for m in names:
        d = os.path.join(args.logs, m)
        if not os.path.isdir(d) or not latest_files(d):
            P(f"[skip] {m}: no samples_*.jsonl under {d}")
            continue
        raw[m], parse_rate[m] = load_model(d)
    if not raw:
        sys.exit("no usable models found")
    P(f"Models: {', '.join(raw)}")
    for m, r in parse_rate.items():
        P(f"  [check] {m}: stored acc re-derived from log-likelihoods on {fmt(r, 2)}% of items")

    # ---- align items across models
    tasks = sorted(set.intersection(*[set(raw[m]) for m in raw]))
    keys = []
    for t in tasks:
        ids = sorted(set.intersection(*[set(raw[m][t]) for m in raw]))
        keys += [(t, i) for i in ids]
    N = len(keys)
    task_arr = np.array([k[0] for k in keys])
    P(f"Aligned items: {N} across {len(tasks)} tasks ({', '.join(tasks)})")
    D = {}
    for m in raw:
        arr = np.array([raw[m][t][i] for t, i in keys], dtype=float)
        D[m] = dict(conf=arr[:, 0], correct=arr[:, 1], pgold=arr[:, 2])

    # ---- 1-3. calibration and selective prediction
    P("\n=== Calibration and selective prediction (all aligned items) ===")
    header = ["model", "n", "accuracy", "mean_conf", "ECE", "ECE_lo", "ECE_hi", "Brier", "NLL",
              "AUROC", "AUROC_lo", "AUROC_hi", "AURC", "AURC_lo", "AURC_hi"] + [f"acc@{int(c*100)}%cov" for c in COVERAGES]
    rows = []
    for m, d in D.items():
        c, y = d["conf"], d["correct"]
        e_lo, e_hi = boot_ci(lambda i: ece(c[i], y[i]), N, args.bootstrap, rng)
        a_lo, a_hi = boot_ci(lambda i: auroc(c[i], y[i]), N, args.bootstrap, rng)
        r_lo, r_hi = boot_ci(lambda i: aurc(c[i], y[i]), N, args.bootstrap, rng)
        nll = -np.nanmean(np.log(np.clip(d["pgold"], 1e-12, 1.0))) if np.isfinite(d["pgold"]).any() else float("nan")
        rows.append([m, N, y.mean(), c.mean(), ece(c, y), e_lo, e_hi, np.mean((c - y) ** 2), nll,
                     auroc(c, y), a_lo, a_hi, aurc(c, y), r_lo, r_hi] + [acc_at_coverage(c, y, cv) for cv in COVERAGES])
    with open(os.path.join(args.out, "calibration_by_model.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    P(f"{'model':<16}{'acc':>7}{'conf':>7}{'ECE [95% CI]':>24}{'AUROC [95% CI]':>24}{'AURC':>8}{'acc@90':>8}{'acc@70':>8}{'acc@50':>8}")
    for r in rows:
        P(f"{r[0]:<16}{r[2]:>7.3f}{r[3]:>7.3f}{fmt(r[4]) + ' [' + fmt(r[5]) + ',' + fmt(r[6]) + ']':>24}"
          f"{fmt(r[9]) + ' [' + fmt(r[10]) + ',' + fmt(r[11]) + ']':>24}{r[12]:>8.3f}{r[16]:>8.3f}{r[18]:>8.3f}{r[19]:>8.3f}")

    # per task + macro
    trows = []
    P("\n=== Per-task accuracy / ECE / AUROC ===")
    for m, d in D.items():
        for t in tasks:
            s = task_arr == t
            trows.append([m, t, int(s.sum()), d["correct"][s].mean(), ece(d["conf"][s], d["correct"][s]),
                          auroc(d["conf"][s], d["correct"][s]), aurc(d["conf"][s], d["correct"][s])])
        mac = [r for r in trows if r[0] == m]
        P(f"{m:<16} macro over tasks: ECE {np.nanmean([r[4] for r in mac]):.3f} | AUROC {np.nanmean([r[5] for r in mac]):.3f}")
    with open(os.path.join(args.out, "calibration_by_task.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "task", "n", "accuracy", "ECE", "AUROC", "AURC"]); w.writerows(trows)

    # paired differences
    pairs = args.pairs or [p for p in ("medilite:base", "medilite_q4:medilite") if all(x in D for x in p.split(":"))]
    prow = []
    if pairs:
        P("\n=== Paired differences (a - b), item bootstrap 95% CI ===")
    for p in pairs:
        a, b = p.split(":")
        ca, ya, cb, yb = D[a]["conf"], D[a]["correct"], D[b]["conf"], D[b]["correct"]
        for label, fn in (("ECE", lambda i: ece(ca[i], ya[i]) - ece(cb[i], yb[i])),
                          ("AUROC", lambda i: auroc(ca[i], ya[i]) - auroc(cb[i], yb[i])),
                          ("AURC", lambda i: aurc(ca[i], ya[i]) - aurc(cb[i], yb[i]))):
            pt = fn(np.arange(N)); lo, hi = boot_ci(fn, N, args.bootstrap, rng)
            prow.append([a, b, label, pt, lo, hi])
            P(f"  {a} - {b}  {label:<6} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if prow:
        with open(os.path.join(args.out, "paired_differences.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["a", "b", "metric", "difference", "ci_lo", "ci_hi"]); w.writerows(prow)

    with open(os.path.join(args.out, "risk_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "coverage", "selective_accuracy"])
        for m, d in D.items():
            cov, sel = risk_coverage(d["conf"], d["correct"])
            for k in np.linspace(0, len(cov) - 1, min(200, len(cov))).astype(int):
                w.writerow([m, f"{cov[k]:.4f}", f"{sel[k]:.4f}"])

    # ---- 4. cascade
    summary = {"models": list(D), "n_items": N}
    verdict_lines = []
    if args.small in D and args.large in D and args.small != args.large:
        S, L = D[args.small], D[args.large]
        measured = args.energy_small is not None and args.energy_large is not None
        eS = args.energy_small if measured else 1.0
        eL = args.energy_large if measured else args.energy_ratio
        unit = "J/item" if measured else "relative units (PROXY)"
        P(f"\n=== Confidence-gated cascade: {args.small} -> {args.large} ===")
        P(f"Energy per item: small {eS:g}, large {eL:g} ({unit}); break-even coverage = E_small/E_large = {eS / eL:.3f}")
        if not measured:
            P("  NOTE: energies are a PROXY. Measure joules per item on the target hardware before drawing conclusions.")

        dev_task = args.dev_task
        if dev_task not in tasks:
            sys.exit(f"--dev-task '{dev_task}' not among aligned tasks: {tasks}")
        dev_idx_all = np.where(task_arr == dev_task)[0]
        perm = np.random.default_rng(args.seed).permutation(len(dev_idx_all))  # split depends only on --seed
        n_dev = int(round(args.dev_frac * len(dev_idx_all)))
        dev = dev_idx_all[perm[:n_dev]]
        test_same = dev_idx_all[perm[n_dev:]]
        test_other = np.where(task_arr != dev_task)[0]
        P(f"Development split: {len(dev)} items of {dev_task}; test (same task): {len(test_same)}; test (other tasks): {len(test_other)}")

        def cascade(idx, tau):
            ans = S["conf"][idx] >= tau
            y = np.where(ans, S["correct"][idx], L["correct"][idx])
            cov = ans.mean()
            return y.mean(), cov, eS + (1 - cov) * eL, (S["correct"][idx][ans].mean() if ans.any() else float("nan"))

        taus = np.unique(np.concatenate([S["conf"][dev], [1.0 + 1e-9]]))
        tau = taus[-1]
        if args.tau_rule == "match":
            target = L["correct"][dev].mean() - args.delta_pp / 100.0
            for t in taus:
                if cascade(dev, t)[0] >= target:
                    tau = t
                    break
            P(f"Selected tau = {tau:.4f} (rule 'match': smallest threshold with dev cascade accuracy >= large accuracy - {args.delta_pp:g} pp)")
        else:
            min_cov = args.go_saving / 100.0 + eS / eL
            if min_cov > 1.0:
                P(f"  NOTE: required coverage {min_cov:.3f} > 1; the --go-saving target cannot be met at any tau. Using tau = min dev confidence.")
                tau = taus[0]
            else:
                best = -1.0
                for t in taus:  # ascending, so ties keep the smaller tau (higher coverage)
                    acc_t, cov_t, _, _ = cascade(dev, t)
                    if cov_t >= min_cov and acc_t > best + 1e-12:
                        best, tau = acc_t, t
            P(f"Selected tau = {tau:.4f} (rule 'maxacc': highest dev cascade accuracy with dev coverage >= {min_cov:.3f})")
        summary.update(tau=float(tau), tau_rule=args.tau_rule, energy_small=eS, energy_large=eL,
                       energy_unit=unit, energy_is_proxy=not measured)

        P(f"\n{'split':<22}{'n':>6}{'acc small':>10}{'acc large':>10}{'acc casc.':>10}{'coverage':>10}{'acc|answered':>13}{'energy':>9}{'saving':>9}")
        res = {}
        for name, idx in (("dev", dev), ("test: same task", test_same), ("test: other tasks", test_other)):
            if len(idx) == 0:
                continue
            acc_c, cov, en, acc_ans = cascade(idx, tau)
            saving = 100 * (1 - en / eL)
            res[name] = dict(n=int(len(idx)), acc_small=float(S["correct"][idx].mean()), acc_large=float(L["correct"][idx].mean()),
                             acc_cascade=float(acc_c), coverage=float(cov), acc_answered_by_small=float(acc_ans),
                             energy=float(en), saving_percent=float(saving))
            P(f"{name:<22}{len(idx):>6}{S['correct'][idx].mean():>10.3f}{L['correct'][idx].mean():>10.3f}{acc_c:>10.3f}"
              f"{cov:>10.3f}{fmt(acc_ans):>13}{en:>9.3f}{saving:>8.1f}%")
        summary["cascade"] = res

        # bootstrap CI on held-out tasks at the fixed tau
        if len(test_other):
            sv, dacc, dgain = [], [], []
            for _ in range(args.bootstrap):
                b = test_other[rng.integers(0, len(test_other), len(test_other))]
                acc_c, cov, en, _ = cascade(b, tau)
                sv.append(100 * (1 - en / eL)); dacc.append(100 * (acc_c - L["correct"][b].mean()))
                dgain.append(100 * (acc_c - S["correct"][b].mean()))
            summary["heldout_saving_ci"] = [float(np.percentile(sv, 2.5)), float(np.percentile(sv, 97.5))]
            summary["heldout_acc_diff_pp_ci"] = [float(np.percentile(dacc, 2.5)), float(np.percentile(dacc, 97.5))]
            summary["heldout_gain_vs_small_pp_ci"] = [float(np.percentile(dgain, 2.5)), float(np.percentile(dgain, 97.5))]
            P(f"Held-out tasks, 95% CI: energy saving {summary['heldout_saving_ci'][0]:.1f} to {summary['heldout_saving_ci'][1]:.1f}%; "
              f"accuracy difference vs large {summary['heldout_acc_diff_pp_ci'][0]:+.2f} to {summary['heldout_acc_diff_pp_ci'][1]:+.2f} pp; "
              f"gain over always-small {summary['heldout_gain_vs_small_pp_ci'][0]:+.2f} to {summary['heldout_gain_vs_small_pp_ci'][1]:+.2f} pp")

        # curve on held-out tasks
        grid = np.unique(np.quantile(S["conf"], np.linspace(0, 1, 101)))
        curve = []
        eval_idx = test_other if len(test_other) else np.arange(N)
        for t in np.concatenate([grid, [1.0 + 1e-9]]):
            acc_c, cov, en, acc_ans = cascade(eval_idx, t)
            curve.append([float(t), float(cov), float(acc_c), float(acc_ans), float(en), float(100 * (1 - en / eL))])
        with open(os.path.join(args.out, "cascade_curve.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["tau", "coverage_small", "accuracy_cascade", "accuracy_answered_by_small", "expected_energy", "saving_percent"])
            w.writerows(curve)

        # decision rule
        auc_s = auroc(S["conf"], S["correct"])
        ho = res.get("test: other tasks", {})
        ok_auc = auc_s >= args.go_auroc
        ok_sav = ho.get("saving_percent", -1e9) >= args.go_saving
        ok_acc = (ho.get("acc_cascade", 0) - ho.get("acc_large", 1)) * 100 >= -args.tol_pp
        gain = (ho.get("acc_cascade", 0) - ho.get("acc_small", 1)) * 100
        ok_gain = gain >= args.min_gain_pp
        go = ok_auc and ok_sav and ok_acc and ok_gain
        P("\n=== Decision rule (fixed in advance) ===")
        P(f"  small-model AUROC {auc_s:.3f} >= {args.go_auroc:g}: {'PASS' if ok_auc else 'FAIL'}")
        P(f"  held-out energy saving {ho.get('saving_percent', float('nan')):.1f}% >= {args.go_saving:g}%: {'PASS' if ok_sav else 'FAIL'}")
        P(f"  held-out accuracy within {args.tol_pp:g} pp of the large model: {'PASS' if ok_acc else 'FAIL'}")
        P(f"  cascade gain over always-small {gain:+.2f} pp >= {args.min_gain_pp:g} pp: {'PASS' if ok_gain else 'FAIL'}")
        P(f"  VERDICT: {'GO' if go else 'STOP'}" + ("" if measured else "  (proxy energies: confirm with measured joules per item before committing)"))
        summary.update(auroc_small=float(auc_s), verdict="GO" if go else "STOP",
                       rule=dict(go_auroc=args.go_auroc, go_saving=args.go_saving, tol_pp=args.tol_pp, min_gain_pp=args.min_gain_pp))
    else:
        P("\n[cascade skipped] --small/--large not both present among the loaded models")

    with open(os.path.join(args.out, "cascade_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            for m, d in D.items():
                cov, sel = risk_coverage(d["conf"], d["correct"])
                ax.plot(cov, sel, label=m, lw=1.6)
            ax.set_xlabel("Coverage"); ax.set_ylabel("Selective accuracy"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            ax.set_title("Risk-coverage curves (all aligned items)")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "risk_coverage.png"), dpi=200); plt.close(fig)
            if "cascade" in summary and "test: other tasks" in summary["cascade"]:
                ho = summary["cascade"]["test: other tasks"]
                fig, ax = plt.subplots(figsize=(6.2, 4.4))
                ax.plot([100 - c[5] for c in curve], [c[2] for c in curve], "-o", ms=2, label="cascade (tau swept)")
                ax.scatter([100 * eS / eL], [ho["acc_small"]], marker="s", label=f"always {args.small}")
                ax.scatter([100.0], [ho["acc_large"]], marker="^", label=f"always {args.large}")
                ax.set_xlabel("Expected energy (% of always-large)"); ax.set_ylabel("Accuracy (held-out tasks)")
                ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("Accuracy-energy frontier")
                fig.tight_layout(); fig.savefig(os.path.join(args.out, "cascade_frontier.png"), dpi=200); plt.close(fig)
        except Exception as e:  # plotting is optional
            P(f"[plots skipped] {e}")

    with open(os.path.join(args.out, "REPORT.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
