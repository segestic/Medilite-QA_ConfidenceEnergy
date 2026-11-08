"""
cross_check_scorer.py
=====================
Separates "different code" from "different hardware" for the option-scoring
disagreement found by diagnose_agreement.py.

It re-scores the fixed item set ON THE LAPTOP CPU with the Scorer class taken
unchanged from run_all_one_session.py (the code that produced the logged Q4
results on the Kaggle GPU), then compares its predictions with
  (a) the laptop predictions from energy_per_item.py  -> same hardware, different code
  (b) the logged GPU predictions                      -> same code, different hardware

Rule fixed before running: if (a) agrees on >= 99% of items, the two
implementations are equivalent and the disagreement with the logs is attributed to
CPU vs GPU execution; if (a) agrees on < 99%, the scoring code in
energy_per_item.py differs from the original and must be corrected.

No energy is measured. Runtime on the paper laptop: about 15 min per model.

Usage
-----
    python cross_check_scorer.py --scorer-script /path/to/run_all_one_session.py \
        --model ../gguf_models/<MediLITE file>.gguf --label medilite_q4 --ref medilite_q4 \
        --items items.jsonl --csv energy_items.csv --out crosscheck_medilite_q4.json
"""

import argparse
import csv
import importlib.util
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorer-script", required=True, help="path to run_all_one_session.py")
    ap.add_argument("--model", required=True, help="path to the .gguf file")
    ap.add_argument("--label", required=True, help="model label in energy_items.csv, e.g. medilite_q4")
    ap.add_argument("--ref", required=True, help="log folder name stored in items.jsonl, e.g. medilite_q4")
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--csv", default="energy_items.csv")
    ap.add_argument("--n-ctx", type=int, default=4096, help="as in the original Q4 scoring")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("orig", args.scorer_script)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    items = [json.loads(l) for l in open(args.items)]
    laptop = {}
    for r in csv.DictReader(open(args.csv)):
        if r["model"] == args.label and r["mode"] == "loglik" and r["rep"] == "1":
            laptop[(r["task"], str(r["doc_id"]))] = (int(r["pred"]), float(r["conf"]))

    scorer = orig.Scorer(args.model, args.n_ctx, add_bos=False)  # same call as the original Q4 stage
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        key = (it["task"], str(it["doc_id"]))
        if key not in laptop or args.ref not in it.get("ref", {}):
            continue
        reqs = [(it["context"], c) for c in it["continuations"]]
        lls = np.array(scorer.score_doc(reqs), dtype=float)
        p = np.exp(lls - lls.max()); p /= p.sum()
        rows.append(dict(task=key[0], doc_id=key[1], gold=it["gold"], orig_cpu_pred=int(np.argmax(lls)),
                         orig_cpu_conf=float(p.max()), laptop_pred=laptop[key][0], laptop_conf=laptop[key][1],
                         logged_gpu_pred=it["ref"][args.ref]["pred"]))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)} items, {(time.time() - t0) / 60:.1f} min")

    n = len(rows)
    a = sum(r["orig_cpu_pred"] == r["laptop_pred"] for r in rows)
    b = sum(r["orig_cpu_pred"] == r["logged_gpu_pred"] for r in rows)
    dconf = max(abs(r["orig_cpu_conf"] - r["laptop_conf"]) for r in rows) if rows else float("nan")
    acc = lambda k: float(np.mean([r[k] == r["gold"] for r in rows]))
    verdict = ("implementations equivalent; disagreement with the logs comes from CPU vs GPU execution"
               if a >= 0.99 * n else "energy_per_item.py scoring differs from the original -- must be corrected")
    summary = dict(label=args.label, items=n,
                   original_code_cpu_vs_laptop_code_cpu=a / n, original_code_cpu_vs_logged_gpu=b / n,
                   max_abs_conf_difference_same_hardware=dconf,
                   acc_original_code_cpu=acc("orig_cpu_pred"), acc_laptop=acc("laptop_pred"),
                   acc_logged_gpu=acc("logged_gpu_pred"), verdict=verdict,
                   minutes=(time.time() - t0) / 60)
    json.dump({"summary": summary, "items": rows}, open(args.out, "w"), indent=2)
    print(f"\n(a) original code on CPU vs energy_per_item.py on CPU: {a}/{n} same ({100 * a / n:.1f}%), "
          f"max |conf diff| {dconf:.2e}")
    print(f"(b) original code on CPU vs logged GPU results:        {b}/{n} same ({100 * b / n:.1f}%)")
    print(f"accuracy: original code CPU {summary['acc_original_code_cpu']:.3f}, laptop "
          f"{summary['acc_laptop']:.3f}, logged GPU {summary['acc_logged_gpu']:.3f}")
    print(f"VERDICT: {verdict}\nwritten to {args.out}")


if __name__ == "__main__":
    main()
