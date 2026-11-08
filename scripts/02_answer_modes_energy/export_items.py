"""
export_items.py
===============
Build the fixed evaluation set for the energy-per-item runs from existing
lm-evaluation-harness per-item logs (no model is run).

For each task, a fixed number of items is drawn at random (seed-controlled).
For every item the exact lm-eval context and answer continuations are taken
from a log folder that stores them (the fp16 logs have an `arguments` field),
and the logged prediction of every model folder is stored as a reference so
that later runs can be checked against the logged results.

Usage
-----
    python export_items.py --logs /content/content/kaggle_output/lm_eval_logs \
        --source medilite --per-task 22 --seed 1234 --out items.jsonl
"""

import argparse
import glob
import json
import os
import re

import numpy as np


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


def gold_index(rec, conts):
    t = rec.get("target")
    if isinstance(t, list) and t:
        t = t[0]
    try:
        return int(t)
    except (TypeError, ValueError):
        pass
    try:
        return [c.strip() for c in conts].index(str(t).strip())
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--source", default="medilite", help="log folder whose records contain `arguments`")
    ap.add_argument("--per-task", type=int, default=22)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="items.jsonl")
    args = ap.parse_args()

    # reference predictions from every model folder
    refs = {}
    for m in sorted(os.listdir(args.logs)):
        d = os.path.join(args.logs, m)
        if not os.path.isdir(d):
            continue
        for task, f in latest_files(d).items():
            with open(f) as fh:
                for n_line, line in enumerate(fh, 1):
                    try:
                        r = json.loads(line)
                        ll = [first_number(x) for x in r["resps"]]
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
                        print(f"[warn] {m}: skipped unreadable line {n_line} of {os.path.basename(f)}")
                        continue
                    refs.setdefault((task, r["doc_id"]), {})[m] = {
                        "pred": int(np.argmax(ll)), "acc": float(r["acc"]) if "acc" in r else None}

    rng = np.random.default_rng(args.seed)
    items, skipped = [], 0
    for task, f in sorted(latest_files(os.path.join(args.logs, args.source)).items()):
        recs = [json.loads(line) for line in open(f)]
        recs = [r for r in recs if get_requests(r)]
        if not recs:
            raise SystemExit(f"{f}: no `arguments` field; choose a --source folder written by lm-eval")
        recs.sort(key=lambda r: r["doc_id"])
        pick = rng.permutation(len(recs))[: args.per_task]
        for i in sorted(pick):
            r = recs[i]
            reqs = get_requests(r)
            conts = [c for _, c in reqs]
            g = gold_index(r, conts)
            if g is None or len({c for c, _ in reqs}) != 1:
                skipped += 1
                continue
            items.append({"task": task, "doc_id": r["doc_id"], "context": reqs[0][0],
                          "continuations": conts, "gold": g,
                          "ref": refs.get((task, r["doc_id"]), {})})
    with open(args.out, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    tasks = sorted({it["task"] for it in items})
    print(f"wrote {len(items)} items from {len(tasks)} tasks to {args.out} (skipped {skipped})")
    for t in tasks:
        print(f"  {t:<28}{sum(it['task'] == t for it in items):>4}")


if __name__ == "__main__":
    main()
