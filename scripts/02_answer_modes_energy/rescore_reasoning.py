"""
rescore_reasoning.py
====================
Re-scores the saved letter and step-by-step reasoning outputs with a
"first answer" rule, as a robustness check on the original "last answer" rule
used by energy_per_item.py, and counts runaway generations.

Why: many reasoning outputs answer the question, then keep generating invented
follow-up questions (often with "[INST]" / "[/INST]" markers) until the token
limit. The original parser took the LAST "Answer: X" in the text, which can
belong to an invented question.

First-answer rule:
  1. cut the output at the first "[INST]", "[/INST]", "<|" or repeated
     "Think step by step" (the start of a runaway continuation);
  2. take the first "Answer: X" whose X is an option label or the exact text
     of an option (e.g. "Answer: Miss");
  3. otherwise take an option label or option text that starts the output;
  4. otherwise: no answer (scored incorrect).

Inputs: energy_items.csv (option scoring correctness, finish reasons),
        texts_medilite_q4.jsonl (generated texts), items.jsonl (options, gold).
Outputs: <out>.csv (per question), <out>_summary.txt (printed summary).

Usage:
  python rescore_reasoning.py --csv energy_items.csv --texts texts_medilite_q4.jsonl \
      --items items.jsonl --model medilite_q4 --out rescored
"""
import argparse, csv, json, re
from scipy.stats import binomtest

CUT = re.compile(r"\[/?INST\]|<\||\n\s*Think step by step", re.IGNORECASE)


def option_texts(item):
    """Map option label -> option text, read from the question prompt."""
    labels = [c.strip() for c in item["continuations"]]
    out = {}
    for lab in labels:
        m = re.search(rf"^\s*{re.escape(lab)}\.\s*(.+)$", item["context"], re.MULTILINE) if len(lab) == 1 else None
        out[lab] = m.group(1).strip().rstrip(".").lower() if m else lab.lower()
    return labels, out


def first_answer(text, labels, opt):
    head = CUT.split(text, maxsplit=1)[0]
    low = [l.lower() for l in labels]
    def match(tok):
        t = tok.strip().strip(".():*").lower()
        if t in low:
            return low.index(t)
        for i, l in enumerate(labels):
            if t and t == opt[l]:
                return i
        return -1
    for m in re.finditer(r"answer\s*(?:is)?\s*[:\-]\s*\(?([^\n]+)", head, re.IGNORECASE):
        cand = m.group(1).strip()
        i = match(cand)
        if i < 0:
            i = match(re.split(r"[\s.)]", cand, maxsplit=1)[0])
        if i >= 0:
            return i
    first = head.strip().split("\n", 1)[0] if head.strip() else ""
    i = match(re.split(r"[\s.)]", first, maxsplit=1)[0]) if first else -1
    if i < 0:
        i = match(first)
    return i


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    p = binomtest(n10, n01 + n10).pvalue if n01 + n10 else 1.0
    return n10, n01, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="energy_items.csv")
    ap.add_argument("--texts", default="texts_medilite_q4.jsonl")
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--model", default="medilite_q4")
    ap.add_argument("--out", default="rescored")
    a = ap.parse_args()

    items = {(it["task"], str(it["doc_id"])): it for it in map(json.loads, open(a.items))}
    rows = {}
    for r in csv.DictReader(open(a.csv)):
        if r["model"] == a.model and r["rep"] == "1":
            rows[(r["mode"], r["task"], str(r["doc_id"]))] = r
    texts = {}
    with open(a.texts) as fh:
        for line in fh:
            t = json.loads(line)
            if t.get("model") == a.model and str(t["rep"]) == "1" and t["mode"] in ("letter", "cot"):
                texts[(t["mode"], t["task"], str(t["doc_id"]))] = t["text"]

    keys = sorted(k for k in items if ("loglik",) + k in rows and ("cot",) + k in rows)
    out, report = [], []
    P = lambda s="": (report.append(s), print(s))
    res = {m: {"orig": [], "first": []} for m in ("letter", "cot")}
    score = [rows[("loglik",) + k]["correct"] == "1" for k in keys]
    for k in keys:
        it = items[k]; labels, opt = option_texts(it)
        rec = dict(task=k[0], doc_id=k[1], gold=labels[it["gold"]], scoring_correct=int(rows[("loglik",) + k]["correct"] == "1"))
        for m in ("letter", "cot"):
            if (m,) + k not in rows:
                continue
            r = rows[(m,) + k]; text = texts.get((m,) + k, "")
            fa = first_answer(text, labels, opt)
            res[m]["orig"].append(r["correct"] == "1"); res[m]["first"].append(fa == it["gold"])
            rec.update({f"{m}_orig_correct": int(r["correct"] == "1"),
                        f"{m}_first_answer": labels[fa] if fa >= 0 else "none",
                        f"{m}_first_correct": int(fa == it["gold"]),
                        f"{m}_finish": r.get("finish", ""),
                        f"{m}_gen_tokens": r.get("gen_tokens", ""),
                        f"{m}_runaway": int(bool(re.search(r"\[/?INST\]", text)) or r.get("finish", "") == "length"),
                        f"{m}_text_missing": int(not text)})
        out.append(rec)

    n = len(keys)
    P(f"questions: {n}; option scoring accuracy {sum(score)/n:.3f}")
    for m in ("letter", "cot"):
        if len(res[m]["orig"]) != n:
            continue
        o, f = res[m]["orig"], res[m]["first"]
        run = sum(r[f"{m}_runaway"] for r in out); miss = sum(r[f"{m}_text_missing"] for r in out)
        P(f"\n[{m}] runaway or truncated outputs: {run}/{n}; texts missing: {miss}")
        P(f"  accuracy, original last-answer rule: {sum(o)/n:.3f}")
        P(f"  accuracy, first-answer rule:        {sum(f)/n:.3f}")
        for name, v in (("original", o), ("first-answer", f)):
            n10, n01, p = mcnemar(score, v)
            P(f"  scoring vs {m} ({name}): scoring-only correct {n10}, {m}-only correct {n01}, "
              f"difference {100*(sum(score)-sum(v))/n:+.1f} points, exact McNemar p={p:.3g}")
    if len(res["letter"]["first"]) == n and len(res["cot"]["first"]) == n:
        n10, n01, p = mcnemar(res["letter"]["first"], res["cot"]["first"])
        P(f"\nletter vs cot (first-answer): letter-only {n10}, cot-only {n01}, exact McNemar p={p:.3g}")

    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in out for k in r}, key=lambda x: (x not in ("task", "doc_id", "gold"), x)))
        w.writeheader(); w.writerows(out)
    open(a.out + "_summary.txt", "w").write("\n".join(report) + "\n")
    print(f"\nwritten {a.out}.csv and {a.out}_summary.txt")


if __name__ == "__main__":
    main()
