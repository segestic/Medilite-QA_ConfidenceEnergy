"""
extract_error_cases.py
======================
Pull out only the reasoning outputs needed for Table 8 (qualitative error
analysis) from the large texts_medilite_q4.jsonl, streaming it line by line.

Selected questions (repetition 1, option scoring vs step-by-step reasoning):
  * "scoring right, reasoning wrong"   -> the 29 questions categorised in Table 8
  * "scoring wrong, reasoning right"   -> the 7 reverse cases, for comparison
Each record gets automatic flags (truncated at the token limit, no parseable
answer, number of option labels mentioned) to speed up manual categorisation.

Outputs
  <prefix>.jsonl  full records (question, options, gold, both answers, full reasoning text)
  <prefix>.md     the same in readable form, small enough to upload or paste

Usage
  python extract_error_cases.py --csv energy_items.csv --texts texts_medilite_q4.jsonl \
      --items items.jsonl --model medilite_q4 --out table8_cases
"""
import argparse
import csv
import json
import re


def label_list(item):
    return [c.strip() for c in item["continuations"]]


def question_text(item):
    ctx = item["context"].rstrip()
    return ctx[: -len("Answer:")].rstrip() if ctx.endswith("Answer:") else ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="energy_items.csv")
    ap.add_argument("--texts", default="texts_medilite_q4.jsonl")
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--model", default="medilite_q4")
    ap.add_argument("--out", default="table8_cases")
    args = ap.parse_args()

    items = {(it["task"], str(it["doc_id"])): it for it in map(json.loads, open(args.items))}

    rows = {}  # (mode, task, doc_id) -> row, repetition 1 only
    for r in csv.DictReader(open(args.csv)):
        if r["model"] == args.model and r["rep"] == "1" and r["mode"] in ("loglik", "cot"):
            rows[(r["mode"], r["task"], str(r["doc_id"]))] = r

    wanted = {}
    for (mode, task, doc), r in rows.items():
        if mode != "loglik" or ("cot", task, doc) not in rows:
            continue
        s_ok, c_ok = r["correct"] == "1", rows[("cot", task, doc)]["correct"] == "1"
        if s_ok != c_ok:
            wanted[(task, doc)] = "scoring right, reasoning wrong" if s_ok else "scoring wrong, reasoning right"

    texts = {}
    with open(args.texts) as fh:  # streamed: the file is never loaded whole
        for line in fh:
            if '"cot"' not in line:
                continue
            t = json.loads(line)
            key = (t["task"], str(t["doc_id"]))
            if t.get("model") == args.model and t["mode"] == "cot" and str(t["rep"]) == "1" and key in wanted:
                texts[key] = t["text"]

    out = []
    for key in sorted(wanted, key=lambda k: (wanted[k], k)):
        it, s, c = items[key], rows[("loglik",) + key], rows[("cot",) + key]
        labels = label_list(it)
        lab = lambda i: labels[int(i)] if 0 <= int(i) < len(labels) else "none"
        text = texts.get(key, "")
        mentioned = sorted({m.upper() if len(m) == 1 else m.lower()
                            for m in re.findall(r"(?<![A-Za-z])(" + "|".join(map(re.escape, labels)) + r")(?![A-Za-z])",
                                                text, re.IGNORECASE)})
        out.append(dict(
            group=wanted[key], task=key[0], doc_id=key[1],
            question=question_text(it), options=labels, gold=lab(it["gold"]),
            scoring_answer=lab(s["pred"]), scoring_confidence=round(float(s["conf"]), 3),
            reasoning_answer=lab(c["pred"]), reasoning_parsed=c["parse_ok"] == "1",
            reasoning_tokens=int(float(c["gen_tokens"])), finish_reason=c.get("finish", ""),
            truncated=c.get("finish", "") == "length", labels_mentioned=mentioned,
            reasoning_text=text if text else "[text not found in texts file]"))

    with open(args.out + ".jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    with open(args.out + ".md", "w") as f:
        for g in ("scoring right, reasoning wrong", "scoring wrong, reasoning right"):
            grp = [o for o in out if o["group"] == g]
            f.write(f"# {g}: {len(grp)} questions\n\n")
            for n, o in enumerate(grp, 1):
                f.write(f"## {n}. {o['task']} #{o['doc_id']}\n")
                f.write(f"gold {o['gold']} | scoring {o['scoring_answer']} (conf {o['scoring_confidence']}) | "
                        f"reasoning {o['reasoning_answer']} | parsed {o['reasoning_parsed']} | "
                        f"{o['reasoning_tokens']} tokens | finish {o['finish_reason']} | "
                        f"labels mentioned {', '.join(o['labels_mentioned']) or 'none'}\n\n")
                q = o["question"]
                f.write("Question: " + (q if len(q) <= 700 else q[:700] + " [...]") + "\n\n")
                f.write("Reasoning output:\n```\n" + o["reasoning_text"].strip() + "\n```\n\n")

    a = [o for o in out if o["group"] == "scoring right, reasoning wrong"]
    print(f"scoring right, reasoning wrong: {len(a)} (paper reports 29)")
    print(f"scoring wrong, reasoning right: {len(out) - len(a)} (paper reports 7)")
    print(f"  among the {len(a)}: truncated {sum(o['truncated'] for o in a)}, "
          f"no parseable answer {sum(not o['reasoning_parsed'] for o in a)}, "
          f"text missing {sum(o['reasoning_text'].startswith('[text not') for o in a)}")
    print(f"written {args.out}.jsonl and {args.out}.md")


if __name__ == "__main__":
    main()
