"""
energy_per_item.py
==================
Measure accuracy, time and CPU-package energy per multiple-choice item for a
GGUF model on CPU (llama-cpp-python), in up to three answering modes:

  loglik  Option log-likelihood scoring on the exact lm-eval context (the mode
          used for the logged accuracies). The context is processed once and
          each option's continuation tokens are scored on top of it.
  letter  Chat-formatted prompt; the model writes only the answer
          (letter, or yes/no/maybe for PubMedQA), greedy, max 8 tokens.
  cot     Chat-formatted prompt; the model reasons step by step and ends with
          "Answer: X", greedy, max --cot-max-tokens tokens.

Energy is read per item from the Intel RAPL package counter
(/sys/class/powercap/intel-rapl:0/energy_uj). Before each mode the idle package
power is measured for --idle-secs seconds; each item is recorded with gross
energy and net energy (gross minus idle power x item time). The KV cache is
cleared before every item, so each item pays its full prompt-processing cost.

RAPL access: on recent kernels energy_uj is readable by root only. Either run
with sudo, or once per boot:  sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj

Usage
-----
    python energy_per_item.py --model MediLITE-QA.Q4_K_M.gguf --label medilite_q4 \
        --items items.jsonl --modes loglik letter cot --ref medilite_q4 --out energy_items.csv

Rows are appended to --out. One JSON line per mode (run settings, software
versions, idle power, timings, agreement with the logged predictions) is
appended to <out>_runs.jsonl. Finished (model, mode, rep, item) rows are skipped,
so an interrupted run can be restarted with the same command.
"""

import argparse
import csv
import json
import os
import platform
import re
import sys
import time

import numpy as np

FIELDS = ["model", "mode", "rep", "task", "doc_id", "gold", "pred", "correct", "parse_ok", "conf",
          "prompt_tokens", "gen_tokens", "seconds", "joules", "idle_w", "joules_net", "finish"]


# ------------------------------------------------------------------ energy
class Rapl:
    def __init__(self, path):
        self.path = path
        name = open(os.path.join(path, "name")).read().strip()
        if not name.startswith("package"):
            print(f"[warn] {path} is '{name}', not a CPU package domain")
        self.max = int(open(os.path.join(path, "max_energy_range_uj")).read())
        self.f = os.path.join(path, "energy_uj")
        self.read()  # fails early if not readable

    def read(self):
        with open(self.f) as fh:
            return int(fh.read())

    def joules(self, a, b):
        d = b - a
        if d < 0:
            d += self.max
        return d / 1e6


def idle_power(rapl, secs):
    a, t0 = rapl.read(), time.perf_counter()
    time.sleep(secs)
    b, t1 = rapl.read(), time.perf_counter()
    return rapl.joules(a, b) / (t1 - t0)


# ------------------------------------------------------------------ prompts and parsing
def question_text(item):
    ctx = item["context"].rstrip()
    if ctx.endswith("Answer:"):
        ctx = ctx[: -len("Answer:")].rstrip()
    labels = [c.strip() for c in item["continuations"]]
    letters = all(len(x) == 1 and x.isalpha() for x in labels)
    if not letters:
        ctx += "\nOptions: " + ", ".join(labels)
    return ctx, labels, letters


def letter_prompt(item):
    q, labels, letters = question_text(item)
    what = "letter of the correct option" if letters else "correct option"
    return f"{q}\n\nReply with only the {what} ({', '.join(labels)}) and nothing else."


def cot_prompt(item):
    q, labels, _ = question_text(item)
    return (f"{q}\n\nThink step by step, briefly. Then write a final line of the form "
            f"'Answer: X', where X is one of: {', '.join(labels)}.")


def _label_regex(labels):
    alt = "|".join(re.escape(x) for x in sorted(labels, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])({alt})(?![A-Za-z])", re.IGNORECASE)


RUNAWAY = re.compile(r"\[/?INST\]|<\||\n\s*Think step by step", re.IGNORECASE)


def option_texts(item):
    """Option label -> option text (lower case), read from the question prompt."""
    labels = [c.strip() for c in item["continuations"]]
    out = {}
    for lab in labels:
        m = re.search(rf"^\s*{re.escape(lab)}\.\s*(.+)$", item["context"], re.MULTILINE) if len(lab) == 1 else None
        out[lab] = m.group(1).strip().rstrip(".").lower() if m else lab.lower()
    return out


def parse_answer(text, labels, mode, item=None):
    """Index of the chosen option, or -1.

    letter: first standalone option label in the output.
    cot:    FIRST-ANSWER rule (version 2): the output is cut at the first sign of a
            runaway continuation ([INST], [/INST], <|, a repeated "Think step by step");
            then the first "Answer: X" whose X is a label or the exact option text is
            taken; otherwise a label or option text that starts the output.
    """
    low = [x.lower() for x in labels]
    opt = option_texts(item) if item else {l: l.lower() for l in labels}
    def match(tok):
        t = tok.strip().strip(".():*").lower()
        if t in low:
            return low.index(t)
        for i, l in enumerate(labels):
            if t and t == opt[l]:
                return i
        return -1
    if mode == "cot":
        head = RUNAWAY.split(text, maxsplit=1)[0]
        for m in re.finditer(r"answer\s*(?:is)?\s*[:\-]\s*\(?([^\n]+)", head, re.IGNORECASE):
            cand = m.group(1).strip()
            i = match(cand)
            if i < 0:
                i = match(re.split(r"[\s.)]", cand, maxsplit=1)[0])
            if i >= 0:
                return i
        first = head.strip().split("\n", 1)[0] if head.strip() else ""
        i = match(re.split(r"[\s.)]", first, maxsplit=1)[0]) if first else -1
        return i if i >= 0 else match(first)
    hits = _label_regex(labels).findall(text)
    return low.index(hits[0].lower()) if hits else -1


# ------------------------------------------------------------------ model calls
def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def last_logprobs(llm):
    """Log-probabilities for the next token after the last evaluated token.
    llama-cpp-python 0.3.x does not keep logits in llm.scores unless logits_all=True
    (which would compute the output layer for every prompt token and distort the
    energy), so they are read from the context directly."""
    ptr = llm._ctx.get_logits_ith(-1)
    return log_softmax(np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).copy())


def encode_pair(llm, context, continuation, add_bos):
    """Mirror lm-eval: trailing spaces of the context move to the continuation."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = llm.tokenize((context + continuation).encode("utf-8"), add_bos=add_bos, special=False)
    ctx = llm.tokenize(context.encode("utf-8"), add_bos=add_bos, special=False)
    return whole[: len(ctx)], whole[len(ctx):]


def run_loglik(llm, item, add_bos):
    pairs = [encode_pair(llm, item["context"], c, add_bos) for c in item["continuations"]]
    ctx = pairs[0][0]
    llm.reset()
    llm.eval(ctx)
    n_ctx = llm.n_tokens
    base = last_logprobs(llm)
    lls, n_gen = [], 0
    for p_ctx, cont in pairs:
        if p_ctx != ctx:  # tokenisation boundary differs for this option: score it from scratch
            llm.reset()
            llm.eval(p_ctx)
            lp = last_logprobs(llm)
        else:
            llm.n_tokens = n_ctx  # rewind to the end of the shared context
            lp = base
        total = 0.0
        for j, tok in enumerate(cont):
            total += lp[tok]
            if j < len(cont) - 1:
                llm.eval([tok])
                n_gen += 1
                lp = last_logprobs(llm)
        lls.append(total)
        if p_ctx != ctx:
            llm.reset()
            llm.eval(ctx)
    probs = np.exp(log_softmax(lls))
    pred = int(np.argmax(lls))
    return pred, float(probs[pred]), len(ctx), n_gen, "", True


PHI3_TEMPLATE = "<|user|>\n{content}<|end|>\n<|assistant|>\n"
PHI3_STOP = ["<|end|>", "<|user|>", "<|endoftext|>"]


def check_phi3_tokens(llm):
    """Abort if the Phi-3.5 control tokens are not single special tokens in this GGUF."""
    for t in ("<|user|>", "<|end|>", "<|assistant|>"):
        ids = llm.tokenize(t.encode("utf-8"), add_bos=False, special=True)
        if len(ids) != 1:
            raise SystemExit(f"'{t}' is not a single special token in this model ({ids}); "
                             f"--prompt-format phi3 cannot be used with this file")


def run_chat(llm, item, mode, cot_max, seed, prompt_format="gguf"):
    labels = [c.strip() for c in item["continuations"]]
    prompt = cot_prompt(item) if mode == "cot" else letter_prompt(item)
    max_tokens = cot_max if mode == "cot" else 8
    llm.reset()
    if prompt_format == "phi3":
        out = llm.create_completion(PHI3_TEMPLATE.format(content=prompt), temperature=0.0, top_k=1,
                                    max_tokens=max_tokens, seed=seed, stop=PHI3_STOP)
        text = out["choices"][0]["text"] or ""
    else:
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         temperature=0.0, top_k=1, max_tokens=max_tokens, seed=seed)
        text = out["choices"][0]["message"]["content"] or ""
    pred = parse_answer(text, labels, mode, item)
    u = out.get("usage", {})
    return (pred, float("nan"), u.get("prompt_tokens", -1), u.get("completion_tokens", -1),
            out["choices"][0].get("finish_reason", ""), pred >= 0, text)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the .gguf file")
    ap.add_argument("--label", required=True, help="short model name written to the CSV, e.g. medilite_q4")
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--modes", nargs="+", default=["loglik", "letter", "cot"], choices=["loglik", "letter", "cot"])
    ap.add_argument("--reps", type=int, default=2, help="repetitions for loglik and letter")
    ap.add_argument("--cot-reps", type=int, default=1)
    ap.add_argument("--cot-max-tokens", type=int, default=384)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--add-bos", action="store_true", help="prepend BOS in loglik mode (lm-eval default: off)")
    ap.add_argument("--chat-format", default=None, help="override the chat template stored in the GGUF")
    ap.add_argument("--prompt-format", choices=("gguf", "phi3"), default="gguf",
                    help="gguf: chat template from the model file (falls back to llama-2 if absent); "
                         "phi3: explicit Phi-3/3.5 template <|user|>...<|end|><|assistant|>, stop at <|end|>")
    ap.add_argument("--warmup", type=int, default=3, help="unrecorded warm-up items per mode")
    ap.add_argument("--idle-secs", type=float, default=30.0)
    ap.add_argument("--cooldown", type=float, default=60.0, help="seconds between modes")
    ap.add_argument("--rapl", default="/sys/class/powercap/intel-rapl:0")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N items (timing test)")
    ap.add_argument("--ref", default=None, help="log folder name to compare loglik predictions with")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-text", default=None, help="optional JSONL file for generated texts")
    ap.add_argument("--out", default="energy_items.csv")
    args = ap.parse_args()

    from llama_cpp import Llama

    items = [json.loads(line) for line in open(args.items)]
    if args.limit:
        items = items[: args.limit]
    rapl = Rapl(args.rapl)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for r in csv.DictReader(fh):
                done.add((r["model"], r["mode"], r["rep"], r["task"], str(r["doc_id"])))
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    tfh = open(args.save_text, "a") if args.save_text else None

    t_load = time.perf_counter()
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_threads=args.threads, n_threads_batch=args.threads,
                n_gpu_layers=0, logits_all=False, seed=args.seed, chat_format=args.chat_format, verbose=False)
    meta = {"label": args.label, "model": os.path.abspath(args.model), "load_seconds": time.perf_counter() - t_load,
            "threads": args.threads, "n_ctx": args.n_ctx, "cpu": platform.processor() or platform.machine(),
            "python": sys.version.split()[0]}
    try:
        import llama_cpp
        meta["llama_cpp_python"] = llama_cpp.__version__
    except Exception:
        pass
    meta["gguf_chat_format"] = getattr(llm, "chat_format", None)
    meta["prompt_format"] = args.prompt_format
    meta["cot_parse_rule"] = "first-answer (v2)"
    if args.prompt_format == "phi3" and any(m in args.modes for m in ("letter", "cot")):
        check_phi3_tokens(llm)
    print(json.dumps(meta))
    runs_path = os.path.splitext(args.out)[0] + "_runs.jsonl"

    for k, mode in enumerate(args.modes):
        if k:
            print(f"cooldown {args.cooldown:.0f} s")
            time.sleep(args.cooldown)
        call = (lambda it: run_loglik(llm, it, args.add_bos) + ("",)) if mode == "loglik" else \
               (lambda it, m=mode: run_chat(llm, it, m, args.cot_max_tokens, args.seed, args.prompt_format))
        for it in items[: args.warmup]:
            call(it)
        idle_w = idle_power(rapl, args.idle_secs)
        print(f"[{args.label} / {mode}] idle package power {idle_w:.2f} W")
        reps = args.cot_reps if mode == "cot" else args.reps
        n_new, agree, n_ref = 0, 0, 0
        t_mode = time.time()
        for rep in range(1, reps + 1):
            for i, it in enumerate(items):
                key = (args.label, mode, str(rep), it["task"], str(it["doc_id"]))
                if key in done:
                    continue
                e0, t0 = rapl.read(), time.perf_counter()
                pred, conf, n_prompt, n_gen, finish, ok, text = call(it)
                t1, e1 = time.perf_counter(), rapl.read()
                sec, joules = t1 - t0, rapl.joules(e0, e1)
                w.writerow(dict(model=args.label, mode=mode, rep=rep, task=it["task"], doc_id=it["doc_id"],
                                gold=it["gold"], pred=pred, correct=int(pred == it["gold"]), parse_ok=int(ok),
                                conf=f"{conf:.6f}", prompt_tokens=n_prompt, gen_tokens=n_gen, seconds=f"{sec:.4f}",
                                joules=f"{joules:.4f}", idle_w=f"{idle_w:.3f}",
                                joules_net=f"{joules - idle_w * sec:.4f}", finish=finish))
                fh.flush()
                if tfh and text:
                    tfh.write(json.dumps({"model": args.label, "mode": mode, "rep": rep, "task": it["task"],
                                          "doc_id": it["doc_id"], "text": text}) + "\n")
                    tfh.flush()
                n_new += 1
                if mode == "loglik" and rep == 1 and args.ref and args.ref in it.get("ref", {}):
                    n_ref += 1
                    agree += int(pred == it["ref"][args.ref]["pred"])
                if (i + 1) % 20 == 0:
                    print(f"  rep {rep}: {i + 1}/{len(items)} items, last {sec:.2f} s, {joules:.1f} J")
        print(f"[{args.label} / {mode}] {n_new} new rows")
        if n_ref:
            print(f"[{args.label} / loglik] prediction agrees with logged '{args.ref}' on "
                  f"{100 * agree / n_ref:.1f}% of {n_ref} items")
        with open(runs_path, "a") as rf:
            rf.write(json.dumps(dict(meta, mode=mode, reps=reps, items=len(items), new_rows=n_new,
                                     idle_w=idle_w, idle_secs=args.idle_secs, warmup=args.warmup,
                                     cot_max_tokens=args.cot_max_tokens, add_bos=args.add_bos,
                                     chat_format=args.chat_format, seed=args.seed,
                                     started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_mode)),
                                     finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     ref=args.ref, ref_items=n_ref,
                                     ref_agreement=(agree / n_ref) if n_ref else None)) + "\n")
    fh.close()
    if tfh:
        tfh.close()
    print(f"written to {args.out} and {runs_path}")


if __name__ == "__main__":
    main()
