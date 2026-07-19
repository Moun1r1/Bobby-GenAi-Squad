#!/usr/bin/env python3
"""Statistically-rigorous SOMA finetune benchmark.

Evaluates BASE and BASE+LoRA on the same held-out set in one process (exact
pairing), then reports:
  * per-model accuracy with a bootstrap 95% CI,
  * the PAIRED accuracy delta (finetuned - base) with a paired-bootstrap 95% CI,
  * a McNemar-style discordant-pair count,
  * a CI-separated gain verdict (lo > gain_min) mirroring proving.confirm_gain,
  * per-kind and per-seed breakdowns.

Emits a JSON verdict the orchestrator consumes to gate the model swap.
"""
import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def canon(line):
    return re.sub(r"^[\W_]+|[\W_]+$", "", line.strip().lower())


_DROP = {"think", "answer", "answers", "output", ""}


def parse_set(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return {c for x in text.splitlines() if (c := canon(x)) and c not in _DROP}


def grade(pred_text, gold):
    pred = parse_set(pred_text)
    g = {canon(x) for x in gold if canon(x)}
    if not g:
        return 1 if not pred else 0
    return 1 if pred == g else 0


def gen_correct(model, tok, rows, max_new):
    out_flags = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        try:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, enable_thinking=False).to(model.device)
        except TypeError:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        out_flags.append(grade(tok.decode(o[0][plen:], skip_special_tokens=True), r["gold"]))
    return out_flags


def boot_ci(vals, B=3000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(B):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]


def paired_boot_ci(base, ft, B=3000, seed=0):
    rng = random.Random(seed)
    n = len(base)
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(ft[i] - base[i] for i in idx) / n
        diffs.append(d)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--gain-min", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(x) for x in open(args.test)]
    if args.limit and len(rows) > args.limit:
        rng = random.Random(0)
        rng.shuffle(rows)                                # deterministic subsample, mixes seeds/kinds
        rows = rows[: args.limit]

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    getattr(base_model, "eval")()
    base = gen_correct(base_model, tok, rows, args.max_new)

    from peft import PeftModel
    ft_model = PeftModel.from_pretrained(base_model, args.adapter)
    getattr(ft_model, "eval")()
    ft = gen_correct(ft_model, tok, rows, args.max_new)

    n = len(rows)
    b_acc, f_acc = sum(base) / n, sum(ft) / n
    delta = f_acc - b_acc
    b_lo, b_hi = boot_ci(base)
    f_lo, f_hi = boot_ci(ft)
    d_lo, d_hi = paired_boot_ci(base, ft)
    only_ft = sum(1 for i in range(n) if ft[i] and not base[i])       # McNemar discordant
    only_bs = sum(1 for i in range(n) if base[i] and not ft[i])
    separated = d_lo > args.gain_min

    def by(key):
        agg = {}
        for i, r in enumerate(rows):
            k = r.get(key, "?")
            a = agg.setdefault(k, [0, 0, 0])
            a[0] += base[i]
            a[1] += ft[i]
            a[2] += 1
        return {k: {"base": round(v[0] / v[2], 3), "ft": round(v[1] / v[2], 3), "n": v[2]} for k, v in agg.items()}

    verdict = {
        "n": n, "base_acc": round(b_acc, 4), "base_ci95": [round(b_lo, 4), round(b_hi, 4)],
        "ft_acc": round(f_acc, 4), "ft_ci95": [round(f_lo, 4), round(f_hi, 4)],
        "delta": round(delta, 4), "delta_ci95": [round(d_lo, 4), round(d_hi, 4)],
        "mcnemar_only_ft": only_ft, "mcnemar_only_base": only_bs,
        "gain_min": args.gain_min, "ci_separated": bool(separated),
        "verdict": "ACCEPT" if separated else "REJECT",
        "by_kind": by("kind"), "by_seed": by("seed"),
    }
    print(json.dumps(verdict, indent=2))
    if args.out:
        open(args.out, "w").write(json.dumps(verdict, indent=2))
    print(f"\n{'='*70}\nBASE {b_acc:.1%} [{b_lo:.1%},{b_hi:.1%}]  ->  FINETUNED {f_acc:.1%} [{f_lo:.1%},{f_hi:.1%}]")
    print(f"DELTA {delta:+.1%}  95% CI [{d_lo:+.1%}, {d_hi:+.1%}]  (McNemar +{only_ft}/-{only_bs})")
    print(f"VERDICT: {'ACCEPT (CI-separated gain)' if separated else 'REJECT (CI includes 0)'}\n{'='*70}")


if __name__ == "__main__":
    main()
