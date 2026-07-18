#!/usr/bin/env python3
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import LLM, Agent, SelfCore, squad_solve

M_VALUES = [int(x) for x in os.environ.get("M_VALUES", "1,10,25").split(",")]
PROBS_PER_M = int(os.environ.get("PROBS_PER_M", "50"))
K_SAMPLES = int(os.environ.get("K_SAMPLES", "3"))
WORKERS = int(os.environ.get("WORKERS", "12"))
GSM = os.environ.get("GSM_PATH", "/tmp/gsm8k_test.jsonl")
EMBED_URL = os.environ.get("BOBBY_EMBED_URL", "http://host.docker.internal:11435/api/embed")
EMBED_MODEL = os.environ.get("BOBBY_EMBED_MODEL", "nomic-embed-text")

SOLVE = "Solve this math word problem step by step. End with a line exactly:\n#### <final number>\n\nProblem: {q}"


def _norm(s):
    s = (s or "").replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except Exception:
        return None


def gold(a):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", a or "")
    return _norm(m.group(1)) if m else None


def extract(t):
    m = re.findall(r"####\s*(-?\$?[\d,]+(?:\.\d+)?)", t or "")
    if m:
        return _norm(m[-1])
    nums = re.findall(r"-?\$?[\d,]+(?:\.\d+)?", t or "")
    return _norm(nums[-1]) if nums else None


def match(a, g):
    return a is not None and g is not None and abs(a - g) < 1e-6


def _one(q, temperature):
    return LLM()([{"role": "user", "content": SOLVE.format(q=q)}], max_tokens=768, temperature=temperature)


def embed(text):
    body = json.dumps({"model": EMBED_MODEL, "input": text[:2000]}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    v = d.get("embeddings", [d.get("embedding")])[0] if isinstance(d.get("embeddings"), list) else d.get("embedding")
    return v


# ── Arm 1: single LLM, the whole book in one chat ─────────────────────────────
def arm_solo(book):
    prompt = ("Solve ALL of the following problems. Show your step-by-step reasoning for EACH problem (same as you "
              "would for one), and after each output a line 'ANSWER <n>: #### <number>' (n = problem number).\n\n")
    for i, p in enumerate(book):
        prompt += f"[Problem {i+1}]\n{p['question']}\n\n"
    out = LLM()([{"role": "user", "content": prompt}], max_tokens=min(16000, 600 * len(book) + 400), temperature=0.0)
    got = {}
    for m in re.finditer(r"ANSWER\s*(\d+)\s*:\s*#*\s*(-?\$?[\d,]+(?:\.\d+)?)", out):
        got[int(m.group(1))] = _norm(m.group(2))
    return {i: got.get(i + 1) for i in range(len(book))}


# ── Arm 2: encoder + LLM — K samples per problem, encoder picks the central one ─
def _medoid_answer(q):
    sols = [_one(q, 0.0)] + [_one(q, 0.8) for _ in range(max(1, K_SAMPLES - 1))]   # greedy anchor + samples (floor)
    ans = [extract(s) for s in sols]
    valid = [(s, a) for s, a in zip(sols, ans) if a is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0][1]
    try:
        import numpy as np
        vecs = np.array([embed(s) for s, _ in valid], dtype=float)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        sim = vecs @ vecs.T                              # pairwise cosine; medoid = highest total similarity
        central = int(sim.sum(axis=1).argmax())
        return valid[central][1]
    except Exception:                                   # embed unavailable → fall back to plain vote
        import collections
        return collections.Counter([a for _, a in valid]).most_common(1)[0][0]


def arm_encoder(book):
    with cf.ThreadPoolExecutor(min(WORKERS, len(book))) as ex:
        outs = list(ex.map(lambda p: _medoid_answer(p["question"]), book))
    return {i: outs[i] for i in range(len(book))}


# ── Arm 3: agent auto-book recursive — the room drains the book, floored verify ─
def _floored_solve(q):
    base = extract(_one(q, 0.0))                         # the base answer = a clean single solve (the floor)
    d1, d2 = extract(_one(q, 0.5)), extract(_one(q, 0.5))
    if d1 is not None and d1 == d2 and d1 != base:       # two independent re-derivations AGREE on a different value
        return d1                                        # → strong evidence the base was wrong; override
    return base                                          # otherwise keep the base (never regress on noise)


def arm_agent(book):
    solved = {}
    agents = [Agent(SelfCore(identity=f"mathematician #{i}", goal="solve your assigned problem exactly"),
                    llm=LLM()) for i in range(3)]

    def work(agent, unit):
        i, q = unit
        solved[i] = _floored_solve(q)                    # bounded, focused, floored verify (can't regress on noise)
        return solved[i]

    squad_solve(agents, [(i, p["question"]) for i, p in enumerate(book)], work,
                verify=None, split=None, harvest=lambda res, acc: acc, max_passes=len(book) + 5)
    return {i: solved.get(i) for i in range(len(book))}


ARMS = [("1. SINGLE LLM (one chat)", arm_solo),
        ("2. ENCODER + LLM (central select)", arm_encoder),
        ("3. AGENT auto-book recursive", arm_agent)]


def run_M(M, pool):
    data = pool[:PROBS_PER_M]
    books = [data[i:i + M] for i in range(0, len(data), M)]
    books = [b for b in books if len(b) == M]
    tot = len(books) * M
    agg = {name: 0 for name, _ in ARMS}

    def one_book(book):
        golds = [gold(p["answer"]) for p in book]
        res = {}
        for name, fn in ARMS:
            got = fn(book)
            res[name] = sum(1 for i in range(len(book)) if match(got.get(i), golds[i]))
        return res

    with cf.ThreadPoolExecutor(max(2, WORKERS // 2)) as ex:
        for res in ex.map(one_book, books):
            for name in agg:
                agg[name] += res[name]
    print(f"  book M={M:<3d} ({len(books)} books, {tot} problems vs control):", flush=True)
    for name, _ in ARMS:
        acc = agg[name] / max(1, tot)
        ci = 1.96 * math.sqrt(acc * (1 - acc) / max(1, tot))
        print(f"      {name:34s} {agg[name]:3d}/{tot} = {100*acc:5.1f}%  (95% CI ±{100*ci:.1f})", flush=True)
    return {"M": M, "problems": tot, **{name: agg[name] / max(1, tot) for name, _ in ARMS}}


def main():
    data = [json.loads(l) for l in open(GSM) if l.strip()]
    import random
    random.Random(0).shuffle(data)
    print(f"== TRIO benchmark · sweep M={M_VALUES} · {PROBS_PER_M} problems/M · K={K_SAMPLES} · "
          f"model={os.environ.get('BOBBY_LLM_MODEL','?')} · exact-match vs control ==", flush=True)
    print("(M=1 is the fairness gate — all three arms must score ~equal on a single problem)\n", flush=True)
    rows = [run_M(M, data) for M in M_VALUES]
    print("\nRESULT " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
