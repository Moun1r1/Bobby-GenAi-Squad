#!/usr/bin/env python3
import concurrent.futures as cf
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import LLM, Agent, SelfCore, squad_solve

M_VALUES = [int(x) for x in os.environ.get("M_VALUES", "1,10,25,50").split(",")]
PROBS_PER_M = int(os.environ.get("PROBS_PER_M", "100"))
WORKERS = int(os.environ.get("WORKERS", "16"))
GSM = os.environ.get("GSM_PATH", "/tmp/gsm8k_test.jsonl")

SOLVE = "Solve this math word problem step by step. End with a line exactly:\n#### <final number>\n\nProblem: {q}"


def _norm(s):
    s = (s or "").replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        return float(s)
    except Exception:
        return None


def gold(ans_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_field or "")
    return _norm(m.group(1)) if m else None


def extract(text):
    m = re.findall(r"####\s*(-?\$?[\d,]+(?:\.\d+)?)", text or "")
    if m:
        return _norm(m[-1])
    nums = re.findall(r"-?\$?[\d,]+(?:\.\d+)?", text or "")
    return _norm(nums[-1]) if nums else None


def match(a, g):
    return a is not None and g is not None and abs(a - g) < 1e-6


# ── SOLO: the whole book in ONE context ───────────────────────────────────────
def solo_book(book):
    prompt = ("Solve ALL of the following problems. Show your step-by-step reasoning for EACH problem (same as you "
              "would for one), and after each problem output a line 'ANSWER <n>: #### <number>' (n = problem number).\n\n")
    for i, p in enumerate(book):
        prompt += f"[Problem {i+1}]\n{p['question']}\n\n"
    # generous per-problem budget (~600/problem, same room a single clean solve gets) so solo is never token-starved;
    # capped at the served ceiling. If a big book still can't fit its reasoning, that's the REAL one-context limit.
    out = LLM()([{"role": "user", "content": prompt}], max_tokens=min(16000, 600 * len(book) + 400), temperature=0.0)
    got = {}
    for m in re.finditer(r"ANSWER\s*(\d+)\s*:\s*#*\s*(-?\$?[\d,]+(?:\.\d+)?)", out):
        got[int(m.group(1))] = _norm(m.group(2))
    correct = sum(1 for i, p in enumerate(book) if match(got.get(i + 1), gold(p["answer"])))
    return correct


# ── BOBBY ROOM: the shared board auto-scales to the book, one focused solve per problem ──
def bobby_book(book):
    solved = {}                                          # the ROOM: idx -> answer (shared board)
    agents = [Agent(SelfCore(identity=f"mathematician #{i}", goal="solve your assigned problem exactly"),
                    llm=LLM()) for i in range(3)]

    def work(agent, unit):
        i, q = unit
        out = agent.llm([{"role": "user", "content": SOLVE.format(q=q)}], max_tokens=768, temperature=0.0)  # == solo solve
        solved[i] = extract(out)
        return solved[i]

    squad_solve(agents, [(i, p["question"]) for i, p in enumerate(book)], work,
                verify=None, split=None, harvest=lambda res, acc: acc, max_passes=len(book) + 5)
    return sum(1 for i, p in enumerate(book) if match(solved.get(i), gold(p["answer"])))


def run_M(M, pool):
    data = pool[:PROBS_PER_M]
    books = [data[i:i + M] for i in range(0, len(data), M)]
    books = [b for b in books if len(b) == M]            # only full books (fair, equal size)
    n = len(books) * M
    # both arms per book, parallel across books
    def one(book):
        return solo_book(book), bobby_book(book), len(book)
    solo_c = bob_c = tot = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for sc, bc, sz in ex.map(one, books):
            solo_c += sc; bob_c += bc; tot += sz
    sa, ba = solo_c / max(1, tot), bob_c / max(1, tot)
    sci = 1.96 * math.sqrt(sa * (1 - sa) / max(1, tot))
    bci = 1.96 * math.sqrt(ba * (1 - ba) / max(1, tot))
    print(f"  book M={M:<3d} ({len(books)} books, {tot} problems): "
          f"SOLO(1 context) {100*sa:5.1f}% ±{100*sci:.1f}  |  BOBBY room {100*ba:5.1f}% ±{100*bci:.1f}  "
          f"|  room−solo {100*(ba-sa):+5.1f} pts", flush=True)
    return {"M": M, "books": len(books), "problems": tot, "solo": sa, "bobby": ba}


def main():
    data = [json.loads(l) for l in open(GSM) if l.strip()]
    import random
    random.Random(0).shuffle(data)
    print(f"== BOOK benchmark · sweep M={M_VALUES} · {PROBS_PER_M} problems/M · model={os.environ.get('BOBBY_LLM_MODEL','?')} "
          f"· exact-match ==\n(solo = whole book in ONE context; bobby = the room/board auto-scales, one clean solve per problem)\n",
          flush=True)
    rows = [run_M(M, data) for M in M_VALUES]
    print("\nRESULT " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
