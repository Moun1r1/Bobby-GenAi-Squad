#!/usr/bin/env python3
"""Reproduce the DECISIVE long-run compaction result using the generic component.

48 steps, hard context-clear every 8 (auto via Agent.compact_every), drift-proof prime enumeration.
PINNED (self-core + progress survive the wipe) vs NAIVE (everything in the wiped window).
Expected: NAIVE restarts low after every wipe (~largest 37); PINNED resumes and climbs (~largest 200+).
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import Agent, SelfCore, LLM

N, COMPACT_EVERY, WINDOW = 48, 8, 5
GOAL = ("list prime numbers in strictly increasing order — each step output the single next prime that is "
        "larger than the largest listed so far")


def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    i = 3
    while i * i <= n:
        if n % i == 0: return False
        i += 2
    return True

def first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group(0)) if m else None


def run(pinned):
    core = SelfCore(identity="Atlas, a precise enumerator", goal=GOAL,
                    constraints=["output ONLY the next prime, digits only", "never repeat or go backward"])
    a = Agent(core, LLM(temperature=0.3), window=WINDOW, pinned=pinned, compact_every=COMPACT_EVERY)
    seen, largest, forward, errors, resume = set(), 1, 0, 0, []
    for i in range(1, N + 1):
        compact_now = i > 1 and (i - 1) % COMPACT_EVERY == 0
        if not pinned and i == 1:
            a.observe(f"Task: {GOAL}.")                      # naive: goal stated once, into the (wipeable) window
        task = "Output ONLY the next prime number (just the digits)."
        if not pinned and compact_now:
            task = "Continue the sequence.\n" + task         # naive resume: nothing pinned to resume from
        out = a.act(task, max_tokens=16)
        n = first_int(out)
        valid = n is not None and is_prime(n) and n > largest and n not in seen
        if valid:
            seen.add(n); a.record(str(n)); largest = n; forward += 1
        else:
            errors += 1
        if compact_now:
            resume.append((i, n, valid))
    return {"largest": largest, "forward": forward, "errors": errors, "resume": resume}


def main():
    print(f"LONG-RUN COMPACTION (via bobby_squad) — {N} steps, wipe every {COMPACT_EVERY}, window {WINDOW}\n")
    with ThreadPoolExecutor(max_workers=2) as ex:
        P = ex.submit(run, True); Nz = ex.submit(run, False)
        P, Nz = P.result(), Nz.result()
    for lbl, r in (("NAIVE ", Nz), ("PINNED", P)):
        print(f"{lbl}: largest {r['largest']:>4} · valid {r['forward']:>2}/{N} · errors {r['errors']} · "
              f"resumes " + " ".join(f"@{s}:{'fwd' if f else 'back'}" for s, _, f in r["resume"]))
    print(f"\nGAIN: largest {Nz['largest']} → {P['largest']} · valid {Nz['forward']} → {P['forward']} · "
          f"clean resumes {sum(f for *_ , f in Nz['resume'])}/{len(Nz['resume'])} → "
          f"{sum(f for *_ , f in P['resume'])}/{len(P['resume'])}")


if __name__ == "__main__":
    main()
