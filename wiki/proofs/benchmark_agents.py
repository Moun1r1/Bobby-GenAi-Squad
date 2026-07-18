#!/usr/bin/env python3
import collections
import concurrent.futures as cf
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import LLM, Agent, SelfCore, squad_solve

K = int(os.environ.get("K", "100"))            # problems
N_ENS = int(os.environ.get("N_ENS", "5"))       # self-consistency samples (generic baseline)
N_COORD = int(os.environ.get("N_COORD", "5"))   # engine-coordination agents on the shared board (no recursion)
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "4"))   # engine-recursive: max attempts before accepting the majority
WORKERS = int(os.environ.get("WORKERS", "8"))
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
    """Same extraction for every arm (fair): prefer the `#### N` line, else the last number in the text."""
    m = re.findall(r"####\s*(-?\$?[\d,]+(?:\.\d+)?)", text or "")
    if m:
        return _norm(m[-1])
    nums = re.findall(r"-?\$?[\d,]+(?:\.\d+)?", text or "")
    return _norm(nums[-1]) if nums else None


def _one(q, temperature):
    return LLM()([{"role": "user", "content": SOLVE.format(q=q)}], max_tokens=768, temperature=temperature)


# ── Arm 1: solo LLM (one call) ────────────────────────────────────────────────
def arm_solo(q):
    return extract(_one(q, 0.0)), 1


# ── Arm 2: orchestration = self-consistency (N samples, majority vote) ─────────
def arm_orchestration(q):
    outs = [extract(_one(q, 0.8)) for _ in range(N_ENS)]
    outs = [a for a in outs if a is not None]
    if not outs:
        return None, N_ENS
    return collections.Counter(outs).most_common(1)[0][0], N_ENS


# ── Arm 3: engine COORDINATION — persistent-self agents on a shared board, NO recursion ──
def arm_engine_coord(q):
    attempts = []                                   # the shared board (agents see each other's answers)
    agents = [Agent(SelfCore(identity=f"mathematician #{i}",
                             goal="solve the problem exactly; output the final number after '####'"),
                    llm=LLM()) for i in range(3)]

    def work(agent, unit):
        prior = [a for a in attempts if a is not None]
        hint = "" if not prior else f" Peers on the board so far: {prior}. Solve it yourself; give YOUR own final answer."
        out = agent.act(f"Solve exactly. End with '#### <final number>'.{hint}\n\nProblem: {unit[1]}", max_tokens=768)
        a = extract(out)
        attempts.append(a)
        return a

    # squad_solve with N_COORD copies of the unit and verify=None → coordination (shared board) WITHOUT recursion
    squad_solve(agents, [("p", q, 0)] * N_COORD, work, verify=None, split=None, harvest=lambda res, acc: acc)
    at = [a for a in attempts if a is not None]
    if not at:
        return None, N_COORD
    return collections.Counter(at).most_common(1)[0][0], N_COORD


# ── Arm 4: engine RECURSIVE team — squad_solve with verify-by-outcome + split ───
def arm_engine(q):
    calls = [0]
    attempts = []                                   # shared memory across the board (the room)
    agents = [Agent(SelfCore(identity=f"mathematician #{i}",
                             goal="solve the problem exactly; output the final number after '####'"),
                    llm=LLM()) for i in range(3)]

    def work(agent, unit):
        prior = [a for a in attempts if a is not None]
        hint = "" if not prior else (f" Independent earlier attempts concluded {prior}; re-derive from scratch, "
                                     f"do NOT assume them — if you get the same value twice it is confirmed.")
        out = agent.act(f"Solve exactly. End with '#### <final number>'.{hint}\n\nProblem: {unit[1]}", max_tokens=768)
        calls[0] += 1
        a = extract(out)
        attempts.append(a)
        return a

    def verify(unit, acc):                           # run-DON'T-ask: two independent derivations agree → done
        at = [a for a in attempts if a is not None]
        return len(at) >= 2 and at[-1] == at[-2]

    def split(unit):                                 # under-verified → re-queue (recursion; depth self-scales)
        return [(unit[0], unit[1], unit[2] + 1)] if unit[2] + 1 < MAX_DEPTH else []

    squad_solve(agents, [("p", q, 0)], work, verify=verify, split=split,
                harvest=lambda res, acc: acc, max_passes=MAX_DEPTH + 1)
    at = [a for a in attempts if a is not None]
    if not at:
        return None, calls[0]
    return collections.Counter(at).most_common(1)[0][0], calls[0]


def score(name, arm, probs):
    correct = total = calls = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(arm, p["question"]): p for p in probs}
        for f in cf.as_completed(futs):
            g = gold(futs[f]["answer"])
            try:
                a, c = f.result()
            except Exception:
                a, c = None, 0
            calls += c
            total += 1
            if a is not None and g is not None and abs(a - g) < 1e-6:
                correct += 1
            if total % 100 == 0:
                print(f"    …{name[:24]:24s} {total}/{len(probs)} done · {correct} correct · {calls} calls", flush=True)
    acc = correct / max(1, total)
    ci = 1.96 * math.sqrt(acc * (1 - acc) / max(1, total))
    print(f"  {name:38s} {correct:3d}/{total:<3d} = {100*acc:5.1f}%  (95% CI ±{100*ci:.1f})  | "
          f"calls {calls:4d} ({calls/max(1,total):.1f}/problem)", flush=True)
    return {"name": name, "correct": correct, "total": total, "acc": acc, "ci": ci, "calls": calls}


def main():
    import random
    data = [json.loads(l) for l in open(GSM) if l.strip()]
    random.Random(0).shuffle(data)
    probs = data[:K]
    catalog = {
        "solo":  ("SOLO LLM (1 call)", arm_solo),
        "sc":    (f"SELF-CONSISTENCY (N={N_ENS})", arm_orchestration),
        "coord": (f"ENGINE coordination (N={N_COORD})", arm_engine_coord),
        "bobby": ("BOBBY ENGINE (recursive)", arm_engine),
    }
    sel = [a.strip() for a in os.environ.get("ARMS", "solo,bobby").split(",") if a.strip() in catalog]
    arms = [(k, catalog[k][0], catalog[k][1]) for k in sel]
    BATCH = int(os.environ.get("BATCH", "50"))
    print(f"== GSM8K · {len(probs)} problems · model={os.environ.get('BOBBY_LLM_MODEL','?')} · exact-match · "
          f"PAIRED, per-batch duo ==\narms: " + "  vs  ".join(lbl for _, lbl, _ in arms) + f" · batch={BATCH}\n", flush=True)

    tally = {k: {"c": 0, "t": 0, "calls": 0} for k, _, _ in arms}
    disc = {"a_win": 0, "b_win": 0}                 # paired discordance (arm0 vs arm1), when exactly 2 arms

    def task(p):                                     # one problem → run EVERY arm on it (paired, same problem)
        g = gold(p["answer"]); r = {}
        for k, _, fn in arms:
            try:
                a, c = fn(p["question"])
            except Exception:
                a, c = None, 0
            r[k] = (c, a is not None and g is not None and abs(a - g) < 1e-6)
        return r

    done = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:      # parallel ACROSS problems → both arms in flight together
        for f in cf.as_completed([ex.submit(task, p) for p in probs]):
            r = f.result()
            for k, (c, ok) in r.items():
                t = tally[k]; t["t"] += 1; t["calls"] += c; t["c"] += 1 if ok else 0
            if len(arms) == 2:
                ok_a, ok_b = r[arms[0][0]][1], r[arms[1][0]][1]
                disc["a_win"] += 1 if (ok_a and not ok_b) else 0
                disc["b_win"] += 1 if (ok_b and not ok_a) else 0
            done += 1
            if done % BATCH == 0:
                line = "  |  ".join(f"{lbl[:20]:20s} {100*tally[k]['c']/tally[k]['t']:5.1f}% ({tally[k]['c']}/{tally[k]['t']})"
                                    for k, lbl, _ in arms)
                extra = (f"  ‖ {arms[1][0]}✓{arms[0][0]}✗={disc['b_win']} · {arms[0][0]}✓{arms[1][0]}✗={disc['a_win']}"
                         if len(arms) == 2 else "")
                print(f"  [@{done:4d}] {line}{extra}", flush=True)

    print("\n== FINAL ==", flush=True)
    results = []
    for k, lbl, _ in arms:
        t = tally[k]; acc = t["c"] / max(1, t["t"]); ci = 1.96 * math.sqrt(acc * (1 - acc) / max(1, t["t"]))
        print(f"  {lbl:32s} {t['c']:4d}/{t['t']:<4d} = {100*acc:5.1f}%  (95% CI ±{100*ci:.1f})  | "
              f"calls {t['calls']} ({t['calls']/max(1,t['t']):.1f}/problem)", flush=True)
        results.append({"arm": k, "name": lbl, "correct": t["c"], "total": t["t"], "acc": acc, "ci": ci, "calls": t["calls"]})
    if len(arms) == 2:
        net = disc["b_win"] - disc["a_win"]
        print(f"\n  PAIRED (same problems): {arms[1][0]} wins {disc['b_win']} where {arms[0][0]} fails · "
              f"{arms[0][0]} wins {disc['a_win']} where {arms[1][0]} fails · net {net:+d} for {arms[1][0]}", flush=True)
    print("\nRESULT " + json.dumps({"arms": results, "discordant": disc}), flush=True)


if __name__ == "__main__":
    main()
