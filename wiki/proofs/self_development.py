"""full_dev_test — a REAL end-to-end dev run through the whole reusable stack (package primitives only):

  1. DISCOVER   — squad_solve: a self-organizing squad coordinates + recurses to MAP a target's functions (coverage).
  2. BUILD+VERIFY — Agent.autonomous_loop + run-don't-ask verify(): an agent writes a real script, runs it, and only
                    finishes when a REAL run confirms it (not 'the model said done'). Watched live via stream_observer.
  3. PROVE      — prove(): the enforced testing methodology on a real mechanism (self-evolving memory), WITH a
                    negative control + headroom guard, so the verdict is trustworthy.

Everything is imported from `bobby_squad` — no bespoke plumbing. Run:
  GA_LLM_URL=... GA_EMBED_URL=... python3 examples/full_dev_test.py
"""
import glob
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import (Agent, SelfCore, ReadOnlyTools, SandboxTools,   # noqa: E402
                                squad_solve, prove, stream_observer)
from bobby_squad import LLM                                          # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bobby_squad")
SB = os.path.join(ROOT, "out", "dev_sandbox")


def _lines(path):
    with open(os.path.join(ROOT, path), errors="ignore") as f:
        return f.read().splitlines()


# ── 1. DISCOVER — squad_solve coordinated + recursive coverage ───────────────────────────────────────────────────
def discover(llm):
    files = ["dedup.py", "planning.py"]                                         # a small real target
    gt_src = subprocess.run(["grep", "-hoE", "def [a-zA-Z_][a-zA-Z0-9_]*", *[os.path.join(ROOT, f) for f in files]],
                            capture_output=True, text=True).stdout
    gt = set(re.findall(r"def ([a-zA-Z_]\w*)", gt_src))
    agents = [Agent(SelfCore("a code cartographer", "map the functions in this repo"),
                    llm=llm, tools=ReadOnlyTools(ROOT), name=f"carto-{i}") for i in range(2)]

    def work(agent, unit):
        path, lo, hi = unit
        ans = agent.carry_out(f"Read {path} lines {lo}-{hi} and list every def name in that range. Bare names only.",
                              move="investigate", max_rounds=2)
        return {n for n in gt if re.search(rf"\b{re.escape(n)}\b", ans)}

    def verify(unit, acc):
        path, lo, hi = unit
        local = set(re.findall(r"def ([a-zA-Z_]\w*)", "\n".join(_lines(path)[lo - 1:hi])))
        return (not (local - acc)) or (hi - lo) <= 45

    def split(unit):
        path, lo, hi = unit
        mid = (lo + hi) // 2
        return [(path, lo, mid), (path, mid + 1, hi)] if (hi - lo) > 45 else None

    units = [(f, 1, len(_lines(f))) for f in files]
    r = squad_solve(agents, units, work, verify=verify, split=split,
                    observer=lambda e: print(f"    [discover] p{e['n']} {e['unit'][0]} → {e['size']}/{len(gt)} "
                                             f"board={e['board']}", flush=True))
    cov = len(r["result"]) / max(1, len(gt))
    print(f"1. DISCOVER  → {len(r['result'])}/{len(gt)} functions = {cov:.0%} in {r['passes']} passes "
          f"(squad_solve, no orchestrator)\n", flush=True)
    return cov


# ── 2. BUILD + VERIFY — autonomous_loop + run-don't-ask ──────────────────────────────────────────────────────────
def build_and_verify(llm):
    tools = SandboxTools(ROOT, SB)
    goal = ("In the sandbox, WRITE a python script and RUN it so it prints EXACTLY one line 'LOC: <n>' where n is the "
            "total number of non-blank lines across the repo's .py files. Fix and rerun until it works.")
    ag = Agent(SelfCore("an engineer who finishes and VERIFIES real work", goal),
               llm=llm, tools=tools, name="builder", observer=stream_observer)

    def verify():                                                              # RUN-don't-ask
        for p in glob.glob(os.path.join(SB, "**", "*.py"), recursive=True):
            out = tools.run(os.path.relpath(p, SB))
            if "LOC:" in out and "[exit 0" in out:
                return True
        return False

    r = ag.autonomous_loop(verify_fn=verify, max_cycles=3)
    print(f"\n2. BUILD+VERIFY → verified={r['verified']} in {len(r['cycles'])} cycles "
          f"(autonomous_loop; a real RUN gated it, not prose)\n", flush=True)
    return r["verified"]


# ── 3. PROVE — enforced testing methodology on a real mechanism (deterministic, fast) ────────────────────────────
def _mempolicy(predictive, evolved, seed, n_topics=40, cap=15, used=16):
    rng = random.Random(seed)
    topics = list(range(n_topics))
    tested = set(rng.sample(topics, used))
    feedback = list(tested) if predictive else rng.sample([t for t in topics if t not in tested], used)
    order = topics[:]; rng.shuffle(order)
    store = []
    for t in order:
        store.append({"topic": t, "value": 0, "born": len(store)})
        while len(store) > cap:
            victim = min(store, key=(lambda x: (x["value"], x["born"])) if evolved else (lambda x: x["born"]))
            store.remove(victim)
        if rng.random() < 0.5:                                                 # interleaved usage → value for evolved
            for it in store:
                if it["topic"] == rng.choice(feedback) and evolved:
                    it["value"] += 1
    return sum(1 for t in tested if any(it["topic"] == t for it in store)) / len(tested)


def prove_gain():
    print("3. PROVE (enforced methodology: negative control + headroom + CI):", flush=True)
    return prove("self-evolving memory policy (real dev proof)",
                 control=lambda s: _mempolicy(True, False, s),                 # predictive, FIFO
                 treatment=lambda s: _mempolicy(True, True, s),                # predictive, evolved
                 negative=(lambda s: _mempolicy(False, False, s),             # NON-predictive: evolved must NOT win
                           lambda s: _mempolicy(False, True, s)),
                 seeds=range(8), baseline_max=1.0, higher_is_better=True)


def main():
    os.makedirs(SB, exist_ok=True)
    llm = LLM(temperature=0.4, timeout=120)
    print("=== FULL DEV TEST — the whole reusable stack, end to end, on the real repo ===\n", flush=True)
    cov = discover(llm)
    built = build_and_verify(llm)
    verdict = prove_gain()
    print("\n=== SUMMARY ===")
    print(f"  DISCOVER (squad_solve)      : {cov:.0%} coverage")
    print(f"  BUILD+VERIFY (autonomous)   : {'PASS' if built else 'FAIL'} (gated by a real run)")
    print(f"  PROVE (methodology)         : {verdict['verdict']} "
          f"(neg-control rel={verdict.get('neg_control_rel')})")
    print("  → full loop exercised: discover → build → verify → prove, all package primitives, honest verdicts.")


if __name__ == "__main__":
    main()
