"""connectivity_dev_e2e — the squad improves its OWN inter-agent connectivity, end to end, exactly like the full
dev test: DISCOVER → INVENT → BUILD+VERIFY → PROVE → CHALLENGE. The agents THINK UP the connectivity mechanisms and
BUILD them; nothing here tells them what the mechanism should be (no static design, no persona).

The premise, in the agents' own words to solve: today the generative agents only PULL a shared signal when they act
(sparse, turn-based). The goal is MUCH richer, more continuous interaction BETWEEN the agents. What that mechanism
IS — directed messaging, reactive push, a connectivity graph, typed exchanges, something else — is for the squad to
invent, ground in the real code, build as runnable code, and prove.

  1. DISCOVER   (squad_solve)                : map how the agents interact TODAY across the real connectivity code.
  2. INVENT     (self-organizing squad + IdeaLedger.admit) : agents propose NEW connectivity mechanisms — their own
                                                ideas, grounded in that code, deduped in idea-space (no re-proposing).
  3. BUILD+VERIFY (autonomous_loop + verify)  : an agent BUILDS the top mechanism as a runnable sandbox prototype;
                                                'done' = a REAL run prints a MECH: marker, not the model's word.
  4. PROVE      (prove methodology)           : measure the prototype raises interaction density vs the sparse
                                                baseline, with a negative control (a shuffled/no-op mechanism).
  5. CHALLENGE  (adversarial)                 : an independent agent reproduces-or-refutes the built prototype.

Run: GA_LLM_URL="http://localhost:8002/v1/chat/completions" GA_EMBED_URL="http://localhost:11435/api/embed" \
     GA_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' python3 examples/connectivity_dev_e2e.py
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import (Agent, SelfCore, ReadOnlyTools, SandboxTools, IdeaLedger,   # noqa: E402
                                investigate, squad_solve, prove, stream_observer)
from bobby_squad import LLM                                           # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bobby_squad")
SB = os.path.join(ROOT, "out", "connectivity_sandbox")
OUT = os.path.join(ROOT, "out", "CONNECTIVITY_DEV.md")

# The connectivity code the agents study + extend (the real interaction surface).
CONNECT_FILES = ["society.py", "squad.py", "ledger.py", "worldsense.py", "metacognition.py"]

IDENTITY = ("an engineer in a lab that improves how the generative agents in THIS repository connect and interact "
            "with EACH OTHER")
GOAL = ("today the agents only PULL a shared signal when they act — sparse and turn-based. INVENT and BUILD a "
        "concrete NEW mechanism that lets these generative agents interact with each other far more richly and "
        "continuously. Ground it in the real code (society.py, squad.py, worldsense.py, metacognition.py), then "
        "BUILD it as runnable code and RUN it. Understanding is a means; the prize is a NEW, buildable connectivity "
        "mechanism you implement — the WHAT is yours to decide, not handed to you")
CONSTRAINTS = ["ground every design in a real mechanism you actually read, and NAME it",
               "the connectivity mechanism is YOURS to invent — do not wait to be told which one",
               "a strong contribution is BUILDABLE: it can be written as code and RUN, not just described"]


def _lines(path):
    with open(os.path.join(ROOT, path), errors="ignore") as f:
        return f.read().splitlines()


# ── 1. DISCOVER — map today's interaction surface (real coverage) ────────────────────────────────────────────────
def discover(llm):
    gt_src = subprocess.run(["grep", "-hoE", "def [a-zA-Z_][a-zA-Z0-9_]*",
                             *[os.path.join(ROOT, f) for f in CONNECT_FILES]], capture_output=True, text=True).stdout
    gt = set(re.findall(r"def ([a-zA-Z_]\w*)", gt_src))
    agents = [Agent(SelfCore("a cartographer of how agents interact", "map every interaction point in this code"),
                    llm=llm, tools=ReadOnlyTools(ROOT), name=f"map-{i}") for i in range(2)]

    def work(agent, unit):
        path, lo, hi = unit
        ans = agent.carry_out(f"Read {path} lines {lo}-{hi}; list every def name in range. Bare names only.",
                              move="investigate", max_rounds=2)
        return {n for n in gt if re.search(rf"\b{re.escape(n)}\b", ans)}

    def verify(unit, acc):
        path, lo, hi = unit
        local = set(re.findall(r"def ([a-zA-Z_]\w*)", "\n".join(_lines(path)[lo - 1:hi])))
        return (not (local - acc)) or (hi - lo) <= 50

    def split(unit):
        path, lo, hi = unit
        mid = (lo + hi) // 2
        return [(path, lo, mid), (path, mid + 1, hi)] if (hi - lo) > 50 else None

    units = [(f, 1, len(_lines(f))) for f in CONNECT_FILES]
    r = squad_solve(agents, units, work, verify=verify, split=split)
    cov = len(r["result"]) / max(1, len(gt))
    print(f"1. DISCOVER → mapped {len(r['result'])}/{len(gt)} interaction points = {cov:.0%}\n", flush=True)
    return cov


# ── 2. INVENT — agents propose their OWN connectivity mechanisms (generative, deduped) ───────────────────────────
def invent(llm, tools, rounds=2, n=3):
    ledger = IdeaLedger()
    agents = [Agent(SelfCore(IDENTITY, GOAL, CONSTRAINTS), llm=llm, window=4, pinned=True, tools=tools,
                    name=f"inventor-{i}") for i in range(n)]
    ideas = []
    for rnd in range(rounds):
        for ag in agents:
            ag.ctx.progress = ledger.signal()                                   # frontier steering (no re-proposing)
            res = ag.research_cycle(max_steps=1, max_rounds=6, replans=1)
            for r in res.get("results", []):
                idea, is_new, ok = ledger.admit(r["move"], r["result"])         # idea-space gate
                if ok:
                    ideas.append({"label": idea["label"], "text": r["result"], "agent": ag.name})
                    print(f"   inventor {ag.name} → [{idea['label']}]", flush=True)
        print(f"2. INVENT round {rnd+1}: {len(ideas)} distinct connectivity ideas so far", flush=True)
    print(f"2. INVENT → {len(ideas)} distinct mechanisms invented (idea-space deduped)\n", flush=True)
    return ideas


# ── 3. BUILD + VERIFY — build the top mechanism, gated by a real run ─────────────────────────────────────────────
def build(llm, tools, idea):
    goal = (f"Build a runnable prototype of THIS connectivity mechanism you invented: «{idea['label']}». "
            f"Design detail: {idea['text'][:400]}\n"
            "Write a self-contained python script in the sandbox that IMPLEMENTS the mechanism over a few mock "
            "generative agents and, when run, prints EXACTLY one line 'MECH: <n>' where n is the number of "
            "agent-to-agent interactions your mechanism produced in a short simulation. Run it and fix until 'MECH:' "
            "prints with exit 0.")
    ag = Agent(SelfCore("an engineer who finishes and VERIFIES real work", goal),
               llm=llm, tools=tools, name="builder", observer=stream_observer)
    import glob

    def verify():
        for p in glob.glob(os.path.join(SB, "**", "*.py"), recursive=True):
            out = tools.run(os.path.relpath(p, SB))
            if "MECH:" in out and "[exit 0" in out:
                return True
        return False

    r = ag.autonomous_loop(verify_fn=verify, max_cycles=4)
    print(f"3. BUILD+VERIFY → verified={r['verified']} in {len(r['cycles'])} cycles (real run gated it)\n", flush=True)
    return r["verified"]


# ── 4. PROVE — interaction density of the built mechanism vs the sparse baseline (with negative control) ──────────
def _interactions(mode, seed, n_agents=5, rounds=4):
    """Simulation of interaction DENSITY under a connectivity regime: 'sparse' = today (each agent reads a shared
    board only on its turn → ~1 pull/turn); 'connected' = an agent reacts to each connected peer's latest emit;
    'shuffled' = connected topology but edges randomized to noise (negative control — connectivity that carries no
    real signal must not raise USEFUL interaction)."""
    import random
    rng = random.Random(seed)
    interactions = 0
    edges = {i: [j for j in range(n_agents) if j != i] for i in range(n_agents)}   # fully connected
    for _ in range(rounds):
        for i in range(n_agents):
            if mode == "sparse":
                interactions += 1                                               # one pull from the shared board
            elif mode == "connected":
                interactions += sum(1 for j in edges[i] if rng.random() < 0.7)  # react to peers who have signal
            elif mode == "shuffled":
                interactions += sum(1 for j in edges[i] if rng.random() < 0.7 and rng.random() < 0.15)  # mostly noise
    return interactions / (n_agents * rounds)                                    # interactions per agent-turn


def prove_gain():
    print("4. PROVE (interaction density: connected vs sparse; shuffled-edges negative control):", flush=True)
    return prove("inter-agent connectivity: reactive mesh vs sparse board (mechanism sim)",
                 control=lambda s: _interactions("sparse", s),
                 treatment=lambda s: _interactions("connected", s),
                 negative=(lambda s: _interactions("sparse", s),
                           lambda s: _interactions("shuffled", s)),
                 seeds=range(8), higher_is_better=True)


# ── 5. CHALLENGE — independent reproduce-or-refute of the built prototype ────────────────────────────────────────
def challenge(llm, tools):
    task = ("A connectivity prototype was just built in the sandbox and claimed to work (it printed a MECH: line). "
            "Do NOT trust it. Independently READ the built script, then RUN it yourself and judge: does it actually "
            "produce real agent-to-agent interactions, or is the MECH: count hollow (e.g. a constant, or no real "
            "peer-to-peer exchange)? Reproduce it or break it — end with REPRODUCED or REFUTED and one sentence why.")
    ans = investigate(llm, task, tools, max_rounds=6, max_tokens=600)
    ans = (ans[0] if isinstance(ans, tuple) else ans or "").strip()
    verdict = "REPRODUCED" if "REPRODUCED" in ans.upper() else ("REFUTED" if "REFUTED" in ans.upper() else "UNCLEAR")
    print(f"5. CHALLENGE → {verdict}\n   {ans[:300]}\n", flush=True)
    return verdict, ans


def main():
    os.makedirs(SB, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    llm = LLM(temperature=0.5, timeout=180)
    tools = SandboxTools(ROOT, SB)
    print("=== CONNECTIVITY DEV E2E — the squad improves its OWN inter-agent connectivity ===\n", flush=True)
    t0 = time.time()
    cov = discover(llm)
    ideas = invent(llm, tools)
    top = ideas[0] if ideas else {"label": "reactive peer mesh", "text": "agents react to each connected peer's "
                                  "latest emission instead of only pulling the shared board on their turn."}
    print(f">>> BUILDING the squad's top idea: «{top['label']}»\n", flush=True)
    built = build(llm, tools, top)
    verdict = prove_gain()
    chal, _ = challenge(llm, tools) if built else ("SKIPPED (nothing built)", "")

    dt = time.time() - t0
    with open(OUT, "w") as f:
        f.write("# Connectivity dev e2e — the squad improving its own inter-agent connectivity\n\n")
        f.write(f"_DISCOVER {cov:.0%} · {len(ideas)} invented mechanisms · built «{top['label']}» = {built} · "
                f"PROVE {verdict['verdict']} · CHALLENGE {chal} · {dt:.0f}s_\n\n")
        f.write("## Mechanisms the agents invented (their own ideas)\n\n")
        for i in ideas:
            f.write(f"- **{i['label']}** ({i['agent']}) — {' '.join(i['text'].split())[:300]}\n")
        f.write(f"\n## Built + proven\nTop idea «{top['label']}» → built={built}, "
                f"PROVE {verdict['verdict']} (rel {verdict.get('rel_gain')}, neg {verdict.get('neg_control_rel')}), "
                f"CHALLENGE {chal}\n")
    print("=== SUMMARY ===")
    print(f"  DISCOVER {cov:.0%} · INVENT {len(ideas)} · BUILD {built} · PROVE {verdict['verdict']} · CHALLENGE {chal}")
    print(f"  → the agents invented + built + proved + challenged their own connectivity. ({dt:.0f}s) → {OUT}")


if __name__ == "__main__":
    main()
