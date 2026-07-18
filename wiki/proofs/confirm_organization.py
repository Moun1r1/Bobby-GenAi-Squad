import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                                          # sibling proofs import each other
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))                        # repo root → import bobby_squad

from bobby_squad import Agent, SelfCore, FindingsMemory, ReadOnlyTools            # noqa: E402,F401
from bobby_squad import LLM                                                       # noqa: E402
import bobby_squad                                                                # noqa: E402

ROOT = os.path.dirname(os.path.abspath(bobby_squad.__file__))                     # the public code corpus = the package
PASSES = 6                                                                        # equal budget for B and C


def ground_truth():
    out = subprocess.run(["grep", "-rhoE", "def [a-zA-Z_][a-zA-Z0-9_]*", "--include=*.py",
                          "--exclude-dir=__pycache__", ROOT], capture_output=True, text=True).stdout
    return set(re.findall(r"def ([a-zA-Z_]\w*)", out))


GT = ground_truth()
GT_RE = {n: re.compile(rf"\b{re.escape(n)}\b") for n in GT}


def found_in(text):
    return {n for n, rx in GT_RE.items() if rx.search(text or "")}


def make_agent(llm, i):
    core = SelfCore(identity=f"a thorough code cartographer (#{i})",
                    goal="find and list, by exact name, the functions defined across this package's .py files")
    return Agent(core, llm=llm, tools=ReadOnlyTools(ROOT), name=f"carto-{i}")


def one_pass(agent, already):
    hint = ("You have already found these — do NOT relist them; grep OTHER files/dirs to find functions NOT in this "
            f"set:\n{sorted(already)[:60]}\n\n") if already else ""
    intention = (hint + "Using grep/ls/read, list as many DISTINCT function names (def <name>) defined anywhere in "
                 "the package as you can. Output the bare names.")
    return found_in(agent.carry_out(intention, move="investigate", max_rounds=4))


def main():
    llm = LLM(temperature=0.5, timeout=120)
    total = len(GT)
    print(f"[confirm] ground truth: {total} distinct functions in the corpus · budget={PASSES} passes\n", flush=True)

    a_cov = one_pass(make_agent(llm, 0), set())
    print(f"A. SOLO 1-pass ...................... {len(a_cov)}/{total} = {len(a_cov)/total:.0%}", flush=True)

    b = set()
    ag = make_agent(llm, 1)
    for _ in range(PASSES):
        b |= one_pass(ag, set())                       # never told what's already found → re-finds the same
    print(f"B. SOLO {PASSES}-pass, NO memory ......... {len(b)}/{total} = {len(b)/total:.0%}", flush=True)

    agents = [make_agent(llm, 10 + i) for i in range(3)]
    shared = set()
    dry = 0
    for p in range(PASSES):
        ag = agents[p % len(agents)]
        new = one_pass(ag, shared) - shared            # sees shared → hunts NEW regions
        shared |= new
        dry = dry + 1 if not new else 0
        print(f"   squad pass {p+1} ({ag.name}): +{len(new)} new → {len(shared)}/{total}", flush=True)
        if dry >= 2:                                   # PLATEAU — nothing new twice → stop (auto-scaled to complexity)
            print("   plateau reached — stopping early", flush=True)
            break
    print(f"C. SQUAD {PASSES}-pass, SHARED+PLATEAU ... {len(shared)}/{total} = {len(shared)/total:.0%}", flush=True)

    print(f"\n[confirm] ORGANIZATION vs RAW COMPUTE (equal passes): squad {len(shared)/total:.0%} vs "
          f"solo-repeat {len(b)/total:.0%}  →  Δ = {(len(shared)-len(b))/total:+.0%}", flush=True)
    print("RULE CONFIRMED: organization beats raw intelligence." if len(shared) > len(b) + 0.05 * total
          else "inconclusive on this run", flush=True)


if __name__ == "__main__":
    main()
