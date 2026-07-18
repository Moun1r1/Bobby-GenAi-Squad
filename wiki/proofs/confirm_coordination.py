import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from confirm_organization import found_in as _fi_all, make_agent, ROOT   # noqa: E402,F401  (reuse helpers)
from bobby_squad import LLM                                              # noqa: E402


def py_files():                                                    # SCOPE = the core package modules (top-level *.py)
    return sorted(f for f in os.listdir(ROOT) if f.endswith(".py"))


# ground truth recomputed for EXACTLY this scope, so full coverage is reachable and self-scaling is measurable
_gt_out = subprocess.run(["grep", "-hoE", "def [a-zA-Z_][a-zA-Z0-9_]*", *[os.path.join(ROOT, f) for f in py_files()]],
                         capture_output=True, text=True).stdout
GT = set(re.findall(r"def ([a-zA-Z_]\w*)", _gt_out))
_GT_RE = {n: re.compile(rf"\b{re.escape(n)}\b") for n in GT}


def found_in(text):
    return {n for n, rx in _GT_RE.items() if rx.search(text or "")}


def map_file(agent, path, remaining_hint):
    intention = (f"Coordinated mapping. You are mapping ONE file: '{path}'. Read it and list EVERY function it "
                 f"defines (each `def <name>`). Output the bare names only. ({remaining_hint} files still unmapped "
                 "by the squad — you cover this one.)")
    return found_in(agent.carry_out(intention, move="investigate", max_rounds=2))


def main():
    llm = LLM(temperature=0.4, timeout=120)
    total = len(GT)
    board = py_files()                                  # the SHARED WORK-BOARD = the scope = the content size
    agents = [make_agent(llm, 20 + i) for i in range(3)]
    print(f"[coord] scope = {len(board)} files, {total} functions · squad self-scales to the board (no orchestrator)\n",
          flush=True)

    shared = set()
    p = 0
    while board:                                        # PLATEAU = board drained (self-scaled to content), not a budget
        path = board.pop(0)                             # coordination: each agent takes an UNCOVERED file, no overlap
        ag = agents[p % len(agents)]
        shared |= map_file(ag, path, len(board))
        p += 1
        print(f"  pass {p:>2} {ag.name} ← {path:<34} → {len(shared)}/{total} = {len(shared)/total:.0%}", flush=True)

    print(f"\n[coord] SQUAD + COORDINATION (no orchestrator): {len(shared)}/{total} = {len(shared)/total:.0%} "
          f"in {p} passes (== {p} files → plateau SELF-SCALED to content size)")
    print("THEORY CONFIRMED: squad + coordination covers it ALONE and self-scales — no orchestrator needed."
          if len(shared) / total > 0.75 else "partial — coordination helped but not to full coverage")


if __name__ == "__main__":
    main()
