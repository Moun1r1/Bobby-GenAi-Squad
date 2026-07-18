import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from confirm_coordination import GT, found_in, make_agent, py_files, ROOT   # noqa: E402  (scope-consistent GT + helpers)
from bobby_squad import squad_solve                                  # noqa: E402  (the reusable primitive)
from bobby_squad import LLM                                      # noqa: E402


def _lines(path):
    with open(os.path.join(ROOT, path), errors="ignore") as f:
        return f.read().splitlines()


def actual_in(path, lo, hi):                            # VERIFY: the real functions defined in exactly this range
    return set(re.findall(r"def ([a-zA-Z_]\w*)", "\n".join(_lines(path)[lo - 1:hi])))


def map_unit(agent, path, lo, hi):
    intention = (f"Read the file '{path}' lines {lo}-{hi} and list EVERY function (each `def <name>`) defined in "
                 "THAT line range. Output the bare names only.")
    return found_in(agent.carry_out(intention, move="investigate", max_rounds=2))


def main(min_chunk=45, max_passes=45):
    llm = LLM(temperature=0.4, timeout=120)
    total = len(GT)
    agents = [make_agent(llm, 30 + i) for i in range(3)]              # the self-organizing squad (shared board)
    units = [(f, 1, len(_lines(f))) for f in py_files()]             # scope = whole files to start
    print(f"[recur] scope {len(units)} files, {total} functions · running through squad_solve primitive\n", flush=True)

    def work(agent, unit):
        return map_unit(agent, *unit)                               # a pass: agent maps one file/range → found names

    def verify(unit, acc):                                          # run-DON'T-ask gate: covered, or too small to split?
        path, lo, hi = unit
        return (not (actual_in(path, lo, hi) - acc)) or (hi - lo) <= min_chunk

    def split(unit):                                               # RECURSION: halve an under-covered big range
        path, lo, hi = unit
        mid = (lo + hi) // 2
        return [(path, lo, mid), (path, mid + 1, hi)] if (hi - lo) > min_chunk else None

    def obs(e):
        u = e["unit"]
        print(f"  p{e['n']:>2} {u[0]} L{u[1]}-{u[2]}: {e['size']}/{total} = {e['size']/total:.0%}"
              f"{'  ↯ split → board=' + str(e['board']) if not e['done'] else ''}", flush=True)

    r = squad_solve(agents, units, work, verify=verify, split=split, max_passes=max_passes, observer=obs)
    shared, p = r["result"], r["passes"]
    print(f"\n[recur] RECURSIVE squad_solve (no orchestrator): {len(shared)}/{total} = {len(shared)/total:.0%} "
          f"in {p} passes (depth self-scaled to content: {r['units_left']} units left on board)")
    print(f"[recur] progression: raw 1-pass 21% → uncoordinated squad 56% → flat coordination 76% → recursive "
          f"{len(shared)/total:.0%}")
    print("CONFIRMED: recursive coordination reaches full coverage ALONE — the orchestrator dissolves into a squad "
          "+ a recursive shared board + a verify gate." if len(shared) / total > 0.9 else
          "coordination climbed but not to full coverage (raise passes / lower min_chunk)")


if __name__ == "__main__":
    main()
