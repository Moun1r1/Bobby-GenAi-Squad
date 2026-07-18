import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import Agent, SelfCore, ReadOnlyTools   # noqa: E402
from bobby_squad.llm import LLM                          # noqa: E402

ROOT = os.path.dirname(HERE)


def main():
    c = {"target": 0, "plan_steps": 0, "cycle": 0, "record": 0, "abstain": 0,
         "move": Counter(), "tool": Counter()}

    def obs(e):
        k = e.get("kind")
        if k == "target":
            c["target"] += 1
        elif k == "plan":
            c["plan_steps"] += len(e.get("steps") or [])
        elif k == "move_start":
            c["move"][(e.get("move") or "unnamed").lower()] += 1
        elif k == "tool":
            c["tool"][e.get("name")] += 1
        elif k == "cycle":
            c["cycle"] += 1
            if not e.get("results"):
                c["abstain"] += 1

    llm = LLM(temperature=0.5, timeout=120)
    agents = [Agent(SelfCore("a researcher mapping this repo's real capabilities",
                             "find and note something worth reusing, grounded in the code"),
                    llm=llm, window=4, pinned=True, tools=ReadOnlyTools(ROOT), name=f"r{i}", observer=obs)
              for i in range(2)]

    print("[engine_trace] 2 agents × 2 cycles — instrumenting every layer\n", flush=True)
    for rnd in range(2):
        for ag in agents:
            r = ag.research_cycle(max_steps=2, max_rounds=5)
            c["record"] += len(r.get("results") or [])
        print(f"  round {rnd+1} done", flush=True)

    total_moves = sum(c["move"].values())
    total_tools = sum(c["tool"].values())
    stats = {
        "targets_self_selected": c["target"],
        "plan_steps_generated": c["plan_steps"],
        "cycles": c["cycle"],
        "moves_total": total_moves,
        "distinct_moves": len(c["move"]),
        "move_distribution": dict(c["move"].most_common()),
        "move_diversity": round(len(c["move"]) / max(1, total_moves), 2),
        "tool_calls": total_tools,
        "tool_distribution": dict(c["tool"].most_common()),
        "records_to_pinned_tier": c["record"],
        "abstentions": c["abstain"],
    }
    print("\n=== ENGINE LAYERS (real counts) ===")
    print(json.dumps(stats, indent=2))
    open(os.path.join(ROOT, "out", "engine_trace.json"), "w").write(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
