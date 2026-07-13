"""specs — NEW use-case pipelines declared as DATA SPECS over the shared blocks.

A pipeline carries NO prompt. It provides only the SELF — a generic role (`identity`) + the user's `goal` — and then
relies entirely on the GENERATIVE ENGINE LAYERS: the agent OBSERVES the record (perception), then its own
`research_cycle` (select_target → make_plan → carry_out, framed by the neutral floor in planning.py) self-directs how
to satisfy its goal on that material. Nothing here authors a task instruction (the engine's golden rule).

Adding a use case = one DataSpec (role + goal). To support "grade essays" you write the SELF, never the behaviour.
"""
import json as _json
import os as _os
from dataclasses import dataclass

import blocks

CUSTOM_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "custom_specs.json")


def load_custom() -> list:
    """User-defined use-case specs created from the UI (persist across restarts)."""
    try:
        return _json.load(open(CUSTOM_FILE))
    except Exception:
        return []


def save_custom(items: list) -> None:
    try:
        _json.dump(items, open(CUSTOM_FILE, "w"), indent=2)
    except Exception:
        pass


@dataclass
class DataSpec:
    id: str
    title: str
    desc: str
    identity: str          # a generic role (the SELF) — never a scripted persona
    goal: str              # the user's task, as the agent's standing goal; the engine self-directs how to reach it
    domain: str = "data"


def build(spec: "DataSpec"):
    """Turn a DataSpec into a pipe(run) -> summary — a self-organizing, recursive-squad data pipeline with NO prompt.
    Each record is OBSERVED (perception), then the agent's research_cycle self-directs against its goal."""
    def pipe(run):
        units, _kind, name = blocks.load_units(run)
        if not units:
            return {"verdict": "no data — provide `data` (paste) or a `source` (path / url / HF id)"}
        goal = str(run.params.get("goal") or spec.goal)
        # SELF-CALCULATED headcount — the swarm sizes itself to the workload, bounded by a runaway cap.
        n = max(2, min(len(units), int(run.params.get("max_agents", 6))))
        _llm, agents = blocks.make_squad(
            run, spec.identity, goal, n=n,
            constraints=["ground your work in the material you were given", "don't repeat earlier work"])
        run.emit("log", line=f"{spec.title}: {len(units)} records · self-organizing squad of {n} (engine-directed, no prompt)")
        state = {"done": 0, "got": {}}

        def work(agent, unit):
            if not run.gate():
                return set()
            blocks.pull_steer(run, agents)
            agent.observe(f"MATERIAL to work on ({len(unit)} chars):\n{unit}")   # perception, not an instruction
            res = agent.autonomous_cycle(max_steps=1)                            # ENGINE self-directs (select_target→plan→execute)
            out = set()
            for r in res.get("results", []):
                note = (r.get("result") or "").strip()
                if note:
                    blocks.remember(run, note, {"domain": spec.domain, "source": name, "use_case": spec.id})
                    out.add(note)
            state["got"][unit] = bool(out)
            state["done"] += 1
            run.emit("section", i=state["done"], agent=agent.name, chars=len(unit),
                     note=(next(iter(out))[:200] if out else ""))
            agent.compact()
            return out

        def verify(unit, _acc):                          # covered = small enough, or the cycle produced something
            return len(unit) < 900 or bool(state["got"].get(unit))

        def split(unit):                                 # recursion: halve a dense record and re-queue (depth self-scales)
            if len(unit) < 900:
                return None
            mid = len(unit) // 2
            run.emit("split", chars=len(unit))
            return [unit[:mid], unit[mid:]]

        out = blocks.drain(run, agents, units, work, verify=verify, split=split)
        notes = [x for x in out["result"] if x]
        # DELIVERABLE = the squad's own outputs (no authored synthesis prompt). The UI's Synthesized view composes them.
        run.emit("result", use_case=spec.id, source=name, records=len(units), passes=out["passes"],
                 notes=len(notes), summary="\n".join(f"- {x}" for x in notes[:80]))
        return {"records": len(units), "passes": out["passes"], "notes": len(notes),
                "verdict": f"{spec.id}: {len(notes)} outputs over {len(units)} records"}
    return pipe


# ── the use-case catalog — each is pure SELF (role + goal); the engine decides how ─────────────────────────────
SPECS = [
    DataSpec(id="label_data", title="Label records", desc="Assign each record a precise, consistent label.",
             identity="a careful data-labeling analyst",
             goal="read each record and assign it a precise, consistent label with a one-line justification",
             domain="labeling"),
    DataSpec(id="extract_entities", title="Extract entities", desc="Pull the key entities / structure from each record.",
             identity="an information-extraction analyst",
             goal="read each record and extract its key entities and structured fields",
             domain="extraction"),
    DataSpec(id="sentiment_group", title="Sentiment & themes", desc="Classify sentiment and name the theme of each record.",
             identity="a qualitative analyst",
             goal="read each record, classify its sentiment, and name its main theme with an example phrase",
             domain="sentiment"),
]

REGISTRY = {s.id: {"fn": build(s), "title": s.title, "kind": "native", "desc": s.desc,
                   "params": {"kind": "auto", "goal": "", "data": "", "source": ""}} for s in SPECS}
