"""blocks — reusable, golden-rule-clean building-blocks for studio pipelines.

Every studio pipeline needs the SAME seams: build a squad, drain operator steer, remember what's learned, and drive a
self-organizing loop over work. Hand-rolling those per pipeline is why adding a use case was a copy-paste chore. These
blocks compose the ENGINE primitives (Agent, squad_solve) once, so a new pipeline is a short composition.

Golden rule: NO scripted behavior, NO hardcoded persona. You pass a generic identity + the user's
goal + a NEUTRAL task frame; the self-directed loop chooses how. Nothing here enumerates what to say.
"""
from typing import Callable, List, Optional


def make_squad(run, identity: str, goal: str, n: int = 3, tools=None, constraints: Optional[List[str]] = None,
               window: int = 3, temperature: float = 0.4, timeout: int = 150):
    """Build N self-organizing agents sharing ONE identity + goal, wired to the run's live observer. Returns (llm, agents).
    Roles/moves are emergent — we never assign a persona per agent (emergent operating model)."""
    from bobby_squad import Agent, SelfCore
    from bobby_squad.llm import LLM
    llm = LLM(temperature=temperature, timeout=timeout)
    agents = [Agent(SelfCore(identity=identity, goal=goal, constraints=constraints or []),
                    llm=llm, window=window, pinned=True, tools=tools, name=f"agent{i}", observer=run.observe)
              for i in range(n)]
    run.agents = {a.name: a for a in agents}
    run.emit("teams", members=[a.name for a in agents])
    return llm, agents


def pull_steer(run, agents) -> None:
    """Drain any human directives injected mid-run into every agent's working memory (memory injection is a legitimate
    lever, not a scripted prompt)."""
    for s in run.take_steer():
        for a in agents:
            a.observe(f"OPERATOR DIRECTIVE: {s}")


def remember(run, text: str, meta: dict) -> None:
    """Persist a learned item to the vector store (deferred import avoids a cycle with runner)."""
    import runner
    runner._store_knowledge(run, text, meta)


def load_units(run):
    """Turn a run's data param into work units (deferred to runner's loader, which already handles text/CSV/JSON/PDF/URL/HF)."""
    import runner
    return runner._load_units(run)


def drain(run, agents, units, work: Callable, verify=None, split=None):
    """Run a self-organizing squad over a recursive shared board until it drains (the proven coordination primitive).
    `work(agent, unit)` does one unit and returns its harvested items. Recursion self-scales via `split`."""
    from bobby_squad import squad_solve
    return squad_solve(agents, list(units), work, verify=verify, split=split, accumulated=set())
