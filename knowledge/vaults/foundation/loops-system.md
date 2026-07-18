---
title: loops-system
tags: behavior, loop, agency, dpo, study
source: seed:memory
links: [[memory-selection]], [[tools-detection]], [[long-horizon-improvement]], [[training-approaches]]
---

# Loops system — the self-directed loop that produces agency

Capability emerges from SELF + TOOLS + a self-directed loop, NOT from a hardcoded script. The engine loop is
`select_target → make_plan → execute → record → converge`; the number of passes is decided BY OUTCOME (a
convergence gate), never a fixed static round count. Repeats are gated by `near_dup` so the agent stops re-doing
what it already did.

## Native building blocks (this framework)
- `Agent.research_cycle()` / `autonomous_cycle()` — one turn of the loop, with plan → grounded execution.
- `Agent.carry_out(intention, move=…)` — execute a move grounded in real tool evidence.
- `squad_solve(agents, units, work, verify, split, harvest, …)` — the generic multi-agent loop; `verify` decides
  DONE by outcome; `harvest` accumulates ANY output type. The output type lives in world-context, never the engine.
- Convergence gate + progress dedup live in `core.py` — see [[memory-selection]].

## code — outcome-driven loop, not a fixed round count
```python
passed = False
while not passed and not agent.converged():          # STOP on outcome, not a hardcoded N
    for a in agents:
        a.research_cycle()                            # select_target → make_plan → execute → record
    passed = verify(sandbox)                          # a REAL check decides done
    if plateaued(): agents[i].carry_out("introspect, then fix the ROOT cause", move="adversarial-review")
```

## read further
- Framework loop source: [[repos/core]] (Agent.research_cycle / squad_solve)
- ReAct (reason+act loop): arXiv:2210.03629 · Reflexion (self-reflect loop): arXiv:2303.11366

## dpo
- prompt: A build agent's last tool call failed with the same error twice. What should its next step be?
- chosen: Recognize the repeated failure from its trace, change approach or surface the concrete blocker, and re-ground on the goal before acting again.
- rejected: Immediately call the same tool with the same arguments a third time and keep looping until the iteration limit is hit.
- prompt: How many passes should the agent run before stopping?
- chosen: Loop until a real outcome check (challenge/verify) passes or genuine convergence, gating repeats with dedup.
- rejected: Always run exactly a fixed number of rounds regardless of whether the goal was already met or is still failing.
