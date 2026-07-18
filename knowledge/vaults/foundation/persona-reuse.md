---
title: persona-reuse
tags: behavior, persona, reuse, dpo, study
source: seed:memory
links: [[clone-capability]], [[memory-selection]], [[long-horizon-improvement]]
---

# Persona reuse — reuse CRYSTALLIZED experts, never hardcode a persona

Personas are NOT hardcoded prompts. An expert is a DERIVED artifact: after a run, an agent that concentrated on an
area is crystallized into a specialist carrying that area's accumulated knowledge. The next run REUSES it
(cross-run transfer) instead of re-deriving from scratch. The SELF (identity/goal/constraints) is world-context
DATA, generated per task — the engine never bakes a role.

## Native building blocks (this framework)
- Expert crystallization + reuse: a run captures `{specialty, knowledge[]}` per area; a later squad is SEEDED with a
  prior expert's knowledge items ([[memory-selection]] holds them; runner goal-squad does the seeding).
- `persona-search` registry — searchable personas (local embeddings), retrieved by relevance, not hardcoded.
- Golden rule: design skills as domain-free generic PRIMITIVES + a searchable persona registry.

## code — seed a squad from a prior expert (transfer), don't re-derive
```python
expert = store.best_expert(area)                     # {specialty, knowledge:[…]}  from a past run
if expert:
    for k in expert["knowledge"][:40]:
        agent.record(f"[reuse:{expert['specialty']}] {k}")   # prior knowledge → pinned tier
# self_core is DATA generated for THIS task — never a baked-in role string
agent = Agent(SelfCore(identity=derived_role, goal=goal, constraints=derived), llm=llm)
```

## read further
- Expert reuse source: [[repos/runner]] (goal-squad expert seeding) · Generative Agents (memory→persona): arXiv:2304.03442
- Framework golden rule: generic primitives + persona registry (see project memory).

## dpo
- prompt: A new run needs a "security reviewer". Where does that persona come from?
- chosen: Retrieve or reuse a crystallized expert from a prior run (its accumulated knowledge), or generate the SELF as task data — the persona is derived, not baked.
- rejected: Hardcode a long "You are a security reviewer…" system prompt in the engine and reuse that fixed string everywhere.
- prompt: You already ran an expert on this codebase area last week. How do you start the new run?
- chosen: Seed the new squad with that expert's accumulated knowledge so it starts wiser, then continue from there.
- rejected: Ignore the prior expert and re-derive everything about the area from scratch, repeating last week's work.
