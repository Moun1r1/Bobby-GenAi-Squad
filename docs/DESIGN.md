# Design

A hybrid: a self-organizing multi-agent layer on top of an event-driven, plugin-based engine whose job is to move
reducible work off the LLM.

## Spine

An **append-only event log** is the single source of truth. Every plane is a projection of it (the blackboard is a
materialized view, router state is a fold, telemetry is a tee). Persistence, provenance, and replay live here, not in
the planes. Code: `bobby_squad/engine.py` (`EventLog`, `PluginRegistry`, `Engine`).

## Five planes

| Plane | Mechanism | Role | Code |
|---|---|---|---|
| Coordination | blackboard / tuple-space | agents post + claim contracts (first-claim-wins); coverage emerges | `blackboard.py`, `squad.py` |
| Control | FSM + ACR router | sequence steps; route each event to the cheapest handler | `fsm.py`, `router.py`, `engine.py` |
| Execution | stateless worker pool | run/prove/train in isolated workers (incl. a GPU worker) | `agent_tools.py` (Sandbox/DgxTools) |
| Evaluation | seeded batch / world-tick + gain-proof | reproducible grounding with a numeric verdict | `harness.py`, `proving.py`, `synthbench.py` |
| Observability | telemetry stream | cost curve, OOD rate, promotions — all from the log | `telemetry.py` |

Only the Control plane has a downward cost curve; the other four keep the LLM in the loop.

## The generative → static → plugin loop (ACR)

Automated Cognitive Reduction. For a capability the LLM keeps exercising:

1. **Discover offline.** From held-out examples, propose a rule — a regex, a numeric reducer op, a code transform, or
   an LLM-authored `def solve(...)`. `bobby_squad/burn_in.py:_distill`.
2. **Prove.** Score the candidate on held-out same-family examples. Freeze only if mean score ≥ threshold (0.9).
   Below threshold → the capability stays on the LLM (fail-safe).
3. **Freeze + route.** Register the frozen handler as a plugin, addressed by capability tag. The `competence_router`
   serves an in-distribution query from the plugin (zero LLM); an out-of-distribution query trips the **OOD gate**
   (diagonal-Mahalanobis distance to the plugin's proof-set embeddings) and abstains to the LLM.

A plugin is any `Callable[[payload], result]` — a regex extractor, a fold, a sandbox-compiled function, or an
LLM-wrapper. The registry dedups functional twins (AST fingerprint) and requires a proof record to register.

## Absorption surface

The engine is designed to take capability from outside and turn it into cheap local calls:

- **Skills / tools → plugins.** An external skill or tool becomes a plugin behind the same router + proof gate.
- **Primitives → a library.** Domain-free skeletons (`extract_matching`, `reduce_integers`, `find_analogous_case`)
  are proven **cross-domain** (must clear the gate on ≥ N domains they weren't co-fit on), filed by category, and
  indexed in a persistent semantic memory so they are found back before re-distilling. `primitive_lib.py`,
  [PRIMITIVES.md](PRIMITIVES.md).
- **Memory.** A knowledge vault (`vault.py`) of linked notes the agents read and write; plus the primitive index and
  correction memory.
- **Compute.** Sandbox and GPU-worker tools let an agent write, run, and train — verdicts come from execution.

## Generative agents from data

An agent is `(self, tools, moves)`; behavior is self-directed, never a per-step prompt. The **self** is a persona
`(identity, goal, memory)` instantiated from data — a use case is a role + goal (`DataSpec`, `specs.py`); the agent
observes the input record and its own cycle (select-target → plan → act) decides how to satisfy the goal. The persona
is persistent: self + accumulated progress live in a pinned tier that context-compaction never evicts (`core.py`), so a
constant character holds across a long session or a multi-round world (`pipe_persona`, `pipe_world`). What the agent
learns is written to a shared **knowledge vault** (`vault.py`) — linked `[[wikilinked]]` notes it reads (navigate the
relevant subgraph) and enriches (new notes, deduped, with provenance) — so knowledge is reusable across agents and
runs. Distilled plugins and vault knowledge are two forms of the same reuse: cheap deterministic capability, and
navigable memory.

## SO-MAS layer

`squad_solve(agents, units, work, verify, split)` drains a recursive shared board: `verify` (run, don't ask) decides
done-vs-split, an under-covered unit is split and re-queued, and the loop stops when the board drains. No orchestrator,
no assigned roles. Code: `squad.py`, `blackboard.py`, `core.py`.
