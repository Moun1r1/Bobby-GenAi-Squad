# Bobby — Self-Organizing Generative Agent Squad

Most "agentic AI" systems still require heavy human scripting: predefined roles, prompt chains, fixed workflows, and rigid orchestrators.  

**Bobby is different.**

We give agents only three things:
- A **persistent Self** (identity + long-term memory)
- A high-level **Goal**
- Access to tools and the ability to observe outcomes

From there, **everything else emerges**: team structure, division of labor, recursion depth, strategy adaptation, and even new sub-goals. Agents clone, mutate goals, run parallel experiments (generative + static variants), evaluate results, and consolidate winning intelligence into a reusable gallery.

### What this enables
- True self-organization and long-horizon autonomy
- Emergent multi-agent collaboration without predefined roles or graphs
- A growing, self-improving library of specialized agents
- Continuous optimization of complex systems (Obsidian vaults, knowledge bases, conflict resolution simulations, software processes, and more)
- Hybrid power: the creativity of generative agents + the reliability of optimized static agents

Pure Python standard library. It talks to any OpenAI-compatible `/v1/chat/completions` endpoint (local or hosted).

---

## Install

```bash
pip install -e .          # from this folder
```

Point it at your model (any OpenAI-compatible server — vLLM, sglang, Ollama, llama.cpp, a hosted API):

```bash
export BOBBY_LLM_URL="http://localhost:8000/v1/chat/completions"
export BOBBY_LLM_MODEL="your-served-model-id"
# optional: embeddings (for the idea-space memory/board); any Ollama-compatible /api/embed
export BOBBY_EMBED_URL="http://localhost:11434/api/embed"
```

## Quickstart

```python
from bobby_squad import Agent, SelfCore, LLM

agent = Agent(SelfCore(identity="a precise generalist", goal="answer exactly what is asked"), llm=LLM())
print(agent.carry_out("Write a 3-line haiku about entropy.", max_rounds=2))
```

See [`examples/quickstart.py`](examples/quickstart.py).

---

## The primitives

Everything is a reusable primitive imported from `bobby_squad` — the examples just wire them together.

- **`Agent` + `SelfCore` (persistent-self)** — an agent whose identity, goal, and accumulated progress live in a
  **pinned tier** that context-compaction never touches, so state from step 1 survives to step N and the prompt
  stays flat. `research_cycle` (self-select target → plan → tool-grounded execute) and `autonomous_loop`
  (loop-until-verified) drive it. No per-step prompt scripts the behavior — the loop chooses.

- **`squad_solve(agents, units, work, verify, split)`** — the coverage methodology. A squad drains a **recursive
  shared board**; `verify` (run-**don't**-ask) decides done-vs-split; a unit that's under-covered is split and
  re-queued; plateau = the board drains. Answers *"did we cover it all?"* — no orchestrator.

- **`prove(name, control, treatment, negative=, baseline_max=, seeds=)`** — the testing methodology. Not just an
  A/B: it enforces **validity** — a *headroom* guard (ceilinged baseline → `INCONCLUSIVE`, not a false `DELETE`), a
  *negative-control* guard (effect appears where it shouldn't → `INVALID`/leak), and *replication* (seeds + 95% CI).
  Verdicts: `WIRE / MARGINAL / DELETE / INCONCLUSIVE / INVALID / DEFER`.

- **`IdeaLedger`** — a shared idea board with a **deterministic identity floor** (near-duplicate proposals are
  repelled, so the swarm never re-generates the same idea) and **emergent, agent-assigned states** (no hardcoded
  lifecycle — the agents label and organize the board themselves). Idea-space novelty gate + active-repulsion
  frontier (surfaces the *most-spread* ideas so agents are pushed toward gaps, not the dense cluster).

- **`BoardTools`** — gives the swarm the tools to organize its own board (`board` / `set_state` / `merge`) as
  self-selected moves.

- **`BehaviorTrace` + `MetaTools` (metacognition)** — an agent reviews a peer's *real behavioral trace* and detects
  its bias (move/area concentration, repetition) and frontier (where novelty collapses) — grounded in deterministic
  signals, not vibes.

- **`WorldSense`** — a sensing layer: the agent checks many "worlds" (peers, files, the idea frontier, time, its own
  affect and self-model) and pulls the salient signals into its reasoning as data, never as a directive.

- **`SemanticMemory`** — a novelty-gated semantic store that **self-governs retention by learned usage** (bounded
  stores evict lowest-value; critical items pinned = a deterministic recall floor). Proven +25% retention / +12.5%
  downstream generation vs a fixed rule, with a passing negative control.

- **`SandboxTools`** — a full sandbox dev loop (`copy_in` / `write` / `edit` / `run` / `test` / `diff`) so an agent
  can write and **run** a real experiment and read the outcome — verdicts from execution, not a rubric.

---

## The design rules it was built on

1. **No static prompts, no hardcoded roles** — capability comes from a rich *self* + real *tools* + an open
   *move-space*; the agent self-selects mine / invent / compose / critique / organize. A "critic" is a *move*, not a
   persona.
3. **Verify by outcome** — a real run / a strict judge, never the model declaring "done" in prose.
4. **Prove, don't claim** — every gain goes through `prove` (headroom + negative control + CI) or it isn't trusted.
5. **Guard-first** — guardable mistakes (identity dedup, recall floor) live in deterministic code; only un-guardable
   generative choices are left to the model.

## Honest caveats

- Outputs are written by whatever model you point it at — a rich, navigable result, not a signed audit. The
  token/coverage numbers are mechanical; spot-check load-bearing claims.
- **Autonomous proving is a capability floor:** small local models generate broad but don't reliably write a fair
  gain-A/B. The pattern that works is a *teaching flywheel* — the local swarm generates, a stronger model proves.

## License

MIT — see [LICENSE](LICENSE).
