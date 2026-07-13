# Bobby GenAi Squad

**A framework for teams of AI agents that organize themselves.** You give a squad of agents a goal and a model; they
split the work, read and research on their own, keep the goal and everything they've learned intact over very long
tasks, prove what actually helped, and can even train a model on what they learned — and **you don't script the
roles, the prompts, or the steps.** The team figures that out.

Pure Python standard library. It talks to any OpenAI-compatible model endpoint — local (vLLM, sglang, Ollama,
llama.cpp) or a hosted API.

## What it actually does

Point it at a big job and a model, and:

- **It reads huge things end to end** — a whole codebase, or a stack of papers — section by section, *without the AI
  forgetting the earlier parts.* (That "forgetting" is the usual context-window problem; here the prompt stays small
  no matter how long the job.)
- **The agents run as a self-organizing team** — they drain a shared to-do board together: no manager, no fixed roles,
  no hand-written workflow. Work that isn't finished splits into more work; when the board is empty, the job is
  covered.
- **They remember and build on it** — what the squad learns goes into a linked knowledge base (like a personal wiki)
  it reads *and writes back to*, so the next run starts smarter than the last, and an idea can carry from one field
  into another.
- **They prove their own gains, honestly** — before trusting any improvement, it runs a real A/B test with controls
  and **throws out what doesn't help.** Most ideas fail that test — that's the point.
- **It can train models, not just call them** — it turns what it proved into fine-tuning data (no hand-labeling) and
  trains a model on an isolated GPU worker, gated by a real pass/fail check.
- **You can watch it happen** — an optional Studio (a web UI) shows the whole run live instead of a wall of logs.

## How it's different from other agent frameworks

| Most agent frameworks | Bobby |
|---|---|
| You script the roles, prompts, and the workflow | The agents **self-organize** — no orchestrator, no fixed roles |
| Agents lose the thread / hit the context limit on long tasks | The goal + everything learned live in a **pinned memory** that context-trimming never wipes |
| "It works!" — no proof | Every improvement is **A/B-tested with real controls**; whatever doesn't help is kept out |
| It only *calls* an LLM | It reads+writes a **knowledge graph** and can **train** a model from what it learned |
| A library you wire together | A **platform** — watch runs live in a Studio, train on your own GPU |

It runs on any CUDA GPU host (a laptop, a cloud VM, a workstation) — nothing is tied to specific hardware.

**Docs:** [What's new](docs/whats-new.md) · [Architecture](docs/architecture.md) · [Interface / screens](docs/interface.md) · [The engine, layer by layer](docs/engine.md)

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

## The platform — engine · Studio · GPU worker

Bobby is three layers, used together or à la carte:

- **The engine** (`bobby_squad`) — the pure-Python primitives below; talks to any OpenAI-compatible endpoint.
- **Studio** — a live control room. A **FastAPI backend** wraps the engine as pipelines with a streaming event API; a
  **Next.js frontend** launches runs and lets you *watch the generative loop in real time* — the shared board
  draining, each agent's target → plan → move → tool stream, the knowledge-vault graph, the proof bench, and a
  realtime GPU/CPU/memory monitor for the training worker.
- **The GPU worker** — an isolated, memory-capped **Docker** container the swarm pushes code to and **trains on**. A
  pre-train safety gate (realtime NVIDIA monitor) refuses to start unless the box has headroom, so a shared GPU is
  never starved; long runs launch in the background and stream their logs back.

Everything is dockerized for local deployment.

<pre class="mermaid">
flowchart LR
  FE["Next.js Studio<br/>(watch live)"] <--> BE["FastAPI backend<br/>(engine as pipelines)"]
  BE <--> ENG["bobby_squad engine<br/>(persistent-self swarm)"]
  ENG <--> VAULT[("knowledge vault<br/>[[linked notes]]")]
  BE -. push code + train .-> GPU["isolated GPU worker<br/>(Docker · gated · background)"]
</pre>

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

## Knowledge vault — a navigable graph the swarm reads *and* writes

The swarm's knowledge lives in an **Obsidian-style vault**: a graph of markdown notes with `[[wikilinks]]`, on disk,
git-versioned and hand-editable. Agents **navigate** it semantically for the context relevant to their current step
(native prefetch — the local subgraph with its links, not a chunk dump) and **enrich** it with what they learn: new
notes, auto-linked into the graph, deduped, with source provenance. Many vaults, **cross-linked**; it **hot-reloads**
when you edit a note. Markdown is the source of truth; embeddings are the index. This is how run *N+1* starts wiser
than run *N* — and it doubles as a curated store of good/bad examples for the training flywheel below.

---

## Training flywheel — generative → static prompt → auto-finetune

The platform closes the loop from *generation* to *weights*, cheapest rung first:

1. **Generative** — the swarm runs its self-directed loop and produces behavior verified by outcome.
2. **Static prompt / skill** — proven behavior is captured and distilled into a reusable prompt or skill. No training,
   just crystallized instruction — often enough on its own.
3. **Auto-finetune (no hand labels)** — a **meta-cognition** module manufactures the data: for each response it
   recognizes the behavior *pattern*, *critiques* it (coherence / correctness / creativity / safety), generates a
   better *alternative*, and builds a *preference pair* (chosen ≻ rejected). Those pairs — plus good/bad pairs
   harvested from the vault, plus pairs auto-harvested from the agent's own scored trajectory (improvement /
   regression / challenge success) — become a **self-DPO** dataset the model trains on. Iterate: retrain → regenerate
   from the improved model → retrain.

Training runs on the isolated GPU worker (LoRA, bf16, gradient-checkpointing, memory-safe), from small dense LMs up to
**MoE foundation models** — LoRA on attention **and the router** with the load-balance aux loss kept on. Every run is
gated by a real **held-out challenge**: learning is *proven* (held-out loss beats base, the router actually adapted),
never assumed.

### Encoders — feed the model *world state*, not chat

Trainable encoders extend the loop past text-in-a-prompt. Each is a tiny head on a **frozen** base with a
self-generating label and a held-out challenge:

- **World layer** — an encoder turns the framework's world-state (vault notes / memory) into *world tokens* prepended
  to the frozen model, so state enters as **embeddings, not re-serialized chat** — fixed-size regardless of world
  size, and differentiable.
- **Encoder bank** — a learned **value head** (a cheap critic that replaces LLM self-critique in the flywheel), a
  learned **retriever** (which memory to load, ranked by *measured utility* — LM-loss reduction — not surface
  similarity), a **trajectory monitor** (looping / drifting / converging, from the deterministic behavior signals),
  and **perception** (non-text observations → world tokens).
- **Self-model** — the coupled core: the world encoder is the *hub*; the value head and trajectory monitor **condition
  on world state**. One `assess()` answers, per step, *{world, am-I-looping, how-good}* — metacognition with no
  hand-written prompts, and the source of the auto-harvested preference pairs above.

<pre class="mermaid">
flowchart LR
  GEN["generative swarm<br/>(proven behavior)"] --> PROMPT["static prompt / skill<br/>(distill · cheapest)"]
  GEN --> META["meta-cognition<br/>pattern · critique · alternative"]
  VAULT[("vault good/bad")] --> META
  TRAJ["scored trajectory"] --> META
  META --> PAIRS["preference pairs<br/>(chosen ≻ rejected)"]
  PAIRS --> DPO["self-DPO on GPU worker<br/>(LoRA · held-out gate)"]
  DPO -->|improved model| GEN
</pre>

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
