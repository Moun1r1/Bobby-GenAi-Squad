---
title: Architecture
---

# Architecture — the components and how they fit

Bobby is a **platform**, not a single library. It has five parts: the **engine** (a Python package), a **knowledge
vault** (a note graph on disk), **Studio** (a FastAPI backend + Next.js frontend), an isolated **GPU worker**
(Docker), and a **training layer** (encoders + a self-DPO flywheel). You can use the engine alone, or run the whole
platform.

<pre class="mermaid">
flowchart TB
  subgraph studio["Studio (control room)"]
    FE["Next.js + tRPC frontend<br/>watch the loop live"]
    BE["FastAPI backend<br/>pipelines · SSE event stream"]
  end
  FE <-->|tRPC| BE
  BE <--> ENG["engine — bobby_squad<br/>persistent-self swarm"]
  ENG <--> VAULT[("knowledge vault<br/>[[linked]] markdown graph")]
  BE -->|SSE| FE
  BE -. push code + train .-> GPU["GPU worker (Docker)<br/>isolated · memory-capped · gated"]
  GPU -. trained adapters .-> BE
  ENG -. any OpenAI endpoint .-> LLM["served model<br/>(vLLM / sglang / Ollama / hosted)"]
</pre>

---

## 1. The engine — `bobby_squad`

Pure-Python primitives; talks to any OpenAI-compatible `/v1/chat/completions`. The load-bearing ideas:

| primitive | file | what it is |
|---|---|---|
| `Agent` + `SelfCore` | `core.py` | persistent-self: identity/goal/progress in a **pinned tier** compaction never touches; a self-directed loop (`select_target → make_plan → carry_out → record`) chooses its own move. Has an optional `recall(task)` hook (native prefetch, §2). |
| `squad_solve` | `squad.py` | recursive coverage — a squad drains a shared board; `verify` (run-don't-ask) decides done-vs-split; plateau = board drains. |
| `IdeaLedger` | `ledger.py` | shared idea board — deterministic identity floor (no re-generation) + emergent agent-assigned states + active-repulsion frontier. |
| `prove` | `proving.py` | enforced test **validity** — headroom guard + negative control + replication/CI → `WIRE / MARGINAL / DELETE / INCONCLUSIVE / INVALID / DEFER`. |
| `SemanticMemory` | `correction_memory.py` | novelty-gated store that self-governs retention by learned usage (deterministic recall floor). |
| `BehaviorTrace` + `MetaTools` | `metacognition.py` | metacognition — detect a peer's bias/frontier from its real trace; the source of the training signal (§6). |
| `SandboxTools` / `ReadOnlyTools` / `DgxTools` | `agent_tools.py` | tool surfaces: local sandbox dev loop, read-only exploration, and the GPU-worker bridge (§5). |
| `RunStats` | `observability.py` | meters LLM + observer events for a run. |

Design rule that governs all of it: **no static prompts, no hardcoded roles** — capability comes from a rich *self*
+ real *tools* + an open move-space; a "critic" is a *move*, not a persona.

---

## 2. The knowledge vault — `KnowledgeVault` + `VaultHub` (`vault.py`)

Not a vector blob — an **Obsidian-style graph of markdown notes** on disk, git-versioned and hand-editable.

- **A note** = frontmatter (`title`, `tags`, `source`, `links`) + a markdown body with `[[wikilinks]]`. `KnowledgeVault`
  parses a directory into a graph (links + backlinks) plus a semantic entry index (`EmbeddingRetriever`, lexical
  fallback).
- **`VaultHub`** manages **many** vaults with **cross-vault links** (`[[vault/note]]`). `navigate(query)` does
  semantic entry across all vaults + bounded link-hop expansion → returns the local subgraph (excerpts + link names),
  not a chunk dump. `enrich(vault, title, body, links)` writes a note back — deduped, auto-linked, with provenance.
- **Recall wire:** `Agent(recall=…)` calls `hub.navigate(step_target)` and injects the result as a step's reference —
  *native prefetch*. This is the seam that lets an agent reason from the vault without any hardcoded prompt.
- **Learned re-rank (optional):** if a trained retrieval encoder (§6) is installed, entry is re-ranked by *measured
  utility* instead of cosine (`stats.recall` → `learned`).
- **Hot-reload:** a per-vault dir signature (files + mtimes) + a shared embed cache → hand-edited or externally-added
  notes go live within ~2 s (or via `/vault/reload`), without a restart; the shared cache means a reload re-parses but
  re-embeds only new text.

Markdown is the **source of truth**; embeddings are the **index**. It's how run *N+1* starts wiser than run *N*, and
it doubles as a curated good/bad store for the training flywheel.

---

## 3. Studio backend — FastAPI (`studio/backend/`)

Wraps the engine as **pipelines** and streams what they do.

- **`runner.py`** — the pipeline registry (`PIPELINES`). Each pipeline is `(Run) -> summary`; agents are wired with
  `observer=run.observe` so every `target/plan/move/tool/cycle` becomes an event. World-context pipelines
  (`_idea_lab`, `_dev_lab`) share one generic engine; the SELF is data. Training pipelines live here too (§6).
- **`app.py`** — the HTTP surface: `POST /runs` (launch), `GET /runs/{id}/stream` (SSE live events), `/vault/{stats,
  graph,note,navigate,list,reload}`, `/dgx/{health,safe,stream}`, plus knowledge/experts/proofs endpoints.
- **`store.py`** — persistence over Qdrant (vector DB) with an in-memory fallback; collections for runs / events /
  knowledge.
- **`dgx_monitor.py`** — the realtime hardware monitor + pre-train safety gate (§5).

---

## 4. Studio frontend — Next.js + tRPC (`studio/frontend/`)

A live control room. tRPC queries proxy to the backend; `useRunEvents` subscribes to the SSE stream. The UI is
organized by **engine layer**, not a flat menu — see **[Interface »](interface)**. The point: you *watch the
generative loop happen* (the board draining, each agent's move/tool stream, the vault graph, the GPU monitor, the
proof bench) instead of reading logs.

---

## 5. The GPU worker — any CUDA GPU host, isolated Docker + safety gate

**Not tied to any specific product.** The worker talks to a GPU host over **SSH + Docker + `nvidia-smi`** — so it's a
workstation, a cloud VM, or a cluster node, whatever CUDA box you have. Training must never crash a shared box, so it
runs in a **dedicated, memory-capped Docker container** with a pre-train gate. (Some class/endpoint names still carry
a `dgx` prefix from where this was first developed — it's a naming artifact, not a hardware requirement.)

- **Worker bridge** (`agent_tools.py`) — `push` (scp + `docker cp` into the worker), `run` (with a `background=True`
  mode), `pull`, `logs`. The worker mounts model weights read-only and a writable workspace.
- **GPU monitor** (`dgx_monitor.py`) — polls GPU/CPU/RAM/disk/containers; `is_safe()` gates every training launch (on
  unified-memory boxes it gates on system-RAM free). Streams to the Compute screen.
- **`_dgx_train` (`runner.py`)** — launches a training script as a **background job** and polls its log for the
  challenge marker, so a big-model load + train is never cut off by an exec timeout.

<pre class="mermaid">
flowchart LR
  BE["backend pipeline"] -->|is_safe gate| MON["GPU monitor"]
  BE -->|push code+data| W["worker container<br/>(--memory cap · any CUDA GPU)"]
  BE -->|background run| W
  W -->|poll logs| BE
  W -->|pull adapter| BE
  MODELS[("/models (ro)")] --- W
  WS[("/workspace (rw)")] --- W
</pre>

---

## 6. Training layer — encoders + the self-DPO flywheel

The platform can turn *proven behavior* into *weights*. See **[What's new »](whats-new)** for the narrative; the
components:

- **World layer** (`world_layer.py`) — `WorldEncoder` (perceiver latents → K "world tokens") + `WorldPrefixLM`
  (prepends them to a **frozen** LM's input embeddings). Feeds world-state as embeddings, not re-serialized chat.
- **Encoder bank** (`encoders.py`) — `ValueHead` (a learned critic), `RetrievalEncoder` (learned recall by utility),
  `TrajectoryMonitor` (looping/drifting/converging from behavior signals). `SelfMonitor` **couples** them: the world
  encoder is the hub; value + monitor condition on world state. `trajectory_dpo(steps)` auto-harvests preference
  pairs from a scored trajectory. `LearnedRetriever` (`learned_retriever.py`) is the torch-free numpy reload used at
  inference in the vault.
- **Training pipelines** (`runner.py`) — each trains on the worker, with a **self-generating label** and a **held-out
  challenge** (learning is proven, not assumed):

| pipeline | trains | label | challenge |
|---|---|---|---|
| `world_layer` | world encoder on a frozen base | vault-note embeddings | held-out with-world loss < without |
| `value_head` | a learned critic | self-DPO chosen ≻ rejected | held-out ranking accuracy |
| `retrieval_encoder` | learned recall | LM-loss reduction (utility) | held-out utility > cosine |
| `trajectory_monitor` | the self-monitor | deterministic regime signals | held-out regime accuracy |
| `perception` | world layer for non-text obs | observation → target | with-obs beats without |
| `self_model` | coupled world+value+monitor | pairs + signals | value & regime both pass |
| `self_dpo` | LoRA on a foundation model | meta-cognition preference pairs | DPO loss drops, margin ↑ |
| `qwen_moe_lora` | MoE LoRA (attn + router) | vault targets, aux-loss on | held-out adapter < base AND router adapted |

<pre class="mermaid">
flowchart LR
  GEN["generative swarm"] --> META["meta-cognition<br/>pattern·critique·alternative"]
  VAULT[("vault good/bad")] --> META
  TRAJ["scored trajectory"] --> META
  META --> PAIRS["preference pairs"]
  PAIRS --> DPO["self-DPO / LoRA<br/>on GPU worker"]
  DPO -->|held-out gate| CH{"proven?"}
  CH -->|yes| MODEL["improved model"] --> GEN
  CH -->|no| DROP["kept out (not wired)"]
</pre>

---

See also: **[What's new »](whats-new)** · **[Interface »](interface)** · **[The engine, layer by layer »](engine)**.

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
</script>
