# Bobby GenAi Squad — Wiki: features, design & proofs

The full picture: what the platform **does** (features), **why it's built that way** (design), and the **runnable
proofs** behind the claims. For deeper docs see [Architecture](../docs/architecture.md),
[What's new](../docs/whats-new.md), and [Interface](../docs/interface.md).

---

## Features — what it does, by area

**Self-organizing swarm (the engine).** A squad of persistent-self agents drains a shared to-do board — no
orchestrator, no fixed roles, no scripted workflow. Each agent runs a self-directed loop (`select_target → make_plan →
carry_out → record`) and picks its own move; `squad_solve` splits under-covered work and re-queues it until the board
drains. `IdeaLedger` keeps a shared idea board with dedup + emergent states + an active-repulsion frontier.

**Long horizon without context blowup.** Identity, goal, and accumulated progress live in a **pinned tier** that
context-compaction never touches, so a run reads a whole codebase or a stack of papers section-by-section while the
prompt stays flat — and an agent *crystallizes into a specialist* grounded in what it read, then carries that
knowledge to other agents and other fields.

**Knowledge vault.** Knowledge is an **Obsidian-style graph** of markdown notes with `[[wikilinks]]`, on disk and
git-versioned. Agents **navigate** it for the local subgraph relevant to their step (native prefetch, not a chunk
dump) and **enrich** it with what they learn — deduped, auto-linked, with provenance. Many vaults, cross-linked;
hot-reloads on edit; markdown is the source of truth, embeddings are the index. So run *N+1* starts wiser than run
*N*. (`VaultHub` / `KnowledgeVault`.)

**Prove, don't claim.** `prove` isn't a bare A/B — it enforces validity: a headroom guard, a negative control, and
replication with a CI, returning `WIRE / MARGINAL / DELETE / INCONCLUSIVE / INVALID / DEFER`. Most proposals fail a
fair test — which is the point.

**Studio (control room).** A FastAPI backend exposes the engine as pipelines with a live SSE event stream; a Next.js
frontend lets you *watch the run happen* — the board draining, each agent's move/tool stream, the vault graph, the
proof bench, and a realtime GPU monitor — organized by engine layer. See [Interface](../docs/interface.md).

**GPU worker + training.** An isolated, memory-capped **Docker** container on **any CUDA GPU host** (workstation,
cloud VM, cluster node — nothing product-specific), with a pre-train safety gate and background runs. On it, the
platform can **train models, not just call them**:

- **Training flywheel — generative → static prompt → auto-finetune.** Proven behavior → distilled prompt/skill →
  **self-DPO**: a meta-cognition module manufactures preference pairs (pattern · critique · alternative · chosen ≻
  rejected) with **no hand labels**, plus vault good/bad pairs and pairs auto-harvested from the agent's own scored
  trajectory.
- **Trainable encoders.** A **world layer** feeds world-state to a frozen model as *embeddings, not re-serialized
  chat*; a learned **value head** (a cheap critic), a learned **retriever** (which memory to load, by utility not
  cosine), a **trajectory monitor** (looping/drifting/converging), and **perception** (non-text → world tokens). The
  **self-model** couples them: the world encoder is the hub; value + monitor condition on world state.
- **MoE foundation LoRA** — dense LMs up to Mixture-of-Experts, LoRA on attention **and the router** with the
  load-balance aux loss.

## Design principles

1. **No static prompts, no hardcoded roles** — capability comes from a rich *self* + real *tools* + an open
   move-space; a "critic" is a *move*, not a persona.
2. **Verify by outcome** — a real run / a strict judge, never the model declaring "done" in prose.
3. **Prove, don't claim** — every gain goes through `prove` (headroom + negative control + CI), or it isn't wired.
4. **Guard-first** — guardable mistakes (identity dedup, recall floor) live in deterministic code; only un-guardable
   generative choices go to the model.
5. **Markdown is the source of truth** — the vault is human-readable, git-versioned, hand-editable; embeddings are
   just the index rebuilt from it.
6. **Learning is proven, not assumed** — every training run is gated by a **held-out challenge** written before it;
   a run that doesn't beat its baseline is kept *out*.

---

## Proofs

Every claim above has a **runnable proof** here. Two kinds:

- **Deterministic** — run with just Python, no model, no network. The verdict is computed live.
- **Endpoint-gated** — point at an OpenAI-compatible LLM (`BOBBY_LLM_URL`) and, for some, an embedder
  (`BOBBY_EMBED_URL`). These run the real agents; the output samples in [`samples/`](samples/) are captured from
  actual runs (real server token counts, real transcripts — not hand-written).

Run a proof from the repo root, e.g. `python wiki/proofs/organization_recursive.py`.

| pipeline | proves | kind | how to run |
|---|---|---|---|
| `proofs/proposals_gain.py` | fair A/Bs: Memory-Gate **WIRE +191%**, Active-Design **WIRE**, CWBU **DELETE** | deterministic | `python wiki/proofs/proposals_gain.py` |
| `proofs/memory_policy_gain.py` | self-evolving memory retention **WIRE**, non-predictive negative control **DELETE** | deterministic | `python wiki/proofs/memory_policy_gain.py` |
| `proofs/organization_recursive.py` | recursive coordination improves coverage vs a solo pass | endpoint | `python wiki/proofs/organization_recursive.py` |
| `proofs/cross_domain.py` | one engine, any behavior the request asks — strict-judge graded | endpoint | `python wiki/proofs/cross_domain.py` |
| `proofs/self_review.py` | metacognition: an agent detects a peer's bias & frontier from its real trace | endpoint | `python wiki/proofs/self_review.py` |
| `proofs/self_development.py` | full dev loop: discover → build+verify → prove | endpoint | `python wiki/proofs/self_development.py` |
| `proofs/self_improve_connectivity.py` | the squad invents + builds + proves its own inter-agent connectivity | endpoint | `python wiki/proofs/self_improve_connectivity.py` |
| `proofs/squad_reads_code.py` | **long-horizon:** a self-organizing squad reads whole codebases **end to end**, section-by-section, self-paced, with a **bounded prompt** (no context blowup) | endpoint + a corpus dir | `HORIZON_APPS=/path/to/repos python wiki/proofs/squad_reads_code.py` |
| `proofs/squad_reads_pdfs.py` | **arXiv knowledge farm:** the squad downloads real arXiv papers, reads them section-by-section, and **builds transferable expert knowledge** — same long-horizon mechanism, on papers instead of code | endpoint + `pip install bobby-genai-squad[papers]` | `python wiki/proofs/squad_reads_pdfs.py` |
| `proofs/transfer_knowledge.py` | **transferable knowledge:** a concept one agent read from a paper is recalled by *another* agent who never read it, and **carried across domains** (physics → number theory) via the shared semantic store | endpoint + embedder | `python wiki/proofs/transfer_knowledge.py` |

## Samples (captured from real runs)

- [`samples/squad_reads_code_output.txt`](samples/squad_reads_code_output.txt) — two agents read `vue-core` and
  `django` **END-TO-END** (13/13, 15/15 sections) and evolved into codebase experts, while the prompt held only the
  current sections. Impossible to fake: the specialist identities cite exact internals of the real source.
- [`samples/arxiv_squad_output.txt`](samples/arxiv_squad_output.txt) — 5 agents each read a **real arXiv paper**
  (IDs verifiable on arxiv.org) section-by-section and evolved into specialists whose expertise cites the papers'
  actual technical cores (Liénard–Wiechert fields / AdS4 Coulomb seed, Morita invariance of Drinfeld centers, the
  quopit Pauli group). A model can't produce those specifics without reading the paper.
- [`samples/transfer_knowledge_output.txt`](samples/transfer_knowledge_output.txt) — Cantor (read only a logic
  paper) correctly explains a *physics* paper it never read, from the shared store (grounded=True), and an agent
  bridges "antipodal matching" from hep-th into number theory. Knowledge transferred across agents and domains.
- [`samples/KNOWLEDGE_MAP_25_oss_repos.md`](samples/KNOWLEDGE_MAP_25_oss_repos.md) — one persistent-self agent
  streamed **25 large OSS codebases** (hermes, redis, django, langchain, llama.cpp, tokio, polars, …) into a pinned
  index. Real, specific findings per repo.
- [`samples/long_horizon_flatness.txt`](samples/long_horizon_flatness.txt) — the mechanical proof for that run:
  pinned prompt stayed **≤ 4689 tokens** across all 25 while the naive counterfactual reached **40075** (~8.5×).
  These are real served `prompt_tokens` — the unfakeable part.

## The platform additions — how they're proven

The proofs above cover the **engine**. The newer platform pieces (the knowledge vault and the training flywheel — see
[What's new](../docs/whats-new.md) and [Architecture](../docs/architecture.md)) hold to the same **"prove, don't
claim"** discipline, just with a different gate:

- **The knowledge vault** is measured with a **with-vs-without gain-proof**: run the same step twice — recall off (the
  bare agent) vs recall on (the navigated vault injected) — and score the difference on a real metric, with a stated
  threshold and a `WIRE / MARGINAL / DELETE / DEFER` verdict. If the vault doesn't help, it isn't wired.
- **Every training run** (the encoders, the world-transformer layer, self-DPO, MoE LoRA) is gated by a **held-out
  challenge** written *before* training — a real pass/fail on data the run never saw (e.g. *held-out loss must beat the
  base model*, *the MoE router must actually adapt*). Learning is **proven per run, never assumed**; a run that
  doesn't beat its baseline is kept *out*, not shipped.

These need a served model (and, for training, a GPU worker), so they aren't one-command deterministic like the engine
proofs — the held-out challenge is the evidence, and it's checked live on each run. Same principle throughout: a gain
is trusted only when a fair, controlled test says so.

## How to trust these

The deterministic proofs recompute their verdicts every run. The endpoint-gated samples were produced by the
agents on a served model; the numbers that can't be faked (server token counts, section-reached counters,
grounded specifics quoted from real source) are the evidence. Re-run any of them and you get the same shape. The
platform additions (vault, training) are gated the same way — a with/without gain-proof or a held-out challenge, not a
claim.
