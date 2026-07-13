---
title: What's new
---

# What truly changed — from a library to a platform

Bobby started as a **Python library**: persistent-self agents, a self-organizing squad, and a `prove` gate, talking
to any OpenAI endpoint. That core is unchanged. What changed is that it grew into a **platform** — you can now watch
the swarm work, give it a navigable memory it writes back to, and let it *train models* from what it proves. Here is
each change and the problem it solves.

## 1. Studio — watch the generative loop, don't read logs

**Before:** a run was a wall of stdout. **Now:** a FastAPI backend exposes each pipeline with a live SSE event stream,
and a Next.js frontend renders it — the shared board draining, each agent's `target → plan → move → tool`, the memory
tiers, the vault graph, the proof bench, and a realtime GPU monitor. *Why it matters:* generative agents fail in ways
you only catch by seeing the trajectory; a control room turns "why did it do that?" from log-archaeology into a live
view. → [Interface](interface).

## 2. Knowledge vault — a graph the swarm reads *and* writes

**Before:** memory was a flat semantic store — chunks in a bag, write-only to a run. **Now:** knowledge is an
**Obsidian-style graph** of markdown notes with `[[wikilinks]]`, on disk and git-versioned. Agents **navigate** it for
the local subgraph relevant to their step (native prefetch, not a chunk dump) and **enrich** it with what they learn —
new notes, auto-linked, deduped, with provenance. Many vaults, cross-linked; hot-reloads on edit. *Why it matters:*
flat recall returns "similar text"; a linked graph returns *the relevant neighborhood with its structure*, and because
the swarm writes back, **run N+1 starts wiser than run N**. Markdown stays the source of truth; embeddings are the
index. → [Architecture §2](architecture).

## 3. The GPU worker — the swarm can train, safely

**Before:** the framework only *called* models. **Now:** it can *train* them, in an isolated, memory-capped Docker
container the swarm pushes code to. A realtime hardware monitor gates every launch (it refuses to start unless the box
has headroom, so a shared GPU is never starved), and long runs go to the background with their logs streamed back.
*Why it matters:* training on a shared box is the fastest way to crash it; the gate + isolation make "let the agents
train something" safe by construction. → [Architecture §5](architecture).

## 4. The training flywheel — generative → static prompt → auto-finetune

The platform closes the loop from *generation* to *weights*, cheapest rung first:

1. **Generative** — the swarm runs and produces behavior verified by outcome.
2. **Static prompt / skill** — distill proven behavior into a reusable prompt or skill. No training — often enough.
3. **Auto-finetune, no hand labels** — a **meta-cognition** module makes the data: for each response it recognizes the
   behavior *pattern*, *critiques* it (coherence / correctness / creativity / safety), generates a better
   *alternative*, and builds a *preference pair* (chosen ≻ rejected). Add the good/bad pairs harvested from the vault,
   plus pairs auto-harvested from the agent's own **scored trajectory** (improvement / regression / challenge
   success), and you have a **self-DPO** dataset — trained on the worker, iterated: retrain → regenerate from the
   improved model → retrain.

*Why it matters:* hand-labeling preference data is the bottleneck in fine-tuning; here the swarm's own metacognition
manufactures it, and a **held-out challenge** proves the model actually improved before anything is wired.

## 5. Trainable encoders — feed the model *world state*, not chat

Chat re-serializes the entire world into tokens every step — lossy, slow, context-bounded. The encoders fix that; each
is a tiny head on a **frozen** base with a self-generating label and a held-out gate:

- **World layer** — an encoder turns world-state (vault notes / memory) into a fixed bank of **world tokens** prepended
  to the frozen model. State enters as embeddings, fixed-size regardless of world size, and differentiable (the model
  can be *trained* to use it).
- **Encoder bank** — a learned **value head** (a cheap deterministic critic that can replace an LLM self-critique
  call), a learned **retriever** (which memory to load, ranked by *measured utility* — LM-loss reduction — not surface
  similarity), a **trajectory monitor** (looping / drifting / converging, learned from the deterministic behavior
  signals), and **perception** (non-text observations → world tokens).
- **Self-model** — the coupled core: the world encoder is the *hub*; the value head and monitor **condition on world
  state**. One `assess()` answers per step *{world, am-I-looping, how-good}* — metacognition with no hand-written
  prompts, and the source of the auto-harvested pairs above.

*Why it matters:* it turns the framework's world-state from a prompt-string you hope the model reads into a trainable
signal the model learns to use.

## 6. MoE foundation training

Training scales from small dense LMs up to **Mixture-of-Experts** foundation models. The MoE recipe is explicit: LoRA
on attention **and the router**, with the load-balancing aux loss kept on — and the challenge checks the router
actually adapted (the common `target_modules=all-linear` silently skips the router and trains nothing). *Why it
matters:* MoE is where open foundation models are going, and the router is exactly the part naïve LoRA misses.

---

## Honest limits

- Outputs are still written by whatever model you point it at — a rich navigable result, not a signed audit.
- **Training is gated, not guaranteed:** the encoders and the flywheel are proven per-run against a held-out
  challenge; a run that doesn't beat its baseline is kept *out*, not wired.
- Large MoE training is memory-bound: it needs a fitting GPU (or a mature MoE-LoRA backend) — the platform gates and
  reports this rather than pretending it fits.

---

See also: **[Architecture »](architecture)** · **[Interface »](interface)** · **[README »](https://github.com/Moun1r1/Bobby-GenAi-Squad)**.
