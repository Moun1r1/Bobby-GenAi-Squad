---
title: world-transformer-layer
tags: architecture, world-model, encoder, avoid-chat, dpo, study
source: seed:memory
links: [[memory-selection]], [[tokenization]], [[gemma-foundation-native]], [[long-horizon-improvement]], [[split-parallelism]]
---

# World transformer layer — feed the WORLD to the model natively, not as chat

Today the agent round-trips its whole world (goal, vault subgraph, memory, tool state) through CHAT TOKENS every
step: serialize → tokenize → attend → detokenize. That's lossy, slow, and context-bounded. The upgrade: a custom
transformer layer that consumes the framework's WORLD-CONTEXT as EMBEDDINGS and injects them directly into the
model — so state enters as vectors, not re-serialized prose. Chat becomes the output channel, not the state bus.

## The idea (native, trainable)
- **World encoder** — a small transformer/encoder that maps structured world-state (vault note vectors, memory-tier
  items, tool/observation state, the SELF) → a set of `world tokens` (soft embeddings), one bank per source.
- **Prefix / adapter injection** — prepend those world tokens as a learned PREFIX (prefix-tuning) or fuse them via a
  cross-attention adapter layer inside the gemma trunk. The base LM weights stay frozen; only the encoder + adapter
  train. This is a real architecture change, trained with our own DPO — not a prompt.
- **Why it beats chat**: fixed-size world bank regardless of world size (no context blow-up), differentiable state
  (the model can be TRAINED to use the world, via [[long-horizon-improvement]] self-DPO), and no serialization loss.
- Native substrate: a Flax `nnx` module wrapping `gm.nn` with an extra cross-attn block; shard the encoder on the
  model axis ([[split-parallelism]]). The world vectors reuse the SAME embedder as the vault ([[memory-selection]]).

## code — world-encoder → prefix, fused into the frozen trunk (sketch)
```python
import flax.nnx as nnx, jax.numpy as jnp
class WorldEncoder(nnx.Module):                      # structured world-state → K learned "world tokens"
    def __init__(self, d_model, k=32, *, rngs):
        self.proj = nnx.Linear(768, d_model, rngs=rngs)     # vault/memory vectors (nomic 768) → model dim
        self.slots = nnx.Param(jnp.zeros((k, d_model)))     # learned query slots (perceiver-style)
        self.attn = nnx.MultiHeadAttention(num_heads=8, in_features=d_model, rngs=rngs)
    def __call__(self, world_vecs):                  # world_vecs: [N,768] from vault + memory tiers
        ctx = self.proj(world_vecs)
        q = jnp.broadcast_to(self.slots.value[None], (ctx.shape[0] or 1, *self.slots.value.shape))
        return self.attn(q, ctx)                      # -> [K, d_model] world tokens, PREPENDED to the gemma prefix
# base gemma frozen; train {WorldEncoder + a cross-attn adapter} with self-DPO (chosen uses world tokens, rejected re-serializes to chat)
```

## read further
- Prefix-Tuning: arXiv:2101.00190 · P-Tuning v2: arXiv:2110.07602
- Perceiver / learned latent queries: arXiv:2103.03206 · Flamingo cross-attn fusion: arXiv:2204.14198
- Native module surface: [[gemma-foundation-native]] (`gm.nn`, Flax `nnx`)

## dpo
- prompt: The agent needs its full world-context (goal, relevant vault notes, memory) available while it acts. How should that state reach the model?
- chosen: Encode the structured world-state into a fixed bank of learned world-token embeddings and inject them as a prefix/adapter into the frozen trunk, so state enters natively as vectors.
- rejected: Serialize the entire world — every note, every memory item — into a giant chat prompt and re-tokenize it from scratch on every single step.
- prompt: You want the model to get BETTER at using world-context over time. Prompt-only or trainable layer?
- chosen: Make the world pathway a trainable layer (world encoder + adapter) optimized with self-DPO, so the model learns to attend to the world state rather than being told about it in prose.
- rejected: Keep stuffing more instructions into the chat prompt hoping the model reads the world text more carefully, with nothing trainable.
