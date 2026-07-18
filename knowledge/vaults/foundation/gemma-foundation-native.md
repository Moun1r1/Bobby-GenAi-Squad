---
title: gemma-foundation-native
tags: gemma, jax, flax, native, moe
source: seed:memory
links: [[split-parallelism]], [[tokenization]], [[training-approaches]], [[perf-memory]]
---

# Gemma foundation — the NATIVE generative approach (JAX/Flax `gemma.gm`)

The native path for real Gemma foundation work is Google's `gemma` library on JAX/Flax — NOT HuggingFace
`transformers`. Reach for `gemma.gm` first; fall back to HF only when a native op is missing.

## Native building blocks
- `gemma.gm.nn` — model defs (`gm.nn.Gemma3_1B`, `Gemma3_4B`, gemma4 / MoE variants). Params are Flax `nnx`
  modules, sharded over a `jax.sharding.Mesh` (see [[split-parallelism]]).
- `gemma.gm.text.ChatSampler` / `Sampler` — native generation (greedy / temperature / top-p). Sampling is a
  first-class object, not `model.generate` — that IS the generative approach.
- `gemma.gm.ckpts` — Orbax checkpoints (the native weight format). Read weights from the read-only dir; write
  adapters/checkpoints only to the writable workspace.
- `gemma.gm.nn.LoRA` — native LoRA of attention/MLP projections; trains as Flax params, no PEFT. See
  [[training-approaches]].
- Tokenizer: `gemma.gm.text.Gemma3Tokenizer` (SentencePiece) — the served model's vocab; see [[tokenization]].

## gemma4 / MoE specifics
- MoE router: adapt router + expert LoRA, keep the shared trunk frozen for memory-safety. Active-params <<
  total-params → size the memory budget on ACTIVE params ([[perf-memory]]).
- JAX/Flax is the right substrate for MoE + custom sharding; HF's MoE support lags the native lib.

## Provenance / hard facts (this project)
- JAX-CUDA verified on the GB10 at runtime — `jax.devices()` reports `CudaDevice`.
- gemma JAX lib trains gemma4 / MoE here. Install can conflict with PyJWT transitive deps — resolve before import;
  import-check `jax`+`flax` at build, confirm `gemma` at runtime.
- The served model (e.g. a Qwen3 MoE in nvfp4/fp8) is exposed via an OpenAI-compatible endpoint (vLLM / sglang)
  that the agents call; the same box can host both the server and a training run if VRAM is managed
  (see [[qwen/qwen-training-todo]]).

## Rule
"Train / adapt a Gemma model" → a `gemma.gm` script on a JAX `Mesh` with `gm.nn.LoRA` + `gm.ckpts`, writing
adapters to the workspace — not a generic `transformers`+`peft` script.

## code — native generate + LoRA fine-tune
```python
import gemma.gm as gm
# load native weights + tokenizer (read-only dir), sample the native way
model = gm.nn.Gemma3_1B()
params = gm.ckpts.load_params("/models/gemma3-1b")            # Orbax
sampler = gm.text.ChatSampler(model=model, params=params, tokenizer=gm.text.Gemma3Tokenizer())
print(sampler.chat("Explain RMSNorm in one line."))

# LoRA fine-tune — adapters are Flax params, base frozen; write ONLY to /workspace
lora_model = gm.nn.LoRA(model=model, rank=8)                  # wraps q/k/v/o + MLP projections
trainer = gm.train.Trainer(model=lora_model, params=params, optimizer=optax.adamw(1e-4))
# … train loop over tokenized (input, target) batches …
gm.ckpts.save_params(new_params, "/workspace/adapter")       # never /models
```

## read further (study the real repos)
- Gemma native lib (the source of truth for `gemma.gm`): https://github.com/google-deepmind/gemma
- Gemma docs + fine-tuning colabs: https://gemma-llm.readthedocs.io  ·  https://ai.google.dev/gemma/docs
- Flax `nnx` (how params/modules work natively): https://flax.readthedocs.io/en/latest/nnx/index.html
- MoE routing reference (gemma4 path): read `gemma/gm/nn/` in the repo above.

Related capability notes: [[loops-system]], [[tools-detection]], [[long-horizon-improvement]].
