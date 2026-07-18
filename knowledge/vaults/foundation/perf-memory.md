---
title: perf-memory
tags: memory, quantization, throughput, gb10, safety
source: seed:memory
links: [[split-parallelism]], [[training-approaches]], [[gemma-foundation-native]]
---

# Performance & memory — the numbers that decide if a run fits and how fast it goes

## The memory equation (why a run OOMs)
Training memory ≈ params + optimizer + gradients + activations + KV cache.
- Full FT, Adam, bf16 ≈ **16 bytes/param** (2 weights + 4 grad + 8 optimizer) — a 4B model ≈ 64 GB before
  activations. Why full-FT of even "small" models is heavy.
- **LoRA** trains ~0.1–1% of params → optimizer/grad cost collapses; frozen base dominates. QLoRA 4-bit quarters
  the base again ([[training-approaches]]).
- **Activations** scale with batch × seq_len × hidden. `gradient_checkpointing=True` trades compute for a big cut.
- **KV cache** scales with batch × seq_len × layers × hidden — cap `max_length`/`num_ctx` ([[tokenization]]).

## Precision / quantization
- **bf16** — training default; half fp32 memory, stable range.
- **fp8** — inference + some training; measured here (KV fp8 + mem-fraction 0.85) → **3.22× serving capacity**
  (1.26M→4.06M tokens), uncalibrated-quality caveat.
- **nvfp4** — 4-bit float the served MoE (qwen36) uses; native to Blackwell/GB10.

## Throughput levers (measured)
- Speculative decoding (matched drafter DFlash/MTP) → **2.40× lossless**. Measure τ ONLY against the byte-identical
  served target + engine — never a bf16/QAT proxy, or the number lies.
- Parallelism defaults are often the real ceiling: `OLLAMA_NUM_PARALLEL=16`+`FLASH_ATTENTION=1` → 6.7×
  ([[split-parallelism]]).
- **flash-attn does NOT build on GB10** (FA3/FA4 fail on arm64/GB10). The JAX path and sglang handle attention
  without it.

## GB10 safety (the box is SHARED)
- Cap the worker: `--memory=48g --cpus=12 --shm-size=8g`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. JAX pre-allocating ~75% once OOM'd the whole DGX.
- Gate on system-RAM free (unified memory) before every launch; the served leader + ollama share the pool — never
  starve them.

## Rule
Compute the memory budget first, choose precision + LoRA to fit with headroom, cap the container, gate on real RAM,
measure throughput against the served target — never launch on vibes.

## code — project the budget BEFORE launching + safe container
```python
def train_gb(params_b, full_ft=False, bytes_per_param=None):
    bpp = bytes_per_param or (16 if full_ft else 2)          # full-FT Adam ≈16 B/param; LoRA base ≈2 (bf16)
    return params_b * 1e9 * bpp / 1e9                        # + add activations + KV separately
assert train_gb(4, full_ft=False) < 40, "won't fit 48G with headroom → drop batch / add checkpointing / QLoRA"
```
```bash
# GB10 worker — hard caps so a JAX/torch run can NEVER take the shared box (gate on RAM, VRAM is N/A here)
docker run --gpus all --memory=48g --cpus=12 --shm-size=8g \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v ~/models:/models:ro -v ga_worker_data:/workspace ga_worker:latest
```

## read further
- Transformer memory math (activations/optimizer): https://blog.eleuther.ai/transformer-math/
- Gradient checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- Quantization (fp8/int4): https://github.com/NVIDIA/TransformerEngine  ·  QLoRA arXiv:2305.14314
- Speculative decoding: arXiv:2211.17192 (measure τ against the SERVED target — see [[training-approaches]])
