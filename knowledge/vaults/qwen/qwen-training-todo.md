---
title: qwen-training-todo
tags: qwen, todo, plan, moe, roadmap, jlens
source: seed:plan-2026-07-13
links: [[qwen/qwen-moe-training]], [[qwen/qwen-optimization]], [[qwen/qwen-runtime-best-practices]], [[foundation/jacobian-lens]], [[foundation/world-transformer-layer]]
---

# Qwen training + inspection — TODO for the squad (each step has a PROOF gate)

Target: adapt **Qwen3-30B-A3B-Instruct-2507** (bf16 MoE, downloadable HF weights) on the GB10. Every step is "done"
only when its numeric PROOF holds (prove-before-wire). Knowledge notes: [[qwen/qwen-moe-training]],
[[qwen/qwen-optimization]], [[qwen/qwen-runtime-best-practices]].

## Tasks
- [x] Download bf16 MoE → `/models/qwen3-30b-a3b-instruct` (~60GB; leader stays up during download).
- [ ] Wire optimized settings into the training templates: `attn_implementation="sdpa"` (flash-attn won't build on
      GB10), `gradient_checkpointing=True`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
      **Proof:** a 30B run loads + runs without OOM/timeout (background+poll `_dgx_train` handles the slow load).
- [ ] Add `qwen_moe_lora` pipe (transformers+peft): LoRA on **attn + experts + router (mlp.gate)**,
      `router_aux_loss_coef=1e-3`, completion-masked.
      **Proof:** held-out eval_loss ↓ AND `aux_loss` moves AND adapter output ≠ base on a probe set (flat aux = the
      documented router-skip no-op).
- [ ] (Optional) install **ms-swift** in the worker for the turnkey standard LoRA/DPO path (`--target_modules all-linear
      all_router --router_aux_loss_coef 1e-3`).
- [ ] Run the world-transformer-layer ([[foundation/world-transformer-layer]]) on the qwen MoE base (custom module →
      raw transformers). **Proof:** held-out with-world loss < without-world.
- [ ] **Jacobian lens ([[foundation/jacobian-lens]])** on the qwen base: `pip install jlens`, fit a lens (~100 prompts),
      `lens.apply(...)` to read the model's J-space workspace. Uses: (a) inspect what qwen holds mid-reasoning / audit
      for prompt-injection + eval-awareness; (b) a training target/signal to align the world-encoder's tokens to the
      model's own J-space. **Proof:** lens surfaces sensible concepts on a probe set; J-space alignment lowers world-layer
      held-out loss vs unaligned.
- [ ] Feed the self-DPO flywheel ([[foundation/long-horizon-improvement]]) with qwen-generated + vault good/bad pairs;
      DPO-train the adapter. **Proof:** margins ↑, held-out ranking ↑.

## Run protocol (memory-critical — when the serving box is also the training box)
1. Stop the model-serving container (the vLLM/sglang server) to free its reserved VRAM — on a big MoE this can be
   ~90GB+; the agents' LLM calls pause while it is down.
2. Train via the pipe (background+poll; the pre-train safety gate checks free memory first).
3. Restart the serving container (sglang/vLLM takes ~2-3 min to reload a large MoE) → the agents' endpoint is back.

## Hard constraints (facts about this box)
- flash-attn does not build on GB10 → `attn_implementation="sdpa"`.
- bitsandbytes has no Blackwell kernels → bf16 LoRA, not 4-bit QLoRA.
- single device → no ZeRO/FSDP (sharding overhead with one GPU).
- the served `qwen36` is nvfp4 (serving format) → not trainable; only the bf16 HF weights are.
- verify learning by held-out loss + aux_loss movement, never train loss alone.
