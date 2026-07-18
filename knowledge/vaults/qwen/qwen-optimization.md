---
title: qwen-optimization
tags: qwen, lora, memory, gb10, hyperparams, frameworks, dpo, study
source: seed:research-2026-07-13
links: [[qwen/qwen-moe-training]], [[foundation/perf-memory]], [[foundation/training-approaches]], [[foundation/tokenization]]
---

# Qwen3 training optimization — hyperparams · GB10 memory · frameworks

Constraints on THIS box (GB10, arm64 Blackwell, 121GB unified): **flash-attn won't build → use sdpa**;
**bitsandbytes not installed → no 4-bit QLoRA → do bf16 LoRA**; a served model reserves most of VRAM, so **stop the
model-serving container first** to free it, then train via the background+poll runner.

## LoRA hyperparameters (idea → proof)
- **idea:** rank 16–32 · alpha = 2×rank · dropout 0.05 · **lr 5e-5→1e-4** (low end for MoE — routers are sensitive) ·
  cosine schedule · warmup_ratio 0.05 · 1–3 epochs · batch 1 × grad-accum 8–16 · max_seq 2048–4096 · sample-packing on ·
  NEFTune α=5 optional.
- **proof:** held-out eval_loss drops and keeps dropping past epoch 1 without train-loss diverging; if held-out flattens
  while train-loss falls → overfit → cut epochs / lower rank.

## Memory on the single GB10 (bf16 LoRA, no 4-bit)
- 30B bf16 base ≈ **61GB weights**; LoRA adapters + Adam states are tiny (only the adapters train).
- **`attn_implementation="sdpa"` + `gradient_checkpointing=True` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**
  → fits in ~100GB with the leader stopped. Activations at seq 2048 / batch 1 are modest (~3.3B active params).
- **Single-device: NO ZeRO/FSDP** — sharding overhead with one GPU is pure loss. If tight: max_seq→1024, lower rank,
  CPU-offload the optimizer.
- **4-bit without bnb:** effectively skip it (bnb Blackwell kernels absent). torchao int4/int8 is the only path and is
  finicky here — unnecessary given the bf16 base already fits ([[foundation/perf-memory]]).

## Qwen3 specifics
- **Loss-mask to completion only** (mask prompt tokens) — else you train on the question.
- **`enable_thinking` must match the data**: training non-think answers with thinking on teaches inconsistent format
  (same flag that returned empty content on the served leader — see [[foundation/thinking-behavior]]).
- Use the model's chat template verbatim.

## Framework choice
- **Standard LoRA/DPO on the MoE → ms-swift** (transformers backend): turnkey Qwen3-MoE template + auto completion-mask +
  `--target_modules all-linear all_router --router_aux_loss_coef 1e-3`. Megatron-SWIFT expert-parallel is NOT worth it
  single-device (its win needs expert_model_parallel_size>1 across GPUs).
- **Custom trainable layer (world-encoder / prefix adapter) → raw transformers+peft** — ms-swift can't wrap a novel
  module. `Qwen3MoeForCausalLM` + `requires_grad_(False)` + inject via `inputs_embeds`, sdpa, gradient_checkpointing.

## code — ms-swift standard LoRA
```bash
CUDA_VISIBLE_DEVICES=0 swift sft --model Qwen/Qwen3-30B-A3B-Instruct-2507 --train_type lora \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --target_modules all-linear all_router --router_aux_loss_coef 1e-3 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --learning_rate 5e-5 \
  --warmup_ratio 0.05 --num_train_epochs 2 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 --max_length 4096 --gradient_checkpointing true \
  --eval_steps 50 --val_dataset <held-out> --output_dir output
```

## read further
- Qwen ms-swift: https://qwen.readthedocs.io/en/latest/training/ms_swift.html
- PEFT quantization (torchao path): https://huggingface.co/docs/peft/en/developer_guides/quantization
- bitsandbytes (Blackwell support status): https://github.com/bitsandbytes-foundation/bitsandbytes

## dpo
- prompt: You're training a 30B bf16 MoE on a single GB10 and want to save memory. Do you reach for ZeRO-3/FSDP and 4-bit QLoRA?
- chosen: No — on one device use plain single-device bf16 LoRA with gradient_checkpointing + sdpa + expandable_segments; 4-bit needs bitsandbytes which isn't available on Blackwell, and sharding has no benefit with one GPU.
- rejected: Yes, enable ZeRO-3/FSDP and QLoRA 4-bit to be safe — even though there's one GPU (sharding is pure overhead) and bitsandbytes has no Blackwell kernels (it'll just fail).
- prompt: What learning rate for LoRA on a Qwen3 MoE, and how do you confirm it's learning?
- chosen: Use a lower LR (5e-5–1e-4) because routers are sensitive, and confirm by held-out eval_loss dropping plus aux_loss moving — not train loss alone.
- rejected: Use a high LR like 3e-4 for faster convergence and declare success when train loss drops, without checking held-out loss or router aux_loss.
