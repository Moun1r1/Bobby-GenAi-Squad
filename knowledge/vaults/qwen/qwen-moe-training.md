---
title: qwen-moe-training
tags: qwen, moe, lora, router, training, dpo, study
source: seed:research-2026-07-13
links: [[foundation/training-approaches]], [[foundation/split-parallelism]], [[qwen/qwen-optimization]], [[foundation/gemma-foundation-native]]
---

# Qwen3-MoE training — the router footgun + how to adapt an MoE

Target model: **Qwen3-30B-A3B-Instruct-2507** (bf16 MoE, 30.5B total / ~3.3B active, 128 experts). The local `qwen36`
is nvfp4 (serving-only, NOT trainable) — train the downloaded **bf16 HF weights** instead (`/models/qwen3-30b-a3b-instruct`).

## The router-skip no-op (idea → proof)
- **idea:** LoRA with `target_modules=all-linear` on Qwen3-MoE touches attention + expert FFNs but **SKIPS the routers**.
- **why it bites:** with the router frozen, gating never adapts → `aux_loss` stays flat → **the adapter output equals the
  base model**. You "trained" nothing.
- **fix:** include the router (ms-swift: add `all_router`; raw peft: add the `mlp.gate` router linears to `target_modules`)
  AND keep the **load-balancing aux loss** on (`router_aux_loss_coef=1e-3`) — dropping it collapses routing onto one expert.
- **proof (the gate):** `aux_loss` MUST move during training AND held-out loss drops AND the adapter output differs from
  base on a fixed probe set. **Flat aux_loss = FAIL, no matter what train loss does.**

## Which modules to adapt
- Default: LoRA on **attention (q/k/v/o) + expert FFN (gate_proj/up_proj/down_proj) + router (mlp.gate)**, aux-loss on.
- **Preserve generality:** freeze the original experts, ADD new experts, train only those (best for not degrading base skills).
- **Cut cost:** adapt only the top-routed experts per layer (~70% fewer params, near-lossless — per-layer routing is skewed).
- Skip router-z-loss unless you see logit blow-up.

## code — transformers + peft (our custom path)
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16",
        attn_implementation="sdpa")            # flash-attn won't build on GB10
m.config.router_aux_loss_coef = 1e-3           # KEEP load-balance loss → no router collapse
m.config.output_router_logits = True
m.gradient_checkpointing_enable()
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","gate"])  # 'gate' = the router
m = get_peft_model(m, lora)
# … completion-masked SFT/DPO … then CHECK aux_loss moved + held-out loss ↓
```

## read further
- ms-swift MoE FAQ (all_router): https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Frequently-asked-questions.md
- Qwen ms-swift training: https://qwen.readthedocs.io/en/latest/training/ms_swift.html
- MoE fine-tuning (aux loss, expert selection): https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-3-training-large-scale-moes
- Expert selection near-lossless: MoE-Sieve arXiv:2603.24044

## dpo
- prompt: You're LoRA-fine-tuning a Qwen3-MoE and set target_modules to all-linear. What must you check and add?
- chosen: Add the router (all_router / mlp.gate) to LoRA and keep the load-balancing aux loss on; verify aux_loss actually moves and held-out loss drops, else the adapter is a silent no-op.
- rejected: Trust that all-linear covers everything, train, and report success from the dropping train loss without checking aux_loss or held-out — even though the router was skipped and the output equals the base.
- prompt: To adapt a pretrained MoE without wrecking its general skills, what's the safest approach?
- chosen: Freeze the original experts and add new experts (or adapt only top-routed experts), keeping the aux load-balance loss so routing doesn't collapse.
- rejected: Full-fine-tune all experts and drop the aux loss to "let it specialize", collapsing the router onto one expert and destroying generality.
