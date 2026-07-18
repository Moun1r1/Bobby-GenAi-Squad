---
title: training-approaches
tags: sft, lora, dpo, grpo, self-dpo
source: seed:memory
links: [[gemma-foundation-native]], [[perf-memory]], [[tokenization]]
---

# Training approaches — which method for which goal

## The ladder (cheapest → heaviest)
- **Prompting / in-context** — no weights change. Try first; often hits the bar. (Measured here: prompt-fixes alone
  reached ≥0.9 pass@1 on 9 reasoning primitives — training was NOT justified. Prove the gap before you train.)
- **SFT** — teach format/behavior from (prompt → response) pairs. Full-FT is heavy ([[perf-memory]]).
- **LoRA / QLoRA** — freeze base, train low-rank adapters (r=8–64) on attention/MLP. QLoRA adds a 4-bit frozen
  base. Default fine-tune: ~0.1–1% of params, memory-safe, mergeable. Native path: `gm.nn.LoRA`
  ([[gemma-foundation-native]]).
- **DPO** — preference optimization from (prompt, chosen, rejected); no reward model. Raises the margin of chosen
  over rejected.
- **GRPO / RLHF** — online RL from a reward signal; heaviest, needs a reward model/verifier + rollouts.

## Self-DPO (the flywheel here — proven)
The agent manufactures its OWN preference data via meta-cognition: produce a response → recognize the behavior
PATTERN → CRITIQUE (coherence/correctness/creativity/safety) → generate a better ALTERNATIVE → build the PAIR
(chosen ≻ rejected). No external labels. Measured on Gemma3-1B LoRA: DPO loss 0.69→0.49, rewards/margins 0→1.69,
accuracy 0→1.0. Iterate: retrain, regenerate pairs from the improved model, repeat.

## Proving learning (never claim done without this)
- Write a REAL acceptance metric BEFORE training: final loss < X, or eval accuracy > Y, on a held-out slice.
- The challenge must be FAIR — never rigged. Loss below chance (DPO < 0.69) is the floor, not the goal; check the
  margin/accuracy actually moved. Verify by OUTCOME: a passing `challenge.py` on a real run.

## Rule
Climb the ladder — prompt → LoRA → DPO → RL — stopping at the first rung that passes a real challenge. Prove the gap
justifies the rung before spending on it.

## code — DPO on preference pairs (trl, memory-safe)
```python
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
# rows: [{"prompt":…, "chosen":…, "rejected":…}]  — from self-critique OR harvested from vault good/bad notes
cfg = DPOConfig(output_dir="/workspace/dpo_out", per_device_train_batch_size=1,
                gradient_accumulation_steps=4, num_train_epochs=3, learning_rate=5e-5,
                bf16=True, beta=0.1, max_length=512, gradient_checkpointing=True)
lora = LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj"])
tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
r = tr.train()                                                # loss should drop below chance 0.69; check margin↑
```
The `chosen ≻ rejected` pairs can come from the meta-cognition self-critique OR from the **good/bad DPO blocks** in
the behavior notes — see [[long-horizon-improvement]] and [[behavior-patterns]].

## read further
- TRL DPO (the trainer used here): https://huggingface.co/docs/trl/dpo_trainer  ·  https://github.com/huggingface/trl
- DPO paper (why no reward model): arXiv:2305.18290
- PEFT/LoRA: https://huggingface.co/docs/peft  ·  LoRA paper arXiv:2106.09685
- GRPO (next rung): https://huggingface.co/docs/trl/grpo_trainer
