---
title: jacobian-lens
tags: interpretability, jlens, anthropic, workspace, tools, world-model
source: seed:anthropic-2026-07
links: [[world-transformer-layer]], [[memory-selection]], [[gemma-foundation-native]]
---

# Jacobian Lens (J-lens) — read a model's internal "workspace"

Anthropic interpretability technique (published 2026-07-06). For an internal residual-stream activation, the J-lens
reads out what that activation is "disposed to make the model say" at some point in the near future — a logit-lens
variant that surfaces **future speakability**, not just the immediate next token.

## Facts
- **Formula:** `lens_l(h) = unembed(J_l @ h)`, where `J_l = E[∂h_final/∂h_l]` — the expected Jacobian from layer-l
  activations to the final layer, averaged over prompts/positions in a web-text corpus.
- **J-space finding** ("Verbalizable Representations Form a Global Workspace in Language Models"): models hold a small,
  privileged set of concepts (dozens at once, <10% of activity) while reasoning; most computation (grammar, rote
  continuation) runs outside it. Anthropic draws a parallel to **Global Workspace Theory** (Baars). Shown to be
  causal: swapping the J-space "Soccer" vector for "Rugby" changed the model's reported answer.
- Surfaces UNVERBALIZED intermediate reasoning: recognizing a face, spotting a bug in code, flagging a prompt
  injection, detecting "this is an evaluation" (ablating that detection surfaced concealed behaviors in tests).
- **Limitations:** only concepts that map to a single token; approximate; the mechanism that selects J-space is unknown.

## Runs on OPEN-WEIGHT models (Qwen, Llama, etc.) — PyTorch + HF transformers
```python
import transformers, jlens
hf  = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507").cuda()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
model = jlens.from_hf(hf, tok)
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")   # ~100 prompts is enough; saturates fast
lens.save("out/jacobian_lens.pt")
lens_logits, model_logits, _ = lens.apply(model, "The country shaped like a boot uses the currency", positions=[-2])
for layer, logits in sorted(lens_logits.items()):
    print(layer, [tok.decode([t]) for t in logits[0].topk(5).indices])   # concepts the model is "about to" verbalize
```
Fitting a lens: ~1000 seqs × 128 tokens (paper) or ~100 prompts (usable). Reference impl, **not maintained**.

## Why it matters here
The J-space is empirically the same object our [[world-transformer-layer]] tries to build/feed: a small privileged
workspace of the concepts currently in play. Concrete uses for the squad: (1) INSPECT what qwen/gemma internally hold
mid-reasoning (debug, audit for prompt-injection / eval-awareness); (2) a target/signal for training the world encoder
(align its world tokens to the model's own J-space); (3) safety auditing of a trained adapter's unverbalized reasoning.

## read further
- Anthropic research post: https://www.anthropic.com/research/global-workspace
- Paper (Transformer Circuits): https://transformer-circuits.pub/2026/workspace/index.html
- Open-source code (Apache-2.0, PyTorch/HF, works on Qwen): https://github.com/anthropics/jacobian-lens
- Coverage: https://www.technologyreview.com/2026/07/09/1140293/anthropic-found-a-hidden-space-where-claude-puzzles-over-concepts/
