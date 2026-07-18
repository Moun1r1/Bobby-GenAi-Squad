---
title: tokenization
tags: tokenizer, sentencepiece, chat-template, context
source: seed:memory
links: [[gemma-foundation-native]], [[perf-memory]]
---

# Tokenization — the model's real input unit

## Basics
- Gemma uses a **SentencePiece** tokenizer (~256k vocab, byte-fallback) — the SAME tokenizer for train and serve.
  Load the served model's tokenizer; never re-derive special tokens.
- A token ≈ 3–4 chars of English; code / non-Latin cost MORE. Budget sequence length in tokens, not characters.

## Gemma chat template (control tokens matter)
- Turn format: `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n…<end_of_turn>`. `<bos>` at start.
  Roles are literally `user` and `model` (not `assistant`).
- Wrong template (missing `<end_of_turn>`, wrong role) silently degrades fine-tuning — the model learns wrong
  boundaries. Use `apply_chat_template`, don't hand-format.
- Think-gated models: if a chat model returns EMPTY content, it's usually waiting on a thinking block — pass
  `{"chat_template_kwargs":{"enable_thinking":false}}` (measured here on the qwen leader).

## Sequence budgeting, truncation vs windowing
- Set `max_length` / `num_ctx` explicitly; the KV cache grows linearly with it — over-budgeting OOMs
  ([[perf-memory]]).
- Long inputs: don't hard-truncate (you lose the tail). Split into **overlapping moving windows** (e.g. 4000 chars /
  400 overlap) — the pattern the memory layer here already uses so ALL content stays retrievable. Generalizes to any
  long modality; text is the implemented case.
- Packing: concatenate short examples up to `max_length` with separators to cut SFT padding waste.

## Rule
Load the served tokenizer, format with its chat template, budget `max_length` in tokens with KV cost in mind, and
window (don't truncate) anything longer than the window.

## code — native tokenizer + correct chat template
```python
import gemma.gm as gm
tok = gm.text.Gemma3Tokenizer()                               # SAME vocab as the served model
ids = tok.encode("<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n", add_bos=True)
# don't hand-format for training — let the template add <bos>/<end_of_turn> and the `model` role:
turns = [{"role": "user", "content": "Hi"}]
prompt = tok.apply_chat_template(turns, add_generation_prompt=True)
# overlength → moving windows, never truncate (keep the tail):
def windows(s, n=4000, ov=400): 
    return [s[i:i+n] for i in range(0, len(s), n-ov)]
```

## read further
- SentencePiece: https://github.com/google/sentencepiece
- Gemma tokenizer + chat template: https://ai.google.dev/gemma/docs/core/prompt-structure
- HF chat templates (the general mechanism): https://huggingface.co/docs/transformers/main/chat_templating
