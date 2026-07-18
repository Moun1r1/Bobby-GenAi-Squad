---
title: thinking-behavior
tags: behavior, thinking, dpo, study
source: seed:memory
links: [[tokenization]], [[loops-system]], [[tools-detection]]
---

# Thinking behavior — when a reasoning span helps, and when it silently breaks you

Think-gated chat templates (qwen, some gemma configs) wrap an internal reasoning span. Two failure modes:
1. Leaving thinking ON when you only read `content` → the model returns reasoning-only and `content` is EMPTY
   (measured here: criteria came back 0 / evidence blank until we disabled it). Fix:
   `GA_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}`.
2. Turning thinking OFF for a genuinely hard multi-step problem → shallow answers. Use it for math/logic/planning.

Rule: disable thinking for structured/extraction/tool-arg outputs where you consume `content`; enable it (or a
Chain-of-Verification pass) for hard reasoning. Match the switch to the task, don't set-and-forget.

## code — set the switch to the task
```python
extraction = {"chat_template_kwargs": {"enable_thinking": False}}   # need clean, non-empty content
hard_reason = {"chat_template_kwargs": {"enable_thinking": True}}    # math/logic/planning
body = hard_reason if task.is_reasoning else extraction
resp = llm(messages, extra_body=body)
assert resp.strip(), "empty content — thinking was left on while reading content"
```
Chain-of-Verification for correctness-critical tasks: generate → ask verification questions → answer them
independently → revise.

## read further
- Chain-of-Thought: arXiv:2201.11903 · Self-Consistency: arXiv:2203.11171 · Chain-of-Verification: arXiv:2309.11495
- qwen thinking flag behavior: Qwen chat-template docs (enable_thinking).

## dpo
- prompt: You call a think-gated model to extract JSON fields and the response `content` comes back empty. What is the fix?
- chosen: The thinking span consumed the turn; pass enable_thinking:false so the model emits the answer into content.
- rejected: Assume the model failed or the endpoint is down and retry the same call repeatedly with thinking still on.
- prompt: Should you disable thinking for every call to be safe?
- chosen: No — disable it for extraction/tool-arg/structured output you consume directly, but keep it (or add Chain-of-Verification) for hard multi-step reasoning.
- rejected: Yes, always disable thinking everywhere, even on hard math/logic where the reasoning span is what makes the answer correct.
