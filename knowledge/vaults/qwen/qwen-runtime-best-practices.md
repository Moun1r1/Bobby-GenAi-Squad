---
title: qwen-runtime-best-practices
tags: qwen, memory, context, tools, function-calling, agentic, serving, sampling
source: seed:qwen-docs-2026-07-13
links: [[qwen/qwen-optimization]], [[foundation/tokenization]], [[foundation/thinking-behavior]], [[foundation/jacobian-lens]]
---

# Qwen3 runtime best practices — memory · tools · agentic · serving · sampling

All values from official Qwen docs / HF cards (cited). Facts, not preferences.

## Memory / context (YaRN)
Native context **32,768** tokens; **131,072 via YaRN**. Exact `config.json`:
```json
"max_position_embeddings": 131072,
"rope_scaling": { "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768 }
```
- `factor` = target ÷ native (65,536 → `factor: 2.0`; 131,072 → `4.0`). Scale to actual need.
- transformers implements **static YaRN** → scaling is constant regardless of input length → **degrades short-context
  (<32K) quality**. Official guidance: **enable YaRN only when you actually need long context.**
- Bug: transformers ≤4.52.3 mis-derives the factor as `max_position_embeddings/original_max_position_embeddings`
  regardless of the stated `factor` — check version.
- Sources: https://huggingface.co/Qwen/Qwen3-8B · https://qwen.readthedocs.io/en/latest/inference/transformers.html

## Tools / function-calling
- Qwen3 uses **Hermes-style** tool calls; the chat template already supports it. Call format:
  `<tool_call>{"name":..,"arguments":{..}}</tool_call>`; tool results returned in a `role:"tool"` message.
- **Qwen3-Coder differs** — nested XML: `<tool_call><function=name><parameter=key>val</parameter></function></tool_call>`.
- **Qwen-Agent** holds the canonical parser + exposes function-calling over an OpenAI-compatible API. `generate_cfg`:
  `'fncall_prompt_type':'nous'` (recommended for Qwen3), `'use_raw_api':True` for the native tool interface. MCP,
  Code-Interpreter, RAG supported natively.
- **Parser rule:** if Qwen-Agent parses, do NOT also set vLLM `--enable-auto-tool-choice --tool-call-parser hermes`
  (double-parse). For server-side parsing use `--tool-call-parser hermes` (Qwen3) / `qwen3_coder` (Coder).
- Sources: https://qwen.readthedocs.io/en/latest/framework/function_call.html · https://github.com/QwenLM/Qwen-Agent

## Agentic / thinking
- Thinking is **on by default**. Hard switch: `enable_thinking=false` via `chat_template_kwargs`. Soft, per-turn:
  append **`/think`** or **`/no_think`** in a user/system message.
- Use **thinking** for math/logic/multi-step tool loops; **non-thinking** for latency/simple chat.
- See [[foundation/thinking-behavior]] (empty-content-on-serving bug when thinking left on).
- Source: https://qwenlm.github.io/blog/qwen3/

## Quantization / serving
- Official formats: **FP8, GPTQ-Int4/Int8, AWQ, GGUF**.
- **Do NOT use greedy decoding** (repetition loops). Model-card sampling:
  - **Thinking / Thinking-2507:** `temperature 0.6, top_p 0.95, top_k 20, min_p 0`
  - **Non-thinking / Instruct-2507:** `temperature 0.7, top_p 0.8, top_k 20, min_p 0`
  - `presence_penalty 0–2` curbs repetition (too high → language mixing).
- vLLM: `vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3`; disable
  thinking `--default-chat-template-kwargs '{"enable_thinking":false}'` or per-request extra_body. Caveat:
  `enable_thinking=false` incompatible with reasoning parser ≤vLLM 0.8.5 (fixed 0.9.0).
- SGLang: `python -m sglang.launch_server --model-path <model> --reasoning-parser qwen3`; disable thinking via
  `"chat_template_kwargs":{"enable_thinking":false}`. (This is how our leader is served.)
- Sources: https://qwen.readthedocs.io/en/latest/deployment/vllm.html · https://qwen.readthedocs.io/en/latest/deployment/sglang.html · https://huggingface.co/Qwen/Qwen3-8B

## Prompting
- Official card gives **no rigid system-prompt template**; sampling params above are the primary lever. The model
  obeys `/think` `/no_think` in system or user messages. Read the EXACT variant's card (some cards historically had
  conflicting "recommended" vs "best-practices" sampling values).

## World / context state — what's official vs framework
- **Official Qwen**: no named long-horizon state-management pattern beyond **YaRN** (long context) + **Qwen-Agent**
  Memory/RAG modules. (Do NOT attribute a specific "world-state" pattern to Qwen — not in the docs.)
- **This framework's approach** for world/context state on qwen: feed it via the [[foundation/world-transformer-layer]]
  (world-encoder prefix, no chat serialization), and INSPECT the model's own concept-workspace with the
  [[foundation/jacobian-lens]] (J-space). These are OUR methods, not Qwen doc guidance — labeled as such.

## Not verified (do not quote as fact)
- Exact numeric **thinking-budget** cap: Qwen-Agent manages it but no official numeric default confirmed.
- Per-token **KV-cache byte** figure: standard transformer math; no official Qwen number.
