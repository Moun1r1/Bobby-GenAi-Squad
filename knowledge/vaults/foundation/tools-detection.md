---
title: tools-detection
tags: behavior, tools, dpo, study
source: seed:memory
links: [[loops-system]], [[thinking-behavior]], [[gemma-foundation-native]]
---

# Tools detection — pick the RIGHT tool, from the RELEVANT set

Tool selection improves when the agent is offered the RELEVANT subset of tool schemas, not all of them: confusable
tools (grep vs find, head vs tail, edit vs patch) cause slips when dumped together. Detect intent → narrow the menu
→ pick. Measured here (gains/proposals_llm_gain.py, tool-gating): relevant-subset beats all-tools.

## Native building blocks (this framework)
- `TOOL_SCHEMAS` / `DGX_SCHEMAS` (agent_tools.py) — the callable tool contracts.
- `ReadOnlyTools` / `SandboxTools` / `DgxTools` — the tool surfaces an agent is handed; give an agent only the
  surface its world-context needs.
- Tool-call dedup: hash(toolName+args), refuse collisions (a repeated identical call is a loop smell — see
  [[loops-system]]).

## code — gate to the relevant tools before offering them
```python
def offer(task, all_tools):
    intent = classify(task)                          # read | search | write | run | train …
    relevant = {n: s for n, s in all_tools.items() if s.intent == intent}
    return relevant or all_tools                     # never offer the confusers alongside the answer
name = model.pick_one(offer(task, TOOL_SCHEMAS))     # then select from the narrowed menu
```

## read further
- Framework tool surfaces: [[repos/agent-tools]]
- Toolformer (learned tool use): arXiv:2302.04761 · Gorilla (API selection): arXiv:2305.15334
- Anthropic tool-use guide: https://docs.anthropic.com/en/docs/build-with-claude/tool-use

## dpo
- prompt: The task is "show me only the last 20 lines of server.log" and every file tool is available. Which tool and why?
- chosen: Pick `tail` — it shows the LAST lines; first narrow to the relevant file-reading subset so head/cat/read don't cause a slip.
- rejected: Pick `head` (it shows the FIRST lines) or dump the whole file with `cat`, guessing from the full undifferentiated tool list.
- prompt: An agent is unsure which of 16 tools fits. What should it do first?
- chosen: Detect the intent, narrow to the 2–3 relevant, non-confusable tools, then choose.
- rejected: Offer the model all 16 tools at once including the confusable ones and hope it picks right.
