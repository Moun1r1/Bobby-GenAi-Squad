---
title: memory-selection
tags: behavior, memory, context, dpo, study
source: seed:memory
links: [[loops-system]], [[persona-reuse]], [[long-horizon-improvement]], [[tokenization]]
---

# Memory selection — what enters context, what survives compaction, what to recall

The proven mechanism (persistent-self): split everything into two tiers and never let compaction touch tier A.
- TIER A — SELF-CORE + PROGRESS: pinned, always injected, immune to compaction (the goal is NEVER lost).
- TIER B — WORKING MEMORY: a scrolling window, wiped on compaction.
Plus a Memory-Gate on compaction (move DISTINCT valuable results into the pinned tier before wiping; `near_dup`
gates out noise) and value-based retention (`SemanticMemory(policy="value")`: retrieval raises value, overflow
evicts the lowest — proven +retention). Recall is SELECTIVE: pull the few relevant items (or vault subgraph), not
everything.

## Native building blocks (this framework)
- `PersistentContext(pinned=True)` — the two tiers; `system_prompt()` injects self-core + pinned progress.
- `compact(consolidate=True)` — Memory-Gate. `SemanticMemory(policy="value")` — value-ranked keep/evict.
- `Agent(recall=…)` — native prefetch: navigate the [[gemma-foundation-native]] knowledge vault for THIS step
  (bounded, attributed) instead of dumping all memory.

## code — pinned goal survives; recall is bounded + selective
```python
ctx = PersistentContext(self_core, pinned=True)      # goal + progress pinned (survive compaction)
ctx.record(verified_finding)                         # → pinned tier, not a compactable turn
ctx.compact(consolidate=True)                        # Memory-Gate: keep distinct value, wipe the rest
# recall injects only the few relevant notes for the step, not the whole store:
ref = vault.navigate(current_target, k=3, hops=1, budget=1800)
```

## read further
- Persistent-self source: [[repos/core]] (PersistentContext / compact) · MemGPT (tiered memory): arXiv:2310.08560
- Value retention proof: gains/retention_gains.py · memory_policy_gain.py

## dpo
- prompt: Context is filling up on a long task. What must NOT be dropped when you compact?
- chosen: Keep the pinned self-core + goal + distinct verified progress (Memory-Gate moves valuable results to the pinned tier); wipe only the working window.
- rejected: Drop the goal and earlier findings along with everything else in one flat window, so the agent forgets what it was doing.
- prompt: You need background to act on the current step. How much memory do you inject?
- chosen: Recall only the few most relevant items / the bounded vault subgraph for this step, keeping context small and attributable.
- rejected: Inject the entire knowledge store into the prompt every step to be sure nothing relevant is missing.
