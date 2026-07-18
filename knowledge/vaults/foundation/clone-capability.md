---
title: clone-capability
tags: behavior, clone, squad, dpo, study
source: seed:memory
links: [[loops-system]], [[persona-reuse]], [[memory-selection]]
---

# Clone capability — spawn sub-agents with isolated context + guaranteed cleanup

An agent scales by CLONING itself into a squad that shares a goal but works in parallel — each clone with its OWN
context window (so they don't poison each other) and its own workspace. The two hard rules: bound the fan-out, and
ALWAYS clean up (worktrees/containers) in a `finally`, not just on success.

## Native building blocks (this framework)
- `squad_solve(agents, split, harvest, …)` — clone N agents over `units`; `split` divides the work, `harvest`
  merges outputs. Each agent is an independent `Agent` with its own `PersistentContext` ([[memory-selection]]).
- Parallel git worktrees for multi-file work: every `git worktree add` pairs with `try/finally` cleanup +
  registration in `worktrees:active`; children write `ExitWorktree({action:'discard'})` in a finally block.
- DGX worker clones = isolated, memory-capped containers ([[perf-memory]]) — never share the base box's memory.

## code — bounded clones with guaranteed cleanup
```python
agents = [Agent(self_core, llm=llm, tools=mk_tools(i)) for i in range(min(n, MAX_FANOUT))]
try:
    results = squad_solve(agents, units, work, verify, split, harvest, max_passes=…)
finally:
    for a in agents: a.tools.dispose()               # cleanup ALWAYS runs — worktree/container discarded
```

## read further
- squad_solve source: [[repos/core]] · git strategy: `.claude/decisions/DEC-2026-05-30-git-strategy.md`
- Multi-agent patterns (map-reduce over agents): AutoGen https://github.com/microsoft/autogen

## dpo
- prompt: You spawn 5 worktrees for parallel edits and one child crashes mid-run. How is cleanup handled?
- chosen: Cleanup runs in a finally block regardless of success or crash, discarding every worktree/container so nothing leaks.
- rejected: Only remove the worktrees on the success path, so the crashed child's worktree leaks and pollutes worktrees:active.
- prompt: A task could use more parallelism. How many clones do you spawn?
- chosen: Spawn up to a bounded fan-out cap, each with an isolated context and workspace, sized to the work and the memory budget.
- rejected: Spawn an unbounded number of clones sharing one context and the base box's memory to go as fast as possible.
