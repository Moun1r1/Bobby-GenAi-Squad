---
title: long-horizon-improvement
tags: behavior, flywheel, self-dpo, dpo, study
source: seed:memory
links: [[memory-selection]], [[persona-reuse]], [[loops-system]], [[training-approaches]], [[behavior-patterns]]
---

# Long-horizon improvement — capture the teaching signal, distill it, self-DPO on bad→good

Getting better over a long horizon is a FLYWHEEL, not a one-pass call: keep the goal across the horizon (pinned
memory), CAPTURE the teaching signal (today it evaporates), and convert every BAD→GOOD correction into a DPO pair
so the model stops repeating the anti-pattern. The behavior notes in this vault each carry a `## dpo` block for
exactly this reason: every capability's KNOWN failure mode is a ready-made `rejected`, its correct form the
`chosen`. The self-DPO pipeline harvests them.

## The loop
1. Run the self-directed loop ([[loops-system]]) with pinned goal + selective recall ([[memory-selection]]).
2. Meta-cognition on each response: recognize the PATTERN → CRITIQUE (coherence/correctness/creativity/safety) →
   generate a better ALTERNATIVE → build the PAIR (chosen ≻ rejected). See [[behavior-patterns]].
3. Add the vault's curated good/bad pairs (every note's `## dpo` block) to the dataset.
4. DPO-train the foundation model ([[training-approaches]]); iterate — regenerate pairs from the improved model.
5. ENRICH the vault with what was learned so the NEXT horizon starts wiser (cross-run transfer, [[persona-reuse]]).

## code — harvest bad→good pairs + self-critique, then DPO
```python
pairs = vault.harvest_dpo()                          # every note's ## dpo block → {prompt, chosen, rejected}
for t in tasks:                                      # + freshly manufactured self-critique pairs
    resp = agent.execute(t)
    p = agent.self_dpo_pair(t, resp)
    if p["improved"]: pairs.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]})
dpo_train(model, pairs)                              # loss↓ below 0.69, margin↑ → the anti-patterns are unlearned
```

## read further
- Self-DPO flywheel proof: pipe_self_dpo (runner.py) — measured Gemma3-1B loss 0.69→0.49, margin 0→1.69.
- Self-Rewarding LMs: arXiv:2401.10020 · Constitutional AI (self-critique): arXiv:2212.08073 · Reflexion: arXiv:2303.11366

## dpo
- prompt: You corrected the same class of agent mistake three runs in a row. How do you make it stop recurring?
- chosen: Capture each bad→good correction as a preference pair and DPO-train on them (plus enrich the vault) so the model unlearns the anti-pattern permanently.
- rejected: Fix it by hand again each time and move on, letting the teaching signal evaporate so the mistake returns next run.
- prompt: How should an agent improve across a long multi-day horizon?
- chosen: Keep the goal pinned, accumulate + selectively recall, and run the capture→distill→self-DPO flywheel so each horizon starts wiser than the last.
- rejected: Make one big one-pass call per task with no memory or capture, forgetting everything and never improving.
