# Extensions (measured)

Two capabilities layered on the core engine, each proof-gated and benchmarked against a control on the same local
Qwen3.6-35B-A3B used for the headline results. Both are opt-in; default behaviour is unchanged, so every existing
proof still reproduces.

---

## 1. Disagreement-gated consensus — Sheaf-ADMM harvest

**Module:** [`bobby_squad/sheaf_consensus.py`](../bobby_squad/sheaf_consensus.py) ·
**Proof:** [`wiki/proofs/test_sheaf_consensus.py`](https://github.com/Moun1r1/Bobby-Self-Organizing-Agent-Squad/blob/main/wiki/proofs/test_sheaf_consensus.py) (13 deterministic checks)

### Problem
`squad_solve`'s default harvest is a set union ([`squad.py`](../bobby_squad/squad.py) `_default_harvest`): every item
any agent proposes is accepted. When several agents cover the *same* content and one is noisy — a weaker model in a
mixed squad, a high-temperature sample, an injected/adversarial input — union keeps that agent's hallucination (high
recall, poor precision).

### Method
A discrete port of the ADMM coordination core in *Learning Multi-Agent Coordination via Sheaf-ADMM*
(Seely, Cupiał, Jones — ICML 2026, arXiv:2605.31005; [code](https://github.com/SakanaAI/sheaf-admm)). Agents are
nodes; the shared "edge space" is the **semantic embedding space** of proposed items (the sheaf restriction map =
identity in embedding space, agreement = cosine ≥ `merge_tau`). Scaled-dual ADMM (`x`-update / sheaf `z`-update /
dual ascent) drives per-candidate consensus toward the support-weighted agreement; a fact is kept when
`z_c ≥ majority`.

**Conditional by construction (safe default).** Consensus only helps when agents *redundantly* cover the same
content. When they instead *partition* the work (disjoint coverage, as in `confirm_coordination` where each agent maps
a different file), majority-consensus would wrongly prune everything — so the harvest first measures agent overlap and
falls back to plain union below `min_overlap`. `make_consensus_harvest(embed=…)` is a drop-in for
`squad_solve(harvest=…)`.

### Result (3-agent fact extraction, F1 vs union, ~0 extra LLM calls)

| injected agent noise (fabrications) | union F1 | consensus F1 | Δ |
|---|---|---|---|
| none | 0.99 | 1.00 | +1 % |
| light | 0.85 | 1.00 | +18 % |
| medium | 0.80 | 0.98 | +23 % |
| heavy | 0.68 | 0.97 | **+42 %** |

Recall stays 1.0 throughout (consensus never drops corroborated facts); precision is what union loses. Parity on
clean/disjoint work — it never hurts.

### Usage
```python
from bobby_squad import squad_solve, make_consensus_harvest
from bobby_squad.retrieval import default_embed
squad_solve(agents, units, work, harvest=make_consensus_harvest(embed=default_embed))
```

---

## 2. SOMA continuous-distillation flywheel

**Module:** [`bobby_squad/soma_flywheel.py`](../bobby_squad/soma_flywheel.py) ·
**Proof:** [`wiki/proofs/test_soma_flywheel.py`](https://github.com/Moun1r1/Bobby-Self-Organizing-Agent-Squad/blob/main/wiki/proofs/test_soma_flywheel.py) (20 deterministic checks)

Closes the loop set out in README §6.5: persist skills across runs, and distill → finetune the base model so more of
the workload becomes reducible over time.

### 2.1 Cheap turn — cross-run skill persistence
`PluginStore` snapshots each proof-gated frozen plugin (its handler recipe — regex / math-op / transform /
LLM-authored code — plus the `OODGate` competence region) and rehydrates them into the next run via
`burn_in.run(preload=store)`. Run k+1 starts warm and skips re-distillation.

Measured (single-sector burn-in, seed 1):

| run 2 | total tokens | distill | f_local | acc |
|---|---|---|---|---|
| cold (re-distils) | 6583 | 1655 | 72 % | 80 % |
| **warm (preloaded)** | **3212** | **0** | **80 %** | 80 % |

**−51 % tokens at equal accuracy**, and it compounds every run. The cold path is byte-identical to the original
(regression-checked), so headline results are unaffected.

### 2.2 Compounding turn — distill → finetune
`DistillationCorpus` emits verified `(prompt → output)` SFT records (kept only if plugin-served or graded correct, so
every label is trustworthy) to a chat-format `.jsonl`; a DPO emitter feeds off the existing `trajectory_dpo` /
`harvest_dpo` pairs. A LoRA fine-tune of a `qwen3-4b` base on that corpus, evaluated with a paired bootstrap CI:

```
BASE 71.8 %  [67.2 %, 76.0 %]  ->  FINETUNED 88.2 %  [85.0 %, 91.2 %]
Δ +16.5 %  95 % CI [+12.2 %, +21.0 %]   (CI excludes 0 => accept; McNemar +79 / -13)
per-family: image 0->100 · math 47->71 · algo 64->75 · extract 85->91 · code 95->100
```

### 2.3 Orchestration (one box, no second machine)
On unified memory you cannot serve and train at once, so the flywheel's compounding turn needs an arbiter that frees
the GPU before training and returns it after. The orchestrator runs the full cycle autonomously:

```
readiness -> OFFLOAD (stop the serving model) -> TRAIN (LoRA) -> EVAL+STATS (paired bootstrap CI)
          -> GATE (accept iff CI-separated gain) -> MERGE -> SWAP+SERVE -> CONFIRM (live endpoint)
```

It records a cursor so it only retrains when the corpus has grown, and on a rejected gate it restarts the previous
model instead of swapping. The gain-gate mirrors `proving.confirm_gain` (a claim is only accepted when the CI
excludes the null).

### Notes
- The finetuned model may prefix a template artifact (empty `<think>` block) from the base model's chat template; the
  graders strip it symmetrically, and cleaning the SFT targets removes it.
- The 35B is fine-tuned model-agnostically by the same loop given an offline training slot; the 4B here is a fast
  stand-in for the measurement.
