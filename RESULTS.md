# Results — every measured number and how to reproduce it

This file exists so the claims aren't taken on faith: each row is a measured
number paired with the exact command that produces it. Results come in two tiers.

- **Tier 1 — zero-infra.** Runs on any laptop with no model, no GPU, no network: a
  deterministic mock model stands in for the LLM, so you can watch the *mechanisms*
  (distill → freeze → route, the OOD fail-safe, coordination, the flywheel logic)
  reproduce byte-for-byte. This is the honest floor — it never trusts a model.
- **Tier 2 — real served model.** The headline cost/accuracy numbers on a live
  OpenAI-compatible endpoint (+ embeddings; + a GPU for the fine-tune). Same code,
  a real model swapped in for the mock.

Grading is exact set-equality (no LLM judge). A gain is only reported when
confidence intervals separate (`bobby_squad/proving.py`).

### Abbreviations
- **ACR** — the distillation operator: discover a rule offline, gain-prove it, freeze it as a zero-token plugin.
- **f_local** (router-local fraction) — share of tasks served by a frozen plugin, i.e. with **0** model tokens.
- **OOD** — out-of-distribution; the competence gate that abstains back to the model on tasks a plugin wasn't proven on.
- **CI** — 95 % confidence interval; **McNemar** — the discordant-pair count (fixed vs broken) for paired model eval.
- **SFT** — supervised fine-tuning; **DPO** — direct preference optimization; **LoRA** — low-rank adapter fine-tuning.

---

## Tier 1 — zero-infra, deterministic (no network, no GPU)

Setup: `pip install -e .` (the Sheaf-ADMM consensus check additionally needs `pip install numpy`). Every command below
was run from a fresh checkout and passes.

| What | Result | Reproduce |
|---|---|---|
| ACR token reduction (single-sector) | **−69 % serving tokens at 100 % accuracy parity**, 2 frozen plugins, router-local 72 %, OOD tripwire holds | `python examples/demo_no_infra.py` |
| Burn-in mechanics | 34 checks (dataset, exact grading, distill, OOD tripwire, report/SVG round-trip) | `python wiki/proofs/test_burn_in.py` |
| Sheaf-ADMM consensus | 13 checks (filters hallucinations under overlap; safe no-op on disjoint work) | `python wiki/proofs/test_sheaf_consensus.py` |
| SOMA flywheel | 20 checks (skill persistence round-trip byte-identical + gate intact; verified-only corpus) | `python wiki/proofs/test_soma_flywheel.py` |
| Cross-domain primitives | generalization + the gate rejecting a non-generalizer | `python wiki/proofs/test_primitives.py` |
| Full suite | 100+ deterministic checks | `for t in test_burn_in test_all_layers test_primitives test_primitive_lib test_ops_world test_sheaf_consensus test_soma_flywheel; do python wiki/proofs/$t.py; done` |

Tier-1 is the honest floor: it shows the distillation → freeze → route mechanism,
the OOD fail-safe, and the coordination/flywheel logic **without trusting any
model**. The mock model is a stand-in; swap in a real one (Tier 2) for real cost.

---

## Tier 2 — real served model (Qwen3.6-35B-A3B, local; GPU for the fine-tune)

Setup:

```bash
export BOBBY_LLM_URL=http://localhost:8000/v1/chat/completions BOBBY_LLM_MODEL=your-model
export BOBBY_EMBED_URL=http://localhost:11434/api/embed BOBBY_EMBED_MODEL=nomic-embed-text
```

| What | Result | Reproduce |
|---|---|---|
| Single-sector burn-in | **−69 % tokens** (6,603 vs 21,202), acc **80 % vs 74 %**, f_local 72 % | `python wiki/proofs/run_burn_in_live.py` |
| Cross-modal (6 modalities) | **−34 % tokens**, acc **96 % vs 93 %**, f_local 58 % | `python wiki/proofs/run_burn_in_mixed_live.py` |
| N=5 seed replication | token −62 % ± 20 %, f_local 64.8 % ± 20 % (95 % CI) | `python wiki/proofs/run_burn_in_sweep.py` |
| SOMA flywheel — persistence turn | **−51 % tokens on a warm run 2** at equal accuracy (distill 1655→0, f_local 72→80 %) | `examples/soma_flywheel/` (see its README) |
| SOMA flywheel — fine-tune turn | qwen3-4b **71.8 % → 88.2 %**, Δ **+16.5 %**, 95 % CI **[+12.2 %, +21.0 %]** (McNemar +79/−13); per-family: image 0→100 · math 47→71 · algo 64→75 · extract 85→91 · code 95→100 | `examples/soma_flywheel/` (LoRA; needs a GPU) |
| Sheaf-ADMM consensus under noise | union→consensus F1: none 0.99→1.00 · light 0.85→1.00 · medium 0.80→0.98 · heavy 0.68→0.97 (**+42 %**) | `examples/soma_flywheel/` bench (needs the endpoint) |

Notes: the fine-tune ran on one GB10 (unified memory) using a LoRA on a 4 B
stand-in — the loop is model-agnostic; scaling to the 35 B needs an offline
training slot. The consensus gain is conditional on agent disagreement (parity on
clean/disjoint work — see `docs/EXTENSIONS.md`).

---

## Raw artifacts

Tier-1 runs write per-ticket signals + a report + an SVG/PNG to `wiki/proofs/out/`
(gitignored — regenerate with the commands above; they are byte-deterministic for
a fixed seed). Tier-2 additionally writes the golden-signal CSV/JSON there.
