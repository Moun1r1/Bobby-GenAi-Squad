# Benchmark — 100-Ticket Burn-In

Measures whether the ACR loop lowers per-task LLM cost as it distills capabilities, without losing accuracy.

## Method

- **Dataset.** A seeded generator emits a ticket stream (`bobby_squad/burn_in.py:generate` / `generate_mixed`). No LLM
  in construction. Grading is exact set-equality (no LLM judge), so results are not arguable.
- **Task types.** Single-sector = pattern extraction. Cross-modal = 6 modalities under one capability: extraction
  (6 sectors), math (sum/max), code (3 transforms), image (ASCII-grid count), algo (Roman-numeral parser, Luhn — an
  LLM-authored function), prose (sentiment — irreducible, must stay on the LLM).
- **Control.** Every run is compared to a **No-ACR control**: the same engine with distillation disabled (LLM every
  ticket). The reduction is measured against the control, not asserted.
- **Gate.** A candidate rule is frozen only if it clears a held-out gain-proof (mean F1 ≥ 0.9). Out-of-competence
  tickets must never be served by a frozen plugin — verified via the OOD tripwire.
- **Cost accounting.** Per-task token cost = serving tokens (0 when a frozen plugin answers). Distillation (writing +
  proving a plugin) is a separate one-time cost, reported separately, included in the total.
- **Replication.** N = 5 seeds, mean ± 95 % CI (Student-t).

Model: Qwen3.6-35B-A3B via sglang (thinking off — extraction is trivial). Embeddings: nomic-embed-text (768-d).

## Results

| Run | Tokens (ACR) | Tokens (No-ACR) | Δ | Accuracy ACR / control | Router-local |
|---|---|---|---|---|---|
| Single-sector, 100 tickets | 6,603 (4,948 serve + 1,655 distill) | 21,202 | −69 % | 80 % / 74 % | 72 % |
| Cross-modal, 180 tickets | 14,613 (10,234 serve + 4,379 distill) | 22,288 | −34 % | 96 % / 93 % | 58 % |

N = 5 (single-sector), mean ± 95 % Student-t CI ($n{=}5$): router-local **64.8 % ± 20.0 %**, token reduction
**62.0 % ± 20.0 %**, accuracy 79.0 % ± 2.2 %.

Per-modality (cross-modal, % frozen / accuracy): extract 44 / 99 · math 67 / 88 · code 67 / 100 · image 67 / 100 ·
algo 67 / 92 · **prose 0 / 100** (irreducible — never distilled, never misrouted).

![single-sector golden signals](screenshots/burn-in-golden-signals.png)

## Reading the numbers

- Local fraction is bounded by `1 − irreducible_fraction − warmup_share`. With a permanent generative floor (prose),
  the achievable ceiling is below 100 %; 58 % is the measured cross-modal value, not a target.
- ACR accuracy exceeds the control on reducible modalities because a frozen deterministic plugin does not make the
  format/arithmetic slips the model occasionally makes.
- The wide N = 5 CI comes from one seed where the model proposed a below-gate regex for one cluster, so that cluster
  correctly stayed on the LLM (1 promotion instead of 2). Distillation reliability is proposal-limited.

## Reproduce

```bash
export BOBBY_LLM_URL=... BOBBY_LLM_MODEL=... BOBBY_EMBED_URL=...
python wiki/proofs/run_burn_in_live.py                 # single-sector (writes signals + report + png)
python wiki/proofs/run_burn_in_mixed_live.py           # cross-modal
python wiki/proofs/run_burn_in_sweep.py                # N=5 CI (BURN_MIXED=1 for the mixed sweep)
python wiki/proofs/test_burn_in.py                     # deterministic mechanics (34 checks, no network)
```

Artifacts land in `wiki/proofs/out/` (per-ticket CSV/JSON, report, golden-signal PNG).
