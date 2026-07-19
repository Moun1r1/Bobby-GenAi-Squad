# Roadmap

Direction, not dates. Ordered by priority. Each item ships with a proof-gated
benchmark (a gain is only claimed when confidence intervals separate).

## Now — usability & reproducibility
- [x] Zero-infra demo (`examples/demo_no_infra.py`) — the ACR moat on a mock model, no GPU/keys.
- [x] `RESULTS.md` — every measured number with its exact reproduce command (Tier 1 no-infra / Tier 2 real model).
- [ ] More worked examples: code-review agent, a research task, a SWE-bench-style loop.
- [ ] "Bobby in 5 minutes" expanded into a short tutorial; API reference for `Agent` + `squad_solve`.
- [ ] Publish to PyPI so install is `pip install bobby-genai-squad` (not source-only).

## Next — external benchmarks (place it against the field, not only its own control)
- [ ] Full SWE-bench via the official per-repo docker harness (real GitHub issues, multi-file patches).
- [ ] GAIA / long-horizon evals with public results and raw artifacts committed under `docs/results/`.
- [ ] External agent baselines (OpenHands, Aider, a ReAct loop) under one resource envelope.

## Coordination (Sheaf-ADMM consensus)
- [x] Disagreement-gated consensus harvest, conditional/safe-default (`bobby_squad/sheaf_consensus.py`).
- [ ] Per-lens verification (correctness / security / repro) instead of majority-only.
- [ ] Wire consensus into the live squad benchmarks under induced heterogeneity.

## Training flywheel (SOMA)
- [x] Cross-run skill persistence (`PluginStore`) — measured −51 % tokens on a warm run.
- [x] Verified SFT corpus emitter + LoRA fine-tune (measured +16.5 %, CI-separated, on a 4 B stand-in).
- [ ] `SelfDPO+` — preference pairs with KL regularization + safety anchors + human-in-loop fallback.
      **Caveat we hold ourselves to:** self-generated preference data often underperforms human data and can
      mode-collapse; this ships only if it clears the gate against a human-data / no-DPO control.
- [ ] Scale the fine-tune turn to the 35 B on an offline training slot; multi-seed CIs over training runs.
- [ ] Automatic primitive mining from behavior traces + router composition search (`extract → evaluate → decide`).

## Hardening
- [x] Optional heavy deps (torch, numpy) guarded so the core installs stdlib-only.
- [x] Studio backend: no exception/stack-trace leakage to clients.
- [ ] Real isolation for the `algo` sandbox (copy-on-write execution plane) — today it is restricted-builtins only.
- [ ] Performance budget: quantify the overhead of metacognition / board / vault I/O and gate regressions in CI.
- [ ] Semantic versioning + migration notes as the training/Studio surfaces evolve.

## Community
- [ ] `help wanted` issues per module; a discussions board.
- [ ] Interactive demo (Hugging Face Space, with a local Streamlit fallback).

Have an opinion on ordering? Open an issue — see [CONTRIBUTING.md](CONTRIBUTING.md).
