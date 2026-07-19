# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-19

First tagged release of the Bobby runtime: cost-decaying LLM agents via proof-gated distillation to
deterministic plugins, with a self-organizing multi-agent (SO-MAS) layer.

### Core
- Event-sourced plugin engine (append-only log; every subsystem a projection).
- ACR distillation operator — proof-gated freezing of reducible capability classes into deterministic
  plugins (**−69 % tokens at 80 % vs 74 % accuracy**, single-sector 100-ticket benchmark vs a
  no-distillation control).
- Competence router + Mahalanobis OOD gate (fail-safe abstain to the model).
- Cross-domain primitives, persistent knowledge vault, SO-MAS recursive squad.

### Added — extensions (measured; see [docs/EXTENSIONS.md](docs/EXTENSIONS.md))
- **Disagreement-gated consensus (Sheaf-ADMM harvest)** — `bobby_squad/sheaf_consensus.py`, a conditional
  drop-in squad harvest that filters squad hallucinations under agent disagreement (+18 %..+42 % F1) and is
  parity on clean/disjoint work. 13 deterministic checks.
- **SOMA continuous-distillation flywheel** — `bobby_squad/soma_flywheel.py`: cross-run skill persistence
  (`PluginStore`, **−51 % tokens** on a warm second run) and a verified SFT corpus emitter
  (`DistillationCorpus`). A LoRA fine-tune on the corpus lifts a `qwen3-4b` base **71.8 % → 88.2 %**
  (Δ +16.5 %, 95 % CI [+12.2 %, +21.0 %]). 20 deterministic checks.

### Security
- Fixed stack-trace / exception exposure in the Studio backend (`py/stack-trace-exposure` ×4): exceptions
  are logged server-side (full traceback via `exc_info`), only generic messages returned to clients.
- Bumped `next` → 15.5.18 and forced `postcss` → ≥ 8.5.10 in `studio/frontend` (closed 53 Dependabot alerts).

### Repo
- Added `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates, `CODEOWNERS`,
  Dependabot config, and a CI workflow running the deterministic proofs.

[Unreleased]: https://github.com/Moun1r1/Bobby-Self-Organizing-Agent-Squad/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Moun1r1/Bobby-Self-Organizing-Agent-Squad/releases/tag/v0.1.0
