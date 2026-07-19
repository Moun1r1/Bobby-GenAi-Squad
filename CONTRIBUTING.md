# Contributing to Bobby

Thanks for your interest. Bobby is a research runtime for cost-decaying LLM agents (proof-gated
distillation to deterministic plugins). The bar for a change is simple and non-negotiable:

> **Every claimed gain is proven against a control, and every mechanism has a deterministic check.**

## Ground rules

- **No unproven claims.** A performance/quality claim must be backed by a benchmark against a control
  (see `wiki/proofs/`), and a gain is only reported when confidence intervals separate. Mirror the
  discipline in `bobby_squad/proving.py` (`confirm_gain`).
- **Deterministic checks for mechanisms.** New mechanisms ship with a no-network deterministic proof in
  `wiki/proofs/test_*.py` (see `test_sheaf_consensus.py`, `test_soma_flywheel.py` for the style).
- **Don't regress the defaults.** New behavior is opt-in; the default path must stay byte-identical so the
  existing reproductions still hold. Cold/​default runs are the regression baseline.
- **Keep subsystems decoupled.** Bobby is event-sourced — each subsystem is a fold over the log. Don't fuse
  independent components into one monolith; it kills ablatability.

## Dev setup

```bash
python -m pip install -e .
export BOBBY_LLM_URL=http://localhost:8000/v1/chat/completions BOBBY_LLM_MODEL=your-model
export BOBBY_EMBED_URL=http://localhost:11434/api/embed
```

Studio (optional UI): see `studio/` (`docker compose up` from `studio/`).

## Running the checks

```bash
# fast, no-network deterministic proofs (must pass before you open a PR)
python wiki/proofs/test_sheaf_consensus.py
python wiki/proofs/test_soma_flywheel.py
for t in test_burn_in test_all_layers test_primitives test_primitive_lib test_ops_world; do
  python wiki/proofs/$t.py
done
```

Live benchmarks (require a served model + embeddings) are under `wiki/proofs/run_*.py`.

## Pull requests

1. Branch from `main` (`fix/…`, `feat/…`, `docs/…`).
2. Keep commits scoped and conventional (`fix(deps): …`, `feat(router): …`, `docs: …`).
3. One concern per commit where practical (e.g. keep security fixes separate from features).
4. Fill in the PR template: what changed, how it was measured, which checks you ran.
5. CI (`.github/workflows/ci.yml`) must be green; address CodeQL / Dependabot findings.

## Code style

- Python: match the surrounding code — dense, comment-the-why, no gratuitous abstraction. `ruff` clean.
- Keep `torch` and `numpy` optional (guarded imports) so `import bobby_squad` works without them.
- No secrets, ever. No exception text or stack traces returned to clients.

## Reporting bugs / requesting features

Use the issue templates (Bug report / Feature request). For anything security-related, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing you agree your contributions are licensed under the repository's [MIT License](LICENSE).
