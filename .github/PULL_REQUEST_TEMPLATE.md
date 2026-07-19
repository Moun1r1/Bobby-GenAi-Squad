<!-- Thanks for contributing to Bobby. Keep PRs scoped; one concern per commit where practical. -->

## What & why
<!-- What does this change and why. Link issues (e.g. Closes #123). -->

## Type
- [ ] fix (bug)
- [ ] feat (new capability)
- [ ] docs
- [ ] chore / ci / deps
- [ ] security

## How it was measured
<!-- REQUIRED for any performance/quality claim: benchmark vs a control, with CI separation.
     Paste the numbers (e.g. base vs treatment, Δ + 95% CI). Say "N/A" for pure docs/chore. -->

## Checks run
- [ ] `python wiki/proofs/test_sheaf_consensus.py`
- [ ] `python wiki/proofs/test_soma_flywheel.py`
- [ ] relevant `wiki/proofs/test_*.py`
- [ ] default/cold path unchanged (no regression to reproduced results)

## Checklist
- [ ] New mechanism ships a deterministic (no-network) proof
- [ ] New behavior is opt-in; defaults are byte-identical
- [ ] No secrets, no exception text / stack traces returned to clients
- [ ] Docs updated if behavior/interface changed
