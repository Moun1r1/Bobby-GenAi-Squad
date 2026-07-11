"""bobby_squad.proving — confirm_gain: prove a gain with ONE deterministic A/B, not a squad, not a vibe.

The canonical quality gate for "does this change actually help?": run control() vs treatment(), measure the
relative gain, emit a WIRE / MARGINAL / DELETE / DEFER verdict against a stated threshold. One instance, one number,
no recursion — bounded and observable. (Formerly gains/gain_test.py; promoted to a reusable package primitive.)
"""
import json
import statistics


def confirm_gain(name, control, treatment, *, higher_is_better=True, samples=1, wire=0.10, marginal=0.02):
    """control()/treatment() each return a numeric metric (run in test mode by the caller). Returns + prints a
    verdict dict; the verdict is DERIVED from the measured number, never asserted.

      WIRE     rel_gain >= wire       — real, worth integrating
      MARGINAL marginal <= rel < wire — small; hold
      DELETE   rel_gain < marginal    — no gain; drop it
      DEFER    a run errored / metric unavailable — can't measure yet
    """
    try:
        b = statistics.mean(float(control()) for _ in range(max(1, samples)))
        t = statistics.mean(float(treatment()) for _ in range(max(1, samples)))
    except Exception as e:
        out = {"name": name, "verdict": "DEFER", "reason": f"could not measure: {e}"}
        print("GAIN " + json.dumps(out))
        return out
    delta = (t - b) if higher_is_better else (b - t)
    rel = (delta / abs(b)) if b else delta
    verdict = "WIRE" if rel >= wire else ("MARGINAL" if rel >= marginal else "DELETE")
    out = {"name": name, "control": round(b, 4), "treatment": round(t, 4), "gain": round(delta, 4),
           "rel_gain": round(rel, 3), "samples": samples, "verdict": verdict}
    print("GAIN " + json.dumps(out))                 # machine-readable line (SandboxTools.run captures the verdict)
    return out


def _stat(fn, seeds):
    xs = [float(fn(s)) for s in seeds]
    m = statistics.mean(xs)
    ci = 1.96 * statistics.pstdev(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0.0   # 95% CI half-width
    return m, ci


def prove(name, control, treatment, *, negative=None, baseline_max=None, higher_is_better=True,
          seeds=range(5), wire=0.10, marginal=0.02):
    """Enforced gain-test METHODOLOGY — not just an A/B, but the guards that make a verdict TRUSTWORTHY (the same
    rigor squad_solve enforces for coverage). `control(seed)` / `treatment(seed)` return a metric per seed.

      • HEADROOM guard — if `baseline_max` is given and the baseline is already at it, there's NO room to improve, so
        a low gain means the TEST is uninformative → INCONCLUSIVE (not a false DELETE). This is exactly the trap that
        made 4 proposals look 'DELETE' when the baseline had ceilinged.
      • NEGATIVE-CONTROL guard — `negative=(neg_control, neg_treatment)`: a condition where the effect must NOT appear
        (e.g. non-predictive usage). If the treatment still 'wins' there, the benchmark LEAKS → INVALID.
      • REPLICATION — run over `seeds`, report mean ± 95% CI, never a single run.

    Verdicts: WIRE / MARGINAL / DELETE / INCONCLUSIVE(ceiling) / INVALID(leak) / DEFER.
    """
    seeds = list(seeds)
    try:
        mb, cib = _stat(control, seeds)
        mt, cit = _stat(treatment, seeds)
    except Exception as e:
        out = {"name": name, "verdict": "DEFER", "reason": f"could not measure: {e}"}
        print("PROVE " + json.dumps(out))
        return out
    delta = (mt - mb) if higher_is_better else (mb - mt)
    rel = (delta / abs(mb)) if mb else delta
    reason, nrel = "", None
    if baseline_max is not None and mb >= baseline_max * 0.98:
        verdict = "INCONCLUSIVE"
        reason = "baseline ceilinged (no headroom) — harden the task before any verdict"
    else:
        if negative is not None:
            nmb, _ = _stat(negative[0], seeds)
            nmt, _ = _stat(negative[1], seeds)
            nrel = (((nmt - nmb) if higher_is_better else (nmb - nmt)) / abs(nmb)) if nmb else 0.0
        if nrel is not None and nrel >= marginal:
            verdict, reason = "INVALID", f"negative control also improved ({nrel:+.2f}) — the test LEAKS"
        else:
            verdict = "WIRE" if rel >= wire else ("MARGINAL" if rel >= marginal else "DELETE")
    out = {"name": name, "control": round(mb, 4), "control_ci": round(cib, 4), "treatment": round(mt, 4),
           "treatment_ci": round(cit, 4), "rel_gain": round(rel, 3), "seeds": len(seeds), "verdict": verdict}
    if negative is not None:
        out["neg_control_rel"] = round(nrel, 3) if nrel is not None else None
    if reason:
        out["reason"] = reason
    print("PROVE " + json.dumps(out))
    return out
