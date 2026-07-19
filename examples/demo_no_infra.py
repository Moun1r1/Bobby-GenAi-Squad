#!/usr/bin/env python3
"""Bobby in 60 seconds — the ACR token reduction, with ZERO infrastructure.

No served model, no GPU, no embeddings endpoint, no network. A deterministic
mock model stands in for the LLM so you can watch the core result reproduce on
any laptop: when the same capability class recurs, Bobby distils it into a frozen
zero-token plugin, and cost bends down while accuracy holds — with an OOD
tripwire that keeps novel tasks on the model.

    python examples/demo_no_infra.py

For the real numbers on a real model (−69 % tokens, etc.) see RESULTS.md; those
need a served OpenAI-compatible endpoint. This demo proves the *mechanism*
offline and is the same machinery exercised by wiki/proofs/test_burn_in.py.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bobby_squad import burn_in as B


def mock_embed(texts):
    """Deterministic stand-in for a real embedder: same ask family → same vector;
    the OOD 'sum' family lands far from the two extraction families."""
    out = []
    for t in texts:
        if "error codes" in t:
            out.append([1.0, 0, 0, 0, 0, 0])
        elif "config keys" in t:
            out.append([0, 1.0, 0, 0, 0, 0])
        else:
            out.append([0, 0, 1.0, 1.0, 1.0, 1.0])
    return out


class MockLLM:
    """A competent-but-costly model stand-in: it solves each ticket exactly, and
    when the distiller asks for candidate regexes it proposes the true patterns.
    Every call 'costs' tokens — which is exactly what distillation removes."""

    def __init__(self):
        self.last_usage = {"total_tokens": 120}

    def __call__(self, messages, max_tokens=400, temperature=0.0):
        content = messages[-1]["content"]
        if "candidate Python `re` regex" in content:                 # distiller asks for rules
            return "ERR-\\d{3,5}\nCFG_[A-Z][A-Z0-9_]{2,}\nNOPE-\\d+"
        blob = content.split("DATA:\n", 1)[1] if "DATA:\n" in content else content
        if "error codes" in content:
            return "\n".join(dict.fromkeys(re.findall(r"ERR-\d{3,5}", blob)))
        if "config keys" in content:
            return "\n".join(dict.fromkeys(re.findall(r"CFG_[A-Z][A-Z0-9_]{2,}", blob)))
        return str(sum(int(x) for x in re.findall(r"\d+", blob)))     # the OOD 'sum' family


def main():
    tickets = B.generate(seed=1)                                      # 100 tickets: 40 A / 40 B / 20 C(OOD)
    print("Running 100 tickets twice on a mock model (no network)...\n")
    acr = B.run(tickets, MockLLM(), mock_embed, distill=True, warmup=4)   # ACR flywheel ON
    ctrl = B.run(tickets, MockLLM(), mock_embed, distill=False)           # no-distillation control

    print(B.render_report(acr, ctrl))

    acr_tok = sum(acr.signals.series("token_cost"))
    ctrl_tok = sum(ctrl.signals.series("token_cost"))
    saved = 100 * (1 - acr_tok / ctrl_tok) if ctrl_tok else 0
    c_all_llm = all(r["route"] == "llm" for r in acr.signals.rows if r["cluster"] == "C")
    print(f"\n  serving tokens : ACR {acr_tok}  vs  control {ctrl_tok}   →  −{saved:.0f}% ")
    print(f"  accuracy       : ACR {acr.accuracy:.0%}  vs  control {ctrl.accuracy:.0%}   (parity or better)")
    print(f"  frozen plugins : {acr.promotions}   router-local fraction {acr.signals.local_fraction():.0%}")
    print(f"  OOD tripwire   : every novel Cluster-C task stayed on the model = {c_all_llm}")
    print("\nThat is the token reduction, offline: the reducible classes migrate off the model; the irreducible ones don't.")


if __name__ == "__main__":
    main()
