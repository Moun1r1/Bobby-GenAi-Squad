#!/usr/bin/env python3
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad import primitive_intel as P

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def embed(texts):
    return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:32]] for t in texts]


def examples(fid, n, seed):
    build = next(f for f in B.FAMILIES if f[0] == fid)[3]
    return [build(B._rng(seed * 100 + i)) for i in range(n)]


# the domain-specific PARAM for each sector (the only per-domain bit; the code is shared)
PATS = {"fin": r"TXN-\d{6}", "health": r"[A-Z]\d{2}\.\d", "security": r"CVE-20\d{2}-\d{4}",
        "legal": r"§\d{3}", "telecom": r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", "aviation": r"[A-Z]{2}\d{3,4}"}
domains = {name: {"param": pat, "examples": examples(name, 6, i + 1)} for i, (name, pat) in enumerate(PATS.items())}

# ── 1) cross-domain proof: ONE extract_matching skeleton generalizes across all 6 sectors ───────────────
base = P.PrimitiveBase()
em = P.extract_matching()
proof = P.dual_distill(base, em, domains, threshold=0.9, min_domains=2)
chk("extract_matching clears the bar on every sector it is bound to", all(s >= 0.9 for s in proof["scores"].values()))
chk("extract_matching GENERALIZES cross-domain (>=2 domains) → promoted to the primitive base",
    proof["generalizes"] and base.has("extract_matching"))
chk("one primitive covers all 6 sectors (composability: 1 code artifact + 6 params, not 6 plugins)",
    base.coverage("extract_matching") == 6)

# the same base serves any sector by binding its param — the low-variance, shared-code win
served = base.serve("extract_matching", PATS["security"], {"blob": examples("security", 1, 99)[0][0]})
chk("primitive base serves a fresh security ticket via (primitive, param)", "CVE-20" in served)

# ── 2) a second primitive: reduce_integers across the math families ─────────────────────────────────────
math_domains = {"sum": {"param": "sum", "examples": examples("math_sum", 6, 7)},
                "max": {"param": "max", "examples": examples("math_max", 6, 8)}}
ri = P.reduce_integers()
proof_ri = P.dual_distill(base, ri, math_domains, threshold=0.9, min_domains=2)
chk("reduce_integers generalizes across sum+max as one code skeleton", proof_ri["generalizes"])

# ── 3) find_analogous_case — a cognitive primitive (analogy/retrieval) as deterministic code ────────────
store = [(name, ask) for name, _, ask, _ in B.FAMILIES if name in PATS]
fa = P.find_analogous_case(embed)
h = fa.bind(store)
target_ask = next(ask for n, _, ask, _ in B.FAMILIES if n == "legal")
chk("find_analogous_case retrieves the analogous domain by similarity (code, not prompt)",
    h({"blob": target_ask}) == "legal")

# ── 4) HONEST negative: irreducible cognitive steps are NOT distilled — they stay on the LLM ─────────────
chk("self_critique / break_down_goal are flagged IRREDUCIBLE (kept generative, not faked into code)",
    "self_critique" in P.IRREDUCIBLE and "break_down_goal" in P.IRREDUCIBLE and not base.has("self_critique"))

# ── 5) a primitive that does NOT generalize must be REJECTED (the gate has teeth) ───────────────────────
# bind extract_matching but give one domain a wrong param → it fails there; with min_domains too high it won't promote
bad_domains = {"fin": {"param": r"TXN-\d{6}", "examples": examples("fin", 6, 1)},
               "mismatch": {"param": r"NOPE-\d+", "examples": examples("health", 6, 2)}}  # wrong pattern for health
base2 = P.PrimitiveBase()
proof_bad = P.cross_domain_proof(em, bad_domains, threshold=0.9, min_domains=2)
chk("cross-domain gate rejects a binding that only works on 1 domain (min_domains=2 not met)",
    not proof_bad["generalizes"] and P.PrimitiveBase().promote(em, proof_bad) is False)

print("\nregistry:", P.__name__)
for n, meta in base.registry().items():
    print("  %-20s %s  passes=%s" % (n, meta["signature"], meta["passed_domains"]))
print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: domain-free primitives generalize cross-domain AS CODE; irreducible ones stay on the LLM.")
