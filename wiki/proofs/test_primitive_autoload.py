#!/usr/bin/env python3
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad import primitive_intel as P

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def examples(fid, n, seed):
    build = next(f for f in B.FAMILIES if f[0] == fid)[3]
    return [build(B._rng(seed * 100 + i)) for i in range(n)]


PATS = {"fin": r"TXN-\d{6}", "health": r"[A-Z]\d{2}\.\d", "security": r"CVE-20\d{2}-\d{4}",
        "legal": r"§\d{3}", "telecom": r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", "aviation": r"[A-Z]{2}\d{3,4}"}
extract_domains = {n: {"param": p, "examples": examples(n, 6, i + 1)} for i, (n, p) in enumerate(PATS.items())}
math_domains = {"sum": {"param": "sum", "examples": examples("math_sum", 6, 7)},
                "max": {"param": "max", "examples": examples("math_max", 6, 8)}}

ROOT = tempfile.mkdtemp()

# ── run "session 1": empty lib → prove + auto-persist extract_matching ──────────────────────────────────
base1 = P.PersistentPrimitiveBase(ROOT)
chk("fresh library starts empty", base1.names() == [])
proof = P.prove_and_persist(base1, "extract_matching", extract_domains)
chk("extract_matching passes the cross-domain gate", proof["generalizes"])
chk("...and is auto-added to the library (added=True)", proof["added"] is True)
chk("a real .py source file was written to the lib", os.path.exists(os.path.join(ROOT, "extract_matching.py")))
chk("registry.json records the promotion + passed domains",
    os.path.exists(os.path.join(ROOT, "registry.json")) and
    len(P.json.load(open(os.path.join(ROOT, "registry.json")))["extract_matching"]["passed_domains"]) == 6)

# ── run "session 2": a BRAND-NEW instance (simulated restart) AUTO-LOADS it with NO re-proof ─────────────
base2 = P.PersistentPrimitiveBase(ROOT)
chk("new process auto-loads the proven primitive from disk (no re-proof)", base2.has("extract_matching"))
served = base2.serve("extract_matching", PATS["security"], {"blob": examples("security", 1, 99)[0][0]})
chk("auto-loaded primitive actually runs (serves a fresh security ticket)", "CVE-20" in (served or ""))

# ── extend the lib: promote a SECOND primitive; a third instance auto-loads BOTH (compounding) ──────────
P.prove_and_persist(base2, "reduce_integers", math_domains)
base3 = P.PersistentPrimitiveBase(ROOT)
chk("library is extendible: a 3rd start auto-loads BOTH primitives (the bank grew)",
    set(base3.names()) == {"extract_matching", "reduce_integers"})
chk("the second auto-loaded primitive runs too (reduce sum over a fresh blob)",
    base3.serve("reduce_integers", "sum", {"blob": examples("math_sum", 1, 55)[0][0]})
    == examples("math_sum", 1, 55)[0][1][0])

# ── the gate has teeth on the persistent path: dedup + non-generalizing are refused ─────────────────────
sig, src = P.PRIMITIVE_SOURCES["extract_matching"]
redup = base3.promote("extract_matching_twin", src, sig, {"generalizes": True, "passed": ["x", "y"]})
chk("dedup: a functional-twin source is refused persistence", redup is False)
nogen = P.prove_and_persist(P.PersistentPrimitiveBase(tempfile.mkdtemp()), "reduce_integers",
                            {"only": math_domains["sum"]}, min_domains=2)  # 1 domain < min_domains → no generalize
chk("gate: a primitive proven on <min_domains is NOT persisted", nogen["generalizes"] is False and nogen["added"] is False)

print("\nLIB @", ROOT)
for n, m in base3.registry().items():
    print("  %-18s %-26s passes=%s" % (n, m["signature"], m["passed_domains"]))
print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: gate-passed primitives self-persist as code and auto-load on the next run — an extendible det. lib.")
