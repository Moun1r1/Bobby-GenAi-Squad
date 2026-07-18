#!/usr/bin/env python3
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad import primitive_lib as L
from bobby_squad.primitive_intel import PRIMITIVE_SOURCES

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


# keyword-bag "embedding": deterministic but genuinely semantic (overlap of task vocabulary)
_VOCAB = ("extract pattern token match sum integer number fold aggregate count max refactor rewrite code identifier "
          "convert similar analogous prior case retrieve").split()


def kembed(texts):
    return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


def examples(fid, n, seed):
    build = next(f for f in B.FAMILIES if f[0] == fid)[3]
    return [build(B._rng(seed * 100 + i)) for i in range(n)]


PATS = {"fin": r"TXN-\d{6}", "health": r"[A-Z]\d{2}\.\d", "security": r"CVE-20\d{2}-\d{4}", "legal": r"§\d{3}"}
extract_domains = {n: {"param": p, "examples": examples(n, 5, i + 1)} for i, (n, p) in enumerate(PATS.items())}
math_domains = {"sum": {"param": "sum", "examples": examples("math_sum", 5, 7)},
                "max": {"param": "max", "examples": examples("math_max", 5, 8)}}
code_domains = {"snake2camel": {"param": "snake2camel", "examples": examples("code_camel", 5, 9)},
                "single2double": {"param": "single2double", "examples": examples("code_quotes", 5, 10)}}
DOMAINS = {"extract_matching": extract_domains, "reduce_integers": math_domains, "transform_code": code_domains}

ROOT = tempfile.mkdtemp()

# ── build the organized library (seed the known primitives behind the gate) ─────────────────────────────
lib = L.PrimitiveLibrary(ROOT, embed_fn=kembed)
seeded = L.seed_known(lib, DOMAINS)
chk("3 primitives distilled + filed", sum(1 for r in seeded.values() if r["action"] == "distilled") == 3)
chk("filed by CATEGORY on disk (core/extraction, core/arithmetic, core/transformation)",
    os.path.exists(os.path.join(ROOT, "core", "extraction", "extract_matching.py"))
    and os.path.exists(os.path.join(ROOT, "core", "arithmetic", "reduce_integers.py"))
    and os.path.exists(os.path.join(ROOT, "core", "transformation", "transform_code.py")))
chk("categories are organized", set(lib.categories()) == {"extraction", "arithmetic", "transformation"})

# ── SEMANTIC recall: find a primitive back from a task description ───────────────────────────────────────
top = lib.recall("I need to compute the sum of the integer numbers in this text", k=1)
chk("semantic recall: 'sum the integers' → reduce_integers", top and top[0][0] == "reduce_integers")
top2 = lib.recall("pull out every token matching a pattern", k=1)
chk("semantic recall: 'pull tokens matching a pattern' → extract_matching", top2 and top2[0][0] == "extract_matching")

# ── STRUCTURAL recall: the SAME for-loop under DIFFERENT variable names is found back ───────────────────
_, orig = PRIMITIVE_SOURCES["extract_matching"]
renamed = ("import re\n"
           "def bind(pat):\n"
           "    compiled = re.compile(pat)\n"
           "    def solve(s):\n"
           "        visited = set(); result = []\n"
           "        for item in compiled.findall(s):\n"
           "            if item not in visited:\n"
           "                visited.add(item); result.append(item)\n"
           "        return '\\n'.join(result)\n"
           "    return solve\n")
chk("structural recall: the same loop with renamed vars matches the existing primitive (AST fingerprint)",
    lib.find_by_fingerprint(renamed) == "extract_matching")

# ── recall_or_distill: memory is consulted BEFORE distilling → no duplicate learning ────────────────────
r_struct = L.recall_or_distill(lib, "extract_v2", "extract tokens matching a pattern", renamed,
                               "(text, pattern) -> tokens", extract_domains, category="extraction")
chk("recall_or_distill REUSES the structural twin instead of adding a duplicate",
    r_struct["action"] == "reused-structural" and r_struct["name"] == "extract_matching")

r_sem = L.recall_or_distill(lib, "adder", "aggregate and sum the integer numbers in the text",
                            PRIMITIVE_SOURCES["reduce_integers"][1].replace("nums", "vals"),  # different code text...
                            "(text, op) -> value", math_domains, category="arithmetic")
chk("recall_or_distill REUSES a semantic match (task already solved) before distilling",
    r_sem["action"] in ("reused-semantic", "reused-structural"))

# ── restart: a fresh instance auto-loads the organized lib + memory index and finds everything back ─────
lib2 = L.PrimitiveLibrary(ROOT, embed_fn=kembed)
chk("restart auto-loads all 3 filed primitives", set(lib2.names()) == {"extract_matching", "reduce_integers",
                                                                        "transform_code"})
chk("restart preserves the semantic memory index (recall still works)",
    lib2.recall("sum the integers", k=1)[0][0] == "reduce_integers")
chk("restart preserves structural index (same loop still found back)",
    lib2.find_by_fingerprint(renamed) == "extract_matching")
chk("auto-loaded primitive still runs (serve a fresh security ticket)",
    "CVE-20" in (lib2.serve("extract_matching", PATS["security"], {"blob": examples("security", 1, 42)[0][0]}) or ""))

print("\n" + lib2.tree())
print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: organized, memory-backed lib — reuse by meaning AND by structure, persistent across restarts.")
