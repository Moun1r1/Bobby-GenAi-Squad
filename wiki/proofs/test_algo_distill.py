#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


import hashlib


def embed(texts):
    return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:32]] for t in texts]


ROMAN_SRC = ("def solve(text):\n"
             "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
             "    total = 0\n    prev = 0\n"
             "    for ch in reversed(text):\n"
             "        v = vals[ch]\n"
             "        if v < prev:\n            total -= v\n"
             "        else:\n            total += v\n            prev = v\n"
             "    return total\n")
LUHN_SRC = ("def solve(text):\n    total = 0\n    alt = False\n"
            "    for ch in reversed(text):\n        d = int(ch)\n"
            "        if alt:\n            d = d * 2\n            if d > 9:\n                d = d - 9\n"
            "        total += d\n        alt = not alt\n"
            "    return 'valid' if total % 10 == 0 else 'invalid'\n")


def _roman_to_int(t):
    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    tot, prev = 0, 0
    for ch in reversed(t):
        x = v[ch]
        tot += -x if x < prev else x
        prev = max(prev, x)
    return tot


class CodeLLM:
    """A stand-in that WRITES the algorithm when asked to distill, and solves individual tickets during warmup."""
    last_usage = {"total_tokens": 200}

    def __call__(self, messages, max_tokens=400, temperature=0.0):
        c = messages[-1]["content"]
        if "def solve" in c:                                    # distillation: author the function
            return LUHN_SRC if "valid" in c.lower() else ROMAN_SRC
        inp = c.split("INPUT:", 1)[1].strip()                   # warmup: solve one ticket
        if "Luhn" in c or "valid" in c:
            return "valid" if B._luhn_ok(inp) else "invalid"
        return str(_roman_to_int(inp))


def _tickets(fid, n, seed):
    fam = next(f for f in B.FAMILIES if f[0] == fid)
    _, kind, ask, build = fam
    out = []
    for i in range(n):
        blob, gold = build(B._rng(seed * 100 + i))
        out.append({"ticket_id": "%s-%d" % (fid, i), "cluster": fid, "kind": kind, "cap": "task", "ask": ask,
                    "prompt": B._PROMPT_TMPL[kind].format(ask=ask, blob=blob), "blob": blob, "gold": gold})
    return out


# interleave roman + luhn tickets
stream = []
rom, luhn = _tickets("algo_roman", 12, 1), _tickets("algo_luhn", 12, 2)
for i in range(12):
    stream.append(rom[i])
    stream.append(luhn[i])

# sanity: regex CANNOT do this — a frozen extractor over the roman input yields nothing useful
reg = B.make_extractor(r"[IVXLCDM]+")
chk("regex baseline cannot solve Roman→int (extracts the glyphs, not the value)",
    reg({"blob": "XLII"}) == "XLII" and B.score(reg({"blob": "XLII"}), ["42"]) == 0.0)

res = B.run(stream, CodeLLM(), embed, distill=True, warmup=4, proof_threshold=0.9)
chk("engine distilled 2 LLM-AUTHORED code plugins (roman + luhn)", res.promotions == 2)
chk("algorithm accuracy is perfect once frozen", res.accuracy >= 0.98)

# the frozen handlers are compiled CODE (have _src), not regex — and provenance captured the LLM's source
plugins = res.engine.registry.active()
srcs = [(p.proof or {}).get("hypothesis") for p in plugins]
code_plugins = [p for p in plugins if hasattr(p.handler, "_src")]
chk("frozen plugins are LLM-authored CODE (handler carries its source), not a pattern", len(code_plugins) == 2)
chk("provenance/hypothesis marks them as llm_authored_code", srcs.count("llm_authored_code") == 2)

# after warmup, the algo tickets route to the frozen code at ZERO llm tokens, and are correct
late = [r for r in res.signals.rows if r["i"] >= 12]
frozen_late = [r for r in late if r["route"] == "frozen"]
chk("post-warmup algo tickets served by frozen code (0 LLM tokens)",
    len(frozen_late) >= 12 and all(r["token_cost"] == 0 for r in frozen_late))
chk("every frozen-served algo ticket is correct (the LLM's code runs deterministically)",
    all(r["correct"] == 1 for r in frozen_late))

# direct check: pull the frozen roman plugin and run the LLM-authored code on a fresh value
rp = next(p for p in code_plugins if "roman" in p.name or _roman_to_int)
val = None
for p in code_plugins:
    out = p.handler({"blob": "MCMXCIV"})                      # 1994 — heavy subtractive parsing
    if out == "1994":
        val = out
chk("the frozen LLM-authored code computes MCMXCIV = 1994 (real algorithm, zero LLM)", val == "1994")

print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: the ACR engine distills LLM-AUTHORED ALGORITHMS (code), not just regex.")
