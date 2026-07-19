#!/usr/bin/env python3
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


# ── deterministic mock embedder: a 6-dim one-hot-ish vector per cluster ask (identical within a cluster) ──
def mock_embed(texts):
    out = []
    for t in texts:
        if "error codes" in t:
            v = [1.0, 0, 0, 0, 0, 0]
        elif "config keys" in t:
            v = [0, 1.0, 0, 0, 0, 0]
        else:                                       # the OOD 'SUM of every integer' ask lands far from A and B
            v = [0, 0, 1.0, 1.0, 1.0, 1.0]
        out.append(v)
    return out


# ── deterministic mock LLM: solves the ask exactly (a stand-in for a competent model) ───────────────────
class MockLLM:
    def __init__(self):
        self.last_usage = {"total_tokens": 120}
        self.n = 0

    def __call__(self, messages, max_tokens=400, temperature=0.0):
        self.n += 1
        content = messages[-1]["content"]
        # a distillation 'propose regexes' request → hand back the true patterns as candidates
        if "candidate Python `re` regex" in content:
            return "ERR-\\d{3,5}\nCFG_[A-Z][A-Z0-9_]{2,}\nNOPE-\\d+"
        # a normal ticket → extract/compute the gold directly from the DATA blob
        blob = content.split("DATA:\n", 1)[1] if "DATA:\n" in content else content
        if "error codes" in content:
            return "\n".join(dict.fromkeys(re.findall(r"ERR-\d{3,5}", blob)))
        if "config keys" in content:
            return "\n".join(dict.fromkeys(re.findall(r"CFG_[A-Z][A-Z0-9_]{2,}", blob)))
        # OOD: sum of every integer
        return str(sum(int(x) for x in re.findall(r"\d+", blob)))


# 1) dataset shape + determinism
t1 = B.generate(seed=1)
t1b = B.generate(seed=1)
chk("dataset has exactly 100 tickets", len(t1) == 100)
chk("clusters are 40 A / 40 B / 20 C", [t["cluster"] for t in t1].count("A") == 40
    and [t["cluster"] for t in t1].count("B") == 40 and [t["cluster"] for t in t1].count("C") == 20)
chk("OOD cluster C injected starting at ticket #80 (0-based #79)",
    all(t["cluster"] != "C" for t in t1[:80]) and all(t["cluster"] == "C" for t in t1[80:]))
chk("generation is byte-deterministic for a fixed seed", [t["gold"] for t in t1] == [tb["gold"] for tb in t1b])
chk("every A/B ticket has a non-empty gold set", all(t["gold"] for t in t1 if t["cluster"] in ("A", "B")))

# 2) grading is exact
chk("score = 1.0 on exact match", B.score("ERR-100\nERR-200", ["ERR-200", "ERR-100"]) == 1.0)
chk("score < 1.0 on a miss", B.score("ERR-100", ["ERR-100", "ERR-200"]) < 1.0)

# 3) ACR flywheel run
acr = B.run(t1, MockLLM(), mock_embed, distill=True, warmup=4)
chk("flywheel promoted >=2 frozen extractors (A and B)", acr.promotions >= 2)
chk("router local fraction is high (A+B mostly frozen)", acr.signals.local_fraction() >= 0.65)
chk("ACR accuracy is perfect with a competent model", acr.accuracy >= 0.999)

# the tripwire: no Cluster C ticket was served by a frozen plugin
c_routes = [r["route"] for r in acr.signals.rows if r["cluster"] == "C"]
chk("OOD tripwire holds: every Cluster C ticket routed to the LLM (never a frozen A/B plugin)",
    all(rt == "llm" for rt in c_routes))
# and A/B tickets after warmup are frozen (cost bent down)
a_late = [r["route"] for r in acr.signals.rows if r["cluster"] == "A"][10:]
chk("Cluster A late tickets are served by the frozen plugin (0 LLM tokens)",
    a_late and all(rt == "frozen" for rt in a_late))

# 4) No-ACR control spends more, same accuracy
ctrl = B.run(t1, MockLLM(), mock_embed, distill=False)
acr_tok = sum(acr.signals.series("token_cost"))
ctrl_tok = sum(ctrl.signals.series("token_cost"))
chk("No-ACR control never distills (0 promotions)", ctrl.promotions == 0)
chk("No-ACR control routes 100% to the LLM", ctrl.signals.local_fraction() == 0.0)
chk("TOKEN REDUCTION: ACR spends strictly fewer tokens than the No-ACR control", acr_tok < ctrl_tok)
chk("TOKEN REDUCTION: token saving is large (>50%)", acr_tok < 0.5 * ctrl_tok)
chk("accuracy parity: ACR >= control", acr.accuracy >= ctrl.accuracy)

# 5) golden signals + report render without error
g = B.golden_signals(acr, ctrl)
chk("golden_signals emits all 6 signal families",
    all(k in g for k in ["token_cost_per_task", "router_local_fraction", "working_context_size",
                         "eval_compute_saved", "wall_clock_time_per_task", "accuracy"]))
chk("working_context_size stays < 5000 tokens/step", g["working_context_size"]["max"] < 5000)
rep = B.render_report(acr, ctrl)
chk("render_report produces a non-empty ASCII report", isinstance(rep, str) and "TOKEN REDUCTION" in rep)

# 6) dependency-free SVG plot + round-trip reload
import tempfile
svg_path = os.path.join(tempfile.mkdtemp(), "gs.svg")
B.plot_svg(acr, ctrl, svg_path)
svg = open(svg_path).read()
chk("plot_svg writes a valid standalone SVG (no matplotlib)", svg.startswith("<svg") and "polyline" in svg)
jp = os.path.join(tempfile.mkdtemp(), "s.json")
acr.signals.to_json(jp)
chk("Signals round-trips through JSON", B.Signals.from_json(jp).local_fraction() == acr.signals.local_fraction())

# ── 7) CROSS-MODAL: extraction across 6 sectors + math + code + image + prose (irreducible) ─────────────
import hashlib


def mixed_embed(texts):
    """Deterministic hash→vector (32-dim, near-orthogonal): identical ask ⇒ identical vector (in-distribution),
    different family ⇒ far (OOD) — a clean stand-in for real embeddings' family separation."""
    out = []
    for t in texts:
        hb = hashlib.sha256(t.encode()).digest()
        out.append([b / 255.0 for b in hb[:32]])
    return out


_ROMAN_SRC = ("def solve(text):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
              "    t=0\n    p=0\n    for ch in reversed(text):\n        x=v[ch]\n"
              "        t+=-x if x<p else x\n        p=max(p,x)\n    return t\n")
_LUHN_SRC = ("def solve(text):\n    t=0\n    alt=False\n    for ch in reversed(text):\n        d=int(ch)\n"
             "        if alt:\n            d*=2\n            if d>9:\n                d-=9\n        t+=d\n        alt=not alt\n"
             "    return 'valid' if t%10==0 else 'invalid'\n")


def _roman_to_int(s):
    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    t, p = 0, 0
    for ch in reversed(s):
        x = v[ch]
        t += -x if x < p else x
        p = max(p, x)
    return t


class MixedLLM:
    """A competent stand-in that solves every modality deterministically (so accuracy is a clean signal)."""
    def __init__(self):
        self.last_usage = {"total_tokens": 130}

    def __call__(self, messages, max_tokens=400, temperature=0.0):
        c = messages[-1]["content"]
        blob = c.split("DATA:\n", 1)[1] if "DATA:\n" in c else (
            c.split("GRID:\n", 1)[1] if "GRID:\n" in c else (
                c.split("REVIEW:\n", 1)[1] if "REVIEW:\n" in c else c))
        if "candidate Python `re` regex" in c:                  # distill proposal — infer regex from the example
            for pat in [r"TXN-\d{6}", r"CVE-20\d{2}-\d{4}", r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", r"[A-Z]\d{2}\.\d",
                        r"§\d{3}", r"[A-Z]{2}\d{3,4}"]:          # answers (not the ask) are in the prompt → match them
                if re.search(pat, c):
                    return pat
            return r"\b[A-Z]{2,}-\d+\b"
        if "sentiment" in c:                                    # prose (irreducible) — classify by phrasing
            low = blob.lower()
            if any(w in low for w in ["disappoint", "broke", "terrible", "stopped"]):
                return "negative"
            if any(w in low for w in ["superb", "exceeded", "delighted", "flawless"]):
                return "positive"
            return "neutral"
        if "CODE:" in c:                                        # any code family — apply the asked transform
            code = c.split("CODE:", 1)[1].strip()
            if "camelCase" in c:
                return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), code)
            if "mutable default" in c:
                return re.sub(r"=\s*(\[\]|\{\}|list\(\)|dict\(\)|set\(\))", "=None", code)
            if "single to double" in c:
                return re.sub(r"'([^']*)'", r'"\1"', code)
            return code
        if "SUM of all" in c:
            return str(sum(int(x) for x in re.findall(r"\d+", blob)))
        if "MAXIMUM" in c:
            return str(max(int(x) for x in re.findall(r"\d+", blob)))
        if "filled (#)" in c:
            return str(blob.count("#"))
        if "def solve" in c:                                    # algo distill — author the function
            return _LUHN_SRC if "valid" in c.lower() else _ROMAN_SRC
        if "Roman numeral" in c:
            return str(_roman_to_int(c.split("INPUT:", 1)[1].strip()))
        if "Luhn checksum" in c:
            return "valid" if B._luhn_ok(c.split("INPUT:", 1)[1].strip()) else "invalid"
        for key, pat in [("transaction", r"TXN-\d{6}"), ("ICD", r"[A-Z]\d{2}\.\d"), ("CVE", r"CVE-20\d{2}-\d{4}"),
                         ("statute", r"§\d{3}"), ("MAC", r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}"), ("flight", r"[A-Z]{2}\d{3,4}")]:
            if key in c:
                return "\n".join(dict.fromkeys(re.findall(pat, blob)))
        return ""


mt = B.generate_mixed(seed=1, per=12)
chk("mixed dataset: 15 families × 12 = 180 tickets", len(mt) == 180)
chk("mixed spans 6 modalities (incl. LLM-authored algo)",
    set(t["kind"] for t in mt) == {"extract", "math", "code", "image", "algo", "prose"})
chk("code modality has 3 distinct transform families (camel, mutdef, quotes)",
    sum(1 for f in B.FAMILIES if f[1] == "code") == 3)
macr = B.run(mt, MixedLLM(), mixed_embed, distill=True, warmup=4, proof_threshold=0.9)
by_kind_frozen = {}
for r in macr.signals.rows:
    by_kind_frozen.setdefault(r["kind"], []).append(r["route"])
chk("REDUCIBLE families distilled to frozen plugins (extract+math+code+image, ~10)", macr.promotions >= 8)
chk("extract sectors route frozen after warmup (no cross-sector misroute)",
    all(rt == "frozen" for rt in by_kind_frozen["extract"][40:]))
chk("math distilled to a frozen reducer", "frozen" in by_kind_frozen["math"])
chk("code distilled to a frozen transform", "frozen" in by_kind_frozen["code"])
chk("image distilled to a frozen grid-reader", "frozen" in by_kind_frozen["image"])
chk("IRREDUCIBLE prose NEVER distilled — stays on the LLM every time (the generative floor)",
    all(rt == "llm" for rt in by_kind_frozen["prose"]))
chk("mixed accuracy holds high with a competent model", macr.accuracy >= 0.98)
mctrl = B.run(mt, MixedLLM(), mixed_embed, distill=False)
chk("cross-modal TOKEN REDUCTION: ACR spends fewer tokens than No-ACR control",
    sum(macr.signals.series("token_cost")) < sum(mctrl.signals.series("token_cost")))
chk("cross-modal accuracy parity: ACR >= control", macr.accuracy >= mctrl.accuracy)

print("\n" + rep)
print("\n[cross-modal] promotions=%d local=%.0f%% acc=%.0f%% | prose stayed LLM: %s"
      % (macr.promotions, macr.signals.local_fraction() * 100, macr.accuracy * 100,
         all(rt == "llm" for rt in by_kind_frozen["prose"])))
print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
