#!/usr/bin/env python3
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import PluginStore, DistillationCorpus  # noqa: E402
from bobby_squad.engine import Engine  # noqa: E402
from bobby_squad.router import OODGate, competence_router  # noqa: E402
from bobby_squad import burn_in as B  # noqa: E402

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


tmp = tempfile.mkdtemp()


def fake_embed(texts):
    # deterministic 8-d embedding: char-histogram buckets (no network)
    out = []
    for t in texts:
        v = [0.0] * 8
        for ch in t:
            v[ord(ch) % 8] += 1.0
        out.append(v)
    return out


# ── build run k: an engine with two proof-gated frozen plugins (regex + math) ──────────────────────────
eng1 = Engine(os.path.join(tmp, "run1"), require_proof=True)
eng1.interceptors = [competence_router(fake_embed)]

rx = B.make_extractor(r"TXN-\d{4}")
gate_rx = OODGate.fit(fake_embed(["find the transaction id", "extract txn code", "get the TXN"]))
chk("run1: promote regex plugin", eng1.promote("extract_frozen_0", rx, tags=["fin_txn"],
    proof={"verdict": "WIRE", "kind": "extract", "hypothesis": r"TXN-\d{4}", "score": 1.0, "competence": gate_rx},
    code="import re\ndef h(p): return re.findall('TXN', p)"))

agg = B.make_aggregator("sum")
gate_m = OODGate.fit(fake_embed(["sum the numbers", "total the amounts", "add them up"]))
chk("run1: promote math plugin", eng1.promote("math_frozen_1", agg, tags=["arith_sum"],
    proof={"verdict": "WIRE", "kind": "math", "hypothesis": "sum", "score": 1.0, "competence": gate_m},
    code="def h(p): reduce ints by sum"))

txn_blob = "log: TXN-0007 ok; retry TXN-0007; new TXN-0042 done"
out1 = eng1.registry.resolve("fin_txn").handler({"blob": txn_blob})
sum1 = eng1.registry.resolve("arith_sum").handler({"blob": "3 and 4 and 5"})

# ── snapshot → save → load into a FRESH store, restore into a FRESH engine (== run k+1 cold start) ──────
store = PluginStore(path=os.path.join(tmp, "skills.json"))
n_snap = store.snapshot(eng1, provenance="run1")
store.save()
chk("snapshot captured both reconstructable plugins", n_snap == 2)
chk("store persisted to disk", os.path.exists(store.path) and json.load(open(store.path)).get("plugins"))

store2 = PluginStore.load(store.path)
eng2 = Engine(os.path.join(tmp, "run2"), require_proof=True)
eng2.interceptors = [competence_router(fake_embed)]
n_restore = store2.restore(eng2)
chk("restore promoted both plugins into a fresh engine", n_restore == 2)

# ── the restored plugins must serve byte-identically ───────────────────────────────────────────────────
p_rx = eng2.registry.resolve("fin_txn")
p_m = eng2.registry.resolve("arith_sum")
chk("restored regex plugin exists + serves identically", p_rx is not None and p_rx.handler({"blob": txn_blob}) == out1)
chk("restored math plugin serves identically", p_m is not None and p_m.handler({"blob": "3 and 4 and 5"}) == sum1)

# ── the OODGate competence region is preserved (routing + OOD fail-safe intact) ─────────────────────────
g0, g1 = gate_rx, p_rx.proof["competence"]
qv = fake_embed(["extract the TXN please"])[0]
chk("restored OODGate is an OODGate", isinstance(g1, OODGate))
chk("restored gate tau matches", abs(g1.tau - g0.tau) < 1e-9)
chk("restored gate scores distances identically", abs(g1.distance(qv) - g0.distance(qv)) < 1e-9)
chk("covered_caps reports both capabilities", store2.covered_caps() == {"fin_txn", "arith_sum"})

# ── DistillationCorpus: only verified traces become training records ────────────────────────────────────
corpus = DistillationCorpus()
chk("plugin-served trace is always kept",
    corpus.record(input=txn_blob, output="TXN-0007\nTXN-0042", capability="fin_txn", source="plugin"))
chk("correct model trace is kept",
    corpus.record(input="sum 3 4 5", output="12", capability="arith_sum", source="model", correct=True))
chk("INCORRECT model trace is dropped (no bad labels)",
    not corpus.record(input="sum 1 2", output="99", capability="arith_sum", source="model", correct=False))
chk("duplicate trace is deduped",
    not corpus.record(input=txn_blob, output="TXN-0007\nTXN-0042", capability="fin_txn", source="plugin"))
chk("empty output is dropped",
    not corpus.record(input="x", output="   ", capability="fin_txn", source="model", correct=True))

st = corpus.stats()
chk("corpus stats: 2 verified records (1 plugin, 1 model)",
    st["n"] == 2 and st["by_source"]["plugin"] == 1 and st["by_source"]["model"] == 1)

sft = os.path.join(tmp, "corpus.jsonl")
n = corpus.emit_sft(sft, style="messages")
lines = [json.loads(x) for x in open(sft)]
chk("emit_sft wrote all records as chat messages", n == 2 and len(lines) == 2)
chk("SFT record has user+assistant turns with the verified output",
    lines[0]["messages"][0]["role"] == "user" and lines[0]["messages"][1]["role"] == "assistant"
    and lines[0]["messages"][1]["content"])

# ── collect_run: join tickets + Signals rows into verified records ──────────────────────────────────────
tickets = [{"blob": "TXN-0009", "gold": ["TXN-0009"], "cap": "fin_txn", "prompt": "extract"},
           {"blob": "2 3", "gold": ["5"], "cap": "arith_sum", "prompt": "sum"},
           {"blob": "bad", "gold": ["Z"], "cap": "fin_txn", "prompt": "extract"}]
rows = [{"route": "frozen", "correct": 1}, {"route": "llm", "correct": 1}, {"route": "llm", "correct": 0}]
c2 = DistillationCorpus()
kept = c2.collect_run(tickets, rows)
chk("collect_run keeps the 2 verified tasks, drops the 1 wrong model task", kept == 2 and c2.stats()["n"] == 2)

print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: frozen skills persist across runs (byte-identical + gate intact); only verified traces train.")
