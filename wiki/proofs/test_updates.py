#!/usr/bin/env python3
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


# 1) needle-preserving navigate (whole_vault/whole_k) — embeddings only, no LLM
from bobby_squad import VaultHub
from bobby_squad.retrieval import default_embed

d = tempfile.mkdtemp()
h = VaultHub(d, embed_fn=default_embed)
h.enrich("doc", "passage 1", "The vault cipher hidden by the archivist is BLUE-MANGO 7.", source="b")
h.enrich("doc", "passage 2", "Filler about weather.", source="b")
h.enrich("kg", "Archivist", "keeper of secret notes", source="b")
block = h.navigate("vault cipher archivist", per_vault_k=3, hops=1, budget=6000, per_note=600,
                   whole_vault="doc", whole_k=2)
chk("navigate.whole_vault keeps the needle verbatim", "BLUE-MANGO 7" in block and "(verbatim)" in block)
chk("navigate backward-compatible (no whole_vault)", isinstance(h.navigate("cipher"), str))

# 2) gain-proof harness deterministic parts
import memory_gains as MG

chk("memory_gains.qa_f1 exact", MG.qa_f1("blue mango", ["blue mango"]) == 1.0)
chk("memory_gains.qa_f1 miss", MG.qa_f1("zzz", ["blue mango"]) == 0.0)
chk("memory_gains BUILTINS present", {"solo", "null", "flat_k", "graph_hop"}.issubset(MG.BUILTINS))
rows = [{"context": "c" + str(i % 3), "input": "q", "answers": ["a"]} for i in range(9)]
tr, he = MG.split(rows)
chk("memory_gains.split held-out nonempty + total preserved", len(he) > 0 and len(tr) + len(he) == 9)
chk("memory_gains verdict cost-inclusive", MG.verdict(MG.Result("c", 30.0, 100, 1.0, 10),
                                                      MG.Result("b", 20.0, 100, 1.0, 10))["verdict"] == "WIRE")

# 3) flywheel code-gen deterministic helpers (no LLM)
import self_improve_engine as SI

good = ("from memory_gains import Method\n"
        "class Candidate(Method):\n"
        "    def prepare(self, ctx): return ctx\n"
        "    def answer(self, mem, q): return ('x', 1)\n")
chk("sie._extract from fenced block", "class Candidate" in SI._extract("```python\n" + good + "```"))
chk("sie._extract from unterminated fence", "class Candidate" in SI._extract("```python\n" + good))
chk("sie._complete True on valid code", SI._complete(good) is True)
chk("sie._complete False on truncated code", SI._complete("class Candidate(Method):\n    def prepare(self, ctx") is False)

# 4) Mem0 comparison shims + scoring
import compare_memory as CM

CM._install_shims()
chk("compare_memory shims install without error", True)
chk("compare_memory.qa_f1 date match", CM.qa_f1("7 may 2023", ["7 May 2023"]) > 0.9)

# 5) named-job registry (jobs.py) — no LLM
from bobby_squad import JobRegistry

jr = JobRegistry(tempfile.mkdtemp())
jr.submit("t1", "echo hi; exit 0", timeout=30)
j1 = jr.wait("t1", poll=0.05, timeout=10)
chk("jobs: done with exit 0", bool(j1) and j1["status"] == "done" and j1["exit"] == 0)
chk("jobs: logs captured", "hi" in jr.logs("t1"))
jr.submit("t2", "sleep 5", timeout=30)
p1 = jr.status("t2")["pid"]
p2 = jr.submit("t2", "sleep 5", timeout=30)["pid"]
chk("jobs: idempotent by name (same pid, no duplicate)", p1 == p2)
jr.cancel("t2")
chk("jobs: cancel → killed", jr.status("t2")["status"] == "killed")
jr.submit("t3", "exit 3", timeout=30)
j3 = jr.wait("t3", poll=0.05, timeout=10)
chk("jobs: failing cmd → failed with exit 3", j3["status"] == "failed" and j3["exit"] == 3)

# 6) AST-dedup (dedup_ast.py) — no LLM
from bobby_squad import AstDedup, fingerprint

dd = AstDedup()
dd.add("def f(x):\n    return x + 1\n")
chk("astdedup: rename + docstring + comment = duplicate",
    dd.is_dup("def f(y):\n    '''doc'''\n    return y + 1  # c\n"))
chk("astdedup: functional difference is NOT a duplicate", not dd.is_dup("def f(x):\n    return x + 2\n"))
chk("astdedup: syntax error → None (not a duplicate)", fingerprint("def f(:") is None)

# 7) THE KERNEL (engine.py) — the cost curve: the LLM fallback stops being called once a proven plugin is promoted
from bobby_squad import Engine, Plugin

eng = Engine(tempfile.mkdtemp())
llm_calls = {"n": 0}


def _llm(payload):
    llm_calls["n"] += 1
    return "LLM:" + payload.get("q", "")


eng.on("translate", _llm)                                          # last-resort handler = the expensive LLM
r1 = eng.emit("translate", {"cap": "translate", "q": "hi"})
chk("engine: falls through to LLM when no plugin",
    r1 == "LLM:hi" and llm_calls["n"] == 1 and eng.stats["by_fallback"] == 1)

code = "def translate(p):\n    return 'FROZEN:' + p['q']\n"
chk("engine: promote a PROVEN frozen plugin",
    eng.promote("translate_v1", lambda p: "FROZEN:" + p["q"], tags=["translate"],
                proof={"verdict": "WIRE", "dF1": 5.0}, code=code))
before = llm_calls["n"]
r2 = eng.emit("translate", {"cap": "translate", "q": "yo"})
chk("engine: routes to the frozen plugin — LLM NOT called again (cost curve)",
    r2 == "FROZEN:yo" and llm_calls["n"] == before and eng.stats["by_interceptor"] >= 1)
chk("engine: unproven plugin rejected (governance gate)",
    not eng.promote("bad", lambda p: 1, tags=["x"], proof=None, code="def g(p):\n    return 1\n"))
twin = "def translate(q):\n    '''d'''\n    return 'FROZEN:' + q['q']  # c\n"
chk("engine: functional-twin plugin rejected (dedup gate)",
    not eng.registry.register(Plugin("twin", lambda p: 0, frozenset(["translate"]), proof={"verdict": "WIRE"}),
                              code=twin))
kinds = [e.kind for e in eng.log.read()]
chk("engine: event-log spine records emits + handled + promotion",
    "translate.handled" in kinds and "SKILL_PROMOTED" in kinds)

# 8) reproducible harness (harness.py) — no LLM
from bobby_squad import Scenario, DataCollector, harness_verdict, Report, ci95

sc = Scenario("s", seed=7)
reps = sc.spawn_replications(5)
chk("harness: replications are seeded + distinct", [r.seed for r in reps] == [7, 8, 9, 10, 11])
m, c = ci95([10.0, 12.0, 11.0, 13.0, 9.0])
chk("harness: ci95 mean+interval", abs(m - 11.0) < 1e-9 and c > 0)
dc = DataCollector()
rep = dc.run("det", lambda s: 50.0 + (s.seed % 3), sc, replications=6)   # deterministic metric
chk("harness: DataCollector reproducible run", rep.n == 6 and rep.per == dc.run("det", lambda s: 50.0 + (s.seed % 3), sc, 6).per)
treat = Report("t", 62.0, 1.0, 5, [])
base = Report("b", 55.0, 1.0, 5, [])
chk("harness.verdict WIRE (CI-separated)", harness_verdict(treat, base)["verdict"] == "WIRE")
chk("harness.verdict INVALID on leaking negative control",
    harness_verdict(treat, base, control=Report("c", 60.0, 1.0, 5, []))["verdict"] == "INVALID")
chk("harness.verdict INCONCLUSIVE when baseline ceilinged",
    harness_verdict(Report("t", 99.0, 1.0, 5, []), Report("b", 97.0, 1.0, 5, []))["verdict"] == "INCONCLUSIVE")
chk("harness.verdict MARGINAL when CIs overlap",
    harness_verdict(Report("t", 56.0, 3.0, 5, []), Report("b", 55.0, 3.0, 5, []))["verdict"] == "MARGINAL")

# 9) synthetic needle generator (synthbench.py) — no LLM, control ≈ 0 by construction
from bobby_squad import synthbench

items = synthbench.make(seed=1, length_chars=3000, needles=2)
chk("synthbench: needle planted in context", all(it["answers"][0] in it["context"] for it in items))
chk("synthbench: answer is UNGUESSABLE (not in the question)", all(it["answers"][0] not in it["question"] for it in items))
chk("synthbench: deterministic by seed", synthbench.make(1, 3000, 2)[0]["answers"] == synthbench.make(1, 3000, 2)[0]["answers"])
chk("synthbench: dataset shape matches harness rows", all({"context", "input", "answers"} <= set(r) for r in synthbench.dataset(2, n_docs=3)))

# 10) FSM control plane (fsm.py) — no LLM
from bobby_squad import FSM, cluster_match

fsm = FSM("draft")
fsm.add("draft", "review", guard=lambda c: c.get("has_spec")).add("review", "ship", guard=("llm", "ok? {x}"))
chk("fsm: deterministic predicate guard fires (no LLM)", fsm.next("draft", {"has_spec": True}) == ("review", "deterministic"))
chk("fsm: predicate false → stuck without an LLM", fsm.next("draft", {"has_spec": False}) == (None, "stuck"))
chk("fsm: is_valid rejects an off-graph transition", fsm.is_valid("draft", "review") and not fsm.is_valid("draft", "ship"))
chk("fsm: LLM-guarded edge is stuck when no llm provided", fsm.next("review", {"x": "y"}) == (None, "stuck"))

f1s = lambda a, b: MG.qa_f1(a, [b])                                # answer-agreement: both are strings
chk("cluster_match: frozen == LLM on held-out → match",
    cluster_match(lambda p: "Ada Lovelace", lambda p: "Ada Lovelace",
                  [{"q": "1"}, {"q": "2"}], f1s, threshold=0.8)["match"])
chk("cluster_match: frozen != LLM → no match",
    not cluster_match(lambda p: "wrong", lambda p: "Ada Lovelace", [{"q": "1"}, {"q": "2"}], f1s)["match"])

# 11) telemetry / observability (telemetry.py) — measures the cost curve from the event-log spine
from bobby_squad import Telemetry

te = Engine(tempfile.mkdtemp())
te.on("cap", lambda p: "LLM")
te.emit("cap", {"cap": "cap"}); te.emit("cap", {"cap": "cap"})            # 2 events served by the LLM fallback
tm = Telemetry(te.log)
cc0 = tm.cost_curve()
chk("telemetry: cost_curve all-LLM before any promotion", cc0["llm_frac"] == 1.0 and cc0["local_frac"] == 0.0)
te.promote("p", lambda p: "FROZEN", tags=["cap"], proof={"verdict": "WIRE"}, code="def h(p):\n    return 'FROZEN'\n")
for _ in range(3):
    te.emit("cap", {"cap": "cap"})                                        # 3 events now served locally
cc1 = tm.cost_curve()
chk("telemetry: cost curve BENDS DOWN after promotion (local_frac rises)",
    cc1["local_frac"] > cc0["local_frac"] and tm.promotions() == 1)
chk("telemetry: snapshot exposes all metrics",
    {"local_frac", "llm_frac", "ood_rate", "promotions", "move_entropy"} <= set(tm.snapshot()))

# 12) surrogate pruner (surrogate.py) — predict gain-proof score from cheap AST features, prune before the real eval
from bobby_squad import Surrogate, code_features
import random as _r

f_loop = code_features("def f(a):\n    return [x for x in a if x > 0]\n")
f_flat = code_features("def f(a):\n    return a\n")
chk("surrogate: code_features deterministic + structural (loops counted)",
    code_features("def f(a):\n    return a\n") == f_flat and f_loop[4] > f_flat[4])

rng = _r.Random(0)
feats, scores = [], []
for _ in range(20):
    calls = rng.randint(0, 10)
    feats.append([100.0, 10.0 + calls, 1.0, float(calls), float(rng.randint(0, 2)), 1.0, 2.0, 3.0])
    scores.append(20.0 + 8.0 * calls)                             # learnable: score = 20 + 8·calls
S = Surrogate(k=3).fit(feats[:15], scores[:15])
hi = [100.0, 20.0, 1.0, 10.0, 1.0, 1.0, 2.0, 3.0]
lo = [100.0, 10.0, 1.0, 0.0, 1.0, 1.0, 2.0, 3.0]
chk("surrogate: predicts higher score for the higher-signal candidate", S.predict(hi)[0] > S.predict(lo)[0])
kept = S.prune([lo, hi, [100.0, 15.0, 1.0, 5.0, 1.0, 1.0, 2.0, 3.0]], keep_frac=0.34, explore_frac=0.34)
chk("surrogate: prune keeps the predicted-best candidate (fail-safe)", 1 in kept)
rep = S.replay(feats, scores, keep_frac=0.5, winners_frac=0.3)
chk("surrogate: replay preserves winners on learnable data (recall ≥ 0.7)", rep["winner_recall"] >= 0.7)

# 13) blackboard-on-log (blackboard.py) — P1 coordination as a projection + Seam-1 claim/version
from bobby_squad import Blackboard

bbe = Engine(tempfile.mkdtemp())
bb = Blackboard(bbe.log)
r_a = bb.post("contract", {"task": "A"}, by="planner")
bb.post("finding", {"note": "x"}, by="researcher")
chk("blackboard: rows are a projection of the log", len(bb.rows()) == 2 and len(bb.rows("contract")) == 1)
chk("blackboard: first claim wins (Seam-1 optimistic concurrency)",
    bb.claim(r_a, "worker1") is True and bb.claim(r_a, "worker2") is False and bb.claimant(r_a) == "worker1")
chk("blackboard: claim reflected in the projected row", bb.rows("contract")[0]["claimed_by"] == "worker1")

print("\n== %d PASS / %d FAIL ==" % (len(ok), len(bad)))
sys.exit(1 if bad else 0)
