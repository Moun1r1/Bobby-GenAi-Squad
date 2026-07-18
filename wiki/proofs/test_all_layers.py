#!/usr/bin/env python3
import hashlib
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import (Engine, Blackboard, Telemetry, Surrogate, code_features, FSM, Scenario, DataCollector,
                         harness_verdict, Plugin)
from bobby_squad.router import OODGate, competence_router
from bobby_squad import burn_in as B

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def embed(texts):
    return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:32]] for t in texts]


class MockLLM:
    last_usage = {"total_tokens": 90}

    def __call__(self, messages, max_tokens=200, temperature=0.0):
        c = messages[-1]["content"].lower()
        if "severity" in c:                                  # the only irreducible hop -> LLM
            return "high" if ("outage" in c or "critical" in c) else "low"
        return ""


EX_ASK = "extract every error code from the incident log"
CT_ASK = "count how many error codes are present"
AS_ASK = "assess the operational severity of the incident note"


def build_engine(root, distill=True):
    eng = Engine(root)
    eng.interceptors = [competence_router(embed)]

    def fallback(p):
        return (MockLLM()([{"role": "user", "content": p["prompt"]}]) or "").strip()
    eng.on("task", fallback)
    if distill:
        eng.promote("extract_err", B.make_extractor(r"ERR-\d+"), tags=["task"],
                    proof={"verdict": "WIRE", "competence": OODGate.fit(embed([EX_ASK] * 4))},
                    code="import re\ndef h(p):\n    return '\\n'.join(re.findall(r'ERR-\\d+', p['blob']))\n")
        eng.promote("count_err", B.make_aggregator("count"), tags=["task"],
                    proof={"verdict": "WIRE", "competence": OODGate.fit(embed([CT_ASK] * 4))},
                    code="def h(p):\n    import re\n    return str(len(re.findall(r'\\d+', p['blob'])))\n")
    return eng


def run_triage(eng, blob, note):
    """The multi-hop reasoning flow, sequenced by an FSM, each hop routed by the engine, coordinated on a blackboard."""
    fsm = FSM("plan")
    fsm.add("plan", "extract", None)
    fsm.add("extract", "count", None)
    fsm.add("count", "assess", None)
    fsm.add("assess", "decide", None)
    fsm.add("decide", "escalate", lambda ctx: ctx.get("count", 0) >= 2)       # deterministic guard on the reasoning
    fsm.add("decide", "monitor", lambda ctx: ctx.get("count", 0) < 2)
    bb = Blackboard(eng.log)
    ctx, state, hops = {}, "plan", 0
    while state not in ("escalate", "monitor") and hops < 8:
        hops += 1
        if state == "extract":
            codes = eng.emit("task", {"cap": "task", "q": EX_ASK, "blob": blob,
                                      "prompt": "Extract every ERR-#### code:\n" + blob})
            ctx["codes"] = codes or ""
            bb.post("extract", ctx["codes"], by="extractor")
        elif state == "count":
            cnt = eng.emit("task", {"cap": "task", "q": CT_ASK, "blob": ctx.get("codes", ""),
                                    "prompt": "Count the lines:\n" + ctx.get("codes", "")})
            m = re.findall(r"\d+", str(cnt))
            ctx["count"] = int(m[0]) if m else 0
            bb.post("count", str(ctx["count"]), by="counter")
        elif state == "assess":
            sev = eng.emit("task", {"cap": "task", "q": AS_ASK, "blob": note,
                                    "prompt": "Assess severity (one word) of this note:\n" + note})
            ctx["severity"] = (sev or "").strip()
            bb.post("assess", ctx["severity"], by="assessor")
        state, _how = fsm.next(state, ctx, MockLLM())
    row = bb.post("decision", {"action": state, "count": ctx.get("count"), "severity": ctx.get("severity")},
                  by="planner")
    ctx["decision"] = state
    ctx["verified"] = bb.claim(row, "verifier")              # a second agent claims the decision (first-claim-wins)
    return ctx


# ── 1) run the full multi-hop reasoning flow once ───────────────────────────────────────────────────────
BLOB = "boot ok\nERR-4041 disk\nheartbeat\nERR-5012 net\ncache warm\nERR-4041 disk"    # 3 codes -> escalate
NOTE = "Customer-facing outage across two regions; checkout is down."
eng = build_engine(tempfile.mkdtemp(), distill=True)
ctx = run_triage(eng, BLOB, NOTE)
chk("multi-hop flow completed to a terminal FSM state", ctx["decision"] in ("escalate", "monitor"))
chk("EXTRACT hop found the error codes (frozen plugin)", "ERR-4041" in ctx["codes"] and "ERR-5012" in ctx["codes"])
chk("COUNT hop counted the codes (frozen plugin)", ctx["count"] >= 2)
chk("FSM deterministic guard fired: count>=2 -> escalate", ctx["decision"] == "escalate")
chk("ASSESS hop (irreducible) produced a severity via the LLM", ctx.get("severity") in ("high", "low"))
chk("Blackboard: verifier claimed the decision row (Seam-1 optimistic concurrency)", ctx["verified"] is True)

# ── 2) assert EACH plane left its fingerprint in the single event log (P0 projection) ────────────────────
kinds = [e.kind for e in eng.log.read()]
chk("P0 spine: EXTRACT + COUNT served locally (task.handled by competence_router)",
    sum(1 for e in eng.log.read("task.handled") if e.payload.get("by") == "competence_router") >= 2)
chk("OOD tripwire: ASSESS abstained -> OOD_DETECTED + served by fallback",
    ("OOD_DETECTED" in kinds) and any(e.payload.get("by") == "fallback" for e in eng.log.read("task.handled")))
chk("P1 blackboard: posts + a claim are on the log", "board.post" in kinds and "board.claim" in kinds)
chk("registry: two frozen plugins promoted (SKILL_PROMOTED)", kinds.count("SKILL_PROMOTED") == 2)

# ── 3) P5 observability: the cost curve is a projection of that same log ─────────────────────────────────
tel = Telemetry(eng.log)
cc = tel.cost_curve()
snap = tel.snapshot()
chk("P5 telemetry: cost curve shows local work (frozen hops) AND llm work (assess)",
    cc["local_frac"] > 0 and cc["llm_frac"] > 0)
chk("P5 telemetry: snapshot exposes ood_rate + promotions", snap["promotions"] == 2 and snap["ood_rate"] > 0)

# ── 4) Surrogate (P4 pre-filter): predict a plugin's gain-proof score from AST features, prune ───────────
srcs = ["import re\ndef h(p): return re.findall('ERR-\\d+', p['blob'])\n",
        "def h(p): return str(len(p['blob']))\n",
        "def h(p):\n    total = 0\n    for x in p['blob'].split():\n        total += 1\n    return total\n",
        "def h(p): return sum(int(x) for x in p['blob'].split() if x.isdigit())\n"]
feats = [code_features(s) for s in srcs]
scores = [0.95, 0.10, 0.20, 0.88]                            # observed gain-proof scores (2 good, 2 dud)
sur = Surrogate(k=2)
sur.fit(feats, scores)
kept = sur.prune(feats, keep_frac=0.5, explore_frac=0.0)
rep = sur.replay(feats, scores, keep_frac=0.5, winners_frac=0.5)
chk("P4 surrogate: predicts + prunes candidates (keeps a subset)", 0 < len(kept) < len(feats))
chk("P4 surrogate: replay reports winner-recall + evals-saved", "winner_recall" in rep and "evals_saved_frac" in rep)

# ── 5) P4 evaluation: harness replicates the whole reasoning scenario -> mean +/- 95% CI + verdict ──────
def metric_local(scn):
    e = build_engine(tempfile.mkdtemp(), distill=scn.params["distill"])
    run_triage(e, BLOB, NOTE)
    return Telemetry(e.log).cost_curve()["local_frac"]


dc = DataCollector()
treat = dc.run("acr", metric_local, Scenario("acr", 1, {"distill": True}), replications=5)
base = dc.run("nodistill", metric_local, Scenario("base", 2, {"distill": False}), replications=5)
v = harness_verdict(treat, base)
chk("P4 harness: N=5 replication with a 95% CI on local_fraction", treat.n == 5 and treat.ci >= 0)
chk("P4 harness: distillation beats the no-distill baseline (CI-separated)",
    treat.mean > base.mean and v["verdict"] in ("WIRE", "MARGINAL"))

# ── 6) scheduler + dedup primitives participate ─────────────────────────────────────────────────────────
eng.jobs.submit("triage_probe", "printf done", timeout=30)
st = eng.jobs.wait("triage_probe", poll=0.05, timeout=15)
chk("scheduler: JobRegistry ran a named probe job to completion (daemonless sentinel)", st.get("status") == "done")
dup = Plugin(name="extract_err_twin", handler=B.make_extractor(r"ERR-\d+"), tags=frozenset(["task"]),
             proof={"verdict": "WIRE", "competence": OODGate.fit(embed([EX_ASK] * 4))})
accepted = eng.registry.register(
    dup, code="import re\ndef h(p):\n    return '\\n'.join(re.findall(r'ERR-\\d+', p['blob']))\n")
chk("dedup: a functional-twin plugin is rejected by AstDedup at register", accepted is False)

print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("ALL PLANES EXERCISED: P0 spine, P1 blackboard, P2 control(FSM)+ACR router+OOD, "
      "P4 evaluation(harness+surrogate), P5 observability + scheduler + dedup")
