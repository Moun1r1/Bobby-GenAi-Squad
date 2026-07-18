#!/usr/bin/env python3
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import Engine, LLM
from bobby_squad.router import OODGate, ood_plugin_router
from bobby_squad.retrieval import default_embed

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


llm = LLM(extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # thinking OFF (fast, content non-empty)
calls = {"n": 0}


def fallback(payload):
    calls["n"] += 1
    return (llm([{"role": "user", "content": payload["q"]}], max_tokens=400, temperature=0.0) or "").strip()


eng = Engine(tempfile.mkdtemp())
eng.interceptors = [ood_plugin_router(default_embed)]            # OOD-aware router in front of the LLM
eng.on("memqa", fallback)                                        # the expensive last resort = the real LLM

train = ["When did Caroline go to the support group?", "Who is Melanie's sister?", "What job did John start?",
         "Where did Sarah move after college?", "When is Tom's anniversary?", "What pet did Lily adopt?",
         "How old is David's daughter?", "Which city did Emma visit?", "What hobby did Mark pick up?",
         "When did Anna change careers?"]
gate = OODGate.fit(default_embed(train))                          # competence region from REAL embeddings
eng.promote("memqa_frozen", lambda p: "FROZEN-ANSWER", tags=["memqa"],
            proof={"verdict": "WIRE", "competence": gate}, code="def h(p):\n    return 'FROZEN-ANSWER'\n")

# 1) in-distribution → frozen plugin, real LLM NOT called
before = calls["n"]
r1 = eng.emit("memqa", {"cap": "memqa", "q": "When did Rachel start her new diet?"})
chk("LIVE: in-distribution query → frozen plugin (real LLM NOT called)",
    r1 == "FROZEN-ANSWER" and calls["n"] == before)

# 2) OOD → abstain to the real LLM + OOD_DETECTED
before = calls["n"]
r2 = eng.emit("memqa", {"cap": "memqa", "q": "Write and unit-test a Python quicksort function."})
chk("LIVE: OOD query → abstains to the real LLM (fallback fired, real answer)",
    calls["n"] == before + 1 and isinstance(r2, str) and len(r2) > 0 and r2 != "FROZEN-ANSWER")
chk("LIVE: OOD_DETECTED event logged on the spine", any(e.kind == "OOD_DETECTED" for e in eng.log.read()))

print("\n== %d PASS / %d FAIL ==" % (len(ok), len(bad)))
print("real LLM calls made:", calls["n"], "| OOD answer preview:", repr((r2 or "")[:60]))
sys.exit(1 if bad else 0)
