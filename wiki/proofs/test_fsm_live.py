#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import FSM, LLM

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


llm = LLM(extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # thinking off → fast YES/NO
fsm = FSM("pending").add("pending", "verified",
                         guard=("llm", "Does this statement claim a task is fully COMPLETE and TESTED? Statement: {text}"))

s1, how1 = fsm.next("pending", {"text": "All 44 unit tests pass and the feature is deployed and verified in production."},
                    llm=llm, max_tokens=8)
chk("LIVE fsm: LLM guard UNLOCKS 'verified' on a complete-and-tested statement", s1 == "verified" and how1 == "llm")

s2, how2 = fsm.next("pending", {"text": "I started sketching a rough draft but nothing works yet."}, llm=llm, max_tokens=8)
chk("LIVE fsm: LLM guard HOLDS (stuck) on an incomplete statement", s2 is None)

print("  complete →", (s1, how1), " | incomplete →", (s2, how2))
print("\n== %d PASS / %d FAIL ==" % (len(ok), len(bad)))
sys.exit(1 if bad else 0)
