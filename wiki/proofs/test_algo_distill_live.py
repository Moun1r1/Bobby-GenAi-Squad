#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad.llm import LLM
from bobby_squad.retrieval import default_embed

llm = LLM(url="http://localhost:8002/v1/chat/completions", model="claude-opus-4-8", temperature=0.0,
          extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def _tickets(fid, n, seed):
    fam = next(f for f in B.FAMILIES if f[0] == fid)
    _, kind, ask, build = fam
    out = []
    for i in range(n):
        blob, gold = build(B._rng(seed * 100 + i))
        out.append({"ticket_id": "%s-%d" % (fid, i), "cluster": fid, "kind": kind, "cap": "task", "ask": ask,
                    "prompt": B._PROMPT_TMPL[kind].format(ask=ask, blob=blob), "blob": blob, "gold": gold})
    return out


stream = []
rom, luhn = _tickets("algo_roman", 10, 1), _tickets("algo_luhn", 10, 2)
for i in range(10):
    stream += [rom[i], luhn[i]]

print("model:", llm.model, "| 20 algorithmic tickets (roman + luhn), warmup 4", flush=True)
res = B.run(stream, llm, default_embed, distill=True, warmup=4, proof_threshold=0.9)
print("summary:", res.summary(), flush=True)

for p in res.engine.registry.active():
    src = getattr(p.handler, "_src", None)
    if src:
        print("\n=== FROZEN LLM-AUTHORED CODE: %s ===" % p.name, flush=True)
        print(src, flush=True)

late = [r for r in res.signals.rows if r["i"] >= 8]
frozen_late = [r for r in late if r["route"] == "frozen"]
checks = {
    "model wrote >=1 code plugin that passed gain-proof": res.promotions >= 1,
    "frozen plugins are compiled code (carry source)": sum(1 for p in res.engine.registry.active()
                                                            if getattr(p.handler, "_src", None)) >= 1,
    "post-warmup tickets served by frozen code at 0 tokens": frozen_late and all(r["token_cost"] == 0 for r in frozen_late),
    "frozen-served tickets are correct": frozen_late and all(r["correct"] == 1 for r in frozen_late),
    "accuracy >= 0.9": res.accuracy >= 0.9,
}
print("\nPASS/FAIL:", flush=True)
for k, v in checks.items():
    print(("  PASS " if v else "  FAIL ") + k, flush=True)
sys.exit(0 if all(checks.values()) else 1)
