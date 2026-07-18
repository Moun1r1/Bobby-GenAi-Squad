#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad.llm import LLM
from bobby_squad.retrieval import default_embed
from bobby_squad import ci95

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("BURN_SEEDS", "1,2,3,4,5").split(",")]
MIXED = os.environ.get("BURN_MIXED", "0") == "1"

llm = LLM(temperature=0.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # endpoint from BOBBY_LLM_URL/MODEL
gen = (lambda s: B.generate_mixed(seed=s, per=12)) if MIXED else (lambda s: B.generate(seed=s))
print("sweep: %d seeds | %s | model %s" % (len(SEEDS), "MIXED" if MIXED else "single-sector", llm.model), flush=True)

rows = []
for s in SEEDS:
    tickets = gen(s)
    acr = B.run(tickets, llm, default_embed, distill=True, warmup=4)
    ctrl = B.run(tickets, llm, default_embed, distill=False)
    at, ct = sum(acr.signals.series("token_cost")), sum(ctrl.signals.series("token_cost"))
    row = {"seed": s, "local_frac": acr.signals.local_fraction(),
           "token_reduction_pct": 100 * (1 - at / ct) if ct else 0.0,
           "acr_acc": acr.accuracy, "ctrl_acc": ctrl.accuracy, "promotions": acr.promotions,
           "acr_tokens": at, "ctrl_tokens": ct}
    rows.append(row)
    print("  seed %d: local=%.0f%% tok_red=%.0f%% acr_acc=%.0f%% ctrl_acc=%.0f%% promo=%d"
          % (s, row["local_frac"] * 100, row["token_reduction_pct"], row["acr_acc"] * 100,
             row["ctrl_acc"] * 100, row["promotions"]), flush=True)


def stat(key, scale=1.0, unit=""):
    xs = [r[key] * scale for r in rows]
    m, ci = ci95(xs)
    return "%.1f ± %.1f%s  (n=%d)" % (m, ci, unit, len(xs))


import json
json.dump({"seeds": SEEDS, "mixed": MIXED, "rows": rows}, open(os.path.join(OUT, "sweep_results.json"), "w"), indent=2)

print("\n" + "═" * 66, flush=True)
print("N=%d REPLICATION — mean ± 95%% CI  (%s)" % (len(SEEDS), "mixed" if MIXED else "single-sector"), flush=True)
print("═" * 66, flush=True)
print("  router_local_fraction : " + stat("local_frac", 100, "%"), flush=True)
print("  token_reduction       : " + stat("token_reduction_pct", 1, "%"), flush=True)
print("  ACR accuracy          : " + stat("acr_acc", 100, "%"), flush=True)
print("  No-ACR accuracy       : " + stat("ctrl_acc", 100, "%"), flush=True)
print("  promotions            : " + stat("promotions"), flush=True)
print("═" * 66, flush=True)
print("results → out/sweep_results.json", flush=True)
