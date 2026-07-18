#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import burn_in as B
from bobby_squad.llm import LLM
from bobby_squad.retrieval import default_embed

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
SEED = int(os.environ.get("BURN_SEED", "1"))
PER = int(os.environ.get("BURN_PER", "12"))

llm = LLM(temperature=0.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # endpoint from BOBBY_LLM_URL/MODEL   # thinking OFF — trivial per-ticket tasks
print("model:", llm.model, "| families:", len(B.FAMILIES), "| per:", PER, flush=True)

tickets = B.generate_mixed(seed=SEED, per=PER)
B.write_dataset(tickets, os.path.join(OUT, "bobby_mixed_burn_in.jsonl"))
print("dataset:", len(tickets), "tickets across", len(set(t["kind"] for t in tickets)), "modalities", flush=True)

print("\n[1/2] ACR flywheel (distillation ON) — live...", flush=True)
acr = B.run(tickets, llm, default_embed, distill=True, warmup=4, proof_threshold=0.9)
print("  done:", acr.summary(), flush=True)
print("\n[2/2] No-ACR control (distillation OFF) — live...", flush=True)
ctrl = B.run(tickets, llm, default_embed, distill=False)
print("  done:", ctrl.summary(), flush=True)

acr.signals.to_csv(os.path.join(OUT, "mixed_acr_signals.csv"))
acr.signals.to_json(os.path.join(OUT, "mixed_acr_signals.json"))
ctrl.signals.to_csv(os.path.join(OUT, "mixed_control_signals.csv"))
report = B.render_report(acr, ctrl)
open(os.path.join(OUT, "mixed_report.txt"), "w").write(report)
B.plot_svg(acr, ctrl, os.path.join(OUT, "mixed_golden_signals.svg"))
print("\n" + report, flush=True)

# per-modality breakdown: reducible → frozen; prose → LLM floor
print("\nPER-MODALITY (route mix + accuracy):", flush=True)
by = {}
for r in acr.signals.rows:
    d = by.setdefault(r["kind"], {"n": 0, "frozen": 0, "correct": 0})
    d["n"] += 1
    d["frozen"] += 1 if r["route"] == "frozen" else 0
    d["correct"] += r["correct"]
for kind in ["extract", "math", "code", "image", "algo", "prose"]:
    d = by.get(kind)
    if d:
        print("  %-8s n=%3d  frozen=%3d (%3.0f%%)  acc=%3.0f%%%s"
              % (kind, d["n"], d["frozen"], 100 * d["frozen"] / d["n"], 100 * d["correct"] / d["n"],
                 "   ← irreducible: stays on the LLM by design" if kind == "prose" else ""), flush=True)

a, c = acr.summary(), ctrl.summary()
prose_frozen = any(r["kind"] == "prose" and r["route"] == "frozen" for r in acr.signals.rows)
checks = {
    "cross-modal cost < No-ACR": a["total_tokens"] < c["total_tokens"],
    "accuracy >= control": a["accuracy"] >= c["accuracy"],
    ">=8 reducible families distilled": a["promotions"] >= 8,
    "prose NEVER distilled (irreducible floor held)": not prose_frozen,
    "context < 5000 tok/step": max(acr.signals.series("context_size")) < 5000,
}
print("\nPASS/FAIL:", flush=True)
for k, v in checks.items():
    print(("  PASS " if v else "  FAIL ") + k, flush=True)
sys.exit(0 if all(checks.values()) else 1)
