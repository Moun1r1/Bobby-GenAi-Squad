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
# Endpoint + model come from BOBBY_LLM_URL / BOBBY_LLM_MODEL (see README §4). The reference numbers used a local
# Qwen3.6-35B-A3B served by sglang. Extraction is trivial → thinking OFF keeps answers short (a cost/throughput
# burn-in, not a reasoning test).
llm = LLM(temperature=0.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print("model:", llm.model, "| url:", llm.url, flush=True)

tickets = B.generate(seed=SEED)
B.write_dataset(tickets, os.path.join(OUT, "bobby_100_burn_in_v1.3.jsonl"))
print("dataset: 100 tickets written (40 A / 40 B / 20 C, OOD @ #80)", flush=True)

print("\n[1/2] ACR flywheel (distillation ON) — running 100 tickets live...", flush=True)
acr = B.run(tickets, llm, default_embed, distill=True, warmup=4, proof_threshold=0.9)
print("  done:", acr.summary(), flush=True)

print("\n[2/2] No-ACR control (distillation OFF) — running 100 tickets live...", flush=True)
ctrl = B.run(tickets, llm, default_embed, distill=False)
print("  done:", ctrl.summary(), flush=True)

# publish artifacts
acr.signals.to_csv(os.path.join(OUT, "acr_signals.csv"))
acr.signals.to_json(os.path.join(OUT, "acr_signals.json"))
ctrl.signals.to_csv(os.path.join(OUT, "control_signals.csv"))
report = B.render_report(acr, ctrl)
open(os.path.join(OUT, "report.txt"), "w").write(report)
svg = B.plot_svg(acr, ctrl, os.path.join(OUT, "golden_signals.svg"))
png = B.plot_png(acr, ctrl, os.path.join(OUT, "golden_signals.png"))

print("\n" + report, flush=True)
print("\nartifacts in", OUT, flush=True)
print("  SVG:", svg, flush=True)
print("  PNG:", png or "(matplotlib unavailable — SVG published instead)", flush=True)

# hard pass/fail per the v1.3 spec (single-seed live proof; N=5 CI is runnable via BURN_SEED sweep)
a, c = acr.summary(), ctrl.summary()
c_frozen_fail = any(r["cluster"] == "C" and r["route"] == "frozen" for r in acr.signals.rows)
checks = {
    "cost < 50% of No-ACR": a["total_tokens"] < 0.5 * c["total_tokens"],
    "accuracy >= control": a["accuracy"] >= c["accuracy"],
    "OOD tripwire held (no C on a frozen plugin)": not c_frozen_fail,
    "context < 5000 tok/step": max(acr.signals.series("context_size")) < 5000,
    ">=2 promotions (A,B distilled)": a["promotions"] >= 2,
}
print("\nPASS/FAIL:", flush=True)
for k, v in checks.items():
    print(("  PASS " if v else "  FAIL ") + k, flush=True)
sys.exit(0 if all(checks.values()) else 1)
