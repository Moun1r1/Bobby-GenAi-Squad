import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import (Agent, SelfCore, ReadOnlyTools, BehaviorTrace, MetaTools,   # noqa: E402
                                investigate, stream_observer)
from bobby_squad import LLM                                          # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bobby_squad")                                     # the repo the squad works on (its own subject)
OUT = os.path.abspath(os.path.join(HERE, "..", "out", "BEHAVIORAL_REVIEW.md"))

# ONE shared identity, no roles — capability from SELF + TOOLS + move-space (the generative rule). The goal steers
# toward diverse investigation so there IS behavior worth reviewing; it never scripts a specific move.
IDENTITY = ("a researcher in a small lab, working alongside peers who share one memory, free to investigate any part "
            "of this generative-agent repository")
GOAL = ("investigate this repository's real capabilities and find the next thing worth improving. Follow your own "
        "curiosity across the whole system — you choose what to look at and how")
CONSTRAINTS = ["ground every claim in the real code you actually read",
               "pursue what YOU find most promising; do not wait to be told what to do"]

# The review task points the agent at the behavioral TOOL and demands signal-grounded judgment — it does NOT tell the
# agent what the bias/frontier IS (that would be the static-prompt trap). The naming is the agent's own.
REVIEW_TASK = (
    "You are reviewing how a PEER agent behaved — to detect the frontier and biases of its intelligence. "
    "First call peers(), then call review_peer('{peer}') to get its REAL behavioral signals. From that evidence ONLY, "
    "state:\n"
    "  1. BIAS — the single strongest way its attention was over-concentrated or self-repeating (cite the exact "
    "signal: move-entropy, area-concentration, or repetition-rate).\n"
    "  2. FRONTIER — where its intelligence STOPPED adding anything new (cite the novelty curve / where it collapsed "
    "to 0 / abstentions).\n"
    "  3. ONE move that would break the bias and push past the frontier.\n"
    "Every claim MUST cite a number from review_peer. Do not invent behavior you cannot see in the signals. "
    "Answer in 4-6 lines.")


def run(n_agents=3, cycles=2):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    llm = LLM(temperature=0.6, timeout=200)
    tools = ReadOnlyTools(ROOT)
    traces = {}
    agents = []
    for i in range(n_agents):
        name = f"researcher-{i}"
        tr = BehaviorTrace(name, echo=stream_observer)          # record AND watch live
        traces[name] = tr
        agents.append(Agent(SelfCore(IDENTITY, GOAL, CONSTRAINTS), llm=llm, window=4, pinned=True,
                            tools=tools, name=name, observer=tr))

    print(f"[review-squad] {n_agents} researchers · Phase 1 WORK (generative) → Phase 2 SELF-REVIEW\n", flush=True)
    t0 = time.time()
    # ── Phase 1 · WORK — behavior accumulates in each trace ──────────────────────────────────────────
    for c in range(cycles):
        for ag in agents:
            ag.research_cycle(max_steps=2, max_rounds=6)
        print(f"[review-squad] work round {c+1}/{cycles} done", flush=True)

    # ── Phase 2 · SELF-REVIEW — each agent reviews the NEXT peer's behavior via the tool ─────────────
    meta = MetaTools(traces)
    print("\n[review-squad] === PHASE 2 · behavioral self-review ===\n", flush=True)
    reviews = []
    for i, ag in enumerate(agents):
        peer = agents[(i + 1) % n_agents].name
        assessment = investigate(llm, REVIEW_TASK.format(peer=peer), meta, max_rounds=4, max_tokens=500)
        assessment = (assessment[0] if isinstance(assessment, tuple) else assessment or "").strip()
        reviews.append((ag.name, peer, assessment))
        print(f"── {ag.name} reviewing {peer} ──\n{assessment}\n", flush=True)

    dt = time.time() - t0
    with open(OUT, "w") as f:
        f.write("# Behavioral self-review — the squad detecting its own bias & frontier\n\n")
        f.write(f"_{n_agents} researchers · {cycles} work cycles · {dt:.0f}s. Each reviewed a peer's REAL behavioral "
                "trace via the `review_peer` tool; every claim is grounded in a deterministic signal._\n\n")
        for name, tr in traces.items():
            s = tr.signals()
            f.write(f"## {name} — behavioral signals (the evidence)\n\n")
            f.write(f"- dominant move `{s['dominant_move']}` · move-entropy **{s['move_entropy']}** "
                    f"(0 = collapsed onto one move)\n")
            f.write(f"- dominant area `{s['dominant_area']}` · area-concentration **{s['area_concentration']}**\n")
            f.write(f"- repetition-rate **{s['repetition_rate']}** · abstentions **{s['abstentions']}** · "
                    f"novelty curve `{s['novelty_curve']}` · frontier @ cycle `{s['frontier_cycle']}`\n")
            f.write("- flags: " + "; ".join(tr.flags()) + "\n\n")
        f.write("## Peer reviews (the detection — agent's words, signal-grounded)\n\n")
        for name, peer, a in reviews:
            f.write(f"### {name} → {peer}\n\n{a}\n\n")
    print(f"[review-squad] done in {dt:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    run(n_agents=int(os.environ.get("RD_AGENTS", "3")), cycles=int(os.environ.get("RD_CYCLES", "2")))
