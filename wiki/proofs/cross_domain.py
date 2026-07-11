"""arbitrary_requests — a GENERALITY test: throw wildly different, unbounded requests at the squad and check it can
behave ANY way the request asks, from ONE unchanged engine. No per-task prompt, no persona, no scripted pipeline —
only the request changes; the swarm self-selects how to handle it.

The requests deliberately span domains, formats, stances and audiences (explain / create / argue-against / design /
teach-a-child / compare) — none of them are about this repo — so a pass proves the generative engine isn't overfit
to the tasks we built (idea-farming / coverage / dev), it does what it's told.

Per request: a small self-organizing pair (shared identity = "do EXACTLY what the request asks"), the 2nd sees the
1st's answer, each self-selects its move. We capture the moves it chose + a cheap relevance signal + the output, so
you can see it behaved the requested way. Run:
  GA_LLM_URL="http://localhost:8002/v1/chat/completions" GA_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
  python3 examples/arbitrary_requests.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from bobby_squad import Agent, SelfCore, ReadOnlyTools, BehaviorTrace, stream_observer   # noqa: E402
from bobby_squad import LLM                                           # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bobby_squad")

# Arbitrary, unbounded — different behavior each time. NONE is about this repo.
REQUESTS = [
    "Explain how a Bloom filter works and give ONE real use case. Be concise.",
    "Write a 3-line haiku about entropy.",
    "Argue AGAINST unit testing — steelman the strongest contrarian case in 3 sentences.",
    "Design a token-bucket rate limiter and name its 3 parameters.",
    "Explain recursion to a 5-year-old in exactly 2 sentences.",
    "Compare optimistic vs pessimistic locking, then give a one-line rule for when to use each.",
]

# ONE identity for all requests — the REQUEST defines the behavior; nothing here scripts what to say.
IDENTITY = "a capable generalist who does EXACTLY what a request asks — the request defines your behavior"
CONSTRAINTS = ["do precisely what the request asks, in the form it asks for", "no preamble, no meta-commentary"]

def grade(llm, req, out):
    """Verify by OUTCOME, the right way for open-ended behavior: a STRICT judge (a separate instance, not the actor)
    checks whether the answer actually satisfies the request — INCLUDING any format / length / stance constraint
    (exact line count, sentence count, number of items, required stance). Keyword overlap was the wrong verifier: a
    correct haiku doesn't contain the word 'haiku'. Returns (passed, verdict_text)."""
    q = (f"REQUEST:\n{req}\n\nANSWER:\n{out}\n\nJudge STRICTLY: does the ANSWER fully and correctly satisfy the "
         "REQUEST, including EVERY format/length/stance constraint it states (e.g. exact number of lines, sentences, "
         "or items; a required stance; a target audience)? If any part is unmet, it FAILS. Reply with ONLY 'PASS' or "
         "'FAIL: <short reason>'.")
    r = (llm([{"role": "user", "content": q}], max_tokens=40) or "").strip()
    return r.upper().startswith("PASS"), r


def handle(llm, req, watch=False):
    trace = BehaviorTrace("responder", echo=stream_observer if watch else None)
    core = SelfCore(identity=IDENTITY, goal=req, constraints=CONSTRAINTS)
    agents = [Agent(core, llm=llm, tools=ReadOnlyTools(ROOT), window=4, pinned=True, name=f"r{i}", observer=trace)
              for i in range(2)]
    out = ""
    for i, ag in enumerate(agents):
        if i > 0 and out:
            ag.observe(f"A peer already answered this request:\n{out}\nImprove it or leave it if it's already right.")
        res = ag.research_cycle(max_steps=1, max_rounds=3, replans=1)
        r = (res.get("results") or [{}])[-1].get("result", "") if res.get("results") else ""
        if r:
            out = r
    return trace.moves, out.strip()


def main():
    llm = LLM(temperature=0.5, timeout=120)
    print("=== ARBITRARY-REQUEST GENERALITY TEST — one engine, any behavior the request asks ===\n", flush=True)
    rows = []
    for req in REQUESTS:
        moves, out = handle(llm, req)
        passed, verdict = grade(llm, req, out)
        rows.append((req, moves, passed, verdict, out))
        print(f"── REQUEST: {req}", flush=True)
        print(f"   self-selected moves: {moves}  · judge: {verdict}", flush=True)
        print(f"   OUTPUT: {' '.join(out.split())[:220]}\n", flush=True)

    ok = sum(1 for r in rows if r[2])
    allmoves = sorted({m for r in rows for m in r[1]})
    print("=== SUMMARY ===")
    print(f"  {ok}/{len(rows)} requests satisfied (STRICT judge — content AND format/stance constraints)")
    print(f"  distinct moves the swarm self-selected across all requests: {allmoves}")
    print("  → same engine, no per-task prompt; behavior came from the request, not a script.")


if __name__ == "__main__":
    main()
