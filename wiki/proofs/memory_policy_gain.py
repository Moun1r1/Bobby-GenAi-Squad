import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from bobby_squad import confirm_gain                                    # noqa: E402
from bobby_squad import SemanticMemory                        # noqa: E402
from bobby_squad import LLM                               # noqa: E402

CAP = 40                                                             # store capacity < 100 topics → real pressure
ITEM = lambda t: f"Topic {t}: the answer is CODE{t}."                # one knowledge item per topic
QUERY = lambda t: f"Topic {t}: what is the recorded answer?"


def build(seed, n_topics=100, used_frac=0.4, critical_frac=0.1, predictive=True):
    """`predictive=True`: usage during the stream is on the SAME topics that are tested at the end (realistic — the
    knowledge you keep referencing is the knowledge you'll need). `predictive=False` (negative control): usage is on
    DISJOINT topics, so learned value points at the WRONG items — a non-leaking benchmark must show NO evolved win."""
    rng = random.Random(seed)
    topics = list(range(n_topics))
    tested = set(rng.sample(topics, int(n_topics * used_frac)))
    critical = set(rng.sample(topics, max(1, int(n_topics * critical_frac))))
    if predictive:
        feedback = list(tested)
    else:
        others = [t for t in topics if t not in tested and t not in critical]
        feedback = rng.sample(others, min(len(others), len(tested)))
    order = topics[:]; rng.shuffle(order)
    events = []
    for t in order:
        events.append(("obs", t))
        if feedback and rng.random() < 0.5:                         # interleaved usage → feedback for the value policy
            events.append(("use", rng.choice(feedback)))
    return tested, critical, events


def load(policy, seed, predictive=True):
    used, critical, events = build(seed, predictive=predictive)
    mem = SemanticMemory(tau=0.999, k=4, capacity=CAP, policy=policy)   # tau high: capacity, not novelty, is the test
    for kind, t in events:
        if kind == "obs":
            mem.add(ITEM(t), critical=(t in critical))
        else:
            mem.retrieve(QUERY(t))                                  # usage teaches "value" (no-op for "fifo")
    return mem, used, critical


def retention(mem, used):
    docs = mem.r.docs
    return sum(1 for t in used if any(f"Topic {t}:" in d for d in docs)) / max(1, len(used))


def answer(llm, query, ctx):
    c = "\n".join(f"- {x}" for x in ctx) or "(nothing retrieved)"
    p = (f"Use ONLY the context to answer. Context:\n{c}\n\n{query}\n"
         "Reply with just the code (e.g. CODE12) or UNKNOWN.")
    return (llm([{"role": "user", "content": p}], max_tokens=16) or "").upper()


def generation(llm, mem, used):
    ok = 0
    for t in sorted(used):
        ctx = mem.retrieve(QUERY(t))
        if re.search(rf"CODE{t}\b", answer(llm, QUERY(t), ctx)):
            ok += 1
    return ok / max(1, len(used))


def main():
    seeds = range(8)
    fifo = [load("fifo", s) for s in seeds]
    value = [load("value", s) for s in seeds]
    fr = statistics.mean(retention(m, u) for m, u, _ in fifo)
    vr = statistics.mean(retention(m, u) for m, u, _ in value)
    floor = all(all(any(f"Topic {t}:" in d for d in m.r.docs) for t in c) for m, _, c in fifo + value)
    print(f"capacity={CAP}/100 · seeds={len(list(seeds))}")
    print(f"[retention]  FIFO {fr:.3f}   ·   EvolvedValue {vr:.3f}   ·   deterministic floor held: {floor}")
    confirm_gain("memory policy — RETENTION (evolved vs fifo)", lambda: fr, lambda: vr, higher_is_better=True)

    # NEGATIVE CONTROL — usage NON-predictive of the held-out test. If evolved still wins here, the benchmark is
    # leaking (value can't help when it points at the wrong items). Expect DELETE/MARGINAL — that VALIDATES the test.
    nf = statistics.mean(retention(m, u) for m, u, _ in (load("fifo", s, predictive=False) for s in seeds))
    nv = statistics.mean(retention(m, u) for m, u, _ in (load("value", s, predictive=False) for s in seeds))
    print(f"[neg-control] FIFO {nf:.3f}   ·   EvolvedValue {nv:.3f}   (usage non-predictive → expect NO win)")
    confirm_gain("NEGATIVE CONTROL — retention, non-predictive usage (must NOT be WIRE)",
                 lambda: nf, lambda: nv, higher_is_better=True)

    # --- LLM generation layer: real end-to-end quality on ONE model instance (test mode) ---
    llm = LLM(temperature=0.0, timeout=60)
    probe = (llm([{"role": "user", "content": "Reply with the single word READY."}], max_tokens=5) or "").upper()
    if "READY" not in probe:
        print("[generation] LLM endpoint unreachable — skipping generation layer (set GA_LLM_URL). retention stands.")
        return
    mf, uf, _ = load("fifo", 0)
    mv, uv, _ = load("value", 0)
    gf = generation(llm, mf, uf)
    gv = generation(llm, mv, uv)
    print(f"[generation] FIFO {gf:.3f}   ·   EvolvedValue {gv:.3f}   (real LLM answers from retrieved memory)")
    confirm_gain("memory policy — GENERATION quality (evolved vs fifo)", lambda: gf, lambda: gv, higher_is_better=True)


if __name__ == "__main__":
    main()
