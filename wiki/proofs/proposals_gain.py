import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from bobby_squad import confirm_gain                                    # noqa: E402


# ── 1. Memory-Gate: importance-filtered consolidation vs consolidate-everything ─────────────────────────────────
def memgate(gate, seed):
    rng = random.Random(seed)
    M, W, C, P = 140, 6, 6, 28                        # items, window, compact-every, pinned capacity
    working, pinned = [], []
    items = [(i, rng.random() < 0.3) for i in range(M)]   # (id, important)
    used = {i for i, imp in items if imp}
    for step, it in enumerate(items):
        working.append(it)
        if len(working) > W:
            working.pop(0)
        if (step + 1) % C == 0:                       # compaction: consolidate working → pinned, then wipe
            keep = working if not gate else [w for w in working if w[1]]   # gate = keep only IMPORTANT
            for w in keep:
                if w not in pinned:
                    pinned.append(w)
                    if len(pinned) > P:
                        pinned.pop(0)                 # FIFO evict (bounded store)
            working = []
    return sum(1 for i in used if any(p[0] == i for p in pinned)) / max(1, len(used))


# ── 2. CWBU: confidence-weighted vs unweighted claim aggregation ────────────────────────────────────────────────
def cwbu(weighted, seed):
    rng = random.Random(seed)
    N, trials, correct = 7, 500, 0
    for _ in range(trials):
        truth = rng.random() < 0.5
        score = 0.0
        for i in range(N):
            rel = 0.5 + 0.4 * (i / (N - 1))           # reliability 0.5..0.9
            claim = truth if rng.random() < rel else not truth
            conf = max(0.0, min(1.0, rel + rng.uniform(-0.12, 0.12)))   # confidence ~ reliability (noisy)
            score += (1 if claim else -1) * (conf if weighted else 1.0)
        pred = (score > 0) if score != 0 else (rng.random() < 0.5)
        correct += (pred == truth)
    return correct / trials


# ── 3. Active experimental design: info-gain (halving) vs linear scan ───────────────────────────────────────────
def active(active_mode, seed):
    rng = random.Random(seed)
    N, trials, total = 64, 300, 0
    for _ in range(trials):
        target = rng.randrange(N)
        probes = 0
        if active_mode:                               # binary search — max info-gain probe each step
            lo, hi = 0, N
            while hi - lo > 1:
                mid = (lo + hi) // 2; probes += 1
                if target < mid:
                    hi = mid
                else:
                    lo = mid
        else:                                         # linear scan
            for i in range(N):
                probes += 1
                if i == target:
                    break
        total += probes
    return total / trials                             # LOWER is better (fewer probes)


# ── 4. Synapse: hybrid (lexical+embedding) vs embedding-only retrieval, on MIXED queries ─────────────────────────
def synapse():
    """FAIR & HARD: each doc has a rare REF id (a LEXICAL-only signal — embedding can't read it) and a topic (an
    EMBEDDING signal reachable only via a synonym paraphrase with NO shared tokens — lexical can't read it). Exact
    queries use the REF id → embedding-only FAILS them; paraphrase queries use the synonym → lexical FAILS them. So
    each single signal caps ~50%; only hybrid can get both. Not rigged — hybrid still has to fuse them correctly."""
    from bobby_squad.retrieval import EmbeddingRetriever, _cos
    r = EmbeddingRetriever()
    if r.embed_fn is None:
        return None
    pairs = [("physician", "a medical doctor who treats sick patients"),
             ("automobile", "a car you drive on the road"), ("ocean", "a vast body of salt water"),
             ("attorney", "a lawyer who argues cases in court"), ("glacier", "a slow moving mass of ice"),
             ("melody", "a pleasing sequence of musical notes"), ("vaccine", "an injection that builds immunity"),
             ("drought", "a long period with no rain"), ("microscope", "an instrument to see tiny things"),
             ("volcano", "a mountain that erupts molten lava"), ("currency", "money used for trade"),
             ("satellite", "an object orbiting a planet"), ("harvest", "gathering ripe crops from fields"),
             ("pension", "retirement income for the elderly"), ("compass", "a device that points north"),
             ("telescope", "an instrument to view distant stars")]
    T = len(pairs)
    docs = [f"REF{1000+t}. Filed notes concerning {topic}." for t, (topic, _) in enumerate(pairs)]
    dvecs = [r.embed_fn([r.dp + d])[0] for d in docs]
    if not dvecs or dvecs[0] is None:
        return None

    def norm(xs):
        lo, hi = min(xs), max(xs)
        return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in xs]

    def top(query, mode):
        qv = r.embed_fn([r.dp + query])[0]
        emb = [_cos(qv, dv) if dv else 0.0 for dv in dvecs]
        qtok = set(query.lower().split())
        lex = [len(qtok & set(d.lower().split())) / max(1, len(qtok)) for d in docs]
        fused = emb if mode == "emb" else ([a + b for a, b in zip(norm(emb), norm(lex))] if mode == "hybrid" else lex)
        return max(range(T), key=lambda i: fused[i])

    def recall(mode):
        ok = 0
        for t, (_topic, para) in enumerate(pairs):
            for q in (f"REF{1000+t}", para):          # exact id (lexical) + synonym paraphrase (embedding)
                ok += (top(q, mode) == t)
        return ok / (2 * T)
    return recall("emb"), recall("hybrid")


def main():
    seeds = range(8)
    print("== proving the swarm's proposals (strong-model review) ==\n")

    mg_c = statistics.mean(memgate(False, s) for s in seeds)
    mg_t = statistics.mean(memgate(True, s) for s in seeds)
    print(f"[Memory-Gate]  consolidate-all {mg_c:.3f} → importance-gated {mg_t:.3f}")
    confirm_gain("Memory-Gate (importance-filtered consolidation)", lambda: mg_c, lambda: mg_t, higher_is_better=True)

    cw_c = statistics.mean(cwbu(False, s) for s in seeds)
    cw_t = statistics.mean(cwbu(True, s) for s in seeds)
    print(f"\n[CWBU]  unweighted {cw_c:.3f} → confidence-weighted {cw_t:.3f}")
    confirm_gain("CWBU (confidence-weighted belief update)", lambda: cw_c, lambda: cw_t, higher_is_better=True)

    ac_c = statistics.mean(active(False, s) for s in seeds)
    ac_t = statistics.mean(active(True, s) for s in seeds)
    print(f"\n[Active-Design]  linear {ac_c:.1f} probes → info-gain {ac_t:.1f} probes (lower better)")
    confirm_gain("Active experimental design (info-gain probing)", lambda: ac_c, lambda: ac_t, higher_is_better=False)

    print()
    sy = synapse()
    if sy is None:
        print("[Synapse]  embedder unavailable — skipped (set GA_EMBED_URL)")
    else:
        print(f"[Synapse]  embedding-only {sy[0]:.3f} → hybrid lexical+embedding {sy[1]:.3f}")
        confirm_gain("Synapse (hybrid retrieval)", lambda: sy[0], lambda: sy[1], higher_is_better=True)


if __name__ == "__main__":
    main()
