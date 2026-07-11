"""metacognition — behavioral self-review for a squad.

The point: give the agents a TOOL to review how ONE OF THEM actually behaved, and detect — on their own — that
peer's intelligence BIAS and FRONTIER. Not a code-test runner and not a static prompt: a `BehaviorTrace` records an
agent's real behavior off the observer stream, deterministic SIGNALS turn that trace into grounded evidence
(move-entropy / area-concentration / repetition → bias; novelty-collapse / abstention → frontier), and `MetaTools`
exposes a peer's evidence so a reviewing agent NAMES the bias/frontier itself. The detection is the agent's; the
evidence is real — a self-model loop over behavior, the thing a single one-pass call can never do to itself.

Bias  = where an agent's attention is over-concentrated or self-repeating (it keeps doing/seeing the same thing).
Frontier = where its intelligence stops ADDING — novelty collapses to zero, it abstains, it plateaus.
"""
import json
import math

from .dedup import near_dup
from .ledger import DEFAULT_AREAS


def _entropy(counts):
    """Normalized Shannon entropy (0 = collapsed onto one option = biased, 1 = perfectly spread)."""
    tot = sum(counts) or 1
    ps = [c / tot for c in counts if c]
    if len(ps) <= 1:
        return 0.0
    h = -sum(p * math.log(p, 2) for p in ps)
    return h / math.log(len(ps), 2)


def area_of(text, areas=DEFAULT_AREAS):
    """Classify a target/behavior into a topic area by keyword hits (same buckets the IdeaLedger uses)."""
    low = (text or "").lower()
    best, bh = "other", 0
    for a, kws in areas.items():
        h = sum(low.count(k) for k in kws)
        if h > bh:
            bh, best = h, a
    return best


class BehaviorTrace:
    """An OBSERVER that records one agent's behavior so a peer can review it. Pass as Agent(observer=trace). It is
    pure instrumentation — it never changes behavior. Optionally `echo=stream_observer` to also watch it live."""

    def __init__(self, name, echo=None):
        self.name = name
        self.echo = echo
        self.targets, self.moves, self.intentions, self.cycles, self.tools = [], [], [], [], []
        self.abstentions = 0

    def __call__(self, e):
        k = e.get("kind")
        if k == "target":
            self.targets.append(e.get("target", ""))
        elif k == "move_start":
            self.moves.append((e.get("move") or "unnamed").strip().lower())
            self.intentions.append(e.get("intention", ""))
        elif k == "move_end":
            self.tools += (e.get("tools") or [])
        elif k == "cycle":
            r = e.get("results", 0)
            self.cycles.append(r)
            if r == 0:
                self.abstentions += 1
        if self.echo:
            try:
                self.echo(e)
            except Exception:
                pass

    def signals(self):
        """Deterministic behavioral evidence — the grounding a reviewer reasons FROM (never fabricated)."""
        mv = {}
        for m in self.moves:
            mv[m] = mv.get(m, 0) + 1
        ar = {}
        for t in self.targets:
            a = area_of(t)
            ar[a] = ar.get(a, 0) + 1
        rep, seen = 0, []
        for it in self.intentions:
            if it and near_dup(it, seen):
                rep += 1
            seen.append(it)
        top_area = max(ar.items(), key=lambda x: x[1]) if ar else ("none", 0)
        frontier = next((i for i, c in enumerate(self.cycles) if c == 0), None)
        return {
            "name": self.name,
            "n_moves": len(self.moves), "n_cycles": len(self.cycles),
            "move_distribution": mv, "move_entropy": round(_entropy(list(mv.values())), 2),
            "dominant_move": max(mv.items(), key=lambda x: x[1])[0] if mv else "none",
            "area_distribution": ar, "dominant_area": top_area[0],
            "area_concentration": round(top_area[1] / max(1, len(self.targets)), 2),
            "repetition_rate": round(rep / max(1, len(self.intentions)), 2),
            "abstentions": self.abstentions, "novelty_curve": self.cycles[:], "frontier_cycle": frontier,
            "tools_used": sorted(set(self.tools)),
        }

    def flags(self):
        """Grounded RED FLAGS the reviewer can confirm/extend — each carries the signal it's derived from, so it's
        evidence, not opinion. The agent still names the bias/frontier in its own words; these just anchor it."""
        s, out = self.signals(), []
        if s["n_moves"] >= 3 and s["move_entropy"] < 0.5:
            out.append(f"MOVE-BIAS: collapsed onto '{s['dominant_move']}' (move-entropy {s['move_entropy']} — low)")
        if s["n_moves"] >= 3 and s["area_concentration"] >= 0.6:
            out.append(f"AREA-FIXATION: {int(s['area_concentration']*100)}% of targets in '{s['dominant_area']}'")
        if s["repetition_rate"] >= 0.3:
            out.append(f"ECHO-BIAS: {int(s['repetition_rate']*100)}% of intentions near-duplicate an earlier one")
        if s["frontier_cycle"] is not None:
            out.append(f"FRONTIER: novelty collapsed to 0 at cycle {s['frontier_cycle']} (added nothing after)")
        if s["abstentions"] >= 2:
            out.append(f"FRONTIER: abstained {s['abstentions']}× (ran out of reachable ideas)")
        return out or ["no strong behavioral flags — attention looks spread and still producing"]


META_SCHEMAS = [
    {"type": "function", "function": {"name": "peers", "description": "List the agents whose real behavior you can "
     "review (name + how much they did), so you can pick one to examine.", "parameters": {"type": "object",
        "properties": {}}}},
    {"type": "function", "function": {"name": "review_peer", "description": "Get ONE peer's REAL behavioral evidence "
     "— deterministic signals (move distribution + entropy, area concentration, repetition rate, novelty curve, "
     "where novelty collapsed, abstentions) plus grounded red-flags and sample targets/intentions. Use this to "
     "detect that peer's intelligence BIAS (over-concentration / self-repetition) and FRONTIER (where it stopped "
     "adding anything). Ground every claim you make in a signal from here.", "parameters": {"type": "object",
        "properties": {"name": {"type": "string", "description": "the peer agent's name"}}, "required": ["name"]}}},
]


class MetaTools:
    """The behavioral-review TOOL surface. Wraps a registry of peer BehaviorTraces and hands a reviewing agent the
    real evidence to detect a peer's bias/frontier. Read-only over behavior; it invents nothing."""

    def __init__(self, registry):
        self.registry = registry                       # {name: BehaviorTrace}
        self.schemas = META_SCHEMAS

    def peers(self):
        rows = [f"{t.name}: {len(t.moves)} moves, {len(t.cycles)} cycles" for t in self.registry.values()]
        return "\n".join(rows) or "(no peers recorded)"

    def review_peer(self, name):
        t = self.registry.get(name)
        if not t:
            return f"(no such peer '{name}'. peers: {', '.join(self.registry) or 'none'})"
        ev = {"signals": t.signals(), "flags": t.flags(),
              "sample_targets": t.targets[:6], "sample_intentions": [i[:90] for i in t.intentions[:6]]}
        return json.dumps(ev, indent=1)

    def run_json(self, name, args):
        try:
            if name == "peers":
                return self.peers()
            if name == "review_peer":
                return self.review_peer(args.get("name", ""))
        except Exception as e:
            return f"(meta error: {e})"
        return f"(unknown tool {name})"
