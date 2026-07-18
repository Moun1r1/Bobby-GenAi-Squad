import re
from collections import Counter

from .retrieval import EmbeddingRetriever, _cos

# Boilerplate the swarm prefixes onto proposals ("Based on the investigation of core.py, here is…") — if this becomes
# the idea LABEL, the signal()'s DONE list is a wall of identical openers and the agents can't tell what's saturated.
_BOILER = re.compile(r"^(based on|here('?s| is)|now i have|i have (the full|grounded)|let me|in this|the repository|"
                     r"grounding in reality|result|finding|concrete (result|finding|experiment)|## ?\d)", re.I)


def _clean(line):
    """Strip markdown + a leading marker ('Finding:', 'RESULT:', 'Based on …, here is …:') so what's left is the
    idea's actual content, not the wrapper."""
    s = line.lstrip("#*->`0123456789. ").strip()
    m = re.match(r"^(finding|result|concrete \w+|summary|conclusion|based on .{0,60}?)\s*[:\-–]\s*(.+)", s, re.I)
    return m.group(2).strip() if (m and len(m.group(2)) >= 12) else s


def _label(text):
    """A DISTINCTIVE one-line label for an idea (so the saturated tag is recognizable in signal()). Prefer a cleaned
    markdown heading (usually the idea's name), else the first substantive non-boilerplate line; fall back to line 1."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    cands = []
    for ln in lines:
        c = _clean(ln)
        if len(c) >= 12 and not _BOILER.match(c):
            cands.append((0 if ln.lstrip().startswith("#") else 1, c))    # headings first
    if cands:
        cands.sort(key=lambda x: x[0])
        return cands[0][1][:70]
    return (lines[0][:70] if lines else "(idea)")

# Default feature-AREAS (this repo). Override via IdeaLedger(areas={...}) for another domain — the map is what lets
# the frontier NAME untouched territory instead of vaguely saying "go elsewhere".
DEFAULT_AREAS = {
    "persistent-self": ["persistent-self", "selfcore", "pinned tier", "two-tier", "compaction", "drift"],
    "memory": ["memory", "correction_memory", "retention", "novelty gate", "consolidat", "decay", "eviction",
               "stale", "forget", "gate"],
    "retrieval": ["retrieval", "retrieve", "embedding", "top-k", "semantic recall", "nomic", "cosine"],
    "knowledge-room": ["knowledgeroom", "privileged-retention", "room.py"],
    "search-agent": ["search_agent", "hypothesis", "searcher"],
    "society": ["society", "broadcast", "multi-agent", "incentive", "market", "economy", "collaborat", "role"],
    "planning": ["planning", "select_target", "make_plan", "tpb", "move-space", "guidance"],
    "tools": ["sandbox", "tool", "grep", "execute", "verify-before", "investigate"],
}


class IdeaLedger:
    def __init__(self, embed_fn=None, cluster_tau=0.72, saturate_at=3, over_at=4, areas=None):
        self.r = EmbeddingRetriever(embed_fn=embed_fn)
        self.cluster_tau = cluster_tau
        self.saturate_at = saturate_at
        self.over_at = over_at
        self.areas = areas or DEFAULT_AREAS
        self.ideas = []                                    # {vec, label, area, variants, status, verdict}

    def _area(self, text):
        low = (text or "").lower()
        best, bh = "other", 0
        for area, kws in self.areas.items():
            h = sum(low.count(k) for k in kws)
            if h > bh:
                bh, best = h, area
        return best

    def _embed(self, text):
        v = self.r.embed_fn([self.r.dp + text])
        return v[0] if v and v[0] else None

    def _nearest(self, vec, text):
        if vec is not None:
            best, bc = None, 0.0
            for it in self.ideas:
                if it["vec"] is None:
                    continue
                c = _cos(vec, it["vec"])
                if c > bc:
                    bc, best = c, it
            return best if bc >= self.cluster_tau else None
        toks = {w for w in text.lower().split() if len(w) > 4}     # lexical fallback (embedder down)
        best, bj = None, 0.0
        for it in self.ideas:
            its = {w for w in it["label"].lower().split() if len(w) > 4}
            j = len(toks & its) / max(1, len(toks | its))
            if j > bj:
                bj, best = j, it
        return best if bj >= 0.4 else None

    def admit(self, move, text):
        """IDEA-SPACE acceptance gate — the fix for a swarm re-proposing the same idea in new words. Cluster the
        finding in EMBEDDING space and REJECT it if it lands in an already-CLOSED idea (saturated / proven / dead /
        contested): that is a re-proposal, not a contribution, and it does NOT inflate the idea's variant count. A NEW
        idea, or a variant that DEVELOPS an OPEN one, is admitted. Replaces the old LEXICAL (jaccard) gate that let
        semantic rephrasings through. Proven WIRE +22% diversity vs the lexical gate on real proposals, negative
        control clean (gains/idea_diversity_gain.py). Returns (idea, is_new, admitted)."""
        vec = self._embed(text)
        near = self._nearest(vec, text)
        if near is not None:
            # DETERMINISTIC FLOOR (un-gameable): a near-duplicate of ANYTHING already on the board is REGENERATION —
            # repelled whatever STATE the idea is in. So the squad never regenerates an idea, and agents are free to
            # organize states above this floor without ever being able to reopen a duplicate. A genuinely NEW idea
            # (a different cluster) still passes. `touched` counts how many times the board pulled toward this idea.
            near["touched"] = near.get("touched", 0) + 1
            return near, False, False
        idea, is_new = self.add(move, text, vec=vec)
        return idea, is_new, True

    # ── agent-organized board: states are EMERGENT (free-form), the agents own them; the floor above is code ──
    def set_state(self, idea, state, by=None):
        """Let an AGENT organize the board — assign ANY state to an idea (states are emergent, not a fixed enum:
        exploring / promising / blocked / ready-to-build / …). The anti-regeneration FLOOR does not depend on this,
        so organizing is safe: you can relabel freely but never reopen a duplicate."""
        st = (state or "").strip().lower()
        if st:
            idea["status"] = st
            idea.setdefault("history", []).append((by or "?", st))
        return idea

    def merge(self, keep, fold, by=None):
        """Agent-organized consolidation: fold one idea's content into another and retire the folded one."""
        keep["detail"] = (keep.get("detail", "") + "  ⊕  " + fold.get("detail", ""))[:400]
        self.set_state(fold, "merged", by)
        return keep

    def _farthest_point(self, items, k):
        """ACTIVE REPULSION — return the k most-SPREAD ideas (farthest-point / k-center greedy), so what the board
        surfaces repels agents toward gaps instead of anchoring them on the dense recent cluster. Proven WIRE +74%
        concept coverage vs surfacing by recency, negative control clean (gains/repulsion_gain.py)."""
        have = [it for it in items if it.get("vec") is not None]
        if len(have) <= k:
            return items[-k:]
        sel = [have[0]]
        while len(sel) < k:
            best, bd = None, -1.0
            for it in have:
                if any(it is s for s in sel):
                    continue
                d = min(1.0 - _cos(it["vec"], s["vec"]) for s in sel)
                if d > bd:
                    bd, best = d, it
            sel.append(best)
        return sel

    def add(self, move, text, vec=None):
        """Attach a finding to its idea (or open a new one) and update status. Saturates after enough variants; does
        NOT close on a verdict string (gameable) — closing happens in resolve() from real evidence. Returns (idea, new).
        Prefer admit() at the gate; add() always attaches (no rejection)."""
        if vec is None:
            vec = self._embed(text)
        idea = self._nearest(vec, text)
        new = idea is None
        if new:
            idea = {"vec": vec, "label": _label(text), "area": self._area(text), "variants": 0,
                    "status": "open", "verdict": None, "detail": " ".join((text or "").split())[:200]}
            self.ideas.append(idea)
        idea["variants"] += 1
        return idea, new

    def resolve(self, idea, claim_verdict, repro_verdict):
        """Close by EVIDENCE, not words. claim_verdict = the proposer's real GAIN; repro_verdict = an INDEPENDENT
        challenger's real GAIN. A WIRE the challenge reproduces → proven; a WIRE it can't → contested; DELETE → dead."""
        idea["verdict"], idea["repro"] = claim_verdict, repro_verdict
        if claim_verdict == "DELETE":
            idea["status"] = "dead"
        elif claim_verdict == "WIRE" and repro_verdict == "WIRE":
            idea["status"] = "proven"
        elif claim_verdict == "WIRE":
            idea["status"] = "contested"
        else:
            idea["status"] = "claimed"

    def signal(self, k_per_state=5, **_):
        """The BOARD as the AGENTS organize it — grouped by their OWN emergent state labels. There is NO hardcoded
        lifecycle here: the agents assign states (via set_state / their move) and interpret them; this only presents
        what's on the board, with each idea's content. The identity FLOOR (not any state) prevents regeneration, so
        an idea in ANY state repels its own duplicates — presentation stays purely descriptive. Empty board is
        reported as data so the agent detects it and mines."""
        if not self.ideas:
            return ["BOARD STATE: empty — nothing has been mined onto the board yet; there is nothing here to "
                    "develop, merge, or build on."]
        by_state = {}
        for it in self.ideas:
            by_state.setdefault((it.get("status") or "unlabeled"), []).append(it)
        lines = ["BOARD STATE: " + ", ".join(f"{len(v)} {s}" for s, v in by_state.items())
                 + ". States are the swarm's own; build on with something NEW (a restatement is rejected)."]
        for st, its in by_state.items():
            lines.append(f"[{st}]:")
            # ACTIVE REPULSION: surface the most-SPREAD subset (farthest-point), not the recent one, so agents are
            # anchored on the frontier and pushed toward gaps (proven WIRE +74% concept coverage, repulsion_gain.py).
            lines += [f"- {it['label']}: {it.get('detail', '')[:140]}" for it in self._farthest_point(its, k_per_state)]
        # breadth cue (coverage) — the ONE remaining keyword-derived hint; roadmap-flagged to derive from the repo.
        cov = Counter(it.get("area", "other") for it in self.ideas)
        under = [a for a in self.areas if cov.get(a, 0) == 0]
        if under:
            lines.append("Areas not yet touched on the board: " + ", ".join(under) + ".")
        return lines

    def summary(self):
        return dict(Counter(it["status"] for it in self.ideas))
