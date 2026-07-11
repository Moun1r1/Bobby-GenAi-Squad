"""worldsense — let a generative agent CHECK many worlds/contexts (signals) and pull the salient ones into its own
reasoning, instead of being confined to one fixed subject.

The AgentSociety WorldStream is the BASELINE world (already built). This is the SENSING layer on top of it:

  • a pluggable registry of signal SOURCES — each is a 'world' the agent can check (the society stream is just one:
    files changing, the idea frontier, what peers are doing, time passing, its own EMOTION, its own SELF-MODEL).
  • a deterministic salience × recency × novelty rank (guard-first: the ranking is code, not a prompt), with dedup
    so the same signal isn't re-perceived (the idea-space lesson, applied to perception).
  • perceive() injects the top signals as OBSERVATION DATA — never a directive — so the agent's own generative loop
    decides what (if anything) to do with them. That is what makes the world INTERACTIVE and EXPANDABLE: add a
    source, the agents start attending to a new world; emit back to the WorldStream and peers sense it in turn.

EMOTION and CONSCIOUSNESS are not special-cased — they are just two more sources on the same bus:
  • EmotionState  — a small affect model (valence/arousal) updated FROM the signals; its mood is itself a signal.
  • SelfModelSource — the agent sensing its OWN behavioral trace (metacognition on self) = a self-awareness signal.
"""
import os
import time

from .dedup import near_dup


def signal(source, text, kind="event", t=None, salience=1.0, meta=None):
    """A perceivable signal from some world. `salience` biases the rank; `t` (epoch secs) decays it by recency."""
    return {"source": source, "kind": kind, "text": (text or "").strip(), "t": t, "salience": salience,
            "meta": meta or {}}


# ── SOURCES — each a 'world'; anything with `.name` and `.poll() -> list[signal]` works ───────────────────────────
class WorldStreamSource:
    """The AgentSociety event stream (baseline world): what OTHER agents just said/did, emotion and all."""
    name = "society"

    def __init__(self, world, exclude=None):
        self.world, self.exclude = world, exclude
        self._seq = getattr(world, "_seq", 0)

    def poll(self):
        out = []
        for e in self.world.read_all(self._seq):
            self._seq = max(self._seq, e["seq"])
            if e["agent"] == self.exclude or e["kind"] in ("system", "state"):
                continue
            emo = (e.get("voice") or {}).get("emotion")
            out.append(signal("society", e.get("text", ""), kind=e["kind"], t=e.get("t"),
                              salience=1.2, meta={"agent": e["agent"], "emotion": emo}))
        return out


class FileChangeSource:
    """A changing CODE world: files touched since the last check — the agent notices its environment move."""
    name = "files"

    def __init__(self, root, exts=(".py", ".md"), window_s=3600):
        self.root, self.exts, self.window, self._seen = root, exts, window_s, {}

    def poll(self):
        out, now = [], time.time()
        for r, ds, fs in os.walk(self.root):
            ds[:] = [d for d in ds if d not in ("__pycache__", ".venv", ".git", "out", "world_state", "data")]
            for fn in fs:
                if not fn.endswith(self.exts):
                    continue
                p = os.path.join(r, fn)
                try:
                    m = os.path.getmtime(p)
                except OSError:
                    continue
                first, changed = p not in self._seen, self._seen.get(p) != os.path.getmtime(p)
                self._seen[p] = m
                if not first and changed and (now - m) <= self.window:      # only real CHANGES after the first scan
                    out.append(signal("files", f"{os.path.relpath(p, self.root)} changed", kind="change",
                                      t=m, salience=1.5))
        return out


class LedgerSource:
    """The idea frontier — reuse IdeaLedger.signal() (open / closed / unexplored) as perceivable signals."""
    name = "frontier"

    def __init__(self, ledger):
        self.ledger = ledger

    def poll(self):
        return [signal("frontier", ln, kind="frontier", salience=0.8) for ln in self.ledger.signal()[:4]]


class PeerSource:
    """Other agents' BEHAVIOR (reuse BehaviorTrace): a peer's area/move/novelty is a social signal."""
    name = "peers"

    def __init__(self, traces, exclude=None):
        self.traces, self.exclude = traces, exclude

    def poll(self):
        out = []
        for name, tr in self.traces.items():
            if name == self.exclude:
                continue
            s = tr.signals()
            out.append(signal("peers", f"{name} is on '{s['dominant_area']}' via '{s['dominant_move']}' "
                              f"(novelty {s['novelty_curve']})", kind="peer", salience=0.9))
        return out


class ClockSource:
    """Time passing is itself a signal — enables temporal / event-driven behavior, not just request/response."""
    name = "clock"

    def __init__(self, every_s=30):
        self.every, self._last = every_s, 0.0

    def poll(self):
        now = time.time()
        if now - self._last < self.every:
            return []
        self._last = now
        return [signal("clock", f"time passing — tick at {int(now)}", kind="tick", t=now, salience=0.3)]


class EmotionState:
    """A small AFFECT model (extend the world with emotion). Valence/arousal drift FROM the signals the agent senses
    (a change/conflict raises arousal; monotony lowers it), decaying toward neutral. It is BOTH a source (its mood is
    a signal others/self can perceive) and an update() the sensor calls with what was just sensed. Deterministic —
    an affect substrate, not a scripted feeling."""
    name = "emotion"
    MOODS = [("driven", 0.5, 0.7), ("curious", 0.4, 0.4), ("calm", 0.3, 0.1),
             ("restless", -0.2, 0.6), ("bored", -0.3, 0.1), ("tense", -0.5, 0.7)]

    def __init__(self, agent_name="self"):
        self.agent, self.valence, self.arousal = agent_name, 0.2, 0.3

    def update(self, sigs):
        change = sum(1 for s in sigs if s["kind"] in ("change", "peer"))
        monotony = sum(1 for s in sigs if s["kind"] in ("tick", "frontier"))
        self.arousal = max(0.0, min(1.0, self.arousal * 0.85 + 0.15 * (change / max(1, len(sigs)))))
        self.valence = max(-1.0, min(1.0, self.valence * 0.9 + 0.1 * ((change - monotony) / max(1, len(sigs)))))

    def mood(self):
        return min(self.MOODS, key=lambda m: (self.valence - m[1]) ** 2 + (self.arousal - m[2]) ** 2)[0]

    def poll(self):
        return [signal("emotion", f"your current affect: {self.mood()} "
                       f"(valence {self.valence:+.2f}, arousal {self.arousal:.2f})", kind="affect", salience=0.7)]


class SelfModelSource:
    """CONSCIOUSNESS as a source: the agent sensing its OWN behavioral trace (metacognition turned inward) — a
    self-awareness signal it can act on. Reuses BehaviorTrace.flags(); no new machinery."""
    name = "self-model"

    def __init__(self, own_trace):
        self.trace = own_trace

    def poll(self):
        flags = self.trace.flags()
        return [signal("self-model", "self-awareness — " + f, kind="reflect", salience=0.85) for f in flags[:2]]


# ── the SENSOR: check every world, rank, dedup, deliver ──────────────────────────────────────────────────────────
class WorldSense:
    """Check all registered worlds and return the most salient, novel signals. Add a source → the agents attend to a
    new world (that's the 'world expansion'). Ranking/novelty is deterministic code; the agent decides what to do."""

    def __init__(self, sources=None):
        self.sources = list(sources or [])
        self.emotion = next((s for s in self.sources if isinstance(s, EmotionState)), None)

    def register(self, src):
        self.sources.append(src)
        if isinstance(src, EmotionState):
            self.emotion = src
        return self

    def sense(self, k=6, seen=None):
        sigs = []
        for src in self.sources:
            try:
                sigs += src.poll()
            except Exception:
                pass
        now = time.time()

        def score(s):
            rec = 1.0 if not s.get("t") else max(0.2, 1.0 / (1.0 + (now - s["t"]) / 60.0))
            return s["salience"] * rec

        sigs.sort(key=score, reverse=True)
        out, kept = [], (seen if seen is not None else [])
        for s in sigs:
            if s["text"] and not near_dup(s["text"], kept):
                out.append(s)
                kept.append(s["text"])
            if len(out) >= k:
                break
        if self.emotion:                                    # affect drifts FROM what was just sensed
            self.emotion.update(out)
        return out


def perceive(agent, sense, k=6, seen=None):
    """Sense the worlds and inject the salient signals as OBSERVATION DATA (context, never a directive) so the
    agent's own generative loop can react to them. Emits a 'perceive' observer event. Returns the sensed signals."""
    sigs = sense.sense(k=k, seen=seen)
    if sigs:
        lines = "\n".join(f"[{s['source']}·{s['kind']}] {s['text'][:140]}" for s in sigs)
        agent.observe("WORLD SIGNALS — current context from the worlds you can sense:\n" + lines)
        emit = getattr(agent, "_emit", None)
        if callable(emit):
            emit("perceive", n=len(sigs), sources=sorted({s["source"] for s in sigs}))
    return sigs
