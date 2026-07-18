import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .dedup_ast import AstDedup, fingerprint
from .jobs import JobRegistry


# ── Plane 0: the event-log spine (single source of truth) ────────────────────────────────────────────
@dataclass
class Event:
    kind: str
    payload: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    cause: Optional[str] = None                    # id of the causing event → a provenance chain


class EventLog:
    """Append-only log. On disk as JSONL so it survives restart and can be replayed. Every plane is a projection."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._mem: List[Event] = []
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    d = json.loads(line)
                    self._mem.append(Event(**d))

    def append(self, ev: Event) -> Event:
        with open(self.path, "a") as f:
            f.write(json.dumps({"kind": ev.kind, "payload": ev.payload, "id": ev.id, "ts": ev.ts,
                                "cause": ev.cause}) + "\n")
        self._mem.append(ev)
        return ev

    def read(self, kind: Optional[str] = None) -> List[Event]:
        return [e for e in self._mem if kind is None or e.kind == kind]

    def project(self, fold: Callable[[Any, Event], Any], init: Any) -> Any:
        """Materialize a view by folding the log (a blackboard, a metric, a state machine — all live here)."""
        acc = init
        for e in self._mem:
            acc = fold(acc, e)
        return acc


# ── Plane 2: the plugin registry (frozen local handlers) ─────────────────────────────────────────────
@dataclass
class Plugin:
    name: str
    handler: Callable[[dict], Any]                 # (payload) -> result — the frozen local fast-path
    tags: frozenset                                # capabilities this plugin can serve
    kind: str = "static"                           # static (frozen) | generative (LLM) | world (tick)
    fingerprint: Optional[str] = None
    proof: Optional[dict] = None                   # gain-proof record (governance gate)
    provenance: str = ""
    deprecated: bool = False


class PluginRegistry:
    """Register a frozen plugin only if it is (a) not a functional duplicate (AstDedup) and (b) carries a proof
    (governance) — unless `require_proof=False` for a trusted built-in. Resolve by capability tags."""

    def __init__(self, require_proof: bool = True):
        self.require_proof = require_proof
        self._plugins: Dict[str, Plugin] = {}
        self._dedup = AstDedup()

    def register(self, plugin: Plugin, code: Optional[str] = None) -> bool:
        """Returns True if registered. Rejects a functional-duplicate (by AST fingerprint of `code`) or an unproven
        plugin. `code` is the plugin's source — used for dedup + a content address."""
        if code is not None:
            fp = fingerprint(code)
            if fp is not None and self._dedup.is_dup(code):
                return False                        # cosmetic/functional twin — reject free
            plugin.fingerprint = fp
            self._dedup.add(code)
        if self.require_proof and plugin.proof is None:
            return False                            # governance: no proof, no promotion
        self._plugins[plugin.name] = plugin
        return True

    def resolve(self, cap: str) -> Optional[Plugin]:
        """The best non-deprecated plugin whose tags cover capability `cap`."""
        cands = self.resolve_all(cap)
        return cands[0] if cands else None

    def resolve_all(self, cap: str) -> List[Plugin]:
        """All non-deprecated plugins whose tags cover `cap` (for competence routing across clusters)."""
        return [p for p in self._plugins.values() if not p.deprecated and cap in p.tags]

    def deprecate(self, name: str) -> None:
        if name in self._plugins:
            self._plugins[name].deprecated = True

    def active(self) -> List[Plugin]:
        return [p for p in self._plugins.values() if not p.deprecated]


# an interceptor: (event, engine) -> a result to SHORT-CIRCUIT, or None to pass through to the next/fallback
Interceptor = Callable[[Event, "Engine"], Optional[Any]]


def plugin_router(ev: Event, engine: "Engine") -> Optional[Any]:
    """The ACR router: if a frozen plugin covers this event's capability (`payload['cap']` or the kind), run it —
    zero LLM. Else return None → the event falls through to the LLM fallback."""
    cap = ev.payload.get("cap", ev.kind)
    p = engine.registry.resolve(cap)
    return p.handler(ev.payload) if p is not None else None


# ── the kernel ───────────────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, root: str, require_proof: bool = True):
        self.root = os.path.abspath(root)
        self.log = EventLog(os.path.join(self.root, "events.jsonl"))
        self.registry = PluginRegistry(require_proof=require_proof)
        self.jobs = JobRegistry(os.path.join(self.root, "jobs"))
        self.interceptors: List[Interceptor] = [plugin_router]     # cheapest-first; plugin router is built-in
        self.handlers: Dict[str, Callable[[dict], Any]] = {}       # kind -> last-resort handler (the LLM)
        self.stats = {"events": 0, "by_interceptor": 0, "by_fallback": 0, "unhandled": 0}

    # wiring
    def use(self, interceptor: Interceptor) -> "Engine":
        self.interceptors.append(interceptor)
        return self

    def on(self, kind: str, handler: Callable[[dict], Any]) -> "Engine":
        """Register the last-resort (expensive) handler for an event kind — typically the LLM."""
        self.handlers[kind] = handler
        return self

    def promote(self, name: str, handler: Callable[[dict], Any], tags, proof: dict, code: Optional[str] = None,
                kind: str = "static", provenance: str = "") -> bool:
        """The flywheel's payoff: register a PROVEN frozen plugin. On success the matching capability is served
        locally forever after (the LLM fallback stops being called for it). Emits SKILL_PROMOTED."""
        ok = self.registry.register(Plugin(name=name, handler=handler, tags=frozenset(tags), kind=kind,
                                            proof=proof, provenance=provenance), code=code)
        self.log.append(Event("SKILL_PROMOTED" if ok else "SKILL_REJECTED",
                              {"name": name, "tags": list(tags), "ok": ok}))
        return ok

    # the loop: one event → cheapest handler that can serve it
    def emit(self, kind: str, payload: Optional[dict] = None, cause: Optional[str] = None) -> Any:
        ev = self.log.append(Event(kind, payload or {}, cause=cause))
        self.stats["events"] += 1
        for icept in self.interceptors:
            r = icept(ev, self)
            if r is not None:
                self.stats["by_interceptor"] += 1
                self.log.append(Event(kind + ".handled", {"by": getattr(icept, "__name__", "interceptor"),
                                                           "result": _safe(r)}, cause=ev.id))
                return r
        h = self.handlers.get(kind)
        if h is not None:
            r = h(ev.payload)                                       # the expensive last resort (LLM)
            self.stats["by_fallback"] += 1
            self.log.append(Event(kind + ".handled", {"by": "fallback", "result": _safe(r)}, cause=ev.id))
            return r
        self.stats["unhandled"] += 1
        self.log.append(Event(kind + ".unhandled", {}, cause=ev.id))
        return None


def _safe(r: Any) -> Any:
    try:
        json.dumps(r)
        return r
    except (TypeError, ValueError):
        return str(r)[:200]
