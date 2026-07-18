import math
from typing import List, Optional, Sequence


class OODGate:
    """Fit on a plugin's proof-set embeddings; score a query's distance to that competence region."""

    def __init__(self, mu: List[float], inv_std: List[float], tau: float):
        self.mu = mu
        self.inv_std = inv_std        # 1/std per dimension (diagonal precision)
        self.tau = tau                # abstain when distance > tau

    @classmethod
    def fit(cls, embs: Sequence[Sequence[float]], k: float = 2.5, ridge: float = 1e-3) -> "OODGate":
        n = len(embs)
        d = len(embs[0])
        mu = [sum(e[j] for e in embs) / n for j in range(d)]
        var = [sum((e[j] - mu[j]) ** 2 for e in embs) / max(1, n - 1) + ridge for j in range(d)]
        inv_std = [1.0 / math.sqrt(v) for v in var]
        # tau = the in-sample distances' mean + k·std → OOD is "meaningfully farther than the proof set itself"
        ds = [cls._dist(mu, inv_std, e) for e in embs]
        m = sum(ds) / n
        sd = (sum((x - m) ** 2 for x in ds) / max(1, n - 1)) ** 0.5
        return cls(mu, inv_std, tau=m + k * sd)

    @staticmethod
    def _dist(mu, inv_std, x) -> float:
        return math.sqrt(sum(((x[j] - mu[j]) * inv_std[j]) ** 2 for j in range(len(mu))))

    def distance(self, x: Sequence[float]) -> float:
        return self._dist(self.mu, self.inv_std, x)

    def is_ood(self, x: Sequence[float]) -> bool:
        return self.distance(x) > self.tau


def ood_plugin_router(embed_fn):
    """Build an OOD-aware interceptor for the Engine. A plugin whose `proof['competence']` holds an `OODGate` is used
    ONLY when the query embedding is in-distribution; otherwise the router returns None (→ LLM fallback) and the engine
    records an `OOD_DETECTED` event. Plugins without a gate behave like the plain router (always eligible)."""
    def _router(ev, engine):
        cap = ev.payload.get("cap", ev.kind)
        p = engine.registry.resolve(cap)
        if p is None:
            return None
        gate: Optional[OODGate] = (p.proof or {}).get("competence") if isinstance(p.proof, dict) else None
        q = ev.payload.get("q")
        if gate is not None and q is not None:
            try:
                emb = embed_fn([q])[0]
            except Exception:
                emb = None
            if emb is not None and gate.is_ood(emb):
                engine.log.append(type(ev)("OOD_DETECTED", {"cap": cap, "plugin": p.name,
                                                            "distance": round(gate.distance(emb), 2),
                                                            "tau": round(gate.tau, 2), "q": q}, cause=ev.id))
                return None                                        # abstain → fall through to the LLM (fail-safe)
        return p.handler(ev.payload)
    _router.__name__ = "ood_plugin_router"
    return _router


def competence_router(embed_fn):
    """Multi-cluster router: among ALL frozen plugins covering the event's capability, route to the one whose
    competence region the query is INSIDE (nearest in-distribution). If the query is OOD for every plugin, abstain
    (→ LLM fallback) and log OOD_DETECTED. This is what lets distinct distilled skills (cluster A vs B) coexist under
    one capability without misapplying one to the other."""
    def _router(ev, engine):
        cap = ev.payload.get("cap", ev.kind)
        plugins = engine.registry.resolve_all(cap)
        if not plugins:
            return None
        q = ev.payload.get("q") or ev.payload.get("text")
        emb = None
        if q is not None:
            try:
                emb = embed_fn([q])[0]
            except Exception:
                emb = None
        best, best_d = None, float("inf")
        for p in plugins:
            gate: Optional[OODGate] = (p.proof or {}).get("competence") if isinstance(p.proof, dict) else None
            if gate is None:
                best, best_d = p, -1.0
                break
            if emb is not None:
                d = gate.distance(emb)
                if not gate.is_ood(emb) and d < best_d:
                    best, best_d = p, d
        if best is None:                                          # all plugins OOD → abstain to the LLM (fail-safe)
            engine.log.append(type(ev)("OOD_DETECTED", {"cap": cap, "q": q, "n_plugins": len(plugins)}, cause=ev.id))
            return None
        return best.handler(ev.payload)
    _router.__name__ = "competence_router"
    return _router
