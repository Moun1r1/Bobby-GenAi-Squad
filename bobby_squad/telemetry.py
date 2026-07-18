import math
from collections import Counter

_CONTROL = {"SKILL_PROMOTED", "SKILL_REJECTED", "OOD_DETECTED"}


class Telemetry:
    def __init__(self, log):
        self.log = log

    def _handled(self):
        return [e for e in self.log.read() if e.kind.endswith(".handled")]

    def _emits(self):
        return [e for e in self.log.read()
                if not e.kind.endswith((".handled", ".unhandled")) and e.kind not in _CONTROL]

    def cost_curve(self) -> dict:
        """The moat metric: what fraction of handled events was served LOCALLY (a frozen plugin, ~free) vs by the LLM
        fallback (expensive). As plugins are promoted, `local_frac` rises → cost-per-event falls."""
        by = Counter(e.payload.get("by") for e in self._handled())
        n = sum(by.values()) or 1
        local = sum(v for k, v in by.items() if k and k != "fallback")
        return {"handled": sum(by.values()), "local_frac": round(local / n, 3),
                "llm_frac": round(by.get("fallback", 0) / n, 3)}

    def ood_rate(self) -> float:
        ood = sum(1 for e in self.log.read() if e.kind == "OOD_DETECTED")
        return round(ood / max(1, len(self._emits())), 3)

    def promotions(self) -> int:
        return sum(1 for e in self.log.read() if e.kind == "SKILL_PROMOTED")

    def move_entropy(self) -> float:
        c = Counter(e.kind for e in self._emits())
        n = sum(c.values()) or 1
        return round(-sum((v / n) * math.log2(v / n) for v in c.values()), 3)

    def snapshot(self) -> dict:
        return {**self.cost_curve(), "ood_rate": self.ood_rate(), "promotions": self.promotions(),
                "move_entropy": self.move_entropy()}
