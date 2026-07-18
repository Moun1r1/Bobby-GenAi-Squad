import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Scenario:
    name: str
    seed: int = 0
    params: dict = field(default_factory=dict)

    def spawn_replications(self, k: int) -> List["Scenario"]:
        return [Scenario(f"{self.name}#{i}", seed=self.seed + i, params=dict(self.params)) for i in range(k)]

    def rng(self) -> random.Random:
        return random.Random(self.seed)


# two-sided 95% Student-t critical values by degrees of freedom (n-1); →1.96 as df→∞.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def ci95(xs: List[float]):
    """Return (mean, half-width of the 95% CI) using the Student-t critical value for n-1 d.o.f. (not the normal z),
    which is the correct interval for the small n typical of replication sweeps."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    df = n - 1
    t = _T95.get(df) or next((v for k, v in sorted(_T95.items()) if k >= df), 1.96)
    return m, t * sd / math.sqrt(n)


@dataclass
class Report:
    name: str
    mean: float
    ci: float
    n: int
    per: List[float]

    def as_dict(self):
        return {"name": self.name, "mean": round(self.mean, 3), "ci": round(self.ci, 3), "n": self.n}


class DataCollector:
    """Run `metric(scenario) -> float` across `replications` SEEDED children → mean ± 95% CI (reproducible)."""

    def run(self, name: str, metric: Callable[[Scenario], float], scenario: Scenario, replications: int = 5) -> Report:
        vals = [float(metric(s)) for s in scenario.spawn_replications(replications)]
        m, ci = ci95(vals)
        return Report(name, m, ci, len(vals), vals)


def verdict(treat: Report, base: Report, control: Optional[Report] = None,
            wire: float = 1.0, baseline_max: float = 95.0) -> dict:
    d = treat.mean - base.mean
    if control is not None and (control.mean - base.mean) >= wire:
        return {"verdict": "INVALID", "reason": "effect leaks into the negative control", "dF1": round(d, 2)}
    if base.mean >= baseline_max:
        return {"verdict": "INCONCLUSIVE", "reason": "baseline at ceiling — no headroom", "dF1": round(d, 2)}
    lo = d - (treat.ci + base.ci)                      # lower bound of the improvement after CIs
    hi = d + (treat.ci + base.ci)
    if d >= wire and lo > 0:
        v = "WIRE"
    elif d <= -wire and hi < 0:
        v = "DELETE"
    else:
        v = "MARGINAL"
    return {"verdict": v, "dF1": round(d, 2), "ci": round(treat.ci + base.ci, 2), "lo": round(lo, 2)}
