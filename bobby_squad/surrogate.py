import ast
import math
from collections import Counter
from typing import List, Tuple


def code_features(src: str) -> List[float]:
    """Cheap, deterministic, endpoint-free features of a candidate's code (no embedding call)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [0.0] * 8
    c = Counter(type(n).__name__ for n in ast.walk(tree))
    return [float(len(src)), float(sum(c.values())),
            float(c.get("FunctionDef", 0) + c.get("AsyncFunctionDef", 0)),
            float(c.get("Call", 0)),
            float(c.get("For", 0) + c.get("While", 0) + c.get("comprehension", 0)),
            float(c.get("If", 0)), float(c.get("Attribute", 0)), float(c.get("Constant", 0))]


class Surrogate:
    def __init__(self, k: int = 3):
        self.k = k
        self.X: List[List[float]] = []
        self.y: List[float] = []
        self.mu: List[float] = []
        self.sd: List[float] = []

    def fit(self, feats: List[List[float]], scores: List[float]) -> "Surrogate":
        self.X = [list(f) for f in feats]
        self.y = list(scores)
        d = len(self.X[0])
        n = len(self.X)
        self.mu = [sum(x[j] for x in self.X) / n for j in range(d)]
        self.sd = [max(1e-6, (sum((x[j] - self.mu[j]) ** 2 for x in self.X) / n) ** 0.5) for j in range(d)]
        return self

    def _std(self, x: List[float]) -> List[float]:
        return [(x[j] - self.mu[j]) / self.sd[j] for j in range(len(x))]

    def predict(self, feat: List[float]) -> Tuple[float, float]:
        """(predicted_score, uncertainty). Distance-weighted k-NN; uncertainty = neighbour-spread + nearest distance."""
        xs = self._std(feat)
        ds = sorted((math.dist(xs, self._std(xi)), yi) for xi, yi in zip(self.X, self.y))[:self.k]
        w = [1.0 / (d + 1e-6) for d, _ in ds]
        sw = sum(w) or 1.0
        mean = sum(wi * yi for wi, (_, yi) in zip(w, ds)) / sw
        var = sum(wi * (yi - mean) ** 2 for wi, (_, yi) in zip(w, ds)) / sw
        return mean, math.sqrt(var) + ds[0][0]

    def prune(self, cand_feats: List[List[float]], keep_frac: float = 0.5, explore_frac: float = 0.2) -> List[int]:
        """Indices of candidates that DESERVE a real eval: the top by predicted score UNION a high-uncertainty tail.
        Everything else is deferred. Fail-safe: uncertain candidates are always kept (never silently dropped)."""
        n = len(cand_feats)
        preds = [self.predict(f) for f in cand_feats]
        keep = max(1, int(round(n * keep_frac)))
        expl = max(1, int(round(n * explore_frac)))
        by_score = sorted(range(n), key=lambda i: -preds[i][0])[:keep]
        by_unc = sorted(range(n), key=lambda i: -preds[i][1])[:expl]
        return sorted(set(by_score) | set(by_unc))

    def replay(self, feats: List[List[float]], scores: List[float], keep_frac: float = 0.5,
               winners_frac: float = 0.3) -> dict:
        """Leave-one-out honesty check on ledger history: if the surrogate had pruned to `keep_frac`, how many of the
        TRUE top-`winners_frac` would still have been evaluated (recall), and how many evals were saved?"""
        n = len(feats)
        order = sorted(range(n), key=lambda i: -scores[i])
        true_winners = set(order[:max(1, int(round(n * winners_frac)))])
        kept = 0
        for i in range(n):                                     # leave-one-out: train on the rest, decide on i
            s = Surrogate(self.k).fit([feats[j] for j in range(n) if j != i], [scores[j] for j in range(n) if j != i])
            others = [feats[j] for j in range(n) if j != i]
            preds_i = s.predict(feats[i])[0]
            rank = sum(1 for f in others if s.predict(f)[0] > preds_i)   # how many others outrank i
            if rank < max(1, int(round(n * keep_frac))):
                kept += 1 if i in true_winners else 0
        recall = kept / max(1, len(true_winners))
        return {"winner_recall": round(recall, 3), "evals_saved_frac": round(1 - keep_frac, 3),
                "n": n, "winners": len(true_winners)}
