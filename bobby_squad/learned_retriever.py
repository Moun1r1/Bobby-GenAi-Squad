import os
from typing import Optional

import numpy as np


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class LearnedRetriever:
    """Loads the exported RetrievalEncoder weights and scores query↔candidate the same way the trained net does:
    two 2-layer MLPs (q, k) + cosine in the learned space. Pure numpy."""

    def __init__(self, path: str):
        w = np.load(path)
        self.w = {k: w[k].astype(np.float32) for k in w.files}
        # sanity: needs q.0/q.2/k.0/k.2 weight+bias (Sequential Linear,GELU,Linear)
        self.ok = all(f"{p}.{i}.weight" in self.w for p in ("q", "k") for i in (0, 2))

    def _mlp(self, x, p):
        h = x @ self.w[f"{p}.0.weight"].T + self.w[f"{p}.0.bias"]
        h = _gelu(h)
        return h @ self.w[f"{p}.2.weight"].T + self.w[f"{p}.2.bias"]

    @staticmethod
    def _norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def score(self, q_vec, cand_vecs) -> np.ndarray:
        """q_vec [d] · cand_vecs [N, d] → learned relevance [N]."""
        qe = self._norm(self._mlp(np.asarray(q_vec, dtype=np.float32), "q"))
        ce = self._norm(self._mlp(np.asarray(cand_vecs, dtype=np.float32), "k"))
        return ce @ qe


def load_retriever(path: str) -> Optional[LearnedRetriever]:
    """Return a LearnedRetriever if the weights file exists and is valid, else None (→ fall back to cosine)."""
    try:
        if path and os.path.exists(path):
            r = LearnedRetriever(path)
            return r if r.ok else None
    except Exception:
        pass
    return None
