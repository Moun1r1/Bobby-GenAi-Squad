from typing import Callable, List, Optional

from .retrieval import EmbeddingRetriever, _cos


class SemanticMemory:
    """A novelty-gated semantic store with top-k retrieval.

    - `add(text)` keeps the item only if it adds a DISTINCT direction (max cosine to what's kept < τ) — so
      recurring/redundant items self-compress (the retention gate). Returns True if kept, False if gated as redundant.
    - `retrieve(situation, k)` returns the k most relevant items — O(k) injected regardless of store size.
    """

    def __init__(self, tau: float = 0.9, k: int = 4, embed_fn: Optional[Callable] = None,
                 capacity: Optional[int] = None, policy: str = "value"):
        self.r = EmbeddingRetriever(embed_fn=embed_fn)
        self.tau = tau
        self.k = k
        # SELF-EVOLVING RETENTION — WIRED AS DEFAULT (proven WIRE, gains/memory_policy_gain.py: +25% retention,
        # +12.5% real-LLM generation vs recency, at equal capacity):
        #   capacity=None → UNBOUNDED (default): no eviction ever; the novelty gate + top-k economy are byte-identical
        #     to before, so the measured memory-gains result is preserved. `value` tracking is a harmless no-op here.
        #   capacity=N → the store SELF-GOVERNS: retrieval raises an item's value, overflow evicts the lowest-value
        #     item (policy="value", default) — i.e. the memory keeps what actually feeds generation. policy="fifo"
        #     falls back to recency eviction. Items add()'d critical=True are pinned = the deterministic recall floor.
        self.capacity = capacity
        self.policy = policy
        self._meta = []                                    # per-doc {value, critical, born}, aligned with r.docs
        self._born = 0

    def _embed(self, text: str):
        v = self.r.embed_fn([self.r.dp + text])
        return v[0] if v and v[0] else None

    def add(self, text: str, critical: bool = False) -> bool:
        text = (text or "").strip()
        if len(text) < 8:
            return False
        vec = self._embed(text)
        if vec is None:
            # no embedding endpoint — fall back to exact dedup via the cache dict (O(1), not an O(N) list scan that
            # would make add_many O(N^2) — audit X-simplify, verified)
            if text in self.r.cache:
                return False
            self.r.cache[text] = None; self.r.docs.append(text); self.r.vecs.append(None)
        elif self.r.vecs and any(ev is not None and _cos(vec, ev) >= self.tau for ev in self.r.vecs):
            return False                                   # redundant with something already kept
        else:
            self.r.cache[text] = vec; self.r.docs.append(text); self.r.vecs.append(vec)
        self._meta.append({"value": 0, "critical": critical, "born": self._born}); self._born += 1
        if self.capacity:                                  # bounded store → evict per policy (critical items pinned)
            while len(self.r.docs) > self.capacity and self._evict():
                pass
        return True

    def _evict(self) -> bool:
        """Remove ONE non-critical item per the policy: 'value' → lowest usage-learned value (self-evolving);
        else → oldest (FIFO/recency). Critical items are the deterministic recall floor and are never evicted."""
        idxs = [i for i, m in enumerate(self._meta) if not m["critical"]]
        if not idxs:
            return False
        if self.policy == "value":
            i = min(idxs, key=lambda j: (self._meta[j]["value"], self._meta[j]["born"]))
        else:
            i = min(idxs, key=lambda j: self._meta[j]["born"])
        text = self.r.docs[i]
        del self.r.docs[i]; del self.r.vecs[i]; del self._meta[i]; self.r.cache.pop(text, None)
        return True

    def add_many(self, texts) -> int:
        return sum(1 for t in texts if self.add(t))

    def is_novel(self, text: str) -> bool:
        """Would this be kept? (novelty check without mutating — the plateau signal.)"""
        vec = self._embed(text)
        if vec is None:
            return text not in self.r.cache
        return not (self.r.vecs and any(ev is not None and _cos(vec, ev) >= self.tau for ev in self.r.vecs))

    def retrieve(self, situation: str, k: Optional[int] = None) -> List[str]:
        hits = self.r.search(situation, k=k or self.k)
        if self.policy == "value":                         # usage teaches the policy: retrieved knowledge earns its
            for h in hits:                                 # place, so it survives eviction (self-governing memory)
                try:
                    self._meta[self.r.docs.index(h)]["value"] += 1
                except (ValueError, IndexError):
                    pass
        return hits

    def as_block(self, situation: str, header: str, k: Optional[int] = None) -> str:
        """Formatted for injection into a prompt/self-core: only the top-k relevant items, never the whole store."""
        hits = self.retrieve(situation, k=k)
        if not hits:
            return ""
        return header + "\n" + "\n".join(f"- {h}" for h in hits)

    def __len__(self) -> int:
        return len(self.r.docs)


# Back-compat / intent-revealing aliases — same mechanism, different role.
CorrectionMemory = SemanticMemory     # past mistakes → retrieve the relevant lesson, don't repeat it
FindingsMemory = SemanticMemory       # accumulated findings → novelty gate for plateau + cross-item dedup
