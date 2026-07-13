"""store — the persistence layer for Bobby Studio. Replaces the ad-hoc out/*.json files with a real vector database.

One store, three collections:
  • runs       — one point per squad run (payload: pipeline, status, timings, summary/verdict). Vector = a summary
                 embedding so runs are themselves semantically searchable ("find runs about number theory").
  • events     — the live observer stream (target/plan/move/tool/cycle…), one point per event, keyed by run.
  • knowledge  — what the squad LEARNED (findings, lessons, knowledge-map nodes) as embeddings → semantic recall and
                 cross-domain transfer, which is the whole point of the engine.

Backed by Qdrant (a dockerised vector DB). If Qdrant isn't reachable it degrades to an in-memory store so the backend
still boots for local dev — but the docker-compose wires a real Qdrant so nothing lives in a JSON file.

Embeddings: uses the engine's EmbeddingRetriever (nomic via BOBBY_EMBED_URL) when available; otherwise a deterministic
hashed bag-of-words vector, so semantic search degrades gracefully offline instead of crashing.
"""
import hashlib
import os
import time
from typing import Dict, List, Optional

DIM = int(os.environ.get("BOBBY_VECTOR_DIM", "768"))    # nomic-embed dim; hash fallback matches so the dim is stable
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTIONS = ("runs", "events", "knowledge", "experts")


def _hash_embed(text: str, dim: int = DIM) -> List[float]:
    """Deterministic bag-of-words hashed vector — a dependency-free fallback embedding (good enough for demo recall,
    replaced by a real model when BOBBY_EMBED_URL is set)."""
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _project(vecs: List[List[float]]) -> List[tuple]:
    """2D projection of embedding vectors. PCA (top-2 singular directions) via numpy; deterministic 2-axis fallback."""
    if not vecs:
        return []
    try:
        import numpy as np
        A = np.array(vecs, dtype=float)
        A = A - A.mean(0)
        _u, _s, vt = np.linalg.svd(A, full_matrices=False)
        P = A @ vt[:2].T
        mn, mx = P.min(0), P.max(0)
        rng = mx - mn
        rng[rng == 0] = 1.0
        P = 2 * (P - mn) / rng - 1
        return [tuple(row) for row in P.tolist()]
    except Exception:
        dim = len(vecs[0])
        def axis(seed: str):
            return [((int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16) % 1000) / 500.0 - 1.0) for i in range(dim)]
        a1, a2 = axis("x"), axis("y")
        raw = [(sum(v[i] * a1[i] for i in range(dim)), sum(v[i] * a2[i] for i in range(dim))) for v in vecs]
        def norm(vals):
            mn, mx = min(vals), max(vals)
            rng = (mx - mn) or 1.0
            return [2 * (x - mn) / rng - 1 for x in vals]
        nx, ny = norm([r[0] for r in raw]), norm([r[1] for r in raw])
        return list(zip(nx, ny))


class _Embedder:
    """Prefers the engine's real embeddings; falls back to the hashed vector. Always returns a fixed-DIM vector."""

    def __init__(self):
        self.fn = None
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from bobby_squad.retrieval import default_embed, embedding_available   # noqa
            if embedding_available():
                self.fn = default_embed
        except Exception:
            self.fn = None

    def __call__(self, text: str) -> List[float]:
        if self.fn is not None:
            try:
                out = self.fn([text or ""])
                v = out[0] if out else None
                if v:
                    # project/pad to a stable DIM so the collection dim is model-independent
                    return list(v[:DIM]) if len(v) >= DIM else list(v) + [0.0] * (DIM - len(v))
            except Exception:
                pass
        return _hash_embed(text)


class Store:
    """Thin wrapper over Qdrant with an in-memory fallback. Same API either way."""

    def __init__(self, url: str = QDRANT_URL):
        self.embed = _Embedder()
        self.client = None
        self._mem: Dict[str, List[dict]] = {c: [] for c in COLLECTIONS}
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self.client = QdrantClient(url=url, timeout=5.0)
            for c in COLLECTIONS:
                if not self.client.collection_exists(c):
                    self.client.create_collection(c, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
        except Exception as e:                                    # no qdrant → in-memory
            self.client = None
            self._why = str(e)

    @property
    def backend(self) -> str:
        return "qdrant" if self.client is not None else "memory"

    def _pid(self, kind: str, key: str) -> int:
        return int(hashlib.md5(f"{kind}:{key}".encode()).hexdigest()[:15], 16)

    def _ensure(self, collection: str) -> None:
        """Create the collection on demand — Qdrant upsert 404s on a collection that was never created, and new
        collections (mem_policy, area experts on a fresh volume) aren't in the boot list."""
        try:
            from qdrant_client.models import Distance, VectorParams
            if not self.client.collection_exists(collection):
                self.client.create_collection(collection, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
        except Exception:
            pass

    def upsert(self, collection: str, key: str, payload: dict, text: str = "") -> None:
        vec = self.embed(text or payload.get("summary") or payload.get("text") or key)
        if self.client is not None:
            from qdrant_client.models import PointStruct
            self._ensure(collection)
            self.client.upsert(collection, [PointStruct(id=self._pid(collection, key), vector=vec, payload=payload)])
        else:
            self._mem[collection] = [p for p in self._mem[collection] if p.get("_key") != key]
            self._mem[collection].append({**payload, "_key": key, "_vec": vec})

    def get_run(self, run_id: str) -> Optional[dict]:
        for p in self.list("runs", limit=1000):
            if p.get("run_id") == run_id:
                return p
        return None

    def list(self, collection: str, limit: int = 200, run_id: str = "") -> List[dict]:
        if self.client is not None:
            flt = None
            if run_id:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                flt = Filter(must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))])
            try:
                pts, _ = self.client.scroll(collection, scroll_filter=flt, limit=limit, with_payload=True)
                rows = [p.payload for p in pts]
            except Exception:                          # collection not created yet (no upsert has happened) → empty
                rows = []
        else:
            rows = [p for p in self._mem[collection] if not run_id or p.get("run_id") == run_id]
        rows.sort(key=lambda r: r.get("ts", 0))
        return rows[:limit] if not run_id else rows

    def delete(self, collection: str, key: str) -> bool:
        """Curate the store — remove one point (a knowledge item, a run)."""
        if self.client is not None:
            try:
                self.client.delete(collection, points_selector=[self._pid(collection, key)])
                return True
            except Exception:
                return False
        before = len(self._mem[collection])
        self._mem[collection] = [p for p in self._mem[collection] if p.get("_key") != key]
        return len(self._mem[collection]) != before

    def delete_run(self, run_id: str) -> None:
        """Remove a run and everything it produced (events + knowledge) — manage the vector DB, not just read it."""
        self.delete("runs", run_id)
        if self.client is not None:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                flt = Filter(must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))])
                for c in ("events", "knowledge"):
                    self.client.delete(c, points_selector=flt)
            except Exception:
                pass
        else:
            for c in ("events", "knowledge"):
                self._mem[c] = [p for p in self._mem[c] if p.get("run_id") != run_id]

    def scatter(self, collection: str, limit: int = 400) -> List[dict]:
        """Project the collection's stored embeddings to 2D for the memory map — pure geometry over vectors the store
        already holds (no model call, no prompt). PCA when numpy is present; otherwise a deterministic 2-axis
        projection so it degrades gracefully. Returns payloads with x,y in [-1,1]."""
        rows: List[tuple] = []
        if self.client is not None:
            try:
                pts, _ = self.client.scroll(collection, limit=limit, with_payload=True, with_vectors=True)
                rows = [(p.payload, list(p.vector) if p.vector else None) for p in pts]
            except Exception:
                rows = []
        else:
            for p in self._mem[collection][:limit]:
                rows.append(({k: v for k, v in p.items() if not k.startswith("_")}, p.get("_vec")))
        vecs = [v for _, v in rows if v]
        coords = _project(vecs)
        out, ci = [], 0
        for payload, vec in rows:
            if vec:
                out.append({**payload, "x": round(coords[ci][0], 4), "y": round(coords[ci][1], 4)})
                ci += 1
        return out

    def search(self, collection: str, query: str, limit: int = 10) -> List[dict]:
        """Semantic search — the payoff of a vector store: 'find knowledge about X' across every run."""
        vec = self.embed(query)
        if self.client is not None:
            try:                                          # qdrant-client ≥1.10: query_points (search() was removed)
                res = self.client.query_points(collection, query=vec, limit=limit, with_payload=True)
                return [{**h.payload, "score": round(h.score, 4)} for h in res.points]
            except AttributeError:                        # older client
                hits = self.client.search(collection, query_vector=vec, limit=limit, with_payload=True)
                return [{**h.payload, "score": round(h.score, 4)} for h in hits]
            except Exception:
                return []
        # in-memory cosine
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))
        scored = [({k: v for k, v in p.items() if not k.startswith("_")}, cos(vec, p["_vec"])) for p in self._mem[collection]]
        scored.sort(key=lambda t: -t[1])
        return [{**p, "score": round(s, 4)} for p, s in scored[:limit]]


_STORE: Optional[Store] = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE


def now() -> float:
    return time.time()
