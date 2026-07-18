import json
import math
import os
import re
import urllib.request
from typing import Callable, List, Optional


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


class LexicalRetriever:
    """BM25-lite over an append-only memory store."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: List[str] = []
        self.toks: List[List[str]] = []
        self.df: dict = {}
        self.total_len = 0

    def add(self, text: str) -> None:
        t = tokenize(text)
        self.docs.append(text); self.toks.append(t); self.total_len += len(t)
        for w in set(t):
            self.df[w] = self.df.get(w, 0) + 1

    def add_many(self, texts) -> None:
        for x in texts:
            self.add(x)

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, k: int = 8) -> List[str]:
        n = len(self.docs)
        if n == 0:
            return []
        avgdl = self.total_len / n
        q = [w for w in set(tokenize(query)) if w in self.df]
        scored = []
        for i, dt in enumerate(self.toks):
            if not dt:
                continue
            dl = len(dt)
            s = 0.0
            for w in q:
                tf = dt.count(w)
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.docs[i] for _s, i in scored[:k]]


# ── embedding recall (paraphrase-robust) ────────────────────────────────────────────────────────────
_e = lambda *ks, d="": next((os.environ[k] for k in ks if os.environ.get(k)), d)  # noqa: E731
EMBED_URL = _e("BOBBY_EMBED_URL", "GA_EMBED_URL", d="http://localhost:11434/api/embed")
EMBED_MODEL = _e("BOBBY_EMBED_MODEL", "GA_EMBED_MODEL", d="nomic-embed-text")


def default_embed(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed via an OpenAI/Ollama-style endpoint (default: nomic-embed-text, set BOBBY_EMBED_URL). Returns None on
    any failure so the caller degrades gracefully to lexical retrieval."""
    if not texts:
        return []
    try:
        body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
        req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read()).get("embeddings")
    except Exception:
        return None


def _cos(a, b) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return (s / (na * nb)) if na and nb else 0.0


def embedding_available(embed_fn: Optional[Callable] = None) -> bool:
    v = (embed_fn or default_embed)(["probe"])
    return bool(v)


class EmbeddingRetriever:
    """Semantic recall with the SAME interface as LexicalRetriever. Matches by meaning, so a raw sentence and its
    paraphrased gloss retrieve each other. Caches vectors by text (embeds each unique string once) so persistence
    and reloads are cheap. nomic-embed needs task prefixes — applied automatically."""

    def __init__(self, embed_fn: Optional[Callable] = None, cache: Optional[dict] = None,
                 doc_prefix: str = "search_document: ", query_prefix: str = "search_query: "):
        self.embed_fn = embed_fn or default_embed
        self.dp, self.qp = doc_prefix, query_prefix
        self.cache: dict = cache or {}          # text -> vector (persistable)
        self.docs: List[str] = []
        self.vecs: List[list] = []

    BATCH = 64                                  # embed in chunks — one giant request times out / gets rejected

    def add_many(self, texts) -> None:
        texts = list(texts)
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(miss), self.BATCH):
            chunk = miss[i:i + self.BATCH]
            vs = self.embed_fn([self.dp + t for t in chunk]) or [None] * len(chunk)
            for t, v in zip(chunk, vs):
                if v:
                    self.cache[t] = v
        for t in texts:
            if t in self.cache:
                self.docs.append(t); self.vecs.append(self.cache[t])

    def add(self, text: str) -> None:
        self.add_many([text])

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, k: int = 8) -> List[str]:
        if not self.vecs:
            return []
        qv = self.embed_fn([self.qp + query])
        if not qv or not qv[0]:
            return []
        q = qv[0]
        scored = sorted(((_cos(q, v), i) for i, v in enumerate(self.vecs)), reverse=True)
        return [self.docs[i] for _s, i in scored[:k]]
