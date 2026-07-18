import json
import os
from typing import Callable, Dict, List, Optional, Tuple

from .dedup_ast import AstDedup, fingerprint
from .primitive_intel import (Primitive, PRIMITIVE_SOURCES, _compile_bind, _norm, cross_domain_proof)

# where each known primitive is filed + a natural-language description (what gets embedded into the memory index)
CATEGORY = {"extract_matching": "extraction", "reduce_integers": "arithmetic", "transform_code": "transformation",
            "find_analogous_case": "retrieval"}
DESCRIPTION = {
    "extract_matching": "extract or pull out all tokens in text that match a given pattern",
    "reduce_integers": "aggregate or fold the integers found in text — sum, max, min, count, product",
    "transform_code": "apply a mechanical source-code refactor or rewrite to a line of code",
    "find_analogous_case": "find the most similar or analogous prior case to a query by semantic similarity",
}


def _cos(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class PrimitiveLibrary:
    """A directory-backed, category-organized, memory-indexed primitive store.

    Layout::

        <root>/
        ├── index.json            # {name: {path, category, signature, description, fingerprint, passed_domains, ...}}
        ├── embeddings.jsonl      # {name, vec} — the semantic memory index (persisted)
        └── core/<category>/<name>.py

    `embed_fn` (optional) turns descriptions/queries into vectors for semantic recall; without it, only structural
    (fingerprint) recall is available.
    """

    def __init__(self, root: str, embed_fn: Optional[Callable] = None, autoload: bool = True):
        self.root = os.path.abspath(root)
        self.embed = embed_fn
        os.makedirs(os.path.join(self.root, "core"), exist_ok=True)
        self.index_path = os.path.join(self.root, "index.json")
        self.emb_path = os.path.join(self.root, "embeddings.jsonl")
        self._index: Dict[str, dict] = json.load(open(self.index_path)) if os.path.exists(self.index_path) else {}
        self._emb: Dict[str, List[float]] = {}
        if os.path.exists(self.emb_path):
            for line in open(self.emb_path):
                if line.strip():
                    d = json.loads(line)
                    self._emb[d["name"]] = d["vec"]
        self._dedup = AstDedup()
        self._fp_to_name: Dict[str, str] = {}
        self._loaded: Dict[str, dict] = {}
        if autoload:
            self.load()

    # ── load / organize ────────────────────────────────────────────────────────────────────────────────
    def load(self) -> List[str]:
        self._loaded, self._fp_to_name = {}, {}
        for name, meta in self._index.items():
            path = os.path.join(self.root, meta["path"])
            if not os.path.exists(path):
                continue
            src = open(path).read()
            bind = _compile_bind(src)
            if bind is None:
                continue
            self._loaded[name] = {"bind": bind, "meta": meta}
            fp = meta.get("fingerprint") or fingerprint(src)
            if fp:
                self._fp_to_name[fp] = name
                self._dedup.add(src)
        return list(self._loaded)

    def by_category(self, category: str) -> List[str]:
        return [n for n, m in self._index.items() if m.get("category") == category]

    def categories(self) -> List[str]:
        return sorted({m.get("category", "misc") for m in self._index.values()})

    def names(self) -> List[str]:
        return list(self._loaded)

    def tree(self) -> str:
        lines = ["primitive_lib/"]
        for cat in self.categories():
            lines.append("  " + cat + "/")
            for n in sorted(self.by_category(cat)):
                lines.append("    %-20s %s" % (n + ".py", self._index[n].get("signature", "")))
        return "\n".join(lines)

    # ── RE-FIND (memory) ─────────────────────────────────────────────────────────────────────────────────
    def find_by_fingerprint(self, src: str) -> Optional[str]:
        """Structural recall: return the name of an existing primitive with the SAME functional AST (same loop/logic,
        even under different variable names) — or None. This is how "the same for-loop" is found back, not re-added."""
        fp = fingerprint(src)
        return self._fp_to_name.get(fp) if fp else None

    def recall(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Semantic recall from the memory index: the top-k primitives whose description is closest to `query`."""
        if self.embed is None or not self._emb:
            return []
        q = _norm(self.embed([query])[0])
        scored = [(n, round(_cos(q, _norm(v)), 3)) for n, v in self._emb.items()]
        return sorted(scored, key=lambda x: -x[1])[:k]

    # ── promote (gate + dedup + file + index + memory) ───────────────────────────────────────────────────
    def promote(self, name: str, src: str, signature: str, proof: dict, category: str = "misc",
                description: str = "", params: Optional[dict] = None) -> dict:
        """Add a primitive iff it (a) is not a structural twin of an existing one, (b) compiles, (c) cross-domain
        GENERALIZES. On a twin, returns {reused: <existing name>} instead of adding. Files it under core/<category>/,
        indexes it, and stores its description embedding in the memory index."""
        twin = self.find_by_fingerprint(src)
        if twin is not None:
            return {"added": False, "reused": twin, "reason": "structural-twin (found it back)"}
        if _compile_bind(src) is None:
            return {"added": False, "reused": None, "reason": "compile-error"}
        if not proof.get("generalizes"):
            return {"added": False, "reused": None, "reason": "failed-cross-domain-gate"}
        rel = os.path.join("core", category, name + ".py")
        os.makedirs(os.path.dirname(os.path.join(self.root, rel)), exist_ok=True)
        with open(os.path.join(self.root, rel), "w") as f:
            f.write("# %s | category=%s | passed=%s\n" % (name, category, proof.get("passed")) + src)
        fp = fingerprint(src)
        self._index[name] = {"path": rel, "category": category, "signature": signature, "description": description,
                             "fingerprint": fp, "passed_domains": proof.get("passed", []),
                             "scores": proof.get("scores", {}), "params": params or {}}
        json.dump(self._index, open(self.index_path, "w"), indent=2)
        if self.embed is not None and description:
            vec = self.embed([description])[0]
            self._emb[name] = vec
            with open(self.emb_path, "a") as f:
                f.write(json.dumps({"name": name, "vec": vec}) + "\n")
        self.load()
        return {"added": True, "reused": None, "reason": "promoted"}

    # ── run ──────────────────────────────────────────────────────────────────────────────────────────────
    def bind(self, name: str, param):
        e = self._loaded.get(name)
        return e["bind"](param) if e else None

    def serve(self, name: str, param, payload: dict) -> Optional[str]:
        e = self._loaded.get(name)
        return e["bind"](param)(payload.get("blob", "")) if e else None

    def registry(self) -> dict:
        return dict(self._index)


def recall_or_distill(lib: PrimitiveLibrary, name: str, description: str, src: str, signature: str,
                      domains: Dict[str, dict], category: str = "misc", recall_threshold: float = 0.75,
                      threshold: float = 0.9, min_domains: int = 2) -> dict:
    """The anti-reinvention front door. BEFORE distilling: (1) structural recall — is this same code already here?
    (2) semantic recall — is a primitive that already does this task in memory above `recall_threshold`? If either
    hits, REUSE it (no re-proof, no duplicate). Otherwise cross-domain-prove `src` and, on pass, add it.

    Returns {action: reused-structural | reused-semantic | distilled | rejected, name, ...}."""
    twin = lib.find_by_fingerprint(src)
    if twin is not None:
        return {"action": "reused-structural", "name": twin, "detail": "same functional AST already in lib"}
    hits = lib.recall(description, k=1)
    if hits and hits[0][1] >= recall_threshold:
        return {"action": "reused-semantic", "name": hits[0][0], "score": hits[0][1]}
    bind = _compile_bind(src)
    if bind is None:
        return {"action": "rejected", "reason": "compile"}

    def _pbind(param):
        solve = bind(param)
        return lambda payload: str(solve(payload.get("blob", "")))
    proof = cross_domain_proof(Primitive(name, signature, _pbind), domains, threshold, min_domains)
    res = lib.promote(name, src, signature, proof, category=category, description=description)
    return {"action": "distilled" if res["added"] else "rejected", "name": name, "proof": proof, **res}


def seed_known(lib: PrimitiveLibrary, domains_by_primitive: Dict[str, Dict[str, dict]], **kw) -> Dict[str, dict]:
    """Prove + file every known PRIMITIVE_SOURCES entry that has a domain set — the initial library build."""
    out = {}
    for name, (signature, src) in PRIMITIVE_SOURCES.items():
        if name not in domains_by_primitive:
            continue
        out[name] = recall_or_distill(lib, name, DESCRIPTION.get(name, name), src, signature,
                                      domains_by_primitive[name], category=CATEGORY.get(name, "misc"), **kw)
    return out
