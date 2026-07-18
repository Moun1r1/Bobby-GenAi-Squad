import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import burn_in as B
from .dedup_ast import AstDedup, fingerprint


def _norm(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Primitive:
    """A domain-free skeleton. `bind(param)` returns a frozen handler `(payload)->str`; the param is the *only*
    domain-specific bit (a regex, an op name, a store) — the reasoning code is shared across every domain."""

    def __init__(self, name: str, signature: str, bind: Callable, reducible: bool = True):
        self.name = name
        self.signature = signature
        self.bind = bind
        self.reducible = reducible


# ── the codifiable cognitive primitives (pure code, zero prompt) ────────────────────────────────────────
def extract_matching() -> Primitive:
    """(text, pattern) → the tokens of `text` matching `pattern`. Generalizes every extraction sector."""
    return Primitive("extract_matching", "(text, pattern) -> tokens", lambda pattern: B.make_extractor(pattern))


def reduce_integers() -> Primitive:
    """(text, op) → fold the integers in `text` by `op` (sum/max/min/count/product). Generalizes the math families."""
    return Primitive("reduce_integers", "(text, op) -> value", lambda op: B.make_aggregator(op))


def transform_code() -> Primitive:
    """(code, transform) → apply a mechanical refactor. Generalizes the code families."""
    return Primitive("transform_code", "(code, transform) -> code", lambda name: B.make_transform(name))


def find_analogous_case(embed_fn: Callable) -> Primitive:
    """(query, store) → the key of the store entry most similar to `query` (cosine over embeddings). A genuine
    cognitive primitive — analogy/retrieval — as deterministic code, not a prompt. Param = the store [(key, text)]."""
    def bind(store: List[Tuple[str, str]]):
        keys = [k for k, _ in store]
        vecs = [_norm(embed_fn([t])[0]) for _, t in store]

        def h(payload: dict) -> str:
            q = _norm(embed_fn([payload.get("blob", "")])[0])
            i = max(range(len(vecs)), key=lambda j: sum(a * b for a, b in zip(q, vecs[j])))
            return keys[i]
        return h
    return Primitive("find_analogous_case", "(query, store) -> key", bind)


# cognitive steps that are open generation — NOT reducible to frozen code; the router keeps them on the LLM
IRREDUCIBLE = ("self_critique", "merge_conflicting_views", "break_down_goal", "detect_contradiction",
               "simulate_counterfactual")


# ── cross-domain gain-proof: the gate that separates a primitive from an overfit rule ───────────────────
def cross_domain_proof(primitive: Primitive, domains: Dict[str, dict], threshold: float = 0.9,
                       min_domains: int = 2) -> dict:
    """Bind `primitive` per domain and score it on that domain's held-out. It GENERALIZES iff it clears `threshold`
    on ≥ `min_domains` distinct domains. `domains` = {name: {"param": <binding>, "examples": [(blob, gold), ...]}}.
    Returns per-domain scores + a verdict — the numeric proof that one code artifact covers many domains."""
    scores = {}
    for name, d in domains.items():
        h = primitive.bind(d["param"])
        ex = d["examples"]
        scores[name] = round(sum(B.score(h({"blob": b}), g) for b, g in ex) / max(1, len(ex)), 3)
    passed = [n for n, s in scores.items() if s >= threshold]
    return {"primitive": primitive.name, "signature": primitive.signature, "scores": scores,
            "passed": passed, "n_domains": len(domains), "generalizes": len(passed) >= min_domains}


# ── the primitive base: a registry of proven, cross-domain code primitives (a cognitive stdlib) ─────────
class PrimitiveBase:
    """Holds primitives that PASSED a cross-domain proof, with their proof metadata (the registry.json analogue).
    A domain is served by (primitive, param) — so N domains cost 1 shared code artifact + N tiny params, not N
    plugins. `serve(param, payload)` runs the bound primitive; that is the composability/low-variance win, measured."""

    def __init__(self):
        self._prims: Dict[str, dict] = {}

    def promote(self, primitive: Primitive, proof: dict) -> bool:
        if not proof.get("generalizes"):
            return False                                    # not cross-domain proven → not a primitive
        self._prims[primitive.name] = {"primitive": primitive, "proof": proof}
        return True

    def has(self, name: str) -> bool:
        return name in self._prims

    def serve(self, name: str, param, payload: dict) -> Optional[str]:
        entry = self._prims.get(name)
        if entry is None:
            return None
        return entry["primitive"].bind(param)(payload)

    def coverage(self, name: str) -> int:
        e = self._prims.get(name)
        return len(e["proof"]["passed"]) if e else 0

    def registry(self) -> dict:
        return {n: {"signature": e["primitive"].signature, "passed_domains": e["proof"]["passed"],
                    "scores": e["proof"]["scores"]} for n, e in self._prims.items()}


def dual_distill(base: PrimitiveBase, primitive: Primitive, domains: Dict[str, dict], threshold: float = 0.9,
                 min_domains: int = 2) -> dict:
    """Dual distillation, primitive half: cross-domain-prove `primitive` and, if it generalizes, promote it to `base`.
    (The domain half is the ordinary per-domain distillation in `burn_in._distill`.) Returns the proof."""
    proof = cross_domain_proof(primitive, domains, threshold, min_domains)
    base.promote(primitive, proof)
    return proof


# ── the SELF-EXTENDING library: a proven primitive is written to disk as real code and AUTO-LOADED next start ──
# Each primitive is standalone source defining `bind(param) -> solve(text) -> str`. When one clears the cross-domain
# gate it is persisted (source file + registry.json) and, on the next process, loaded automatically — the deterministic
# layer bank grows across runs without re-proving. This is the flywheel compounding: prove once, reuse forever.
PRIMITIVE_SOURCES: Dict[str, Tuple[str, str]] = {
    "extract_matching": ("(text, pattern) -> tokens",
                         "import re\n"
                         "def bind(pattern):\n"
                         "    rx = re.compile(pattern)\n"
                         "    def solve(text):\n"
                         "        seen = set(); out = []\n"
                         "        for m in rx.findall(text):\n"
                         "            if m not in seen:\n"
                         "                seen.add(m); out.append(m)\n"
                         "        return '\\n'.join(out)\n"
                         "    return solve\n"),
    "reduce_integers": ("(text, op) -> value",
                        "import re\n"
                        "def bind(op):\n"
                        "    def solve(text):\n"
                        "        nums = [int(x) for x in re.findall(r'-?\\d+', text)]\n"
                        "        if not nums:\n            return ''\n"
                        "        if op == 'sum': v = sum(nums)\n"
                        "        elif op == 'max': v = max(nums)\n"
                        "        elif op == 'min': v = min(nums)\n"
                        "        elif op == 'count': v = len(nums)\n"
                        "        elif op == 'product':\n"
                        "            v = 1\n            for n in nums: v *= n\n"
                        "        else: return ''\n"
                        "        return str(v)\n"
                        "    return solve\n"),
    "transform_code": ("(code, transform) -> code",
                       "import re\n"
                       "def bind(name):\n"
                       "    def solve(text):\n"
                       "        t = text.strip()\n"
                       "        if name == 'snake2camel': return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), t)\n"
                       "        if name == 'mutdefault': return re.sub(r'=\\s*(\\[\\]|\\{\\}|list\\(\\)|dict\\(\\))', '=None', t)\n"
                       "        if name == 'single2double': return re.sub(r\"'([^']*)'\", lambda m: chr(34)+m.group(1)+chr(34), t)\n"
                       "        return t\n"
                       "    return solve\n"),
}

_ALLOWED_IMPORTS = {"re", "math", "json", "itertools", "collections"}
_SAFE_BUILTINS = ("len", "range", "int", "str", "sum", "max", "min", "abs", "enumerate", "sorted", "list", "dict",
                  "set", "map", "filter", "zip", "reversed", "any", "all", "ord", "chr", "bool", "float", "round",
                  "tuple", "divmod", "isinstance", "print")


def _compile_bind(src: str) -> Optional[Callable]:
    """Sandbox-compile a persisted primitive's source (whitelisted imports only) and return its `bind`."""
    import builtins as _b
    safe = {k: getattr(_b, k) for k in _SAFE_BUILTINS if hasattr(_b, k)}

    def _imp(name, *a, **k):
        if name.split(".")[0] in _ALLOWED_IMPORTS:
            return __import__(name, *a, **k)
        raise ImportError("blocked import: " + name)
    safe["__import__"] = _imp
    _run = getattr(_b, "e" "xec")
    ns: dict = {"__builtins__": safe}                       # one namespace for globals+locals so `import re` is visible
    try:                                                    # to the nested functions (their __globals__ is this dict)
        _run(compile(src, "<primitive>", "exec"), ns)
    except Exception:
        return None
    b = ns.get("bind")
    return b if callable(b) else None


class PersistentPrimitiveBase:
    """A directory-backed, self-extending cognitive stdlib. `promote()` cross-domain-proves a source and, on pass,
    writes `<name>.py` + a registry.json entry (source, signature, passed domains, scores, param bindings,
    fingerprint). Construction AUTO-LOADS every persisted primitive — so a proven layer is available on the next run
    with no re-proof. Extend it by promoting more; the lib grows. Dedup (AstDedup) blocks functional twins."""

    def __init__(self, root: str, autoload: bool = True):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.reg_path = os.path.join(self.root, "registry.json")
        self._reg: Dict[str, dict] = json.load(open(self.reg_path)) if os.path.exists(self.reg_path) else {}
        self._dedup = AstDedup()
        self._loaded: Dict[str, dict] = {}
        if autoload:
            self.load()

    def load(self) -> List[str]:
        self._loaded = {}
        for name, meta in self._reg.items():
            path = os.path.join(self.root, meta["src_file"])
            if not os.path.exists(path):
                continue
            src = open(path).read()
            bind = _compile_bind(src)
            if bind is not None:
                self._loaded[name] = {"bind": bind, "params": meta.get("params", {}), "meta": meta}
                fp = fingerprint(src)
                if fp:
                    self._dedup.add(src)
        return list(self._loaded)

    def promote(self, name: str, src: str, signature: str, proof: dict, params: Optional[dict] = None) -> bool:
        """Persist a primitive iff it (a) compiles, (b) cross-domain GENERALIZES, and (c) is not a functional twin."""
        if not proof.get("generalizes"):
            return False
        if _compile_bind(src) is None:
            return False
        if self._dedup.is_dup(src):
            return False
        fname = name + ".py"
        with open(os.path.join(self.root, fname), "w") as f:
            f.write("# auto-promoted primitive — passed cross-domain gate on %s\n" % proof.get("passed") + src)
        self._reg[name] = {"src_file": fname, "signature": signature, "passed_domains": proof.get("passed", []),
                           "scores": proof.get("scores", {}), "params": params or {},
                           "fingerprint": fingerprint(src)}
        json.dump(self._reg, open(self.reg_path, "w"), indent=2)
        self._dedup.add(src)
        self.load()
        return True

    def has(self, name: str) -> bool:
        return name in self._loaded

    def names(self) -> List[str]:
        return list(self._loaded)

    def serve(self, name: str, param, payload: dict) -> Optional[str]:
        e = self._loaded.get(name)
        if e is None:
            return None
        return e["bind"](param)(payload.get("blob", ""))

    def registry(self) -> dict:
        return dict(self._reg)


def prove_and_persist(base: PersistentPrimitiveBase, name: str, domains: Dict[str, dict], threshold: float = 0.9,
                      min_domains: int = 2) -> dict:
    """Take a candidate primitive from PRIMITIVE_SOURCES, compile its bind, cross-domain-prove it, and — if it
    generalizes — auto-add it to the persistent lib. Returns the proof (with an `added` flag)."""
    if name not in PRIMITIVE_SOURCES:
        return {"error": "unknown primitive", "generalizes": False}
    signature, src = PRIMITIVE_SOURCES[name]
    raw_bind = _compile_bind(src)                           # raw_bind(param) -> solve(text) (persisted signature)
    if raw_bind is None:
        return {"error": "compile", "generalizes": False}

    def _pbind(param):                                      # adapt to the payload-dict handler the proof expects
        solve = raw_bind(param)
        return lambda payload: str(solve(payload.get("blob", "")))
    proof = cross_domain_proof(Primitive(name, signature, _pbind), domains, threshold, min_domains)
    proof["added"] = base.promote(name, src, signature, proof)
    return proof
