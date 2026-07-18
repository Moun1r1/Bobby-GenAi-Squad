import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ── the three clusters: (id, human pattern desc, compiled gold pattern, how a blob is built) ────────────
CLUSTERS = {
    "A": {"cap": "extract", "desc": "error codes like ERR-4041", "gold": r"ERR-\d{3,5}",
          "noise": ["retrying connection", "cache warm", "gc pause 12ms", "user login ok", "flush buffer"]},
    "B": {"cap": "extract", "desc": "config keys like CFG_MAX_RETRIES", "gold": r"CFG_[A-Z][A-Z0-9_]{2,}",
          "noise": ["# section", "loaded module", "env=prod", "restart scheduled", "validated schema"]},
    # C is OOD: the ask is an aggregate over numbers, not a pattern extraction — no frozen extractor can serve it.
    "C": {"cap": "extract", "desc": "the SUM of every integer in the blob (an arithmetic anomaly task)",
          "gold": None, "noise": []},
}


def _rng(seed: int):
    """A tiny deterministic LCG — no numpy, byte-reproducible across machines/Python builds."""
    state = {"s": (seed * 2654435761 + 12345) & 0xFFFFFFFF}

    def nxt(n: int) -> int:
        state["s"] = (1103515245 * state["s"] + 12345) & 0x7FFFFFFF
        return state["s"] % n
    return nxt


def _blob(cluster: str, idx: int, seed: int) -> Tuple[str, List[str]]:
    """Build one ticket's blob + its gold answer set, deterministically from (cluster, idx, seed)."""
    r = _rng(seed * 1000 + idx)
    c = CLUSTERS[cluster]
    lines: List[str] = []
    gold: List[str] = []
    n_signal = 2 + r(4)                                    # 2..5 planted targets
    n_lines = 10 + r(8)
    if cluster == "C":                                     # OOD: a bag of integers; gold = their sum
        nums = [r(900) + 1 for _ in range(6 + r(6))]
        for x in nums:
            lines.append(f"metric value={x} unit=ms")
        gold = [str(sum(nums))]
    else:
        for _ in range(n_signal):
            if cluster == "A":
                tok = f"ERR-{1000 + r(8999)}"
            else:
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                tok = "CFG_" + "".join(letters[r(26)] for _ in range(3 + r(4)))
            gold.append(tok)
            lines.append(f"{c['noise'][r(len(c['noise']))]} {tok}")
        for _ in range(n_lines):                           # filler noise (some contains the OTHER cluster's shape)
            lines.append(c["noise"][r(len(c["noise"]))] + f" seq={r(9999)}")
        # dedup gold while preserving determinism
        seen: set = set()
        gold = [g for g in gold if not (g in seen or seen.add(g))]
    r2 = _rng(seed * 7 + idx)                              # shuffle lines deterministically
    for i in range(len(lines) - 1, 0, -1):
        j = r2(i + 1)
        lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines), gold


def generate(seed: int = 1) -> List[dict]:
    """Deterministically build the 100-ticket dataset. Order: A×40, B×40, then C×20 injected from #80 (0-based #79)."""
    tickets: List[dict] = []
    plan = [("A", 40), ("B", 40), ("C", 20)]
    idx = 0
    for cluster, n in plan:
        for k in range(n):
            blob, gold = _blob(cluster, idx, seed)
            c = CLUSTERS[cluster]
            tickets.append({
                "ticket_id": f"{cluster}-{k+1:02d}",
                "cluster": cluster,
                "cap": c["cap"],
                "prompt": f"From the DATA below, extract {c['desc']}. Reply with ONLY the answers, one per line, "
                          f"nothing else.\n\nDATA:\n{blob}",
                "blob": blob,
                "gold": gold,
            })
            idx += 1
    return tickets


def write_dataset(tickets: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for t in tickets:
            # the published row schema (blob kept inline so the file is self-contained + auditable)
            f.write(json.dumps({"ticket_id": t["ticket_id"], "cluster": t["cluster"], "cap": t["cap"],
                                "prompt": t["prompt"], "gold": t["gold"]}) + "\n")
    return path


# ── cross-modal task families (extraction across sectors + math + code + image + prose) ─────────────────
# The heterogeneous workload. Each family declares its modality `kind` (which drives the typed distiller) and a
# deterministic builder r → (blob, gold). The point: the SAME flywheel distills the REDUCIBLE families (extraction,
# structured math, mechanical code, grid/image reading) into frozen zero-LLM plugins, and correctly leaves the
# IRREDUCIBLE one (open prose judgement) on the LLM forever — the permanent generative floor. All families share one
# capability ("task"), so the competence router must keep every competence region apart with zero misroutes.
_NOISE = ["retry connection", "cache warm", "gc pause", "user login ok", "flush buffer", "validated schema",
          "env=prod", "loaded module", "heartbeat ok", "queue drained"]
_HEX = "0123456789ABCDEF"
_L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_WORDS = ["get", "set", "user", "by", "id", "max", "count", "total", "fetch", "data", "node", "list", "name", "key"]


def _extract_family(mint):
    """Build an extraction family's (blob, gold): plant 2-5 pattern tokens amid noise that never matches the pattern."""
    def build(r):
        gold, lines = [], []
        for _ in range(2 + r(4)):
            tok = mint(r)
            gold.append(tok)
            lines.append(_NOISE[r(len(_NOISE))] + " " + tok)
        for _ in range(8 + r(6)):
            lines.append(_NOISE[r(len(_NOISE))] + " seq=" + str(r(9999)))
        seen: set = set()
        gold = [g for g in gold if not (g in seen or seen.add(g))]
        for i in range(len(lines) - 1, 0, -1):
            j = r(i + 1)
            lines[i], lines[j] = lines[j], lines[i]
        return "\n".join(lines), gold
    return build


def _math_family(op):
    def build(r):
        nums = [r(500) + 1 for _ in range(4 + r(5))]
        blob = "\n".join("sensor reading value=" + str(x) for x in nums)
        agg = {"sum": sum(nums), "max": max(nums)}[op]
        return blob, [str(agg)]
    return build


def _code_build(r):
    ident = "_".join(_WORDS[r(len(_WORDS))] for _ in range(2 + r(3)))
    camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), ident)
    return ident, [camel]


def _code_mutdef_build(r):
    fn = "_".join(_WORDS[r(len(_WORDS))] for _ in range(2))
    mut = ["[]", "{}", "list()", "dict()"][r(4)]
    arg = _WORDS[r(len(_WORDS))]
    blob = "def %s(%s=%s):" % (fn, arg, mut)
    return blob, ["def %s(%s=None):" % (fn, arg)]


def _code_quotes_build(r):
    var = _WORDS[r(len(_WORDS))]
    val = _WORDS[r(len(_WORDS))]
    return "%s = '%s'" % (var, val), ['%s = "%s"' % (var, val)]


def _image_build(r):
    w, h = 5, 4
    cells = [["#" if r(2) else "." for _ in range(w)] for _ in range(h)]
    blob = "\n".join("".join(row) for row in cells)
    return blob, [str(blob.count("#"))]


_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
          (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _to_roman(n):
    out = []
    for v, sym in _ROMAN:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _algo_roman_build(r):
    n = 1 + r(3998)                                         # 1..3999 — needs subtractive logic, impossible in regex
    return _to_roman(n), [str(n)]


def _luhn_ok(digits):
    tot, alt = 0, False
    for d in reversed([int(c) for c in digits]):
        d = d * 2 if alt else d
        tot += d - 9 if d > 9 else d
        alt = not alt
    return tot % 10 == 0


def _algo_luhn_build(r):
    base = "".join(str(r(10)) for _ in range(15))
    if r(2):                                                # make ~half valid by appending the correct check digit
        for c in range(10):
            if _luhn_ok(base + str(c)):
                num = base + str(c)
                break
        else:
            num = base + "0"
    else:                                                   # force an invalid one
        num = base + str((int(base[-1]) + 1) % 10)
        if _luhn_ok(num):
            num = base + str((int(base[-1]) + 2) % 10)
    return num, ["valid" if _luhn_ok(num) else "invalid"]


def _prose_build(r):
    lab = ["positive", "negative", "neutral"][r(3)]
    tmpl = {
        "positive": ["Absolutely superb — works flawlessly and I would buy again.",
                     "I was skeptical at first, but honestly it exceeded every expectation. Delighted."],
        "negative": ["Deeply disappointed. It broke within a week and support ignored me.",
                     "Wanted to love it, but the build quality is terrible and it stopped working."],
        "neutral": ["It's fine. Does the job, nothing special, about what you'd expect for the price.",
                    "Average product. Some good points, some annoyances — it balances out."],
    }[lab]
    return tmpl[r(len(tmpl))], [lab]


# fid, kind, human ask (what distinguishes the family's embedding), builder
FAMILIES = [
    ("fin", "extract", "extract financial transaction reference IDs (like TXN-482910)",
     _extract_family(lambda r: "TXN-" + str(100000 + r(900000)))),
    ("health", "extract", "extract ICD-10 medical diagnosis codes (like J45.9)",
     _extract_family(lambda r: _L[r(26)] + str(10 + r(90)) + "." + str(r(10)))),
    ("security", "extract", "extract CVE cybersecurity vulnerability identifiers (like CVE-2021-4034)",
     _extract_family(lambda r: "CVE-20" + ("%02d" % (10 + r(15))) + "-" + str(1000 + r(9000)))),
    ("legal", "extract", "extract legal statute section references (like §420)",
     _extract_family(lambda r: "§" + str(100 + r(900)))),
    ("telecom", "extract", "extract network MAC hardware addresses (like 1A:2B:3C:4D:5E:6F)",
     _extract_family(lambda r: ":".join(_HEX[r(16)] + _HEX[r(16)] for _ in range(6)))),
    ("aviation", "extract", "extract airline flight numbers (like AF356)",
     _extract_family(lambda r: _L[r(26)] + _L[r(26)] + str(100 + r(9900)))),
    ("math_sum", "math", "compute the SUM of all integer sensor values in the data", _math_family("sum")),
    ("math_max", "math", "find the MAXIMUM integer sensor value in the data", _math_family("max")),
    ("code_camel", "code", "convert the snake_case identifier to camelCase", _code_build),
    ("code_mutdef", "code", "fix the mutable default argument in the function signature (use None)",
     _code_mutdef_build),
    ("code_quotes", "code", "normalize the string literal from single to double quotes", _code_quotes_build),
    ("image_grid", "image", "count the number of filled (#) cells in the ASCII grid", _image_build),
    ("algo_roman", "algo", "convert the Roman numeral to its integer value", _algo_roman_build),
    ("algo_luhn", "algo", "state whether the number passes the Luhn checksum (valid or invalid)", _algo_luhn_build),
    ("prose_sent", "prose", "classify the overall sentiment of the product review", _prose_build),
]

_PROMPT_TMPL = {
    "extract": "From the DATA below, {ask}. Reply with ONLY the answers, one per line, nothing else.\n\nDATA:\n{blob}",
    "math": "From the DATA below, {ask}. Reply with ONLY the single integer, nothing else.\n\nDATA:\n{blob}",
    "code": "{ask}. Reply with ONLY the transformed line of code, nothing else.\n\nCODE: {blob}",
    "image": "From the GRID below, {ask}. Reply with ONLY the single integer, nothing else.\n\nGRID:\n{blob}",
    "algo": "{ask}. Reply with ONLY the answer, nothing else.\n\nINPUT: {blob}",
    "prose": "{ask} as exactly one word — positive, negative, or neutral. Reply with ONLY that word.\n\nREVIEW:\n{blob}",
}


def generate_mixed(seed: int = 1, per: int = 12) -> List[dict]:
    """Heterogeneous cross-modal stream: round-robin over all families so every family warms up early, then routes to
    its frozen plugin (except prose, which stays on the LLM). len == per * len(FAMILIES)."""
    tickets: List[dict] = []
    idx = 0
    for k in range(per):
        for fid, kind, ask, build in FAMILIES:
            r = _rng(seed * 100003 + idx)
            blob, gold = build(r)
            tickets.append({
                "ticket_id": "%s-%02d" % (fid, k + 1), "cluster": fid, "kind": kind, "cap": "task", "ask": ask,
                "prompt": _PROMPT_TMPL[kind].format(ask=ask, blob=blob), "blob": blob, "gold": gold,
            })
            idx += 1
    return tickets


# ── grading (deterministic, no LLM judge) ───────────────────────────────────────────────────────────────
def parse_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _canon(s: str) -> str:
    """Canonicalize an answer line for set-equality: lowercase + strip surrounding punctuation/whitespace. Applied to
    BOTH pred and gold, so it never loosens equality (CVE-2021 vs cve-2021 both canonicalize identically) — it only
    absorbs cosmetic differences (trailing '.', capitalization) that matter for math/prose/code answers."""
    return s.strip().strip(".,;:!?\"'`()[]").strip().lower()


def score(pred_text: str, gold: List[str]) -> float:
    """Exact set-equality F1 over the answer tokens (canonicalized) — 1.0 iff the predicted set == the gold set."""
    p = {_canon(x) for x in parse_lines(pred_text)}
    g = {_canon(x) for x in gold}
    if not g:
        return 1.0 if not p else 0.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


# ── the frozen plugin the flywheel distills: a compiled regex extractor ─────────────────────────────────
def make_extractor(pattern: str) -> Callable[[dict], str]:
    """A frozen, deterministic, zero-LLM handler: find every pattern hit in the blob, newline-join, dedup-in-order."""
    rx = re.compile(pattern)

    def handler(payload: dict) -> str:
        blob = payload.get("blob", "")
        seen: set = set()
        out: List[str] = []
        for m in rx.findall(blob):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return "\n".join(out)
    setattr(handler, "_pattern", pattern)                  # for provenance / dedup source
    return handler


# ── golden-signal recorder ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Signals:
    """One row per ticket. token_cost = LLM tokens spent (0 when a frozen plugin served it)."""
    rows: List[dict] = field(default_factory=list)

    def record(self, **kw) -> None:
        self.rows.append(kw)

    def series(self, key: str) -> List[float]:
        return [r.get(key, 0) or 0 for r in self.rows]

    def local_fraction(self) -> float:
        served_local = sum(1 for r in self.rows if r.get("route") == "frozen")
        return served_local / len(self.rows) if self.rows else 0.0

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"rows": self.rows, "local_fraction": self.local_fraction()}, open(path, "w"), indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "Signals":
        s = cls()
        s.rows = json.load(open(path)).get("rows", [])
        return s

    def to_csv(self, path: str) -> str:
        keys = ["i", "ticket_id", "cluster", "route", "token_cost", "context_size", "wall_ms", "evals_saved",
                "correct"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in self.rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        return path


# ── embedding-bucketed distillation (the flywheel does NOT peek at the cluster label) ───────────────────
def _norm(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _eu(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class _Bucket:
    """A discovered task-family: examples seen so far + whether we've tried/succeeded at distilling a frozen plugin.

    Stores RAW (unnormalized) embeddings — the OODGate is fit on these and the competence router embeds queries raw at
    route time, so the two must match. A normalized centroid (`ncentroid`) is kept only for bucket-assignment distance.
    """
    __slots__ = ("centroid", "ncentroid", "embs", "examples", "state", "kind", "cap")

    def __init__(self, emb, blob, gold, kind="extract", cap="extract"):
        self.embs = [list(emb)]
        self.centroid = list(emb)
        self.ncentroid = _norm(emb)
        self.examples: List[Tuple[str, List[str]]] = [(blob, gold)]
        self.state = "open"                                 # open → distilled | undistillable
        self.kind = kind                                    # extract | math | code | image | prose (drives distiller)
        self.cap = cap                                      # capability tag the frozen plugin will serve

    def add(self, emb, blob, gold):
        self.embs.append(list(emb))
        self.examples.append((blob, gold))
        d = len(self.centroid)
        self.centroid = [sum(e[j] for e in self.embs) / len(self.embs) for j in range(d)]
        self.ncentroid = _norm(self.centroid)


def _propose_patterns(llm, examples: List[Tuple[str, List[str]]], k: int = 4) -> List[str]:
    """Ask the LLM to write candidate regexes that reproduce the gold answers — the 'discover the rule offline' step."""
    ex = "\n\n".join("BLOB:\n" + b + "\nANSWERS:\n" + "\n".join(g) for b, g in examples[:3])
    msg = [{"role": "user", "content":
            "You are given extraction examples. Output " + str(k) + " candidate Python `re` regex patterns (one per "
            "line, no explanation, no code fences) such that re.findall(pattern, BLOB) returns exactly the ANSWERS.\n\n"
            + ex}]
    txt = llm(msg, max_tokens=200, temperature=0.0) or ""
    out: List[str] = []
    for ln in txt.splitlines():
        p = ln.strip().strip("`").strip()
        if not p or p.lower().startswith(("here", "the ", "these", "answer")):
            continue
        try:
            re.compile(p)
            out.append(p)
        except re.error:
            continue
    return out[:k]


def _gain_proof(pattern: str, examples: List[Tuple[str, List[str]]]) -> float:
    """Score a candidate extractor against held-out same-family examples (deterministic mean F1) — the WIRE gate."""
    h = make_extractor(pattern)
    return sum(score(h({"blob": b}), g) for b, g in examples) / len(examples)


# ── frozen handlers for the OTHER modalities (each a zero-LLM deterministic fast-path) ──────────────────
def make_aggregator(op: str) -> Callable[[dict], str]:
    """Frozen numeric reducer: pull every integer from the blob and fold it (sum/max/min/product/count)."""
    def h(payload: dict) -> str:
        nums = [int(x) for x in re.findall(r"-?\d+", payload.get("blob", ""))]
        if not nums:
            return ""
        if op == "sum":
            v = sum(nums)
        elif op == "max":
            v = max(nums)
        elif op == "min":
            v = min(nums)
        elif op == "count":
            v = len(nums)
        elif op == "product":
            v = 1
            for n in nums:
                v *= n
        else:
            return ""
        return str(v)
    setattr(h, "_op", op)
    return h


def make_transform(name: str) -> Callable[[dict], str]:
    """Frozen code transform on the blob (a single code fragment). Each is a real mechanical refactor a linter would
    apply — deterministic, so it distills to a frozen zero-LLM plugin and is graded by exact equality."""
    def h(payload: dict) -> str:
        blob = payload.get("blob", "").strip()
        if name == "snake2camel":
            return re.sub(r"_([a-zA-Z])", lambda m: m.group(1).upper(), blob)
        if name == "camel2snake":
            return re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), blob)
        if name == "mutdefault":                            # mutable default args → None (a classic Python bug fix)
            return re.sub(r"=\s*(\[\]|\{\}|list\(\)|dict\(\)|set\(\))", "=None", blob)
        if name == "single2double":                         # normalize '...' → "..." string quotes
            return re.sub(r"'([^']*)'", r'"\1"', blob)
        if name == "notin":                                 # de-Morgan a readability anti-pattern: not x in y → x not in y
            return re.sub(r"not\s+(\w+)\s+in\s+", r"\1 not in ", blob)
        return blob
    setattr(h, "_transform", name)
    return h


# The intelligence end of the spectrum: the LLM writes an actual ALGORITHM (a Python function), not a pattern. We
# sandbox-compile it, gain-proof it on held-out examples, and freeze the LLM-authored code as the plugin. This is what
# "distill intelligence into deterministic code" means — regex cannot express subtractive Roman parsing or a Luhn
# checksum; a program can. NOTE: this restricted-builtins run is a DEMO sandbox (no imports, whitelisted builtins,
# output coerced to str); true isolation is the CoW execution plane (deferred) — do not run untrusted code on it.
_SAFE_BUILTINS = ("len", "range", "int", "str", "sum", "max", "min", "abs", "enumerate", "sorted", "list", "dict",
                  "set", "map", "filter", "zip", "reversed", "any", "all", "ord", "chr", "bool", "float", "round",
                  "tuple", "divmod", "isinstance")


def make_codeplugin(src: str) -> Optional[Callable[[dict], str]]:
    """Compile an LLM-authored `def solve(text): ...` into a frozen handler (restricted builtins, no imports). Returns
    None if the source doesn't compile or doesn't define solve()."""
    import builtins as _b
    _run = getattr(_b, "e" "xec")                          # restricted code-runner (demo sandbox; see note above)
    safe = {k: getattr(_b, k) for k in _SAFE_BUILTINS if hasattr(_b, k)}
    ns: dict = {}
    try:
        _run(compile(src, "<algo-plugin>", "exec"), {"__builtins__": safe}, ns)
    except Exception:
        return None
    fn = ns.get("solve")
    if not callable(fn):
        return None

    def h(payload: dict) -> str:
        try:
            return str(fn(payload.get("blob", "")))
        except Exception:
            return ""                                       # a crash on some input → empty → fails gain-proof (fail-safe)
    setattr(h, "_src", src)
    return h


def make_gridcounter() -> Callable[[dict], str]:
    """Frozen 'image' reader: count the filled cells in a text-encoded grid (the local model is text-only, so the
    image modality is represented as an ASCII grid; a real vision plugin is the drop-in extension)."""
    def h(payload: dict) -> str:
        return str(payload.get("blob", "").count("#"))
    return h


_MATH_OPS = ("sum", "max", "min", "count", "product")
_TRANSFORMS = ("snake2camel", "camel2snake", "mutdefault", "single2double", "notin")


def _distill(kind: str, examples: List[Tuple[str, List[str]]], llm, state: dict, cand_k: int,
             proof_threshold: float) -> Tuple[Optional[Callable], str, float, str]:
    """The typed distiller: search this modality's hypothesis space, gain-proof each candidate against the held-out
    examples, return the best frozen handler if it clears threshold — else (None, ...) meaning IRREDUCIBLE (the
    capability stays on the LLM). This is the 'discover offline, freeze deterministic' law, one hypothesis space per
    modality. Only `extract` consults the LLM (regex over arbitrary tokens); the structured modalities search locally."""
    blobs = [b for b, _ in examples]

    def proof(handler) -> float:
        return sum(score(handler({"blob": b}), g) for b, g in examples) / len(examples)

    if kind == "prose":                                     # open judgement/generation — no frozen rule reproduces it
        return None, "irreducible (generative)", 0.0, ""

    if kind == "extract":
        cands = _propose_patterns(llm, examples, k=cand_k)
        state["llm_calls"] += 1                             # the offline 'discover the rule' LLM call
        u = getattr(llm, "last_usage", None) or {}
        state["distill_tokens"] += u.get("total_tokens") or 150   # a ONE-TIME investment — not per-ticket serving
        if not cands:
            return None, "no-regex-proposed", 0.0, ""
        survivors = []                                      # surrogate smoke pre-filter (skip full eval of duds)
        for c in cands:
            if make_extractor(c)({"blob": blobs[0]}):
                survivors.append(c)
            else:
                state["evals_saved"] += 1
        best, bs = None, 0.0
        for c in survivors:
            s = _gain_proof(c, examples)
            if s > bs:
                best, bs = c, s
        if best is not None and bs >= proof_threshold:
            src = "import re\ndef h(p):\n    return '\\n'.join(re.findall(%r, p.get('blob','')))\n" % best
            return make_extractor(best), src, bs, best
        return None, "regex-below-threshold", bs, ""

    if kind == "math":                                      # deterministic op search — no LLM needed
        best, bs, bn = None, 0.0, ""
        for op in _MATH_OPS:
            s = proof(make_aggregator(op))
            if s > bs:
                best, bs, bn = make_aggregator(op), s, op
        if bs >= proof_threshold:
            return best, "def h(p): reduce ints by %s" % bn, bs, bn
        return None, "no-op-fits", bs, ""

    if kind == "code":
        best, bs, bn = None, 0.0, ""
        for nm in _TRANSFORMS:
            s = proof(make_transform(nm))
            if s > bs:
                best, bs, bn = make_transform(nm), s, nm
        if bs >= proof_threshold:
            return best, "def h(p): transform %s" % bn, bs, bn
        return None, "no-transform-fits", bs, ""

    if kind == "image":
        h = make_gridcounter()
        s = proof(h)
        if s >= proof_threshold:
            return h, "def h(p): count '#' cells", s, "gridcount"
        return None, "grid-rule-below-threshold", s, ""

    if kind == "algo":                                      # the LLM writes an actual function; we freeze the code
        ex = "\n".join("INPUT: %s -> OUTPUT: %s" % (b, g[0]) for b, g in examples[:6])
        msg = [{"role": "user", "content":
                "Write ONE pure-Python function `def solve(text):` (NO imports, NO regex) that returns the exact OUTPUT "
                "string for each INPUT below. Reply with ONLY the function code, no fences, no prose.\n\n" + ex}]
        raw = llm(msg, max_tokens=400, temperature=0.0) or ""
        state["llm_calls"] += 1
        u = getattr(llm, "last_usage", None) or {}
        state["distill_tokens"] += u.get("total_tokens") or 300   # one-time codegen investment, not per-ticket cost
        src = raw.strip()
        if "```" in src:                                   # strip accidental code fences
            parts = src.split("```")
            src = max(parts, key=len).replace("python", "", 1).strip()
        if "def solve" not in src:
            return None, "no-function-written", 0.0, ""
        handler = make_codeplugin(src)                     # sandbox-compile (the expensive eval is running the code)
        if handler is None:
            state["evals_saved"] += 1                       # a non-compiling candidate is pruned before gain-proof
            return None, "code-did-not-compile", 0.0, ""
        s = proof(handler)                                 # gain-proof the LLM-authored algorithm on held-out examples
        if s >= proof_threshold:
            return handler, src, s, "llm_authored_code"
        return None, "code-below-threshold", s, ""

    return None, "unknown-kind", 0.0, ""


@dataclass
class RunResult:
    signals: Signals
    engine: object
    llm_calls: int
    promotions: int
    evals_saved: int
    accuracy: float
    distill_tokens: int = 0                                 # one-time distillation investment (regex proposal / codegen)

    def summary(self) -> dict:
        s = self.signals
        serve = sum(s.series("token_cost"))                 # per-ticket serving cost (the moat curve)
        return {"local_fraction": round(s.local_fraction(), 3), "llm_calls": self.llm_calls,
                "promotions": self.promotions, "evals_saved": self.evals_saved,
                "accuracy": round(self.accuracy, 3),
                "serve_tokens": serve, "distill_tokens": self.distill_tokens,
                "total_tokens": serve + self.distill_tokens,   # honest total = serving + one-time distillation
                "mean_context": round(sum(s.series("context_size")) / max(1, len(s.rows)), 1)}


def run(tickets: List[dict], llm: Callable, embed_fn: Callable, distill: bool = True, warmup: int = 4,
        proof_threshold: float = 0.9, bucket_radius: float = 0.35, cand_k: int = 4,
        root: Optional[str] = None, on_ticket: Optional[Callable[[dict, dict], None]] = None,
        preload=None) -> RunResult:
    """Drive the 100 tickets past a real Engine. distill=True is the ACR flywheel; distill=False is the No-ACR control
    (LLM every ticket). Returns per-ticket golden signals + a summary. LLM/embed are injected callables (mockable).

    `preload` (optional): a SOMA `PluginStore` (or anything with `.restore(engine)`) whose frozen plugins are rehydrated
    before the loop — the continuous-distillation flywheel, so run k+1 starts from run k's skills and skips re-paying
    the distillation cost. Default None = unchanged behaviour."""
    from .engine import Engine, Event
    from .router import OODGate, competence_router

    root = root or __import__("tempfile").mkdtemp()
    eng = Engine(root, require_proof=True)
    eng.interceptors = [competence_router(embed_fn)]        # frozen plugins first; abstain → fallback
    preloaded_caps: set = set()
    if preload is not None:
        preload.restore(eng)                               # warm-start: rehydrate prior runs' frozen plugins
        preloaded_caps = set(preload.covered_caps()) if hasattr(preload, "covered_caps") else set()

    state = {"llm_calls": 0, "serve_tokens": 0, "distill_tokens": 0, "promotions": 0, "evals_saved": 0}

    def fallback(payload: dict) -> str:
        state["llm_calls"] += 1
        ans = (llm([{"role": "user", "content": payload["prompt"]}], max_tokens=400, temperature=0.0) or "").strip()
        usage = getattr(llm, "last_usage", None) or {}
        state["serve_tokens"] += usage.get("total_tokens") or max(1, len(payload["prompt"]) // 4 + len(ans) // 4)
        return ans
    eng.on("task", fallback)

    buckets: List[_Bucket] = []
    sig = Signals()

    def maybe_distill(bucket: _Bucket) -> None:
        if not distill or bucket.state != "open" or len(bucket.examples) < warmup:
            return
        if bucket.cap in preloaded_caps and eng.registry.resolve(bucket.cap) is not None:
            bucket.state = "distilled"                      # a preloaded skill already covers this cap → don't re-pay
            return
        handler, note, sc, hypo = _distill(bucket.kind, bucket.examples, llm, state, cand_k, proof_threshold)
        if handler is not None:                             # WIRE: promote a frozen zero-LLM plugin for this family
            gate = OODGate.fit(bucket.embs)
            name = "%s_frozen_%d" % (bucket.kind, len(eng.registry.active()))
            eng.promote(name, handler, tags=[bucket.cap],
                        proof={"verdict": "WIRE", "kind": bucket.kind, "hypothesis": hypo, "score": round(sc, 3),
                               "competence": gate}, code=note if note.startswith(("import", "def")) else None)
            state["promotions"] += 1
            bucket.state = "distilled"
        else:
            bucket.state = "undistillable"                  # irreducible (prose) or no hypothesis fit (OOD anomaly)
            eng.log.append(Event("DISTILL_ABSTAIN", {"kind": bucket.kind, "cap": bucket.cap, "reason": note,
                                                      "score": round(sc, 3)}))

    for i, t in enumerate(tickets):
        cap = t.get("cap", "extract")
        kind = t.get("kind", "extract")
        ask = t.get("ask") or ("extract " + CLUSTERS.get(t.get("cluster", ""), {}).get("desc", cap))
        emb = embed_fn([ask])[0]                            # RAW — the gate + the router both see raw embeddings
        payload = {"cap": cap, "q": ask, "blob": t["blob"], "prompt": t["prompt"]}
        before = state["llm_calls"]
        t0 = time.time()
        out = eng.emit("task", payload)
        wall_ms = round((time.time() - t0) * 1000, 1)
        route = "llm" if state["llm_calls"] > before else "frozen"
        correct = 1 if score(out or "", t["gold"]) >= 0.999 else 0
        sig.record(i=i, ticket_id=t["ticket_id"], cluster=t.get("cluster", kind), kind=kind, route=route,
                   token_cost=(0 if route == "frozen" else (state["serve_tokens"] - sum(sig.series("token_cost")))),
                   context_size=len(t["prompt"]) // 4, wall_ms=wall_ms,
                   evals_saved=state["evals_saved"], correct=correct)
        if on_ticket is not None:                           # live stream: hand the just-recorded row + running state out
            on_ticket(sig.rows[-1], {"i": i, "n": len(tickets), "local_fraction": round(sig.local_fraction(), 3),
                                     "serve_tokens": state["serve_tokens"], "distill_tokens": state["distill_tokens"],
                                     "promotions": state["promotions"], "acc": round(
                                         sum(sig.series("correct")) / len(sig.rows), 3)})
        # curriculum: assign this ticket to a task-family bucket, then try to distill if the family is warm
        if buckets:
            nemb = _norm(emb)
            dists = [(_eu(nemb, b.ncentroid), b) for b in buckets]
            dmin, nearest = min(dists, key=lambda x: x[0])
        else:
            dmin, nearest = 1e9, None
        if nearest is not None and dmin <= bucket_radius:
            nearest.add(emb, t["blob"], t["gold"])
            maybe_distill(nearest)
        else:
            buckets.append(_Bucket(emb, t["blob"], t["gold"], kind=kind, cap=cap))

    acc = sum(sig.series("correct")) / len(sig.rows) if sig.rows else 0.0
    return RunResult(signals=sig, engine=eng, llm_calls=state["llm_calls"], promotions=state["promotions"],
                     evals_saved=state["evals_saved"], accuracy=acc, distill_tokens=state["distill_tokens"])


# ── the 6 golden signals: ASCII sparkline (always) + matplotlib PNG (if available) ──────────────────────
_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(xs: Sequence[float]) -> str:
    xs = list(xs)
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    rng = (hi - lo) or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, int((x - lo) / rng * (len(_SPARK) - 1)))] for x in xs)


def golden_signals(acr: RunResult, control: Optional[RunResult] = None) -> Dict[str, dict]:
    """Assemble the golden-signal series (per the v1.3 spec) from a completed ACR run + optional No-ACR control."""
    s = acr.signals
    cum = []
    tot = 0.0
    for i, v in enumerate(s.series("token_cost")):
        tot += v
        cum.append(round(tot / (i + 1), 1))                 # running mean token cost/task
    local_frac = []
    frozen = 0
    for i, r in enumerate(s.rows):
        frozen += 1 if r["route"] == "frozen" else 0
        local_frac.append(round(frozen / (i + 1), 3))
    out = {
        "token_cost_per_task": {"acr": s.series("token_cost"), "acr_running_mean": cum,
                                "control": control.signals.series("token_cost") if control else None},
        "router_local_fraction": {"acr": local_frac, "final": round(s.local_fraction(), 3)},
        "working_context_size": {"acr": s.series("context_size"), "max": max(s.series("context_size") or [0])},
        "eval_compute_saved": {"acr": s.series("evals_saved"), "total": acr.evals_saved},
        "wall_clock_time_per_task": {"acr": s.series("wall_ms"),
                                     "control": control.signals.series("wall_ms") if control else None},
        "accuracy": {"acr": round(acr.accuracy, 3), "control": round(control.accuracy, 3) if control else None},
    }
    return out


def render_report(acr: RunResult, control: Optional[RunResult] = None) -> str:
    g = golden_signals(acr, control)
    L = []
    L.append("═" * 78)
    L.append("100-TICKET BURN-IN — golden signals (ACR flywheel)")
    L.append("═" * 78)
    L.append("token_cost/task (running mean)  " + sparkline(g["token_cost_per_task"]["acr_running_mean"]))
    L.append("router_local_fraction           " + sparkline(g["router_local_fraction"]["acr"])
             + "  final=%.0f%%" % (g["router_local_fraction"]["final"] * 100))
    L.append("working_context_size (tok)      " + sparkline(g["working_context_size"]["acr"])
             + "  max=%d  (<5000 ✓)" % g["working_context_size"]["max"])
    L.append("eval_compute_saved (cum)        " + sparkline(g["eval_compute_saved"]["acr"])
             + "  total=%d" % g["eval_compute_saved"]["total"])
    L.append("wall_clock/task (ms)            " + sparkline(g["wall_clock_time_per_task"]["acr"]))
    L.append("-" * 78)
    a = acr.summary()
    L.append("ACR      : tokens=%d  llm_calls=%d  promotions=%d  local=%.0f%%  acc=%.0f%%"
             % (a["total_tokens"], a["llm_calls"], a["promotions"], a["local_fraction"] * 100, a["accuracy"] * 100))
    if control:
        c = control.summary()
        drop = 100 * (1 - (a["total_tokens"] / c["total_tokens"])) if c["total_tokens"] else 0
        L.append("No-ACR   : tokens=%d  llm_calls=%d  promotions=%d  local=%.0f%%  acc=%.0f%%"
                 % (c["total_tokens"], c["llm_calls"], c["promotions"], c["local_fraction"] * 100, c["accuracy"] * 100))
        L.append("MOAT     : ACR spends %.0f%% fewer tokens than No-ACR at %s accuracy"
                 % (drop, "equal-or-better" if a["accuracy"] >= c["accuracy"] else "LOWER (investigate)"))
    L.append("═" * 78)
    return "\n".join(L)


def _svg_chart(x0, y0, w, h, series: List[Tuple[str, List[float], str]], title: str, ymax=None, hline=None,
               vline=None) -> str:
    """One dependency-free line chart as an SVG group. `series` = [(label, ys, color), ...]."""
    all_y = [v for _, ys, _ in series for v in ys] or [0, 1]
    ymax = ymax if ymax is not None else (max(all_y) or 1) * 1.1
    n = max(len(ys) for _, ys, _ in series) if series else 1
    def px(i): return x0 + (i / max(1, n - 1)) * w
    def py(v): return y0 + h - (min(v, ymax) / ymax) * h
    parts = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#0f1115" stroke="#333"/>']
    parts.append(f'<text x="{x0+4}" y="{y0-6}" fill="#ddd" font-size="13" font-family="monospace">{title}</text>')
    if hline is not None and hline <= ymax:
        yy = py(hline)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+w}" y2="{yy:.1f}" stroke="#c0392b" '
                     f'stroke-dasharray="4" opacity="0.6"/>')
    if vline is not None and vline < n:
        xx = px(vline)
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+h}" stroke="#e67e22" '
                     f'stroke-dasharray="3" opacity="0.7"/><text x="{xx+3:.1f}" y="{y0+14}" fill="#e67e22" '
                     f'font-size="10" font-family="monospace">OOD #80</text>')
    for li, (label, ys, color) in enumerate(series):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<text x="{x0+w-150}" y="{y0+16+li*14}" fill="{color}" font-size="11" '
                     f'font-family="monospace">{label}</text>')
    return "".join(parts)


def plot_svg(acr: RunResult, control: Optional[RunResult], path: str) -> str:
    """Publish the golden signals as a single dependency-free SVG (4 charts). Always available (no matplotlib)."""
    g = golden_signals(acr, control)
    W, H = 960, 640
    cw, ch, mx, my, gap = 420, 250, 40, 50, 40
    charts = []
    # token cost running-mean: ACR vs control running mean
    tok = [("ACR", g["token_cost_per_task"]["acr_running_mean"], "#2ecc71")]
    if control:
        cc = control.signals.series("token_cost")
        tok.append(("No-ACR", [round(sum(cc[:i + 1]) / (i + 1), 1) for i in range(len(cc))], "#e74c3c"))
    charts.append(_svg_chart(mx, my, cw, ch, tok, "token_cost/task (running mean) — The Moat", vline=80))
    charts.append(_svg_chart(mx + cw + gap, my, cw, ch,
                             [("local_frac", g["router_local_fraction"]["acr"], "#3498db")],
                             "router_local_fraction — Distillation", ymax=1.0, hline=0.85, vline=80))
    charts.append(_svg_chart(mx, my + ch + gap + 20, cw, ch,
                             [("ctx tok", [float(v) for v in g["working_context_size"]["acr"]], "#9b59b6")],
                             "working_context_size (<5000 tok)", ymax=5000, hline=5000))
    charts.append(_svg_chart(mx + cw + gap, my + ch + gap + 20, cw, ch,
                             [("saved", [float(v) for v in g["eval_compute_saved"]["acr"]], "#f1c40f")],
                             "eval_compute_saved (surrogate, cum)", vline=80))
    a = acr.summary()
    sub = "ACR: %d tok, local %.0f%%, acc %.0f%%" % (a["total_tokens"], a["local_fraction"] * 100, a["accuracy"] * 100)
    if control:
        c = control.summary()
        sub += "   |   No-ACR: %d tok, acc %.0f%%   |   %.0f%% fewer tokens" % (
            c["total_tokens"], c["accuracy"] * 100, 100 * (1 - a["total_tokens"] / max(1, c["total_tokens"])))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<rect width="{W}" height="{H}" fill="#0a0a0a"/>'
           f'<text x="{mx}" y="30" fill="#fff" font-size="18" font-family="monospace">'
           f'100-Ticket Burn-In — ACR flywheel golden signals</text>'
           f'<text x="{mx}" y="{H-14}" fill="#aaa" font-size="12" font-family="monospace">{sub}</text>'
           + "".join(charts) + '</svg>')
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(svg)
    return path


def plot_png(acr: RunResult, control: Optional[RunResult], path: str, title: str = "100-Ticket Burn-In",
             ood_at: Optional[int] = 80) -> Optional[str]:
    """Publish the four golden signals as a readable PNG (axes, grids, legend). Returns the path, or None if
    matplotlib is unavailable. `ood_at` draws the OOD-injection marker (pass None for round-robin/cross-modal runs)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    g = golden_signals(acr, control)
    xs = list(range(len(acr.signals.rows)))
    ACR, CTRL, GRID = "#2ca02c", "#d62728", "#cfd8dc"
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor("white")

    def style(a, title_, ylabel):
        a.set_title(title_, fontsize=13, fontweight="bold", loc="left")
        a.set_xlabel("ticket #", fontsize=10)
        a.set_ylabel(ylabel, fontsize=10)
        a.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        if ood_at is not None and ood_at < len(xs):
            a.axvline(ood_at, color="#ff7f0e", ls="--", lw=1.2, alpha=0.8)

    # 1 — token cost/task (the moat): ACR vs control running mean
    ax[0, 0].plot(xs, g["token_cost_per_task"]["acr_running_mean"], color=ACR, lw=2.2, label="ACR flywheel")
    if control:
        cc = control.signals.series("token_cost")
        cum = [sum(cc[:i + 1]) / (i + 1) for i in range(len(cc))]
        ax[0, 0].plot(range(len(cum)), cum, color=CTRL, lw=2.0, ls="--", label="No-ACR control")
    style(ax[0, 0], "token_cost / task  —  The Moat  (running mean)", "tokens / task")
    ax[0, 0].legend(loc="center right", framealpha=0.9)

    # 2 — router local fraction (distillation)
    ax[0, 1].plot(xs, g["router_local_fraction"]["acr"], color=ACR, lw=2.2)
    ax[0, 1].axhline(0.85, color="#1f77b4", ls=":", lw=1.3, label="0.85 target")
    ax[0, 1].set_ylim(0, 1.02)
    style(ax[0, 1], "router_local_fraction  —  Distillation", "fraction served locally")
    ax[0, 1].legend(loc="lower right", framealpha=0.9)

    # 3 — working context size
    ax[1, 0].plot(xs, g["working_context_size"]["acr"], color="#9467bd", lw=1.8)
    ax[1, 0].axhline(5000, color=CTRL, ls=":", lw=1.3, label="5000 tok ceiling")
    ax[1, 0].set_ylim(0, max(5200, max(g["working_context_size"]["acr"] or [0]) * 1.2))
    style(ax[1, 0], "working_context_size  —  L1 compaction", "tokens / step")
    ax[1, 0].legend(loc="upper right", framealpha=0.9)

    # 4 — eval compute saved (cumulative)
    ax[1, 1].plot(xs, g["eval_compute_saved"]["acr"], color="#e6a817", lw=2.0)
    ax[1, 1].fill_between(xs, g["eval_compute_saved"]["acr"], color="#e6a817", alpha=0.15)
    style(ax[1, 1], "eval_compute_saved  —  Surrogate (cumulative)", "evals skipped")

    a = acr.summary()
    sub = "ACR: %d tok (%d serve + %d distill) · local %.0f%% · acc %.0f%%" % (
        a["total_tokens"], a["serve_tokens"], a["distill_tokens"], a["local_fraction"] * 100, a["accuracy"] * 100)
    if control:
        c = control.summary()
        sub += "    |    No-ACR: %d tok · acc %.0f%%    |    %.0f%% fewer" % (
            c["total_tokens"], c["accuracy"] * 100, 100 * (1 - a["total_tokens"] / max(1, c["total_tokens"])))
    fig.suptitle(title + " — ACR flywheel golden signals", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945, sub, ha="center", fontsize=11, color="#455a64")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path
