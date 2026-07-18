#!/usr/bin/env python3
import collections
import hashlib
import json
import os
import re
import shutil
import string
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import LLM, VaultHub
from bobby_squad.retrieval import EmbeddingRetriever, default_embed

N = int(os.environ.get("N", "25"))
FILE = os.environ.get("FILE", "/tmp/mh_2wikimqa.jsonl")
READ = int(os.environ.get("READ", "1500"))               # passage size (chars)
TOPK = int(os.environ.get("TOPK", "6"))                  # bounded passages a method may see
HELD_FRAC = float(os.environ.get("HELD_FRAC", "0.5"))    # fraction of questions reserved as held-out eval
BUDGET = int(os.environ.get("BUDGET", "9000"))
PER_NOTE = int(os.environ.get("PER_NOTE", "1500"))
WIRE_F1 = float(os.environ.get("WIRE_F1", "1.0"))        # ΔF1 (points) to call an improvement
CHEAP = float(os.environ.get("CHEAP", "0.7"))            # "meaningfully cheaper" = tokens < CHEAP × baseline


# ── deterministic LongBench QA-F1 ──────────────────────────────────────────────────────────────────
def _normalize(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, golds):
    best, pt = 0.0, _normalize(pred).split()
    for g in golds:
        gt = _normalize(g).split()
        common = sum(min(pt.count(w), gt.count(w)) for w in set(pt) if w in gt)
        if pt and gt and common:
            prec, rec = common / len(pt), common / len(gt)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _ptoks(t):
    return len(t) // 4


def _q(r):
    return r.get("input") or r.get("question")


def _golds(r):
    return r["answers"] if isinstance(r["answers"], list) else [r["answers"]]


ANSWER = ("Answer the question as concisely as possible using ONLY the information given. Reply with just the "
          "answer — no explanation.\nQuestion: {q}\nAnswer:")


# Reasoning-model config (per Qwen/sglang docs): with thinking ON the reasoning can eat the whole budget before
# </think>, so `content` returns empty unless max_tokens is generous (Qwen reserves ~32k for output). THINK=1
# (default) = full capability. THINK=0 = the documented "hard switch" (enable_thinking:false) — fast but caps
# reasoning; use only for extractive/needle tasks where no reasoning is needed.
THINK = os.environ.get("THINK", "1") == "1"
ANS_TOKENS = int(os.environ.get("ANS_TOKENS", "16384" if THINK else "96"))


def _ask(prompt):
    eb = {} if THINK else {"chat_template_kwargs": {"enable_thinking": False}}   # {} → thinking ON (overrides env)
    return (LLM(extra_body=eb)([{"role": "user", "content": prompt}], max_tokens=ANS_TOKENS, temperature=0.0) or "").strip()


# ── METHOD interface — the pluggable unit the discovery loop generates ─────────────────────────────
class Method:
    """prepare(ctx) → a memory built once per document; answer(mem, question) → (text, prompt_tokens)."""
    name = "base"

    def prepare(self, ctx):
        return ctx

    def answer(self, mem, question):
        raise NotImplementedError


class Solo(Method):
    """CONTROL — the full document in one call. The number every method is measured against."""
    name = "solo"

    def answer(self, mem, question):
        p = f"Material:\n{mem[:480000]}\n\n" + ANSWER.format(q=question)
        return _ask(p), _ptoks(p)


class Null(Method):
    """NEGATIVE CONTROL — no context at all. F1 must be ≈0 or the task is guessable / the harness is broken."""
    name = "null"

    def prepare(self, ctx):
        return None

    def answer(self, mem, question):
        p = ANSWER.format(q=question)
        return _ask(p), _ptoks(p)


def _passages(ctx):
    return [ctx[i:i + READ] for i in range(0, len(ctx), READ)] or [""]


class FlatK(Method):
    """Top-k VERBATIM passages (cosine) into the answer step — needle-preserving. Proven WIRE (+51.7 NIAH)."""
    name = "flat_k"

    def prepare(self, ctx):
        secs = _passages(ctx)
        r = EmbeddingRetriever(embed_fn=default_embed)
        r.add_many(secs)
        return (r, secs)

    def answer(self, mem, question):
        r, secs = mem
        top = r.search(question, TOPK) or secs[:TOPK]
        p = "Excerpts from the document:\n\n" + "\n\n---\n\n".join(top) + "\n\n" + ANSWER.format(q=question)
        return _ask(p), _ptoks(p)


class GraphHop(Method):
    """Answer over the NAVIGATED subgraph — passages linked by shared entities, entry + link-hop (cheaper, noisier)."""
    name = "graph_hop"

    def prepare(self, ctx):
        d = tempfile.mkdtemp(prefix="mg_")
        hub = VaultHub(d, embed_fn=default_embed)
        secs = _passages(ctx)
        ent = [set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", s)) for s in secs]
        for i, s in enumerate(secs):
            links = [f"doc/passage-{j+1}" for j in range(len(secs)) if j != i and ent[i] & ent[j]][:5]
            hub.enrich("doc", f"passage {i+1}", s, source="document", links=links or None)
        return (hub, d)

    def answer(self, mem, question):
        hub, _ = mem
        block = hub.navigate(question, per_vault_k=TOPK, hops=2, budget=BUDGET, per_note=PER_NOTE)
        p = f"Retrieved from the document (linked passages):\n{block}\n\n" + ANSWER.format(q=question)
        return _ask(p), _ptoks(p)

    def cleanup(self, mem):
        shutil.rmtree(mem[1], ignore_errors=True)


BUILTINS = {m.name: m for m in [Solo(), Null(), FlatK(), GraphHop()]}


# ── evaluation ─────────────────────────────────────────────────────────────────────────────────────
class Result:
    __slots__ = ("name", "f1", "tok", "latency", "n")

    def __init__(self, name, f1, tok, latency, n):
        self.name, self.f1, self.tok, self.latency, self.n = name, f1, tok, latency, n

    def as_dict(self):
        return {"name": self.name, "f1": self.f1, "tok_per_q": self.tok, "latency_per_q": round(self.latency, 2),
                "n": self.n}


def split(rows):
    """Deterministic held-out split — group by document, reserve the LAST ceil(HELD_FRAC·k) questions of each doc as
    held-out. No RNG → byte-stable re-runs; the discoverer tunes on `train`, is graded on `held`."""
    by_doc = collections.OrderedDict()
    for r in rows:
        by_doc.setdefault(hashlib.md5(r["context"].encode()).hexdigest(), []).append(r)
    train, held = [], []
    for qs in by_doc.values():
        cut = len(qs) - max(1, int(len(qs) * HELD_FRAC + 0.999)) if len(qs) > 1 else 0
        train += qs[:cut]
        held += qs[cut:]
    return train, held


def evaluate(method, held):
    """Run one method over the held-out set. prepare() is cached per document (built once, reused across its
    questions — the reuse economics), and each answer is timed. Returns a Result (F1 · tokens/Q · latency/Q)."""
    mem_cache = {}
    f1 = tok = 0.0
    lat = 0.0
    n = 0
    try:
        for r in held:
            key = hashlib.md5(r["context"].encode()).hexdigest()
            if key not in mem_cache:
                mem_cache[key] = method.prepare(r["context"])
            t0 = time.time()
            a, pt = method.answer(mem_cache[key], _q(r))
            lat += time.time() - t0
            tok += pt
            f1 += qa_f1(a, _golds(r))
            n += 1
    finally:
        if hasattr(method, "cleanup"):
            for m in mem_cache.values():
                try:
                    method.cleanup(m)
                except Exception:
                    pass
    n = max(1, n)
    return Result(method.name, 100 * f1 / n, tok / n, lat / n, n)


def verdict(cand, base):
    """Cost-inclusive verdict of `cand` vs `base`: WIRE if it clears +WIRE_F1 in F1, OR reaches parity while being
    meaningfully cheaper (tokens < CHEAP × base). DELETE if it loses ≥WIRE_F1 without a cost win. Else MARGINAL."""
    d = cand.f1 - base.f1
    cheaper = cand.tok < CHEAP * base.tok
    if d >= WIRE_F1 or (abs(d) < WIRE_F1 and cheaper):
        v = "WIRE"
    elif d <= -WIRE_F1 and not cheaper:
        v = "DELETE"
    else:
        v = "MARGINAL"
    per_k = (d / (cand.tok / 1000.0)) if cand.tok else 0.0
    return {"verdict": v, "dF1": round(d, 1), "dF1_per_ktok": round(per_k, 3),
            "tok_ratio": round(cand.tok / max(1, base.tok), 2)}


def run(methods=None, baseline="solo"):
    rows = [json.loads(l) for l in open(FILE) if l.strip()][:N]
    train, held = split(rows)
    names = methods or ["flat_k", "graph_hop"]
    print(f"== MEMORY GAINS · {os.path.basename(FILE)} · held-out {len(held)}/{len(rows)} Q · "
          f"model={os.environ.get('BOBBY_LLM_MODEL','?')} · temp=0 ==\n", flush=True)

    base = evaluate(BUILTINS[baseline], held)
    ctrl = evaluate(BUILTINS["null"], held)
    print(f"  {base.name:10s} F1 {base.f1:5.1f}  · {int(base.tok):>7,} tok/Q · {base.latency:4.1f}s/Q  · BASELINE", flush=True)
    sane = ctrl.f1 < 15.0
    print(f"  {ctrl.name:10s} F1 {ctrl.f1:5.1f}  · negative control — {'OK (≈0)' if sane else 'BROKEN: task guessable!'}", flush=True)
    if not sane:
        print("  ⚠ verdicts below are UNTRUSTWORTHY until the control reads ≈0.", flush=True)

    out = {"baseline": base.as_dict(), "control": ctrl.as_dict(), "control_sane": sane, "candidates": []}
    for nm in names:
        m = BUILTINS.get(nm)
        if m is None:
            continue
        res = evaluate(m, held)
        v = verdict(res, base)
        print(f"  {res.name:10s} F1 {res.f1:5.1f}  · {int(res.tok):>7,} tok/Q · {res.latency:4.1f}s/Q  · "
              f"ΔF1 {v['dF1']:+5.1f} · {v['tok_ratio']}× cost · {v['verdict']}", flush=True)
        out["candidates"].append({**res.as_dict(), **v})
    print("\nRESULT " + json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    run(os.environ.get("METHODS", "flat_k,graph_hop").split(","))
