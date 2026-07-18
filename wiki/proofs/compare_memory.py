#!/usr/bin/env python3
import collections
import json
import os
import re
import string
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import VaultHub  # noqa: E402
from bobby_squad.retrieval import default_embed  # noqa: E402

DATA = os.environ.get("LOCOMO_PATH", "/tmp/mem_locomo.jsonl")
NCONV = int(os.environ.get("NCONV", "1"))                 # conversations (each has its own memory + many questions)
NQ = int(os.environ.get("NQ", "20"))                      # questions per conversation
READ = int(os.environ.get("READ", "1500"))
TOPK = int(os.environ.get("TOPK", "6"))
SYSTEMS = os.environ.get("SYSTEMS", "solo,ours,mem0").split(",")

# ── shims: make Mem0 run on the LOCAL model + embeddings, for free ───────────────────────────────────
def _install_shims():
    # 1) thinking-off on EVERY openai chat call (covers Mem0's extraction + answer)
    try:
        from openai.resources.chat import completions as _c
        if not getattr(_c.Completions, "_bobby_patched", False):
            _orig = _c.Completions.create

            def _create(self, *a, **k):
                eb = k.get("extra_body") or {}
                eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
                k["extra_body"] = eb
                return _orig(self, *a, **k)
            _c.Completions.create = _create
            _c.Completions._bobby_patched = True
    except Exception as e:
        print(f"  [shim] openai patch skipped: {e}", flush=True)
    # 2) route Mem0's ollama embedder to our default_embed (same embeddings as ours)
    try:
        import mem0.embeddings.ollama as O

        def _noop(self):
            return None

        def _embed(self, text, memory_action=None):
            v = default_embed([text])
            return v[0] if v else [0.0] * 768
        O.OllamaEmbedding._ensure_model_exists = _noop
        O.OllamaEmbedding.embed = _embed
    except Exception as e:
        print(f"  [shim] mem0 embedder patch skipped: {e}", flush=True)


# ── scoring ──────────────────────────────────────────────────────────────────────────────────────────
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


ANSWER = ("Answer the question as concisely as possible using ONLY the information given. Reply with just the "
          "answer — no explanation.\nQuestion: {q}\nAnswer:")


# Full-capability answer config (same for EVERY system → fair). Reasoning model: thinking ON needs a generous budget
# or content returns empty (Qwen reserves ~32k). THINK=0 → fast capped mode. Mem0's INTERNAL extraction keeps
# thinking-off via the openai shim (its own design/small budget); this is only the shared ANSWER step.
THINK = os.environ.get("THINK", "1") == "1"
ANS_TOKENS = int(os.environ.get("ANS_TOKENS", "16384" if THINK else "96"))


def _ask(prompt):
    from bobby_squad import LLM
    eb = {} if THINK else {"chat_template_kwargs": {"enable_thinking": False}}
    return (LLM(extra_body=eb)([{"role": "user", "content": prompt}], max_tokens=ANS_TOKENS, temperature=0.0) or "").strip()


# ── observability ────────────────────────────────────────────────────────────────────────────────────
class Obs:
    """Per-system stage counters + failure samples — so the report shows WHERE a system breaks."""

    def __init__(self, name):
        self.name = name
        self.f1 = self.tok = self.lat = 0.0
        self.n = 0
        self.build_err = self.empty_recall = self.empty_ans = self.ans_err = 0
        self.samples = []

    def note(self, stage, detail):
        if len(self.samples) < 6:
            self.samples.append(f"{stage}: {detail}")

    def add(self, f1, tok, lat, n_ret, ans):
        self.n += 1
        self.f1 += f1
        self.tok += tok
        self.lat += lat
        if n_ret == 0:
            self.empty_recall += 1
        if not ans:
            self.empty_ans += 1

    def row(self):
        n = max(1, self.n)
        return (f"  {self.name:8s} F1 {100*self.f1/n:5.1f} · {int(self.tok/n):>6,} tok/Q · {self.lat/n:4.1f}s/Q · "
                f"empty-recall {self.empty_recall}/{n} · empty-ans {self.empty_ans}/{n} · "
                f"build-err {self.build_err} · ans-err {self.ans_err}")


# ── adapters (build memory once per conversation, then answer each question) ─────────────────────────
def solo_build(ctx):
    return ctx


def solo_answer(mem, q):
    p = f"Conversation:\n{mem[:480000]}\n\n" + ANSWER.format(q=q)
    return _ask(p), _ptoks(p), 1


def ours_build(ctx):
    import tempfile
    d = tempfile.mkdtemp(prefix="cmp_")
    hub = VaultHub(d, embed_fn=default_embed)
    secs = [ctx[i:i + READ] for i in range(0, len(ctx), READ)] or [""]
    ent = [set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", s)) for s in secs]
    for i, s in enumerate(secs):
        links = [f"doc/passage-{j+1}" for j in range(len(secs)) if j != i and ent[i] & ent[j]][:5]
        hub.enrich("doc", f"passage {i+1}", s, source="conversation", links=links or None)
    return hub


def ours_answer(hub, q):
    block = hub.navigate(q, per_vault_k=TOPK, hops=2, budget=9000, per_note=1500, whole_vault="doc", whole_k=TOPK)
    p = f"Recalled from the conversation:\n{block}\n\n" + ANSWER.format(q=q)
    return _ask(p), _ptoks(block), (block.count("##"))


def mem0_build(ctx, conv_id):
    from mem0 import Memory
    cfg = {"llm": {"provider": "openai", "config": {"model": os.environ.get("BOBBY_LLM_MODEL", "local"),
                   "openai_base_url": os.environ["BOBBY_LLM_URL"].replace("/chat/completions", ""), "temperature": 0.0}},
           "embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text", "embedding_dims": 768,
                        "ollama_base_url": "http://host.docker.internal:11435"}},
           # our embeddings are 768-dim (nomic) — the store must match, or it defaults to OpenAI's 1536 and crashes
           "vector_store": {"provider": "qdrant", "config": {"embedding_model_dims": 768,
                            "path": "/tmp/mem0_qdrant_" + conv_id, "on_disk": False}}}
    m = Memory.from_config(cfg)
    # ingest per session-chunk (fewer extraction calls than per-turn); user_id keys the conversation
    for i in range(0, len(ctx), 4000):
        m.add(ctx[i:i + 4000], user_id=conv_id)
    return m


def mem0_answer(m, q, conv_id):
    hits = m.search(q, filters={"user_id": conv_id}, top_k=TOPK)   # v2 API: filters + top_k, not top-level user_id
    mems = hits.get("results", hits) if isinstance(hits, dict) else hits
    facts = [h.get("memory", "") if isinstance(h, dict) else str(h) for h in (mems or [])]
    block = "\n".join(f"- {f}" for f in facts)
    p = f"Recalled memories:\n{block}\n\n" + ANSWER.format(q=q)
    return _ask(p), _ptoks(block), len(facts)


def main():
    _install_shims()
    convs = collections.OrderedDict()
    for r in (json.loads(l) for l in open(DATA) if l.strip()):
        convs.setdefault(r["context"], []).append(r)
    conv_items = list(convs.items())[:NCONV]
    print(f"== MEMORY HEAD-TO-HEAD · LoCoMo · {len(conv_items)} conv × ≤{NQ} Q · model={os.environ.get('BOBBY_LLM_MODEL','?')} "
          f"· systems={SYSTEMS} ==\n", flush=True)

    obs = {s: Obs(s) for s in SYSTEMS}
    for ci, (ctx, qs) in enumerate(conv_items):
        cid = f"conv{ci}"
        built = {}
        for s in SYSTEMS:
            try:
                t0 = time.time()
                built[s] = {"solo": lambda: solo_build(ctx), "ours": lambda: ours_build(ctx),
                            "mem0": lambda: mem0_build(ctx, cid)}[s]()
                print(f"  [{s}] conv{ci} memory built in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                obs[s].build_err += 1
                obs[s].note("build", f"{type(e).__name__}: {str(e)[:80]}")
                print(f"  [{s}] conv{ci} BUILD FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
        for r in qs[:NQ]:
            q, golds = r["question"], r["answers"]
            for s in SYSTEMS:
                if s not in built:
                    continue
                try:
                    t0 = time.time()
                    ans, tok, nret = {"solo": lambda: solo_answer(built[s], q),
                                      "ours": lambda: ours_answer(built[s], q),
                                      "mem0": lambda: mem0_answer(built[s], q, cid)}[s]()
                    obs[s].add(qa_f1(ans, golds), tok, time.time() - t0, nret, ans)
                except Exception as e:
                    obs[s].ans_err += 1
                    obs[s].note("answer", f"{type(e).__name__}: {str(e)[:80]}")

    print("\n== RESULT (identical local model + embeddings; only the memory layer differs) ==", flush=True)
    for s in SYSTEMS:
        print(obs[s].row(), flush=True)
        for smp in obs[s].samples:
            print(f"        ↳ {smp}", flush=True)
    print("\nRESULT " + json.dumps({s: {"f1": 100 * obs[s].f1 / max(1, obs[s].n), "n": obs[s].n,
                                        "empty_recall": obs[s].empty_recall, "empty_ans": obs[s].empty_ans,
                                        "build_err": obs[s].build_err, "ans_err": obs[s].ans_err} for s in SYSTEMS}),
          flush=True)


if __name__ == "__main__":
    main()
