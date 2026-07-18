"""quickstart — set up Bobby and see the three things that make it different, in one runnable file.

────────────────────────────────────────────────────────────────────────────────────────────────────
SETUP
────────────────────────────────────────────────────────────────────────────────────────────────────
1. Install (from this folder):
       pip install -e .

2. Point it at any OpenAI-compatible model endpoint — local (vLLM / sglang / Ollama / llama.cpp) or hosted:
       export BOBBY_LLM_URL="http://localhost:8000/v1/chat/completions"
       export BOBBY_LLM_MODEL="your-served-model-id"

   Optional — embeddings, for the semantic memory / primitive re-find (any Ollama-compatible /api/embed):
       export BOBBY_EMBED_URL="http://localhost:11434/api/embed"

   Reasoning model (Qwen3-class)? Keep thinking ON with an adequate budget, OR disable it for speed:
       export BOBBY_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

3. Run:
       python examples/quickstart.py

Sections 1–2 call your model; section 3 (the ACR engine) runs with NO model so it always works — it shows the
routing + distillation mechanism deterministically. Section 4 needs BOBBY_EMBED_URL for real semantic recall.
────────────────────────────────────────────────────────────────────────────────────────────────────
"""
import os

from bobby_squad import Agent, SelfCore, LLM, squad_solve
from bobby_squad import Engine, OODGate, burn_in
from bobby_squad.router import competence_router

LLM_READY = bool(os.environ.get("BOBBY_LLM_URL"))
EMBED_READY = bool(os.environ.get("BOBBY_EMBED_URL"))
print("setup: LLM endpoint %s | embeddings %s\n" % ("SET" if LLM_READY else "unset (sections 1-2 will error)",
                                                    "SET" if EMBED_READY else "unset (section 4 skipped)"))


# ── 1) one persistent-self agent — behavior comes from the request, not a script ────────────────────────
if LLM_READY:
    llm = LLM(temperature=0.4)
    solo = Agent(SelfCore(identity="a precise generalist", goal="do exactly what the request asks"), llm=llm)
    print("1) SOLO AGENT\n", solo.carry_out("Write a 3-line haiku about entropy.", max_rounds=2), "\n")

    # ── 2) a self-organizing squad covering a target — recursive shared board, no orchestrator ──────────
    TARGET = set(range(1, 21))
    agents = [Agent(SelfCore("an enumerator", "list the numbers in the given range"), llm=llm, name="a%d" % i)
              for i in range(2)]

    def work(agent, unit):
        lo, hi = unit
        ans = agent.carry_out("List every integer from %d to %d, comma-separated, digits only." % (lo, hi),
                              max_rounds=1)
        return {n for n in TARGET if str(n) in ans.replace(",", " ").split()}

    def verify(unit, acc):
        lo, hi = unit
        return set(range(lo, hi + 1)) <= acc or (hi - lo) <= 4          # covered, or small enough to accept

    def split(unit):
        lo, hi = unit
        mid = (lo + hi) // 2
        return [(lo, mid), (mid + 1, hi)] if (hi - lo) > 4 else None

    r = squad_solve(agents, [(1, 20)], work, verify=verify, split=split)
    print("2) SQUAD COVERAGE: %d/%d in %d passes (no orchestrator)\n" % (len(r["result"]), len(TARGET), r["passes"]))
else:
    print("1-2) skipped — set BOBBY_LLM_URL to run the live agent + squad demos.\n")


# ── 3) the ACR engine — route each event to the cheapest handler; freeze the LLM's work into local code ──
# No model needed: we register a proven frozen plugin, then show in-distribution events served locally (zero LLM)
# while an out-of-distribution event trips the OOD tripwire and abstains to the (here, stub) LLM fallback.
print("3) ACR ENGINE (deterministic)")
eng = Engine(os.path.join(os.path.dirname(__file__), ".qs_engine"), require_proof=True)


def stub_embed(texts):                                  # a tiny deterministic 'embedding' so the demo needs no server
    import hashlib
    return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


eng.interceptors = [competence_router(stub_embed)]
llm_calls = {"n": 0}


def llm_fallback(payload):                              # stands in for the expensive LLM last-resort
    llm_calls["n"] += 1
    return "LLM-HANDLED: " + payload.get("q", "")


eng.on("task", llm_fallback)

# promote a frozen extractor the flywheel would have written+proven — competence region = the asks it was proven on
ask = "extract error codes from the log"
eng.promote("extract_errors", burn_in.make_extractor(r"ERR-\d+"), tags=["task"],
            proof={"verdict": "WIRE", "competence": OODGate.fit(stub_embed([ask] * 4))},
            code="import re\ndef h(p):\n    return '\\n'.join(re.findall(r'ERR-\\d+', p['blob']))\n")

in_dist = eng.emit("task", {"cap": "task", "q": ask, "blob": "boot ok\nERR-4041 disk\nERR-5012 net"})
before = llm_calls["n"]
ood = eng.emit("task", {"cap": "task", "q": "translate this poem into French", "blob": "..."})
print("   in-distribution  -> %r  (LLM calls: %d — served locally, zero cost)" % (in_dist, before))
print("   out-of-dist      -> %r  (OOD tripwire abstained to the LLM fallback)\n" % ood)


# ── 4) the primitive library — find a capability BACK before re-distilling (semantic + structural memory) ─
if EMBED_READY:
    import tempfile
    from bobby_squad import primitive_lib as L
    from bobby_squad.retrieval import default_embed

    lib = L.PrimitiveLibrary(tempfile.mkdtemp(), embed_fn=default_embed)
    ex = lambda fid, n, s: [next(f for f in burn_in.FAMILIES if f[0] == fid)[3](burn_in._rng(s * 100 + i))
                            for i in range(n)]
    domains = {"fin": {"param": r"TXN-\d{6}", "examples": ex("fin", 5, 1)},
               "security": {"param": r"CVE-20\d{2}-\d{4}", "examples": ex("security", 5, 2)},
               "legal": {"param": r"§\d{3}", "examples": ex("legal", 5, 3)}}
    L.seed_known(lib, {"extract_matching": domains})
    hit = lib.recall("pull out the tokens matching a pattern", k=1)
    print("4) PRIMITIVE LIBRARY — semantic recall found %r back from memory (no re-distill).\n"
          % (hit[0][0] if hit else None))
else:
    print("4) skipped — set BOBBY_EMBED_URL for real semantic recall. See docs/PRIMITIVES.md.\n")

print("Next: docs/BURN-IN.md (the benchmark), docs/PRIMITIVES.md (distill-to-code), docs/runtime-architecture.md.")
