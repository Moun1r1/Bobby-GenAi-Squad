#!/usr/bin/env python3
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLISHED = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PUBLISHED)
sys.path.insert(0, HERE)
import memory_gains as H                                   # noqa: E402  the Step-1 harness (teacher)
from memory_gains import evaluate, verdict, split, BUILTINS  # noqa: E402
from bobby_squad import Agent, SelfCore, LLM, VaultHub, SemanticMemory, CorrectionMemory  # noqa: E402
from bobby_squad.agent_tools import SandboxTools  # noqa: E402
from bobby_squad.retrieval import default_embed  # noqa: E402

MAX_ITERS = int(os.environ.get("MAX_ITERS", "6"))         # runaway backstop only — the loop stops at PLATEAU
PATIENCE = int(os.environ.get("PATIENCE", "3"))           # plateau: stop after this many iterations with no new WIRE
LEDGER_DIR = os.environ.get("LEDGER_DIR", os.path.join(tempfile.gettempdir(), "method_ledger"))

# the import header a candidate file needs — the researcher is told to start cand.py with exactly this
PREAMBLE = ("import re, tempfile\n"
            "from memory_gains import Method, _passages, _ask, _ptoks, ANSWER, TOPK\n"
            "from bobby_squad import VaultHub\n"
            "from bobby_squad.retrieval import EmbeddingRetriever, default_embed\n")

SMOKE = """import sys
sys.path.insert(0, %r)
from cand import Candidate
m = Candidate()
mem = m.prepare("Ada Lovelace, daughter of Lord Byron, wrote the first algorithm for the Analytical Engine in 1843.")
a, tok = m.answer(mem, "Who wrote the first algorithm?")
assert isinstance(a, str) and isinstance(tok, int), "answer must return (str, int)"
print("SMOKE OK:", repr(a[:50]), "tok", tok)
""" % HERE

TASK = (
    "You improve the MEMORY method of a long-context QA engine. The current best retrieves the top-K verbatim "
    "passages by cosine and stuffs them into the answer prompt.\n\n"
    "Invent ONE genuinely NEW method — a different way to rank/select passages, keep/drop/consolidate memory, or "
    "structure the answer step — that would beat the current best on held-out F1 PER TOKEN (not just more cost). "
    "Your recalled notes list methods already tried; pick a DIFFERENT direction.\n\n"
    "Write the COMPLETE file `cand.py`. It MUST start with exactly these imports:\n" + PREAMBLE +
    "then define `class Candidate(Method):` with `prepare(self, ctx)` (build a memory once) and "
    "`answer(self, mem, question)` returning `(answer_text: str, prompt_tokens: int)`. You may use: _passages(ctx), "
    "_ask(prompt), _ptoks(text), ANSWER (a str with a {q} field), TOPK, EmbeddingRetriever(embed_fn=default_embed), "
    "VaultHub.\n\n"
    "Then RUN `smoke.py` and FIX `cand.py` until it prints `SMOKE OK`. Do NOT score or grade the method yourself — "
    "the gain-proof does that. When smoke passes, state the one-line mechanism you invented."
)


def _extract(text):
    t = text or ""
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
    if m:
        src = m.group(1)
    else:                                                  # unterminated fence (model truncated) — strip fences by hand
        src = re.sub(r"```\s*$", "", re.sub(r"^\s*```(?:python)?\s*", "", t))
    return src if "class Candidate" in src else ""


def _withhdr(src):
    return src if src.lstrip().startswith(("import", "from")) else PREAMBLE + src


def _complete(src):
    """AUTO completion signal — the code is done when it PARSES, not when a token budget is hit. Drives continuation
    without a magic max_tokens: a cut-off class raises SyntaxError → keep going; a valid one stops."""
    if not src or "class Candidate" not in src:
        return False
    try:
        compile(_withhdr(src), "<cand>", "exec")
        return True
    except SyntaxError:
        return False


def _mech(src, out):
    """The one-line mechanism the model actually stated — from the method's docstring (`Method:`/`Mechanism:`) or the
    prose after the code block — NOT a stray code line."""
    for pat in (r"Method:\s*([^\n]+)", r"[Mm]echanism:\s*([^\n]+)"):
        m = re.search(pat, src or "")
        if m and len(m.group(1).strip(' "')) > 6:
            return m.group(1).strip(' "')[:120]
    tail = (re.split(r"```", out or "")[-1] or "").strip()
    return (tail.splitlines()[0][:120] if tail.strip() else "candidate")


def _generate(researcher, prompt, cont=3):
    """Generate until the code COMPILES (auto), not until a fixed budget. Generous ceiling so the model stops at its
    own end; if it was cut off (won't parse), ask for the complete file and re-check — bounded only by a safety cap."""
    out = researcher.act(prompt, max_tokens=2000)
    src = _extract(out)
    tries = 0
    while not _complete(src) and tries < cont:
        out = researcher.act("Your cand.py was incomplete or didn't parse. Output the COMPLETE file as ONE python code "
                             "block — keep it concise (well under 50 lines, minimal comments).", max_tokens=2000)
        src = _extract(out) or src
        tries += 1
    return out, src


def generate_candidate(researcher, tools, sbx, max_fix=3):
    """GENERATIVE content, controlled plumbing, self-repair driven by the REAL smoke error (not a fixed script):
    the researcher invents the code (recall-grounded → self-selects a novel direction), we write+run it, and on any
    failure we feed the actual error back so it FIXES. Generation is completion-driven (compiles), not budget-capped.
    Returns (mech, built_method) or (mech, None). The researcher never grades — smoke only checks it BUILDS + answers."""
    out, src = _generate(researcher, TASK)
    mech = _mech(src, out)
    for _ in range(max_fix):                                # bounded self-repair backstop; exits on SMOKE OK
        if not src:
            return mech, None
        tools.write("cand.py", _withhdr(src))
        res = tools.run("smoke.py")
        if "SMOKE OK" in res:
            return _mech(src, out), build_from(os.path.join(sbx, "cand.py"))
        _out, src = _generate(researcher, f"Running smoke.py on your cand.py failed:\n{res[:700]}\n\nFix cand.py and "
                              "output ONLY the complete corrected file as one python code block.")
    return mech, None


def build_from(path):
    """Import the researcher's cand.py (write→build). Returns an instance or None (fail-safe)."""
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        spec = importlib.util.spec_from_file_location("cand_" + os.path.basename(os.path.dirname(path)), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Candidate() if hasattr(mod, "Candidate") else None
    except Exception as e:
        return None if not os.environ.get("DEBUG") else (_ for _ in ()).throw(e)


class MethodLedger:
    """STEP 2 — second-order memory. WIRE → `methods` vault (novelty-gated); DELETE → CorrectionMemory."""

    def __init__(self, root):
        self.hub = VaultHub(root, embed_fn=default_embed)
        self.wired = SemanticMemory(tau=0.9, embed_fn=default_embed)
        self.rejected = CorrectionMemory(tau=0.9, embed_fn=default_embed)

    def recall(self, task):
        block = self.hub.navigate(task, per_vault_k=4, hops=1, budget=1800, per_note=300)
        tried = "\n".join(f"- already tried (rejected): {t}" for t in self.rejected.retrieve(task, k=5))
        return (block + ("\n\n" + tried if tried else "")) or tried

    def record(self, name, mech, src, res, v):
        tag = f"{name}: {mech}"
        if v["verdict"] == "WIRE" and self.wired.add(tag):
            self.hub.enrich("methods", name, f"mechanism: {mech}\nverdict: WIRE ΔF1 {v['dF1']:+} · "
                            f"{v['tok_ratio']}× cost\n\n## code\n{src[:1200]}", source="self-improve",
                            tags=["method", "wire"])
            return "wired"
        if v["verdict"] == "DELETE":
            self.rejected.add(tag)
            return "rejected"
        return "marginal"


def promote_method(engine, name, method, code, proof, tags=("memqa",), embed_fn=None, competence_qs=None):
    """Bridge flywheel → kernel: register a PROVEN memory method as a frozen engine plugin. The handler takes an event
    payload {ctx, q} → answer (memory built once per ctx, then reused). With embed_fn + competence_qs, attach an OOD
    gate so the frozen plugin serves only in-distribution queries and abstains to the LLM elsewhere. Returns True if
    it registered (dedup + governance gates in the registry). This is how a discovered skill stops costing LLM calls."""
    from bobby_squad.router import OODGate
    pr = dict(proof)
    if embed_fn and competence_qs:
        pr["competence"] = OODGate.fit(embed_fn(competence_qs))
    prepared = {}

    def handler(payload):
        ctx = payload.get("ctx", "")
        if ctx not in prepared:
            prepared[ctx] = method.prepare(ctx)
        out = method.answer(prepared[ctx], payload["q"])
        return out[0] if isinstance(out, tuple) else out

    return engine.promote(name, handler, tags=tags, proof=pr, code=code, kind="static", provenance="flywheel")


def discover():
    rows = [json.loads(l) for l in open(H.FILE) if l.strip()][:H.N]
    _train, held = split(rows)
    ledger = MethodLedger(LEDGER_DIR)
    sbx = tempfile.mkdtemp(prefix="sipe_")
    shutil.copy(os.path.join(HERE, "memory_gains.py"), os.path.join(sbx, "memory_gains.py"))
    with open(os.path.join(sbx, "smoke.py"), "w") as f:
        f.write(SMOKE)
    tools = SandboxTools(repo_root=os.path.join(PUBLISHED, "bobby_squad"), sandbox_root=sbx, timeout=120)
    researcher = Agent(SelfCore(identity="a memory-engine researcher who WRITES and self-repairs retrieval/answer "
                                "methods, then lets an external gain-proof judge them",
                                goal="discover a memory method that beats the current best on the held-out gain-proof"),
                       llm=LLM(temperature=0.6), tools=tools, recall=ledger.recall)

    best = evaluate(BUILTINS["flat_k"], held)
    ctrl = evaluate(BUILTINS["null"], held)
    print(f"== SELF-IMPROVE (generative) · {os.path.basename(H.FILE)} · held-out {len(held)} Q ==", flush=True)
    print(f"  best=flat_k F1 {best.f1:.1f} · {int(best.tok)} tok/Q | control(null) F1 {ctrl.f1:.1f} "
          f"{'OK' if ctrl.f1 < 15 else 'BROKEN — need more held-out Q; verdicts noisy'}\n", flush=True)

    kept, dry, it = [], 0, 0
    while dry < PATIENCE and it < MAX_ITERS:
        it += 1
        # GENERATIVE: the researcher self-selects a direction (recall = what it already tried), invents the code, and
        # self-repairs it against the real smoke error until it BUILDS.
        candpath = os.path.join(sbx, "cand.py")
        mech, m = generate_candidate(researcher, tools, sbx)
        if m is None:
            print(f"  iter {it}: candidate did not build → skip", flush=True)
            researcher.record(f"iteration {it}: my candidate failed to build; try a simpler, importable design")
            dry += 1
            continue
        m.name = f"cand{it}"
        os.makedirs(os.path.join(LEDGER_DIR, "code"), exist_ok=True)   # persist full code so winners can be re-graded
        shutil.copy(candpath, os.path.join(LEDGER_DIR, "code", f"{m.name}.py"))
        try:
            res = evaluate(m, held)                          # DETERMINISTIC teacher grades on held-out
        except Exception as e:
            print(f"  iter {it}: run error {str(e)[:50]} → skip", flush=True)
            researcher.record(f"iteration {it}: candidate ran with an error; make answer() robust")
            dry += 1
            continue
        v = verdict(res, best)
        status = ledger.record(m.name, mech, open(candpath).read(), res, v)
        print(f"  iter {it}: {m.name} F1 {res.f1:.1f} · {int(res.tok)} tok/Q · ΔF1 {v['dF1']:+} · "
              f"{v['verdict']} → {status} · {mech[:55]}", flush=True)
        researcher.record(f"iteration {it}: tried '{mech}' → {v['verdict']} (F1 {res.f1:.1f} vs best {best.f1:.1f})")
        if v["verdict"] == "WIRE" and res.f1 > best.f1:
            best = res
            kept.append({"name": m.name, "mechanism": mech, **res.as_dict(), **v})
            dry = 0
        else:
            dry += 1

    stop = "plateau" if dry >= PATIENCE else "max-iters backstop"
    print(f"\n== DONE ({stop}) · {it} iters · kept {len(kept)} WIRE · best F1 {best.f1:.1f} ==", flush=True)
    for k in kept:
        print(f"  + {k['name']} F1 {k['f1']:.1f} ΔF1 {k['dF1']:+} · {k['mechanism']}", flush=True)
    print("\nRESULT " + json.dumps({"kept": kept, "best_f1": best.f1, "iters": it, "stop": stop,
                                    "ledger": LEDGER_DIR}), flush=True)
    shutil.rmtree(sbx, ignore_errors=True)
    return kept


if __name__ == "__main__":
    discover()
