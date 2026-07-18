#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from bobby_squad import Agent, SelfCore                       # noqa: E402
from bobby_squad.correction_memory import CorrectionMemory    # noqa: E402
from bobby_squad.llm import LLM                               # noqa: E402
from bobby_squad.retrieval import default_embed               # noqa: E402
from datasets import load_dataset                             # noqa: E402

DS = os.environ.get("BENCH_DS", "google-research-datasets/mbpp")
CONFIG = os.environ.get("BENCH_CONFIG", "sanitized")
SPLIT = os.environ.get("BENCH_SPLIT", "test")
N = int(os.environ.get("BENCH_N", "100"))
CYCLES = int(os.environ.get("BENCH_CYCLES", "2"))            # in-task self-fix budget (verify-by-outcome)
RECALL_K = int(os.environ.get("BENCH_RECALL_K", "3"))       # reusable recipes injected per task


class CountingLLM:
    """Wrap the LLM to sum tokens across every call the Agent makes."""
    def __init__(self, inner):
        self._i = inner
        self.total = 0

    def __getattr__(self, k):
        return getattr(self._i, k)

    def _acc(self):
        u = getattr(self._i, "last_usage", {}) or {}
        self.total += (u.get("total_tokens") or 0)

    def __call__(self, *a, **k):
        r = self._i(*a, **k); self._acc(); return r

    def chat(self, *a, **k):
        r = self._i.chat(*a, **k); self._acc(); return r


base = LLM(temperature=0.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})   # match bare baseline
llm = CountingLLM(base)
ds = load_dataset(DS, CONFIG, split=SPLIT)
tasks = list(ds)[:N]

# ONE agent + ONE reusable memory for the WHOLE stream — this is the long horizon.
core = SelfCore(identity="a careful Python engineer who reuses what they have already learned",
                goal="write a Python solution that passes its tests, reusing earlier recipes when they apply",
                constraints=["output only the complete Python source, no prose, no code fences",
                             "define the exact functions the tests call"])
mem = CorrectionMemory(embed_fn=default_embed, tau=0.9, k=RECALL_K)   # semantic, novelty-gated, unbounded

# THE REUSE SEAM, WIRED NATIVELY (not a manual observe): `recall(task)` returns a BOUNDED (top-k) reference block that
# the framework injects fresh for the current step and that is NOT part of the growing context. `compact_every=1`
# auto-compacts (wipes the working tier) at the start of every step, so per-task context stays FLAT regardless of how
# far into the stream we are — this is what lets reuse lower cost instead of inflating it.
def _recall(task: str) -> str:
    return mem.as_block(task, header="Reusable recipes from earlier tasks you solved (reuse if they apply):",
                        k=RECALL_K)
ag = Agent(core, llm=llm, window=4, pinned=True, name="dev", compact_every=1, recall=_recall)
_embed_live = bool((default_embed(["probe"]) or [None])[0])
print("engine run: %s[%s] · %d tasks · <=%d cycles · recall_k=%d · embed=%s · model %s"
      % (DS, SPLIT, len(tasks), CYCLES, RECALL_K, "on" if _embed_live else "OFF(exact-dedup)", base.model), flush=True)


def _strip(code: str) -> str:
    code = (code or "").strip()
    if "```" in code:
        parts = code.split("```")
        code = max(parts, key=len)
        if code.lstrip().startswith("python"):
            code = code.lstrip()[len("python"):]
    return code.strip() + "\n"


def grade(code: str, task, timeout: int = 15) -> tuple:
    """Execution verdict: write the agent's code + the task's tests, run in a clean subprocess."""
    d = tempfile.mkdtemp(prefix="agent_")
    open(os.path.join(d, "solution.py"), "w").write(code)
    open(os.path.join(d, "test_task.py"), "w").write(
        "\n".join(task.get("test_imports") or []) + "\nfrom solution import *  # noqa\n"
        + "\n".join(task["test_list"]) + "\nprint('ALL_PASS')\n")
    try:
        p = subprocess.run([sys.executable, "test_task.py"], cwd=d, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout + p.stderr)
        return (p.returncode == 0 and "ALL_PASS" in out), out[-400:]
    except Exception as e:                                    # noqa: BLE001
        return False, str(e)[:200]


def _recipe(prompt: str, code: str) -> str:
    """A compact, reusable recipe distilled from a solved task: gist -> the working signature + first lines."""
    sig = next((ln for ln in code.splitlines() if ln.strip().startswith("def ")), "")
    return "PROBLEM: %s\nWORKING APPROACH: %s | %s" % (prompt.strip()[:140], sig.strip(), code.strip()[:200])


def solve(task) -> tuple:
    prompt = task.get("prompt") or task.get("text") or ""
    tests = "\n".join(task["test_list"])
    reused = bool(_recall(prompt))                            # native recall (bounded top-k) fires for this task
    base = ("TASK: %s\nThe solution must pass these tests:\n%s\n"
            "Write the COMPLETE Python source that makes the tests pass." % (prompt, tests))
    ok, cycles, code, fb = False, 0, "", ""
    for c in range(CYCLES):
        cycles = c + 1
        # failure feedback rides in the task turn (NOT the working tier), so compact_every=1 can wipe freely and
        # per-task context stays flat — recipes come from the bounded native recall seam, not a growing context.
        t = base + (("\n\nYour previous attempt FAILED when the tests ran:\n%s\nReturn the CORRECTED source only." % fb)
                    if fb else "")
        code = _strip(ag.act(t, max_tokens=650))
        ok, out = grade(code, task)
        if ok:
            break
        fb = out[:350]
    if ok:
        mem.add(_recipe(prompt, code))                        # distill a reusable recipe (novelty gate self-compresses)
    return ok, cycles, reused


rows = []
passed = 0
for i, t in enumerate(tasks):
    ok, cyc, reused = solve(t)
    passed += ok
    rows.append({"i": i, "task_id": t.get("task_id"), "passed": ok, "cycles": cyc, "reused": reused,
                 "tokens_cum": llm.total, "mem": len(mem)})
    print("  [%2d/%d] pass@1 %.0f%%  (task %s: %s in %d cyc · reuse=%s · mem=%d)"
          % (i + 1, len(tasks), 100 * passed / (i + 1), t.get("task_id"), "PASS" if ok else "fail", cyc,
             "Y" if reused else "-", len(mem)), flush=True)

n = len(tasks)
half = n // 2
tok_h1 = rows[half - 1]["tokens_cum"] if half else 0
tok_h2 = llm.total - tok_h1
p_h1 = sum(r["passed"] for r in rows[:half])
p_h2 = sum(r["passed"] for r in rows[half:])
reuse_hits = sum(1 for r in rows if r["reused"])

out_path = os.path.join(HERE, "out", "mbpp_agent_signals.json")
json.dump({"dataset": "%s/%s" % (DS, CONFIG), "n": n, "max_cycles": CYCLES, "recall_k": RECALL_K,
           "embed": _embed_live, "passed": passed, "pass_rate": round(passed / n, 3), "tokens": llm.total,
           "mem_final": len(mem), "reuse_hits": reuse_hits,
           "first_half": {"n": half, "passed": p_h1, "tokens": tok_h1},
           "second_half": {"n": n - half, "passed": p_h2, "tokens": tok_h2}, "rows": rows},
          open(out_path, "w"), indent=2)

print("\n" + "=" * 70)
print("%s — LONG-HORIZON FRAMEWORK (one persistent agent + reusable memory):" % DS.upper())
print("  pass@1 = %d/%d = %.1f%%  ·  %d tokens  ·  memory kept %d recipes  ·  reuse fired on %d/%d tasks"
      % (passed, n, 100 * passed / n, llm.total, len(mem), reuse_hits, n))
print("  COST DECAY (the thesis) — first half vs second half of the same stream:")
print("    1st half: %d/%d pass · %d tok · %.0f tok/task" % (p_h1, half, tok_h1, tok_h1 / max(half, 1)))
print("    2nd half: %d/%d pass · %d tok · %.0f tok/task" % (p_h2, n - half, tok_h2, tok_h2 / max(n - half, 1)))
print("  signals → out/mbpp_agent_signals.json")
print("=" * 70)
