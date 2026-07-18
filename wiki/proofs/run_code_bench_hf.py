#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad.llm import LLM

from datasets import load_dataset   # noqa: E402

DS = os.environ.get("BENCH_DS", "mbpp")
CONFIG = os.environ.get("BENCH_CONFIG", "sanitized")
SPLIT = os.environ.get("BENCH_SPLIT", "test")
N = int(os.environ.get("BENCH_N", "100"))

ds = load_dataset(DS, CONFIG, split=SPLIT)
tasks = list(ds)[:N]
llm = LLM(temperature=0.0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print("dataset: %s/%s[%s] · %d tasks · model %s" % (DS, CONFIG, SPLIT, len(tasks), llm.model), flush=True)

tokens = {"n": 0}


def _strip(code: str) -> str:
    code = (code or "").strip()
    if "```" in code:
        parts = code.split("```")
        code = max(parts, key=len)
        if code.lstrip().startswith("python"):
            code = code.lstrip()[len("python"):]
    return code.strip() + "\n"


RETRIES = int(os.environ.get("BENCH_RETRIES", "2"))       # verify-by-outcome: feed the failing test output back, retry


def solve(t, feedback: str = "") -> str:
    prompt = ("Write a correct Python solution. Return ONLY the code — no explanation, no fences.\n\n"
              "TASK: %s\n\nIt must pass these tests:\n%s" % (t.get("prompt") or t.get("text"), "\n".join(t["test_list"])))
    if feedback:
        prompt += ("\n\nYour previous attempt FAILED when the tests were run:\n%s\nReturn the CORRECTED code only."
                   % feedback)
    out = llm([{"role": "user", "content": prompt}], max_tokens=700, temperature=0.0) or ""
    u = getattr(llm, "last_usage", None) or {}
    tokens["n"] += u.get("total_tokens") or (len(prompt) // 4 + len(out) // 4)
    return _strip(out)


def grade(code: str, t, timeout: int = 15):
    body = "\n".join(t.get("test_imports") or []) + "\n" + code + "\n" + "\n".join(t["test_list"]) + "\n"
    d = tempfile.mkdtemp(prefix="mbpp_")
    open(os.path.join(d, "t.py"), "w").write(body)
    try:
        p = subprocess.run([sys.executable, "t.py"], cwd=d, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr)[-500:]
    except Exception as e:                                  # noqa: BLE001
        return False, str(e)[:200]


import json                                                 # noqa: E402
rows = []
pass_at = [0] * (RETRIES + 1)                               # cumulative solved within a=0,1,2 retries (the gain curve)
for i, t in enumerate(tasks):
    fb, solved_at = "", None
    for a in range(RETRIES + 1):
        ok, out = grade(solve(t, fb), t)
        if ok:
            solved_at = a
            break
        fb = out
    for a in range(RETRIES + 1):
        if solved_at is not None and solved_at <= a:
            pass_at[a] += 1
    rows.append({"i": i, "task_id": t.get("task_id"), "solved_at": solved_at})
    if (i + 1) % 10 == 0 or solved_at is None:
        print("  [%3d/%d] pass@1 %.0f%% · with-retries %.0f%%  (task %s solved_at=%s)"
              % (i + 1, len(tasks), 100 * pass_at[0] / (i + 1), 100 * pass_at[RETRIES] / (i + 1),
                 t.get("task_id"), solved_at), flush=True)

n = len(tasks)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "mbpp_signals.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump({"dataset": "%s/%s" % (DS, CONFIG), "n": n, "retries": RETRIES, "pass_at": pass_at,
           "tokens": tokens["n"], "rows": rows}, open(out_path, "w"), indent=2)

print("\n" + "=" * 66)
print("%s (execution-graded, N=%d) — the verify-by-outcome GAIN:" % (DS.upper(), n))
for a in range(RETRIES + 1):
    lbl = "pass@1 (no retry)" if a == 0 else "with %d retr%s" % (a, "y" if a == 1 else "ies")
    print("  %-20s %d/%d = %.1f%%%s" % (lbl, pass_at[a], n, 100 * pass_at[a] / n,
          "   (+%.1f pts from execution feedback)" % (100 * (pass_at[a] - pass_at[0]) / n) if a else ""))
print("  tokens: %d · signals → out/mbpp_signals.json" % tokens["n"])
print("=" * 66)
