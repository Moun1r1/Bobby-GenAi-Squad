import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Task:
    id: str
    category: str
    issue: str           # the "ticket": a natural-language description of the wrong behavior
    buggy: str           # the buggy module source
    gold: str            # a known-correct source (for harness self-test / an upper bound; never shown to the agent)
    test: str            # a self-running test (calls its assertions at import; exit 0 iff correct)
    module: str = "mod.py"


def grade(task: "Task", source: str, timeout: int = 20) -> dict:
    """Write `source` as the module + the task's test into a temp dir and RUN the test. Pass iff exit code 0."""
    d = tempfile.mkdtemp(prefix="swe_")
    open(os.path.join(d, task.module), "w").write(source)
    open(os.path.join(d, "test_task.py"), "w").write(task.test)
    try:
        p = subprocess.run([sys.executable, "test_task.py"], cwd=d, capture_output=True, text=True, timeout=timeout)
        return {"passed": p.returncode == 0, "output": (p.stdout + p.stderr)[-1200:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "timeout (possible infinite loop)"}
    except Exception as e:                                   # noqa: BLE001
        return {"passed": False, "output": "harness error: %s" % e}


def run(tasks: List["Task"], fix: Callable[["Task"], str], on_task: Optional[Callable[[dict], None]] = None) -> dict:
    """Run each task: verify the buggy version fails, ask `fix` for a patch, and grade it by execution."""
    rows = []
    for t in tasks:
        base = grade(t, t.buggy)                             # sanity: the bug must actually fail the test
        try:
            patched = fix(t) or ""
        except Exception as e:                               # noqa: BLE001
            patched = ""
            fix_err = str(e)[:120]
        else:
            fix_err = ""
        res = grade(t, patched) if patched else {"passed": False, "output": "no patch produced: " + fix_err}
        row = {"id": t.id, "category": t.category, "buggy_fails": not base["passed"], "fixed": res["passed"],
               "output": res["output"]}
        rows.append(row)
        if on_task:
            on_task(row)
    n = len(rows)
    passed = sum(1 for r in rows if r["fixed"])
    valid = sum(1 for r in rows if r["buggy_fails"])         # tasks whose bug genuinely fails (dataset integrity)
    return {"rows": rows, "n": n, "passed": passed, "pass_rate": round(passed / n, 3) if n else 0.0,
            "dataset_valid": valid == n}


# ── the task set — real, self-contained bug-fixes; each buggy version fails its test, each gold passes ──────
TASKS: List[Task] = [
    Task("chunk_partial", "boundary",
         "chunk(items, size) must include the final partial chunk; it currently drops it.",
         buggy=("def chunk(items, size):\n"
                "    out = []\n"
                "    for i in range(0, len(items) - size + 1, size):\n"
                "        out.append(items[i:i + size])\n"
                "    return out\n"),
         gold=("def chunk(items, size):\n"
               "    return [items[i:i + size] for i in range(0, len(items), size)]\n"),
         test=("from mod import chunk\n"
               "assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\n"
               "assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]\n"
               "assert chunk([], 2) == []\n")),
    Task("mutable_default", "mutable-default",
         "collect(x, acc=[]) accumulates across calls because of a shared mutable default; each call must start fresh.",
         buggy=("def collect(x, acc=[]):\n"
                "    acc.append(x)\n"
                "    return acc\n"),
         gold=("def collect(x, acc=None):\n"
               "    if acc is None:\n"
               "        acc = []\n"
               "    acc.append(x)\n"
               "    return acc\n"),
         test=("from mod import collect\n"
               "assert collect(1) == [1]\n"
               "assert collect(2) == [2]\n")),
    Task("none_in_sum", "none-guard",
         "total(prices) must skip missing (None) entries; it currently raises on them.",
         buggy=("def total(prices):\n"
                "    return sum(p for p in prices)\n"),
         gold=("def total(prices):\n"
               "    return sum(p for p in prices if p is not None)\n"),
         test=("from mod import total\n"
               "assert total([1,2,3]) == 6\n"
               "assert total([1, None, 3]) == 4\n")),
    Task("median_edges", "empty-edge",
         "median(xs) must handle an even count (average the two middles) and an empty list (return None).",
         buggy=("def median(xs):\n"
                "    s = sorted(xs)\n"
                "    return s[len(s) // 2]\n"),
         gold=("def median(xs):\n"
               "    if not xs:\n"
               "        return None\n"
               "    s = sorted(xs); n = len(s)\n"
               "    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n"),
         test=("from mod import median\n"
               "assert median([3,1,2]) == 2\n"
               "assert median([1,2,3,4]) == 2.5\n"
               "assert median([]) is None\n")),
    Task("bsearch_last", "boundary-search",
         "bsearch(a, x) returns -1 for the largest element; the loop bound is off by one.",
         buggy=("def bsearch(a, x):\n"
                "    lo, hi = 0, len(a) - 1\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if a[mid] == x:\n"
                "            return mid\n"
                "        if a[mid] < x:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1\n"),
         gold=("def bsearch(a, x):\n"
               "    lo, hi = 0, len(a) - 1\n"
               "    while lo <= hi:\n"
               "        mid = (lo + hi) // 2\n"
               "        if a[mid] == x:\n"
               "            return mid\n"
               "        if a[mid] < x:\n"
               "            lo = mid + 1\n"
               "        else:\n"
               "            hi = mid - 1\n"
               "    return -1\n"),
         test=("from mod import bsearch\n"
               "a = [1,3,5,7,9]\n"
               "assert bsearch(a,1) == 0\n"
               "assert bsearch(a,9) == 4\n"
               "assert bsearch(a,5) == 2\n"
               "assert bsearch(a,4) == -1\n")),
    Task("flatten_recur", "recursion",
         "flatten(xs) must recurse into nested lists; it currently appends a sublist instead of flattening it.",
         buggy=("def flatten(xs):\n"
                "    out = []\n"
                "    for x in xs:\n"
                "        if isinstance(x, list):\n"
                "            out.append(x)\n"
                "        else:\n"
                "            out.append(x)\n"
                "    return out\n"),
         gold=("def flatten(xs):\n"
               "    out = []\n"
               "    for x in xs:\n"
               "        if isinstance(x, list):\n"
               "            out.extend(flatten(x))\n"
               "        else:\n"
               "            out.append(x)\n"
               "    return out\n"),
         test=("from mod import flatten\n"
               "assert flatten([1,[2,3],[4,[5,6]]]) == [1,2,3,4,5,6]\n"
               "assert flatten([1,2,3]) == [1,2,3]\n")),
]
