#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import swe_bench as S

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


chk("task set is non-trivial (>=6 tasks, categorized)", len(S.TASKS) >= 6 and all(t.category for t in S.TASKS))

# (i) + (ii): each buggy fails, each gold passes — the grader is real (runs the test)
buggy_all_fail = all(not S.grade(t, t.buggy)["passed"] for t in S.TASKS)
gold_all_pass = all(S.grade(t, t.gold)["passed"] for t in S.TASKS)
chk("every BUGGY module fails its test (bugs are real)", buggy_all_fail)
chk("every GOLD module passes its test (fixes are valid)", gold_all_pass)

# (iii): run() with the gold fixer → 100% pass, dataset valid
res_gold = S.run(S.TASKS, fix=lambda t: t.gold)
chk("run(gold-fixer) solves 100% by execution", res_gold["pass_rate"] == 1.0)
chk("dataset integrity flag set (all buggy versions fail first)", res_gold["dataset_valid"])

# (iv): a no-op fixer (returns the buggy code) solves 0%
res_noop = S.run(S.TASKS, fix=lambda t: t.buggy)
chk("run(no-op fixer) solves 0% (a non-fix is graded as failed)", res_noop["pass_rate"] == 0.0)

# (v): a fixer that only handles one category solves exactly those tasks (run() grades per-task)
def one_category_fix(t):
    return t.gold if t.category == "mutable-default" else t.buggy
res_partial = S.run(S.TASKS, fix=one_category_fix)
solved = {r["id"] for r in res_partial["rows"] if r["fixed"]}
chk("a single-category fixer solves exactly its tasks, nothing else", solved == {"mutable_default"})

# note (honest): the naive `=[]`→`=None` regex from the burn-in does NOT fully fix mutable-default in real code —
# it also needs the None-guard — so even the 'reducible' category resists a one-line distilled rule here.
import re
naive = re.sub(r"=\s*\[\]", "=None", next(t for t in S.TASKS if t.category == "mutable-default").buggy)
chk("naive regex fix is INCOMPLETE for real code (a None-guard is still required)",
    not S.grade(next(t for t in S.TASKS if t.category == "mutable-default"), naive)["passed"])

print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: SWE-bench-style tasks graded by execution — real bugs, valid golds, non-fix = fail.")
