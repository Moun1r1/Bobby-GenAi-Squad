#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bobby_squad import sheaf_consensus, make_consensus_harvest  # noqa: E402

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def f1(pred, gt):
    pred, gt = set(pred), set(gt)
    tp = len(pred & gt)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


GT = {"a", "b", "c", "d"}

# ── 1) redundant coverage + one singleton hallucination → consensus filters it ─────────────────────────
props = [["a", "b", "c"], ["a", "b", "d"], ["a", "c", "d", "X"]]     # X = 1-agent hallucination
union = set().union(*map(set, props))
res = sheaf_consensus(props, embed=None)
chk("engages consensus when agents overlap (mode=consensus)", res.mode == "consensus")
chk("consensus drops the singleton hallucination 'X'", "X" not in res.accepted)
chk("consensus keeps every majority-supported fact (a,b,c,d)", GT.issubset(set(res.accepted)))
chk("consensus F1 > union F1 under a hallucination",
    f1(res.accepted, GT) > f1(union, GT) and f1(res.accepted, GT) == 1.0)

# ── 2) clean redundant coverage (no disagreement) → parity, nothing pruned ─────────────────────────────
clean = [["a", "b", "c", "d"], ["a", "b", "c", "d"], ["a", "b", "c", "d"]]
rc = sheaf_consensus(clean, embed=None)
chk("clean agreement → consensus == union (no wrongful pruning)", set(rc.accepted) == GT)

# ── 3) SAFETY: agents partition disjoint work → fall back to union, prune NOTHING ───────────────────────
# mirrors confirm_coordination: each agent maps a different file's functions, zero overlap.
disjoint = [["a", "b"], ["c", "d"], ["e", "f"]]
rd = sheaf_consensus(disjoint, embed=None)
chk("disjoint coverage detected → mode=union (conditional fallback)", rd.mode == "union")
chk("disjoint coverage → keeps ALL items (recall preserved)",
    set(rd.accepted) == {"a", "b", "c", "d", "e", "f"})
chk("disjoint redundancy measured ~0", rd.redundancy < 0.34)

# forcing consensus off the conditional would wrongly prune the disjoint set → proves the guard matters
forced = sheaf_consensus(disjoint, embed=None, conditional=False)
chk("without the conditional guard, consensus WOULD over-prune disjoint work",
    len(forced.accepted) < 6)

# ── 4) drop-in squad harvest matches squad_solve's harvest(result, acc) contract ───────────────────────
h = make_consensus_harvest(embed=None)
acc = set()
for r in props:
    acc = h(r, acc)
chk("make_consensus_harvest returns a set and filters the hallucination", isinstance(acc, set) and "X" not in acc)
chk("make_consensus_harvest keeps the corroborated facts", GT.issubset(acc))

# ── 5) degenerate inputs are safe ──────────────────────────────────────────────────────────────────────
chk("single agent → union (no consensus possible)", sheaf_consensus([["a", "b"]], embed=None).mode == "union")
chk("empty proposals → empty accepted", sheaf_consensus([[], []], embed=None).accepted == [])

print("\n%d passed, %d failed" % (len(ok), len(bad)))
if bad:
    print("FAILURES:", bad)
    sys.exit(1)
print("PROVEN: consensus harvest filters squad hallucinations under overlap, and is a safe no-op on disjoint work.")
