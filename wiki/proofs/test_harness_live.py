#!/usr/bin/env python3
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import memory_gains as MG
from bobby_squad import Scenario, DataCollector, harness_verdict, synthbench

ok, bad = [], []


def chk(name, cond):
    (ok if cond else bad).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def metric_for(mname):
    m = MG.BUILTINS[mname]

    def _metric(sc: Scenario) -> float:
        rows = synthbench.dataset(sc.seed, n_docs=2, length_chars=6000, needles=1)
        f1 = 0.0
        for r in rows:
            mem = m.prepare(r["context"])
            out = m.answer(mem, r["input"])
            a = out[0] if isinstance(out, tuple) else out
            f1 += MG.qa_f1(a, r["answers"])
        return 100 * f1 / max(1, len(rows))
    return _metric


dc = DataCollector()
base = Scenario("bench", seed=100)
R = 3
solo = dc.run("solo", metric_for("solo"), base, replications=R)
flatk = dc.run("flat_k", metric_for("flat_k"), base, replications=R)
null = dc.run("null", metric_for("null"), base, replications=R)

print(f"  solo   F1 {solo.mean:.1f} ±{solo.ci:.1f}   (n={solo.n})")
print(f"  flat_k F1 {flatk.mean:.1f} ±{flatk.ci:.1f}")
print(f"  null   F1 {null.mean:.1f} ±{null.ci:.1f}   (negative control)")

chk("LIVE: harness produced real CI reports for all arms", solo.n == R and flatk.n == R and null.n == R)
chk("LIVE: synthetic benchmark is CLEAN (null control ≈0, unguessable)", null.mean < 15.0)
chk("LIVE: solo actually uses the context (needle found)", solo.mean > 40.0)
v = harness_verdict(flatk, solo, control=null)
chk("LIVE: verdict not INVALID (control does not leak)", v["verdict"] != "INVALID")
print("  flat_k vs solo verdict:", v)

print("\n== %d PASS / %d FAIL ==" % (len(ok), len(bad)))
sys.exit(1 if bad else 0)
