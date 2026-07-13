"""LIVE end-to-end integration test — pipeline GENERATION + auto-managed cache / logic / memory / system pipelines.

Exercises the real running backend (default http://localhost:8091):
  1. GENERATE new use-case pipelines from pure SELF (role + goal) — the psychology/conflict use cases.
  2. Verify they register, carry the `custom` flag, and PERSIST (cache).
  3. Run the built-in `goal` pipeline on a psychology-style goal → criteria + evidence cards + a crystallized expert.
  4. Run a GENERATED pipeline on real records → engine-directed sections + stored knowledge (memory auto-managed).
  5. Verify the auto-managed substrate grew: /stats knowledge, /experts, /knowledge/scatter (vector memory), /config (cache).

Golden rule: generated pipelines are SELF-only (identity + goal), engine self-directs, NO prompt.

Needs the backend + a reachable model. If a run produces no events in time, the model is unavailable → SKIP (not FAIL).
Run:  python tests/test_integration.py            (or set BOBBY_BASE=...)
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BOBBY_BASE", "http://localhost:8091")


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                              headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get(p): return _req("GET", p)
def post(p, b): return _req("POST", p, b)
def delete(p): return _req("DELETE", p)


# The psychology / conflict-management use cases, expressed as pure SELF (role + goal) — no prompt.
USE_CASES = [
    {"id": "conflict_patterns", "title": "Conflict patterns",
     "identity": "a behavioral psychologist who models human conflict dynamics",
     "goal": "read each conflict scenario, model the psychological drivers and biases at play, and extract a reusable resolution strategy",
     "domain": "psychology"},
    {"id": "memory_cards", "title": "Memory cards",
     "identity": "a knowledge curator",
     "goal": "distill each finding into a polished, self-contained reusable memory card with a cross-domain insight",
     "domain": "knowledge"},
]

# Conflict records the generated pipeline will read (JSON rows → work units).
CONFLICT_DATA = json.dumps([
    {"scenario": "Two engineers clash over a rewrite; one feels ownership threatened, the other feels blocked."},
    {"scenario": "A manager and report disagree on scope; the report feels unheard, the manager feels deadline pressure."},
    {"scenario": "Two teammates avoid a needed hard conversation for weeks; resentment compounds silently."},
])

PSYCH_GOAL = ("Identify and clearly articulate three distinct, high-value benefits of unit tests, with deep "
              "explanations of their role as executable documentation, safety mechanisms, and enablers of loose "
              "coupling and refactoring; draw psychological/organizational analogies (safety, cognitive-load reduction).")


def wait_done(rid, timeout=150):
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout:
        d = get(f"/runs/{rid}")
        n = len(d.get("events", []))
        if n:
            last = n
        if d.get("status") in ("done", "error"):
            return d
        time.sleep(3)
    return get(f"/runs/{rid}")


def kinds(d):
    from collections import Counter
    return dict(Counter(e["kind"] for e in d.get("events", [])))


def main():
    results = []
    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  {'✓' if ok else '✗'} {name}{(' — ' + detail) if detail else ''}")

    # 0) backend up
    h = get("/health")
    check("backend up", h.get("ok"), f"{h.get('pipelines')} pipelines · store {h.get('store')}")

    # 1) GENERATE pipelines from SELF
    for uc in USE_CASES:
        delete(f"/pipelines/{uc['id']}")               # clean slate
        r = post("/pipelines/spec", uc)
        check(f"generate '{uc['id']}'", r.get("ok"), r.get("error", ""))

    # 2) registered + custom flag + persisted
    cat = {p["id"]: p for p in get("/pipelines")}
    for uc in USE_CASES:
        p = cat.get(uc["id"])
        check(f"'{uc['id']}' in catalog + custom", bool(p) and p.get("custom"))

    know0 = get("/stats").get("knowledge", 0)

    # 3) run the built-in goal pipeline on a psychology-style goal
    g = post("/runs", {"pipeline": "goal", "params": {"goal": PSYCH_GOAL, "agents": 1}})
    gd = wait_done(g["run_id"])
    gk = kinds(gd)
    crit = next((e for e in gd["events"] if e["kind"] == "criteria"), None)
    cards = [e for e in gd["events"] if e["kind"] == "card" and e.get("cstate") == "verified" and e.get("evidence")]
    experts = [e for e in gd["events"] if e["kind"] == "expert"]
    if not gd.get("events"):
        check("goal run (LLM)", False, "SKIP — model unavailable"); return report(results, skipped=True)
    check("goal · criteria derived", bool(crit and crit.get("criteria")), f"{len(crit['criteria']) if crit else 0} criteria")
    check("goal · evidence cards (engine-directed)", len(cards) >= 1, f"{len(cards)} cards · e.g. {cards[0]['evidence'][:70] if cards else ''}")
    check("goal · expert crystallized (memory)", len(experts) >= 1, experts[0].get("specialty", "")[:60] if experts else "")

    # 4) run a GENERATED pipeline on real records
    c = post("/runs", {"pipeline": "conflict_patterns", "params": {"data": CONFLICT_DATA, "kind": "json", "agents": 1}})
    cd = wait_done(c["run_id"])
    sections = [e for e in cd["events"] if e["kind"] == "section" and e.get("note")]
    res = next((e for e in cd["events"] if e["kind"] == "result"), None)
    check("generated pipeline runs (engine-directed)", len(sections) >= 1, f"{len(sections)} sections · {kinds(cd)}")
    check("generated pipeline result", bool(res), f"{res.get('notes', 0)} notes" if res else "")

    # 5) auto-managed memory + cache
    know1 = get("/stats").get("knowledge", 0)
    check("memory auto-managed (knowledge grew)", know1 > know0, f"{know0} → {know1}")
    exp = get("/experts").get("experts", [])
    check("experts persisted", len(exp) >= 1, f"{len(exp)} experts")
    pts = get("/knowledge/scatter?limit=50").get("points", [])
    check("vector memory projects", len(pts) >= 1, f"{len(pts)} points")
    post("/config", {"agents": 4, "patience": 3, "max_units": 80})
    cfg = get("/config")
    check("config cache round-trips", cfg.get("agents") == 4)

    # cleanup generated pipelines
    for uc in USE_CASES:
        delete(f"/pipelines/{uc['id']}")

    return report(results)


def report(results, skipped=False):
    ok = sum(1 for _, o, _ in results if o)
    total = len(results)
    print(f"\n{ok}/{total} checks passed" + (" (LLM SKIPPED)" if skipped else ""))
    fails = [n for n, o, _ in results if not o]
    if fails and not skipped:
        print("FAILED:", fails)
    sys.exit(0 if (ok == total or skipped) else 1)


if __name__ == "__main__":
    main()
