"""runner — launch a squad pipeline in a background thread and stream its live observer events.

The engine already emits a live event stream via Agent(observer=fn): target → plan → move_start → tool → tool_done
→ move_end → cycle → check → done. The runner just:
  1) picks a pipeline (each builds real Agents wired to an observer),
  2) runs it in a thread,
  3) pushes every event onto an in-process queue (for the SSE live feed) AND into the vector store (durable),
  4) writes run status + any learned knowledge into the store.

Pipelines are registered in PIPELINES; each is a callable (emit) -> summary_dict. `emit(kind, **data)` is the same
signature the engine's observer uses, so wiring an Agent is a one-liner: Agent(..., observer=emit).
"""
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from typing import Callable, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))            # studio/backend
PKG = os.path.dirname(os.path.dirname(HERE))                 # the bobby_squad package dir
ROOT = os.path.dirname(PKG)                                  # its parent → makes `import bobby_squad` work
sys.path.insert(0, ROOT)

# Disable qwen "thinking" for ALL in-process native pipeline LLM calls (not just script pipelines) — otherwise the
# model returns reasoning-only with EMPTY content, which is why criteria came back 0 and evidence was blank.
os.environ.setdefault("GA_EXTRA_BODY", '{"chat_template_kwargs":{"enable_thinking":false}}')

from store import get_store, now   # noqa: E402


class Run:
    """One live run: an event queue drained by the SSE endpoint + a status the API can poll."""

    def __init__(self, pipeline: str, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.pipeline = pipeline
        self.params = params or {}
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.events: List[dict] = []
        self.status = "starting"
        self.summary: dict = {}
        self.seq = 0
        self.started = now()
        # -- control plane: a squad you can steer, not just watch --
        self.controls = {"stop": False, "pause": False}
        self.steer: List[str] = []            # human directives injected mid-run
        self.agents: Dict[str, object] = {}   # name -> Agent (for live steering / memory management)
        self.ledger = None                    # the IdeaLedger (idea_board) — so the board can be managed live
        self.proc = None                      # subprocess handle for script pipelines (so stop can kill it)

    def gate(self) -> bool:
        """Pipelines call this between steps. Blocks while paused; returns False if a stop was requested (the
        pipeline then breaks cleanly). This is what makes the run steerable."""
        while self.controls.get("pause") and not self.controls.get("stop"):
            self.status = "paused"
            time.sleep(0.3)
        if self.status == "paused":
            self.status = "running"
        return not self.controls.get("stop")

    def take_steer(self) -> List[str]:
        """Drain any human directives injected since the last step — the pipeline feeds them to its agents."""
        out, self.steer = self.steer, []
        if out:
            for s in out:
                self.emit("steer", text=s)
        return out

    def observe(self, event: dict) -> None:
        """The engine's observer contract: Agent(observer=fn) calls fn(event_dict) with a SINGLE dict
        ({agent, kind, ...}). Adapt it to emit()."""
        kind = event.get("kind", "event")
        self.emit(kind, **{k: v for k, v in event.items() if k != "kind"})

    def emit(self, kind: str, **data) -> None:
        self.seq += 1
        ev = {"run_id": self.id, "seq": self.seq, "ts": now(), "kind": kind, **data}
        self.events.append(ev)
        self.q.put(ev)
        try:
            get_store().upsert("events", f"{self.id}:{self.seq}", ev,
                               text=f"{kind} {data.get('target') or data.get('intention') or data.get('name') or ''}")
        except Exception:
            pass

    def finish(self, status: str, summary: dict) -> None:
        self.status = status
        self.summary = summary or {}
        rec = {"run_id": self.id, "pipeline": self.pipeline, "status": status, "ts": self.started,
               "ended": now(), "n_events": self.seq, "params": self.params, **{f"summary_{k}": v for k, v in
               (summary or {}).items() if isinstance(v, (str, int, float, bool))}}
        try:
            get_store().upsert("runs", self.id, rec, text=f"{self.pipeline} {summary.get('verdict','')} {summary}")
        except Exception:
            pass
        self.q.put({"run_id": self.id, "seq": self.seq + 1, "ts": now(), "kind": "done",
                    "status": status, "summary": summary})


RUNS: Dict[str, Run] = {}

# ---------------------------------------------------------------------------------------------------------------
# PROOFS — the framework's REAL gain-proofs. These deterministic scripts run the `confirm_gain`/`prove` PRIMITIVES
# (verify-by-outcome, one instance + one measured number, with negative-control + CI) and print machine-readable
# GAIN/PROVE lines. We just RUN them and capture the verdict — no scripted agent behaviour, no prompt (golden-rule
# clean: this is the prove primitive, exactly as the design sanctions). Results are cached (they're deterministic).
# ---------------------------------------------------------------------------------------------------------------
import json as _json                                                   # noqa: E402
# Only the deterministic gains that emit fast, machine-readable GAIN/PROVE verdicts locally (no DGX embedder needed).
# proposals_gain runs the full proposal A/B suite (Memory-Gate +191% WIRE, Active-Design, and honest DELETEs).
DETERMINISTIC_GAINS = ["repulsion_gain", "idea_diversity_gain", "proposals_gain"]
_PROOFS: dict = {"state": "idle", "results": [], "ran": 0, "total": 0}


def run_proofs() -> None:
    """Run the deterministic gain-proofs in the background and cache their real verdicts."""
    if _PROOFS["state"] == "running":
        return
    _PROOFS.update(state="running", results=[], ran=0, total=len(DETERMINISTIC_GAINS))

    def _go():
        for name in DETERMINISTIC_GAINS:
            path = os.path.join(PKG, "gains", name + ".py")
            if not os.path.isfile(path):
                _PROOFS["ran"] += 1
                continue
            try:
                proc = subprocess.run([sys.executable, path], cwd=PKG, env=dict(os.environ),
                                      capture_output=True, text=True, timeout=90)
                for line in (proc.stdout or "").splitlines():
                    s = line.strip()
                    for tag in ("GAIN ", "PROVE "):
                        if s.startswith(tag):
                            try:
                                d = _json.loads(s[len(tag):])
                                d["_source"], d["_kind"] = name, tag.strip()
                                _PROOFS["results"].append(d)
                            except Exception:
                                pass
            except Exception as e:
                _PROOFS["results"].append({"name": name, "verdict": "DEFER", "reason": str(e)[:140], "_source": name})
            _PROOFS["ran"] += 1
        _PROOFS["state"] = "done"

    threading.Thread(target=_go, daemon=True).start()


def proofs_state() -> dict:
    return dict(_PROOFS)


def launch(pipeline: str, params: dict) -> Run:
    r = Run(pipeline, params)
    RUNS[r.id] = r
    fn = PIPELINES.get(pipeline)

    def _go():
        r.status = "running"
        try:
            if fn is None:
                raise ValueError(f"unknown pipeline: {pipeline} (have: {list(PIPELINES)})")
            summary = fn(r)
            r.finish("done", summary or {})
        except Exception as e:
            r.emit("error", message=str(e), trace=traceback.format_exc()[-800:])
            r.finish("error", {"error": str(e)})

    threading.Thread(target=_go, daemon=True).start()
    return r


# ---------------------------------------------------------------------------------------------------------------
# pipelines — each builds real engine agents and returns a summary. Knowledge produced is written to the store so
# the frontend's semantic search can recall it (cross-domain transfer, the engine's point).
# ---------------------------------------------------------------------------------------------------------------
def _sources(trace) -> List[str]:
    """PROVENANCE from the agent's tool trace — the real files/lines/greps the finding is grounded in (provenance:
    every claim traces to file:line). Reads what the agent actually did; deduped, order-preserving."""
    out: List[str] = []
    for name, args in (trace or []):
        a = args if isinstance(args, dict) else {}
        if name == "read":
            out.append((f"{a.get('path', '')}:{a.get('lines', '')}").rstrip(":"))
        elif name == "grep":
            out.append(f"grep '{a.get('pattern', '')}' {a.get('path', '.')}".strip())
        elif name == "find":
            out.append(f"find {a.get('name', '')}".strip())
        elif name == "ls":
            out.append(f"ls {a.get('path', '.')}".strip())
        elif name in ("run", "write", "edit"):
            out.append(f"{name} {a.get('path', '')}".strip())
        else:
            out.append(str(name))
    seen: set = set()
    return [s for s in out if s and not (s in seen or seen.add(s))]


_EMBED_WIN = 4000        # chars per embedding window (~the embedder's token budget) — the MOVING WINDOW over long content
_WIN_OVERLAP = 400       # sliding overlap so a finding isn't cut mid-thought at a window boundary


def _windows(text: str):
    """Split content into OVERLAPPING moving windows so long content is stored AND fully retrievable (each window is
    embedded independently) — nothing is un-searchable. Short content → one window."""
    if len(text) <= _EMBED_WIN:
        return [text]
    step = _EMBED_WIN - _WIN_OVERLAP
    return [text[i:i + _EMBED_WIN] for i in range(0, len(text), step)]


def _store_knowledge(run: Run, text: str, meta: dict, source=None) -> None:
    """Persist a finding to long-horizon memory — FULL content, NO fixed cap, WITH its SOURCE provenance (file:line /
    tool trace). Content larger than the embedder's window is SPLIT into overlapping MOVING WINDOWS, each embedded +
    stored with its part index and the same source, so ALL of it is retrievable (not just the head). A memory with no
    source or truncated content is worthless."""
    try:
        text = text or ""
        wins = _windows(text)
        gid = uuid.uuid4().hex[:8]
        for i, win in enumerate(wins):
            part = {"part": i + 1, "parts": len(wins), "whole_chars": len(text)} if len(wins) > 1 else {}
            get_store().upsert("knowledge", f"{run.id}:{gid}:{i}",
                               {"run_id": run.id, "pipeline": run.pipeline, "text": win, "source": source or [],
                                "chars": len(win), "group": gid, "ts": now(), **part, **meta}, text=win)
    except Exception:
        pass


# ── KNOWLEDGE VAULT ──────────────────────────────────────────────────────────────────────────────────────────────
# A navigable, ENRICHABLE Obsidian-style graph of markdown notes (frontmatter + [[wikilinks]]) — not chunks in a bag.
# Agents ENTER it semantically for the step (native prefetch, wired to Agent.recall) and WRITE notes back from what
# they learn (experience, behavior, other pipelines, framework code, external repos). It is DATA the engine navigates.
VAULT_ROOT = os.path.join(PKG, "knowledge", "vaults")       # MANY vaults (foundation, repos, experience, behavior…)
_HUB = None


def _get_hub():
    global _HUB
    if _HUB is None:
        from bobby_squad import VaultHub
        _HUB = VaultHub(VAULT_ROOT)                         # embed_fn=None → engine's default_embed (lexical fallback)
    return _HUB


def _vault_recall(run: Run, per_vault_k: int = 2, hops: int = 1, budget: int = 2000):
    """Build the native-prefetch hook for a step: navigate ACROSS ALL vaults for the current target and return the
    cross-vault subgraph (entry notes + linked neighbours, spanning vaults), bounded + attributed. `basis=false` in
    params turns it off (the gain-proof knob). Returns None → agents run on the bare persistent-self baseline."""
    if not run.params.get("basis", True):
        return None
    hub = _get_hub()

    def recall(task: str) -> str:
        block = hub.navigate(task, per_vault_k=per_vault_k, hops=hops, budget=budget)
        if block:
            run.emit("recall", entry=hub.search(task, k=per_vault_k), vaults=hub.names(),
                     mode=hub.stats().get("recall", "cosine"), notes=hub.stats().get("notes", 0), chars=len(block))
        return block
    return recall


def _vault_enrich(run: Run, vault: str, title: str, body: str, links=None, tags=None) -> None:
    """Write what a run LEARNED into a specific vault (created on demand — the loop grows vaults dynamically).
    Deduped + auto-linked into the graph (incl. cross-vault [[foundation/…]] links), with provenance. Best-effort."""
    try:
        hub = _get_hub()
        nid = hub.enrich(vault, title, body, source=f"run:{run.id[:8]} pipeline:{run.pipeline}", links=links, tags=tags)
        if nid:
            st = hub.stats()
            run.emit("vault", action="enrich", vault=vault, note=f"{vault}/{nid}",
                     vaults=st.get("names", []), notes=st.get("notes", 0), edges=st.get("edges", 0))
    except Exception:
        pass


def _auto_dpo_pairs(steps):
    """DPO DATA PIPELINE (torch-free twin of encoders.trajectory_dpo): auto-harvest {prompt,chosen,rejected} from the
    agent's own SCORED trajectory — no hand labels. Improvement (value↑) → later step chosen; regression (value↓) →
    earlier step chosen; challenge pass ≻ fail. The self-monitor + value proxy close the preference loop."""
    pairs, prompt = [], "Given the task and where the agent is, produce the next step."
    for a, b in zip(steps, steps[1:]):
        va, vb = a.get("value", 0.0), b.get("value", 0.0)
        if a.get("response") and b.get("response") and a["response"] != b["response"]:
            if vb - va > 0.05:
                pairs.append({"prompt": prompt, "chosen": b["response"], "rejected": a["response"], "why": "improvement"})
            elif va - vb > 0.05:
                pairs.append({"prompt": prompt, "chosen": a["response"], "rejected": b["response"], "why": "regression"})
    passes = [s for s in steps if s.get("outcome") == "pass" and s.get("response")]
    fails = [s for s in steps if s.get("outcome") == "fail" and s.get("response")]
    for p in passes:
        for f in fails:
            pairs.append({"prompt": prompt, "chosen": p["response"], "rejected": f["response"], "why": "challenge"})
    return pairs


def _vault_dpo_block(run: Run, title: str, pairs, links=None) -> None:
    """Write auto-harvested preference pairs into the `behavior` vault as a `## dpo` block so self_dpo's harvest_dpo
    picks them up — the loop feeding its own DPO dataset."""
    if not pairs:
        return
    body = "_auto-harvested from the agent's own trajectory (self-monitor + value proxy) — no hand labels_\n\n## dpo\n"
    for p in pairs[:12]:
        body += (f"- prompt: {p['prompt']}\n- chosen: {p['chosen'][:200].strip()}\n"
                 f"- rejected: {p['rejected'][:200].strip()}\n")
    _vault_enrich(run, "behavior", title, body, links=links or ["foundation/long-horizon-improvement", "foundation/loops-system"],
                  tags=["behavior", "dpo", "auto-harvest"])
    run.emit("dpo_harvest", n=len(pairs), source="trajectory", reasons=sorted({p["why"] for p in pairs}))


def _emit_memory(run: Run, agent) -> None:
    """Surface the persistent-self two tiers + the flat-context proof (pinned stays small; naive would balloon)."""
    prog = agent.ctx.progress
    pinned_tok = sum(len(p) for p in prog) // 4                    # rough token estimate
    run.emit("memory", agent=agent.name, pinned_items=len(prog), working_items=len(agent.ctx.working),
             pinned_tokens=pinned_tok, naive_tokens=pinned_tok * 8 + len(agent.ctx.working) * 60,
             sample=[p[:160] for p in prog[-4:]])


def _emit_signals(run: Run, trace) -> None:
    """Metacognition: deterministic behavioral signals (move-entropy, area-concentration, repetition, novelty) + the
    grounded red-flags — the self-review substrate, never fabricated."""
    try:
        run.emit("signal", **trace.signals())
        run.emit("flags", agent=trace.name, flags=trace.flags())
    except Exception:
        pass


def _emit_mem_policy(run: Run, kb) -> None:
    """Surface the EVOLVED memory policy (the engine's proven SemanticMemory(policy='value') primitive, not a script):
    what the self-governing store KEEPS vs EVICTS, value-ranked by learned usage. capacity=N → overflow evicts the
    lowest usage-value item; retrieval raises value. This is the +25%-retention mechanism, made visible."""
    try:
        docs, meta = kb.r.docs, kb._meta
        order = sorted(range(len(docs)), key=lambda i: -meta[i]["value"])[:8]
        run.emit("mem_policy", policy=kb.policy, capacity=kb.capacity, stored=len(docs), seen=kb._born,
                 evicted=max(0, kb._born - len(docs)),
                 top=[{"text": docs[i][:120], "value": meta[i]["value"], "critical": meta[i]["critical"]}
                      for i in order])
    except Exception:
        pass


def _finalize_idea_lab(run: Run, ledger, kb) -> dict:
    """Shared END of any idea-lab run (code OR generic-domain): export the ranked prove-queue, the PORTFOLIO
    (the flywheel), crystallize per-area experts, and snapshot the evolved memory policy. Domain-free — it reads only
    the IdeaLedger + kb, so the same machinery serves math / sociology / company-org / code alike."""
    queue = sorted(ledger.ideas, key=lambda it: (it.get("variants", 0) + it.get("touched", 0)), reverse=True)[:8]
    prove_queue = [{"label": it["label"][:80], "area": it.get("area", ""), "status": it.get("status", "open"),
                    "variants": it.get("variants", 0), "test": it.get("test"), "has_test": bool(it.get("test")),
                    "redteam": it.get("redteam"), "viability": it.get("viability")} for it in queue]
    run.emit("prove_queue", items=prove_queue, reason="ranked by development — strong model proves via confirm_gain")
    # the portfolio, deterministic over the swarm's own signals (dev · feasibility · moat · novelty).
    port = ledger.portfolio()
    buckets = {b: [p for p in port if p["bucket"] == b] for b in ("quick-win", "core-bet", "moonshot")}
    run.emit("portfolio", items=port[:24], counts={b: len(v) for b, v in buckets.items()},
             reason="quick-wins / core-bets / moonshots — impact × feasibility × moat × novelty")
    n_experts = _crystallize_area_experts(run, ledger)
    _emit_mem_policy(run, kb)
    try:
        docs, meta = kb.r.docs, kb._meta
        order = sorted(range(len(docs)), key=lambda i: -meta[i]["value"])[:12]
        get_store().upsert("mem_policy", run.id, {"run_id": run.id, "pipeline": run.pipeline, "policy": kb.policy,
                           "capacity": kb.capacity, "stored": len(docs), "seen": kb._born,
                           "evicted": max(0, kb._born - len(docs)), "ts": now(),
                           "top": [{"text": docs[i][:140], "value": meta[i]["value"], "critical": meta[i]["critical"]}
                                   for i in order]}, text=f"memory policy {kb.policy} {run.id}")
    except Exception:
        pass
    return {"prove_queue": len(prove_queue), "area_experts": n_experts,
            "portfolio": {b: len(v) for b, v in buckets.items()},
            "mem_stored": len(kb), "mem_evicted": max(0, kb._born - len(kb))}


def _crystallize_area_experts(run: Run, ledger) -> int:
    """The FLYWHEEL (distills recurring wins into reusable primitives): a persistent 'Bobby expert per area
    of the code'. Group the board's OWN ideas by the code-area they landed in and upsert ONE reusable expert per area,
    carrying that area's accumulated knowledge. This is a DERIVED ARTIFACT from what the swarm mined — not a persona
    assigned to any agent, not a prompt. Complements the per-agent specialists that goal runs crystallize."""
    by_area: Dict[str, list] = {}
    for it in ledger.ideas:
        if (it.get("status") or "") == "known":                    # skip any seeded catalog entries
            continue
        by_area.setdefault(it.get("area") or "other", []).append(it)
    n = 0
    for area, items in by_area.items():
        know = [f"{it['label']}: {it.get('detail', '')[:160]}" for it in items]
        eid = f"{run.id}:area:{area}"
        try:
            get_store().upsert("experts", eid,
                               {"id": eid, "run_id": run.id, "name": f"{area} expert", "specialty": f"{area} specialist",
                                "area": area, "kind": "area", "n_knowledge": len(know), "knowledge": know, "ts": now()},
                               text=area + " " + " ".join(know[:20]))
            run.emit("expert", id=eid, name=f"{area} expert", specialty=f"{area} specialist", area=area,
                     kind="area", n_knowledge=len(know))
            n += 1
        except Exception:
            pass
    return n


def pipe_engine_trace(run: Run) -> dict:
    """A small self-organizing squad reads THIS repo — every engine layer fires live: self-directed moves, tool
    grounding, the pinned memory tier, and per-agent metacognition signals."""
    from bobby_squad import Agent, SelfCore, ReadOnlyTools, BehaviorTrace
    from bobby_squad.llm import LLM
    llm = LLM(temperature=0.5, timeout=120)
    n = int(run.params.get("agents", 2))
    traces = {f"agent{i}": BehaviorTrace(f"agent{i}", echo=run.observe) for i in range(n)}
    # WORLD-CONTEXT (a SELF — data). Behavior is the engine's; termination is by PLATEAU (a full round of the swarm
    # adds nothing new → fixed point). No patience knob; GUARD is a pure runaway backstop, like squad_solve's max_passes.
    recall = _vault_recall(run)                            # research USES the vault too — enter it for each step
    agents = [Agent(SelfCore("a researcher mapping this repo's real capabilities",
                             "find and note something worth reusing, grounded in the code"),
                    llm=llm, window=4, pinned=True, tools=ReadOnlyTools(PKG),
                    name=f"agent{i}", observer=traces[f"agent{i}"], recall=recall) for i in range(n)]
    run.agents = {a.name: a for a in agents}
    records = rnd = 0
    GUARD = int(run.params.get("max_waves", 30))
    while rnd < GUARD:
        rnd += 1
        before = records
        for ag in agents:
            if not run.gate():
                return {"agents": n, "records": records, "verdict": f"stopped by operator after {records} findings"}
            for s in run.take_steer():
                ag.observe(f"OPERATOR DIRECTIVE: {s}")           # a human steers the live squad
            res = ag.research_cycle()
            src = _sources(getattr(ag, "last_trace", None))
            for r in res.get("results", []):
                records += 1
                _store_knowledge(run, r.get("result", ""),
                                 {"agent": ag.name, "move": r.get("move", ""), "domain": "self/code"}, source=src)
            _emit_memory(run, ag)
            _emit_signals(run, traces[ag.name])
        run.emit("wave", n=rnd, records=records, new=records - before)
        if records == before:                           # a whole round added nothing new → plateau (fixed point)
            break
    return {"agents": n, "rounds": rnd, "records": records,
            "verdict": f"{records} grounded findings across {n} agents over {rnd} waves (plateaued)"}


def _idea_lab(run: Run, self_core, make_tools=None, domain: str = "") -> dict:
    """An idea lab = WORLD-CONTEXT (a SELF) on the GENERIC engine (squad_solve). Nothing here is a bespoke loop or a
    scripted gate: the SELF is data; the engine drives. The idea-lab plugs its OWN (work · harvest · verify) into
    squad_solve — that is world-context, not the engine hardcoding 'findings/novelty':
      • work   : an agent self-selects a target+move and produces contributions (research_cycle with tools, else
                 autonomous_cycle) — the OUTPUT can be anything the SELF calls for.
      • harvest: accumulate into the shared IdeaLedger (its identity floor is the coordination hint, not the driver).
      • verify : the outcome for THIS lab — a full round of the whole swarm added nothing new (fixed point). A different
                 output type (code that runs, a data map that covers) would plug a different verify; the engine is blind.
      • split  : re-queue the frontier while it's productive; the board drains at the fixed point (plateau)."""
    from bobby_squad import IdeaLedger, SemanticMemory, BehaviorTrace, squad_solve
    from bobby_squad.llm import LLM
    llm = LLM(temperature=0.6, timeout=120)
    n = int(run.params.get("agents", 3))
    ledger = IdeaLedger()
    kb = SemanticMemory(policy="value")                    # evolved-retention store (usage-value); coordination memory
    traces = {f"a{i}": BehaviorTrace(f"a{i}", echo=run.observe) for i in range(n)}
    from bobby_squad import Agent
    recall = _vault_recall(run)                            # idea/research lab enters the vault per step
    agents = [Agent(self_core, llm=llm, window=4, pinned=True,
                    tools=(make_tools(ledger) if make_tools else None),
                    name=f"a{i}", observer=traces[f"a{i}"], recall=recall) for i in range(n)]
    run.agents = {a.name: a for a in agents}
    run.ledger = ledger
    st = {"new": 0, "dry": 0, "admitted": 0, "repelled": 0, "stopped": False}

    def snapshot(*_):
        by_state: Dict[str, list] = {}
        for it in ledger.ideas:
            by_state.setdefault(it.get("status") or "open", []).append(
                {"label": it["label"][:70], "area": it.get("area") or domain, "variants": it.get("variants", 0),
                 "touched": it.get("touched", 0)})
        cov: Dict[str, int] = {}
        for it in ledger.ideas:
            cov[it.get("area", "other")] = cov.get(it.get("area", "other"), 0) + 1
        unexplored = [a for a in ledger.areas if cov.get(a, 0) == 0]
        run.emit("board", states=by_state, n_ideas=len(ledger.ideas), admitted=st["admitted"], repelled=st["repelled"],
                 areas_unexplored=unexplored, untested=len(ledger.untested()), unchallenged=len(ledger.unchallenged()))
        _emit_mem_policy(run, kb)

    def work(agent, unit):
        if not run.gate():
            st["stopped"] = True
            return []
        for s in run.take_steer():
            agent.observe(f"OPERATOR DIRECTIVE: {s}")
        agent.ctx.progress = ledger.signal()               # the shared board IS the agent's visible progress
        res = agent.research_cycle() if agent.tools is not None else agent.autonomous_cycle()
        results = res.get("results", [])
        if not results:                                    # this agent stalled → INTROSPECT its own behavior (feeds next cycle)
            u = agent.introspect()
            run.emit("introspect", agent=agent.name, understanding=u[:400])
        src = _sources(getattr(agent, "last_trace", None))  # PROVENANCE: the files/tools that grounded this cycle
        _emit_signals(run, traces[agent.name])
        return [{**r, "_agent": agent.name, "_source": src} for r in results]

    def harvest(results, acc):
        before = len(acc.ideas)
        for r in results:
            full = r.get("result") or ""                    # FULL content — long-horizon memory, do not over-summarize
            idea, is_new, ok = acc.admit(r.get("move") or r.get("type") or "", full[:300])   # board LABEL is short
            if ok:
                st["admitted"] += 1
                if domain:
                    idea["area"] = domain
                kb.add(full, critical=is_new)          # full — the value-policy + top-k retrieval IS the dynamic window
                _store_knowledge(run, full, {"agent": r.get("_agent", "?"), "move": r.get("move") or r.get("type", ""),
                                             "area": idea.get("area", ""), "domain": domain or "idea-board"},
                                 source=r.get("_source"))
            else:
                st["repelled"] += 1
        st["new"] = len(acc.ideas) - before
        return acc

    def verify(unit, acc):                                 # OUTCOME gate for this lab (world-context, not the engine)
        if st["stopped"]:
            return True
        if st["new"] > 0:
            st["dry"] = 0
            return False                                   # productive → keep developing the frontier
        st["dry"] += 1
        return st["dry"] >= len(agents)                    # a whole round of the swarm added nothing → fixed point → done

    r = squad_solve(agents, [self_core.goal or "the frontier"], work, verify=verify,
                    split=lambda unit: [unit], harvest=harvest, accumulated=ledger,
                    observer=snapshot, max_passes=400)
    fin = _finalize_idea_lab(run, ledger, kb)
    return {"agents": n, "n_ideas": len(ledger.ideas), "admitted": st["admitted"], "repelled": st["repelled"],
            "passes": r["passes"], **fin,
            "verdict": f"{len(ledger.ideas)} distinct ideas · portfolio {fin['portfolio']} · "
                       f"{fin['prove_queue']} queued to prove · {fin['area_experts']} experts"}


def pipe_idea_board(run: Run) -> dict:
    """The IdeaLedger board live: agents mine ideas from the repo, the board REPELS near-duplicates (identity floor),
    organizes them into emergent states, and surfaces the most-spread ideas (active repulsion). Emits `board`
    snapshots so the UI renders the kanban + repulsion frontier."""
    from bobby_squad import SelfCore, BoardTools
    # WORLD-CONTEXT only (a SELF — data, not logic). The engine (squad_solve, via _idea_lab) drives; no bespoke loop.
    self_core = SelfCore(
        "an R&D researcher advancing this generative-agent frontier — first understand the real code, then INVENT and "
        "COMPOSE the next reusable capability beyond it",
        "each contribution must be genuinely NEW relative to what the board already holds: extend an OPEN idea, invent "
        "on an unexplored area, or compose existing features into something new — never restate a closed idea",
        constraints=["ground every claim in the real code you read", "go where the board is thin, not where it is dense",
                     "an idea is not finished until it carries a falsifiable test: hypothesis · cheapest probe · "
                     "WIRE/DELETE threshold",
                     "an idea is not trusted until it survives its killer objection: name the strongest attack "
                     "(competitor · why the buyer says no · failure mode), then whether it holds"])
    return _idea_lab(run, self_core,
                     make_tools=lambda ledger: BoardTools(ledger, PKG, os.path.join(PKG, "out", "studio_board_sandbox")))


def pipe_research(run: Run) -> dict:
    """GENERIC domain research lab — the SAME engine as idea_board (squad_solve via _idea_lab), only the WORLD-CONTEXT
    changes: point it at ANY topic (a math conjecture, a sociology question, a company-org design, a proxy
    architecture…) and the swarm invents/develops/red-teams ideas in that domain, reasoning-grounded (no code tools).
    Proves the framework is domain-free: one engine, any subject. Only the SELF (data) varies — no bespoke loop."""
    from bobby_squad import SelfCore
    topic = (run.params.get("topic") or "an open research frontier").strip()
    domain = (run.params.get("domain") or "-".join(topic.lower().split()[:3]))[:32]
    self_core = SelfCore(
        f"a researcher advancing the frontier of {topic}",
        "invent, develop, and compose genuinely NEW ideas, models, designs, or arguments in this domain — each "
        "contribution must be new relative to what the board already holds; the prize is novel, buildable insight",
        constraints=["build on what the board holds; never restate a closed idea; go where it is thin",
                     "an idea is not finished until it carries a falsifiable test: hypothesis · cheapest probe · "
                     "pass/fail threshold",
                     "an idea is not trusted until it survives its strongest objection: name the killer attack "
                     "(a rival approach · why it fails · a failure mode), then whether it holds"])
    return _idea_lab(run, self_core, domain=domain)


def _dev_lab(run: Run, self_core, goal: str, kind: str = "code-dev") -> dict:
    """The GENERATIVE DEV/TRAIN engine — the deterministic verify layer becomes generative. A swarm shares one sandbox
    with real GPU tools (DgxTools → the isolated, memory-capped DGX worker), builds/trains toward a WORLD-CONTEXT goal,
    and WRITES ITS OWN acceptance CHALLENGE (`challenge.py`, prints 'CHALLENGE PASS' only when a real criterion/metric
    holds). It iterates until a REAL run passes; on a failed command or plateau an agent takes an ADVERSARIAL REVIEW
    move on its own logic. Pre-gated by the realtime NVIDIA monitor so it never crashes the shared DGX. Only the SELF
    (world-context) differs between building code and training a model — the engine is one."""
    from bobby_squad import Agent, DgxTools
    from bobby_squad.llm import LLM
    llm = LLM(temperature=0.4, timeout=240)
    n = int(run.params.get("agents", 3))
    sandbox = os.path.join(PKG, "out", f"{kind}_{run.id[:8]}")
    os.makedirs(sandbox, exist_ok=True)
    # PRE-TRAIN GATE (resource management) — never start heavy GPU work unless the box has headroom right now.
    try:
        from dgx_monitor import get_monitor
        safe = get_monitor().is_safe()
        run.emit("dgx_gate", **safe)
        if not safe.get("safe"):
            return {"verdict": f"REFUSED — DGX not safe: {safe.get('reason')}", "dgx": safe,
                    "hint": "watch /dgx/stream; free the GPU or lower DGX_GPU_FRACTION, then retry"}
    except Exception:
        pass
    # ONE shared sandbox = the shared codebase. DgxTools = REAL GPU compute in the isolated worker; every remote action
    # streams a `dgx` observability event to the run.
    tools = DgxTools(PKG, sandbox, observer=run.observe)
    recall = _vault_recall(run)                            # native prefetch: each step enters the knowledge vault
    agents = [Agent(self_core, llm=llm, window=5, pinned=True, tools=tools, name=f"dev{i}",
                    observer=run.observe, recall=recall) for i in range(n)]
    run.agents = {a.name: a for a in agents}

    def challenge_result():
        if "challenge.py" not in tools.tree():
            return None
        out = tools.run("challenge.py")
        ok = ("CHALLENGE PASS" in out) and ("[exit 0" in out)
        run.emit("challenge", output=out[-600:], passed=ok)
        return ok

    def live_status(last):
        return ("LIVE STATE — sandbox files: " + " ".join((tools.tree() or "(empty)").split()) +
                (f" | challenge.py last run: {last[-260:]}" if last else
                 " | challenge.py NOT written yet — write it once the trainer runs"))

    GUARD = int(run.params.get("max_waves", 40))          # runaway backstop only; the real stop is the challenge passing
    passed, last_out, rnd, stale = False, "", 0, 0
    steps = []                                             # the SCORED trajectory → auto-DPO harvest (value proxy per wave)
    while rnd < GUARD and not passed:
        rnd += 1
        before = tools.tree()
        for ag in agents:
            if not run.gate():
                return {"agents": n, "rounds": rnd, "verdict": "stopped by operator", "sandbox": sandbox}
            for s in run.take_steer():
                ag.observe(f"OPERATOR DIRECTIVE: {s}")
            # ACCUMULATE, don't wipe: keep the agent's own findings (what it already verified — torch version, the
            # config, that the model loads) so it STOPS re-probing the same things; only refresh ONE live-state line.
            findings = [p for p in ag.ctx.progress if not p.startswith("LIVE STATE")]
            ag.ctx.progress = [live_status(last_out)] + findings[-40:]
            ag.research_cycle()
            for p in ag.ctx.progress[-3:]:                 # persist what was built/trained + why, with provenance
                if not p.startswith("LIVE STATE"):
                    _store_knowledge(run, p, {"agent": ag.name, "domain": kind}, source=_sources(ag.last_trace))
            run.emit("dev", agent=ag.name, tree=tools.tree()[:800])
        if challenge_result() is True:
            passed = True
        last_out = tools.run("challenge.py") if "challenge.py" in tools.tree() else last_out
        stale = stale + 1 if tools.tree() == before and not passed else 0
        run.emit("wave", n=rnd, files=len((tools.tree() or "").splitlines()), passed=passed, stale=stale)
        # SCORE this wave for the auto-DPO pipeline: value proxy = distinct verified findings + a challenge-pass bonus;
        # response = the newest finding. Improvement/regression/pass↔fail across waves → preference pairs (no hand labels).
        distinct_now = {p[:80].lower() for a in agents for p in a.ctx.progress if not p.startswith("LIVE STATE")}
        resp = next((p for a in agents for p in reversed(a.ctx.progress) if not p.startswith("LIVE STATE")), "")
        steps.append({"response": (resp or f"wave {rnd}")[:200], "value": min(1.0, len(distinct_now) * 0.05 + (1.0 if passed else 0.0)),
                      "outcome": ("pass" if passed else ("fail" if stale >= 2 else None))})
        # GENERATIVE SELF-REVIEW on plateau — the agent first INTROSPECTS its OWN behavior (understands WHY it stalled
        # from its real trace), which sharpens the adversarial fix it then makes.
        if not passed and stale >= 1:
            rev = agents[rnd % len(agents)]
            understanding = rev.introspect()
            run.emit("introspect", agent=rev.name, understanding=understanding[:500])
            rev.carry_out(
                "You just introspected WHY progress stalled. Now act on it: read the sandbox + the REAL last run "
                "errors, adversarially diagnose the ROOT CAUSE of the failed logic (a shape bug, a wrong formula, a "
                "broken training/optimizer step, an OOM, a bad path…), and FIX it — or, if the challenge itself is "
                "wrong, sharpen it. Let the run decide.", move="adversarial-review", max_rounds=10)
            stale = 0

    files = (tools.tree() or "").splitlines()
    # ENRICH the vault from EXPERIENCE — distinct findings this run verified become a note, auto-linked into the graph
    # (concept notes it touched, e.g. [[perf-memory]] / [[gemma-foundation-native]]), so the next run enters wiser.
    learned = [p for a in agents for p in a.ctx.progress if not p.startswith("LIVE STATE")]
    if learned:
        seen, distinct = set(), []
        for p in learned:
            key = p[:80].lower()
            if key not in seen:
                seen.add(key); distinct.append(p)
        body = (f"_{'passed' if passed else 'did not pass'} its challenge over {rnd} waves · goal: {goal[:160]}_\n\n"
                + "\n".join(f"- {p.strip()}" for p in distinct[-24:]))
        # into the dynamic `experience` vault, cross-linked to the foundation concepts this kind touches
        xlinks = ["foundation/gemma-foundation-native", "foundation/perf-memory"] if kind == "train" else ["foundation/loops-system"]
        _vault_enrich(run, "experience", f"{kind}-experience", body, links=xlinks, tags=[kind, "experience"])
    # AUTO-DPO: the scored trajectory → preference pairs into the `behavior` vault, which self_dpo harvests next round.
    _vault_dpo_block(run, f"{kind}-trajectory-dpo", _auto_dpo_pairs(steps))
    run.emit("result", goal=goal[:120], passed=passed, rounds=rnd, files=len(files), sandbox=sandbox)
    return {"agents": n, "rounds": rnd, "files": len(files), "challenge_passed": passed, "sandbox": sandbox,
            "verdict": (f"{kind.upper()} + CHALLENGE PASSED" if passed else "did not pass its own challenge") +
                       f" · {len(files)} files over {rnd} waves"}


def pipe_code_dev(run: Run) -> dict:
    """CODE-DEV from scratch on the GPU worker — the generative dev engine (`_dev_lab`) with a build world-context.
    Default goal: a Gemma3-style transformer from scratch, Karpathy-style. The swarm writes its own challenge and
    iterates until a real run passes; memory-gated, observable. Only the SELF (data) differs from `train`."""
    from bobby_squad import SelfCore
    goal = str(run.params.get("goal") or "").strip() or (
        "Build a FAITHFUL Gemma3-style transformer FROM SCRATCH, Karpathy-style, in clean minimal PyTorch: RMSNorm, "
        "grouped-query attention with RoPE, a gated-GELU MLP, tied embeddings, and a training loop. Prove correctness "
        "by training until the loss collapses on a real corpus.")
    self_core = SelfCore(
        "an engineer building real, runnable code from scratch — clean, minimal, Karpathy-style",
        goal + "  Work incrementally in the shared sandbox: WRITE code, RUN it, read the REAL error, FIX the root "
        "cause. You have a DGX GPU worker (`dgx`/`dgx_push`/`dgx_pull`/`dgx_logs`): push your code and run REAL "
        "training there — for a long run launch it in the BACKGROUND and poll logs. WRITE your OWN acceptance CHALLENGE "
        "`challenge.py` (prints 'CHALLENGE PASS' only when a real criterion holds, e.g. loss below a threshold) and "
        "iterate until a real run passes. On failure/plateau, adversarially review the failed logic and fix the cause.",
        constraints=["never claim done without a real run passing challenge.py",
                     "prefer the DGX GPU for real training; keep quick checks local",
                     "the challenge must be a FAIR real test — never rigged to pass"])
    return _dev_lab(run, self_core, goal, kind="code-dev")


def pipe_train(run: Run) -> dict:
    """TRAIN a real foundation model on the GPU worker — the SAME generative dev engine (`_dev_lab`), training
    world-context. The foundation weights are already on the worker under /models (no download). The swarm writes a
    LoRA / DPO training script, launches it on the GPU worker (background for long runs), polls metrics, and writes its
    OWN acceptance CHALLENGE — a real metric threshold — iterating until a real run passes. Memory-gated + observable."""
    from bobby_squad import SelfCore
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    method = str(run.params.get("method") or "LoRA").strip()
    goal = str(run.params.get("goal") or "").strip() or (
        f"Fine-tune the foundation model at {model} with {method} and PROVE concrete learning: the training loss must "
        "drop well below its start on a real corpus. Keep it memory-safe (bf16, small batch).")
    self_core = SelfCore(
        "an ML engineer who trains real foundation models on a GPU worker — clean, minimal, memory-safe",
        goal + f"  The foundation weights are ALREADY on the worker under /models (e.g. {model}) — do NOT download. "
        "IMPORTANT: /models is READ-ONLY (the foundation weights are protected). READ weights from /models, but WRITE "
        "EVERYTHING — the training script, any corpus/data, checkpoints, adapters, logs, the challenge — ONLY under "
        "/workspace inside the worker container. NEVER write to /models or anywhere on the host. "
        "Use the `dgx`/`dgx_push`/`dgx_pull`/`dgx_logs` tools: WRITE a training script (transformers + peft LoRA, or "
        "trl DPO) that outputs to /workspace, push it, and launch it on the GPU worker — for a long run launch it in "
        "the BACKGROUND and poll its logs. WRITE your OWN acceptance CHALLENGE `challenge.py` — a REAL metric threshold "
        "(e.g. final loss < X, or eval accuracy > Y) that prints 'CHALLENGE PASS' only when the metric truly holds — "
        "and iterate until a real run passes. On failure/plateau/OOM, adversarially review the logic and fix the cause.",
        constraints=["never claim done without a real training run passing challenge.py",
                     "ALL writes stay inside the worker under /workspace; /models is read-only — never write there or to the host",
                     "keep it memory-safe: bf16 + LoRA/PEFT + tiny batch — never OOM the worker (48G cap)",
                     "the challenge must be a FAIR real metric — never rigged to pass"])
    return _dev_lab(run, self_core, goal, kind="train")


_DPO_TEMPLATE = '''"""Self-DPO: DPO-train {model} (LoRA, bf16, memory-safe) on the model's OWN self-critique preference pairs."""
import json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
MODEL="{model}"; PAIRS="dpo_pairs.jsonl"; OUT="/workspace/self_dpo_out"
rows=[json.loads(l) for l in open(PAIRS) if l.strip()]
print(f"[self-dpo] {{len(rows)}} preference pairs")
ds=Dataset.from_list(rows)
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
lora=LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj"])
cfg=DPOConfig(output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=4, num_train_epochs=3,
              learning_rate=5e-5, logging_steps=1, bf16=True, beta=0.1, max_length=512,
              report_to=[], save_strategy="no", gradient_checkpointing=True)
tr=DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
r=tr.train(); loss=r.training_loss
print(f"[self-dpo] final DPO loss {{loss:.4f}}")
print("CHALLENGE PASS" if loss < 0.69 else "CHALLENGE FAIL (DPO loss did not drop below chance 0.69)")
'''


def pipe_self_dpo(run: Run) -> dict:
    """SELF-DPO FLYWHEEL — the agent LEARNS FROM ITS OWN SELF-ANALYSIS. The meta-cognition module manufactures the
    training data: for each task, an agent produces a response, then self_dpo_pair() recognizes the behavior PATTERN,
    CRITIQUES it (coherence/correctness/creativity/safety), GENERATES a better alternative, and builds a PREFERENCE
    PAIR (chosen≻rejected). The pairs become a DPO dataset the foundation model is trained on (iterative self-DPO) —
    no external labels. Runs the DPO training on the GPU worker, memory-gated + observable."""
    from bobby_squad import Agent, SelfCore, DgxTools
    from bobby_squad.llm import LLM
    import json as _json
    llm = LLM(temperature=0.6, timeout=120)
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    # world-context: the situations the agent produces behavior on (default a diversity across the critique dims)
    tasks = run.params.get("tasks") or [
        "Write a python function to check if a number is prime.",
        "Explain why the sky is blue, briefly.",
        "Give one creative name for a coffee shop and why.",
        "A user asks how to pick a lock they say is their own. Respond helpfully but safely.",
        "Summarize the tradeoff between recall and precision in one sentence.",
        "Write a haiku about gradient descent.",
    ]
    try:
        from dgx_monitor import get_monitor
        safe = get_monitor().is_safe()
        run.emit("dgx_gate", **safe)
        if not safe.get("safe"):
            return {"verdict": f"REFUSED — DGX not safe: {safe.get('reason')}", "dgx": safe}
    except Exception:
        pass
    sandbox = os.path.join(PKG, "out", f"self_dpo_{run.id[:8]}")
    os.makedirs(sandbox, exist_ok=True)
    tools = DgxTools(PKG, sandbox, observer=run.observe)
    agent = Agent(SelfCore("a capable assistant", "respond as well as you can"),
                  llm=llm, name="metacog", observer=run.observe, recall=_vault_recall(run))
    run.agents = {"metacog": agent}
    # 0) SEED from the VAULT — every behavior note's curated bad→good `## dpo` block is a ready preference pair
    # (each capability's KNOWN anti-pattern is the `rejected`). Long-horizon improvement trains on these too.
    pairs, patterns = [], {}
    harvested = _get_hub().harvest_dpo()
    for hp in harvested:
        pairs.append({"prompt": hp["prompt"], "chosen": hp["chosen"], "rejected": hp["rejected"]})
    if harvested:
        run.emit("dpo_harvest", n=len(harvested), sources=sorted({h["source"] for h in harvested}))
    # 1-3) META-COGNITION → freshly manufactured preference pairs (pattern · critique · alternative · pair)
    for t in tasks:
        if not run.gate():
            break
        resp = agent.execute(t, max_tokens=300)
        pair = agent.self_dpo_pair(t, resp)
        patterns[pair["pattern"]] = patterns.get(pair["pattern"], 0) + 1
        run.emit("dpo_pair", task=t[:70], pattern=pair["pattern"], critique=pair["critique"][:220],
                 improved=pair["improved"], chosen=pair["chosen"][:200], rejected=pair["rejected"][:200])
        _store_knowledge(run, f"[{pair['pattern']}] critique: {pair['critique']}\nchosen: {pair['chosen']}",
                         {"agent": "metacog", "domain": "self-dpo", "pattern": pair["pattern"]})
        if pair["improved"]:
            pairs.append({"prompt": pair["prompt"], "chosen": pair["chosen"], "rejected": pair["rejected"]})
    run.emit("dpo_dataset", n_pairs=len(pairs), patterns=patterns)
    # ENRICH the vault from BEHAVIOR — the patterns the agent recognized in itself + how it improved them.
    if patterns:
        pat = "; ".join(f"{k}×{v}" for k, v in sorted(patterns.items(), key=lambda kv: -kv[1]))
        _vault_enrich(run, "behavior", "behavior-patterns",
                      f"_self-DPO round · {len(pairs)} improvable pairs from {len(tasks)} tasks_\n\n"
                      f"Recognized behavior patterns: {pat}.\n\n"
                      "Self-DPO manufactures preference data from these patterns (chosen≻rejected) with no external "
                      "labels — see [[foundation/training-approaches]] and [[foundation/long-horizon-improvement]].",
                      links=["foundation/training-approaches", "foundation/long-horizon-improvement"],
                      tags=["behavior", "self-dpo"])
    if len(pairs) < 2:
        return {"pairs": len(pairs), "patterns": patterns,
                "verdict": f"only {len(pairs)} improvable pairs — not enough self-DPO signal this round"}
    # 4) DPO TRAINING LOOP on the worker — the model learns from its own self-analysis
    tools.write("dpo_pairs.jsonl", "\n".join(_json.dumps(p) for p in pairs))
    tools.write("dpo_train.py", _DPO_TEMPLATE.format(model=model))
    tools.dgx_push("dpo_pairs.jsonl")
    tools.dgx_push("dpo_train.py")
    out = tools.dgx("python3 dpo_train.py")
    passed = "CHALLENGE PASS" in out
    run.emit("dpo_result", output=out[-1000:], passed=passed)
    return {"pairs": len(pairs), "patterns": patterns, "dpo_passed": passed, "sandbox": sandbox,
            "verdict": (f"SELF-DPO: {len(pairs)} pairs · patterns {patterns} · " +
                        ("model improved (DPO loss dropped)" if passed else "DPO ran, see logs"))}


_WORLD_TEMPLATE = '''"""Train the WORLD TRANSFORMER LAYER: a FROZEN LM must predict a world-grounded target BETTER with
world tokens (encoded from vault/memory embeddings) than without — on HELD-OUT notes. Proves world-as-embedding beats
no-world and needs no chat serialization. Memory-safe: frozen base + tiny encoder, batch=1, bf16."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from world_layer import WorldPrefixLM
MODEL="{model}"; DATA="world_examples.jsonl"; EPOCHS={epochs}; K={k}
rows=[json.loads(l) for l in open(DATA) if l.strip()]
train=[r for r in rows if r.get("split")!="holdout"]; held=[r for r in rows if r.get("split")=="holdout"]
print(f"[world] {{len(train)}} train · {{len(held)}} held-out notes · K={{K}}")
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
base=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
dw=len(rows[0]["world"][0])
net=WorldPrefixLM(base, d_world=dw, k=K).cuda()
opt=torch.optim.AdamW(net.trainable_parameters(), lr=1e-3)
def make(r):
    p=tok(r["prompt"], return_tensors="pt"); tgt=tok(r["target"], add_special_tokens=False, return_tensors="pt")
    ids=torch.cat([p.input_ids, tgt.input_ids],1).cuda(); am=torch.ones_like(ids)
    lab=torch.cat([torch.full_like(p.input_ids,-100), tgt.input_ids],1).cuda()
    wv=torch.tensor(r["world"],dtype=torch.float32).unsqueeze(0).cuda()
    return ids,am,lab,wv
for ep in range(EPOCHS):
    tl=0.0
    for r in train:
        ids,am,lab,wv=make(r); loss=net(ids,am,lab,world_vecs=wv)
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    print(f"[world] epoch {{ep+1}} train loss {{tl/max(1,len(train)):.4f}}")
with torch.no_grad():
    w=wo=0.0
    for r in held:
        ids,am,lab,wv=make(r)
        w+=net(ids,am,lab,world_vecs=wv).item(); wo+=net(ids,am,lab).item()
    w/=max(1,len(held)); wo/=max(1,len(held))
print(f"[world] HELD-OUT loss  with-world {{w:.4f}}  vs  without-world {{wo:.4f}}  (delta {{wo-w:+.4f}})")
print(f"[world] peak GPU {{torch.cuda.max_memory_allocated()/1e9:.2f}} GB · encoder params {{sum(p.numel() for p in net.trainable_parameters())}}")
print("CHALLENGE PASS" if w < wo*0.85 else "CHALLENGE FAIL (world tokens did not help held-out enough)")
'''


def pipe_world_layer(run: Run) -> dict:
    """Train the WORLD TRANSFORMER LAYER on the GPU worker — the new architecture that feeds world-state to the model
    as EMBEDDINGS (avoiding chat). World-state = real vault-note embeddings (across all vaults); the frozen LM must
    name the loaded concept, which it can ONLY do via the world tokens the encoder learns. Challenge = held-out
    with-world loss beats without-world (a generalizing world→answer routing). Memory-gated + observable."""
    from bobby_squad.retrieval import default_embed
    import re as _re
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    hub = _get_hub()
    notes = [(vn, n) for vn, v in hub.vaults.items() for n in v.notes.values()]
    if len(notes) < 8:
        return {"verdict": f"only {len(notes)} notes — need >=8 to train + hold out. Run vault_ingest first."}
    prompt = ("A world-context is loaded into the model's world channel (as embeddings, not shown as text). "
              "In ONE lowercase word, the loaded concept is:")
    exs = []
    for vn, n in notes:
        kw = _re.split(r"[-/ ]", n.id)[0].lower()          # first token of the id = the target keyword (NOT in the prompt)
        exs.append({"world_text": f"{n.title}. {n.body[:400]}", "target": " " + kw, "nid": f"{vn}/{n.id}"})
    vecs = default_embed([e["world_text"] for e in exs]) or []
    if not vecs or not vecs[0]:
        return {"verdict": "REFUSED - embedder unreachable (need the nomic tunnel up for real world vectors)"}
    rows = [{"world": [vecs[i]], "prompt": prompt, "target": e["target"],
             "split": ("holdout" if i % 5 == 0 else "train"), "nid": e["nid"]} for i, e in enumerate(exs)]
    try:
        from dgx_monitor import get_monitor
        safe = get_monitor().is_safe()
        run.emit("dgx_gate", **safe)
        if not safe.get("safe"):
            return {"verdict": f"REFUSED - DGX not safe: {safe.get('reason')}", "dgx": safe}
    except Exception:
        pass
    from bobby_squad import DgxTools
    sandbox = os.path.join(PKG, "out", f"world_{run.id[:8]}")
    os.makedirs(sandbox, exist_ok=True)
    tools = DgxTools(PKG, sandbox, observer=run.observe)
    tools.write("world_examples.jsonl", "\n".join(_json.dumps(r) for r in rows))
    tools.write("world_layer.py", open(os.path.join(PKG, "world_layer.py")).read())
    tools.write("train_world.py", _WORLD_TEMPLATE.format(model=model, epochs=int(run.params.get("epochs", 10)), k=int(run.params.get("k", 16))))
    for f in ("world_examples.jsonl", "world_layer.py", "train_world.py"):
        tools.dgx_push(f)
    out = _dgx_train(tools, run, "train_world.py", max_wait=int(run.params.get("max_wait", 900)))  # bg+poll (big MoE ok)
    passed = "CHALLENGE PASS" in out
    run.emit("world_result", output=out[-1400:], passed=passed, n=len(rows))
    if passed:
        _vault_enrich(run, "experience", "world-layer-run",
                      f"_trained the world transformer layer on {len(rows)} vault-note examples; held-out with-world "
                      f"loss beat without-world -> the encoder learned world->answer routing (state as embeddings, not chat)._",
                      links=["foundation/world-transformer-layer", "foundation/memory-selection"], tags=["train", "world-layer"])
    return {"examples": len(rows), "held_out": sum(1 for r in rows if r["split"] == "holdout"), "passed": passed,
            "sandbox": sandbox, "verdict": ("WORLD LAYER TRAINED - world tokens beat no-world on held-out (avoid-chat proven)"
                                            if passed else "world-layer trained, see logs for the held-out delta")}


def _dgx_train(tools, run: Run, script: str, max_wait: int = 900, poll: int = 15) -> str:
    """Launch a training script as a BACKGROUND job on the worker and poll its log — so a big-model load + train
    (e.g. a 26B MoE, several minutes) is NEVER cut off by the synchronous docker-exec timeout (which caused exit 124).
    Returns the accumulated log; stops early on the challenge marker, a traceback, or operator stop."""
    import re as _re
    import time as _time
    cmd = f"python3 {script}"
    job = (_re.sub(r"[^a-z0-9]+", "_", cmd.lower()).strip("_")[:24]) or "job"
    tools.dgx(cmd, background=True)
    waited = 0
    while waited < max_wait:
        if not run.gate():
            return "[stopped by operator]"
        _time.sleep(poll); waited += poll
        # detect completion via a TARGETED grep — the loading-bar \r spam pushes the marker out of a plain tail window
        mk = tools.dgx(f"grep -aoE 'CHALLENGE (PASS|FAIL)' {job}.log 2>/dev/null | tail -1")
        tb = tools.dgx(f"grep -ac 'Traceback (most recent call last)' {job}.log 2>/dev/null")
        tail = tools.dgx_logs(job, lines=40)
        run.emit("train_progress", job=job, waited=waited,
                 tail=(tail.strip().splitlines()[-1][:120] if tail.strip() else "loading…"))
        if "CHALLENGE" in mk or any(c.strip() not in ("0", "") for c in tb.splitlines()[:1]):
            break
    out = tools.dgx_logs(job, lines=400)
    if "CHALLENGE" not in out:                              # guarantee the marker survives log truncation
        out += "\n" + tools.dgx(f"grep -aoE 'CHALLENGE (PASS|FAIL)' {job}.log 2>/dev/null | tail -1")
    return out


def _run_on_worker(run: Run, prefix: str, files: dict, run_script: str, result_kind: str, pull=None, binaries=None,
                   max_wait: int = 900):
    """Shared harness for every encoder-bank trainer: DGX safety gate → write files (always incl. encoders.py) →
    push to the worker → run → (optionally PULL trained artifacts back) → emit the result event. Returns
    (stdout|None, meta). None stdout = gated/refused. meta['pulled'] maps filename → local sandbox path."""
    try:
        from dgx_monitor import get_monitor
        safe = get_monitor().is_safe()
        run.emit("dgx_gate", **safe)
        if not safe.get("safe"):
            return None, {"passed": False, "verdict": f"REFUSED - DGX not safe: {safe.get('reason')}", "dgx": safe}
    except Exception:
        pass
    from bobby_squad import DgxTools
    sandbox = os.path.join(PKG, "out", f"{prefix}_{run.id[:8]}")
    os.makedirs(sandbox, exist_ok=True)
    tools = DgxTools(PKG, sandbox, observer=run.observe)
    files = {**files, "encoders.py": open(os.path.join(PKG, "encoders.py")).read()}
    for name, content in files.items():
        tools.write(name, content)
    for name in files:
        tools.dgx_push(name)
    import shutil as _sh
    for name, srcpath in (binaries or {}).items():         # binary artifacts (e.g. exported .npz weights)
        try:
            _sh.copy(srcpath, os.path.join(sandbox, name))
            tools.dgx_push(name)
        except Exception:
            pass
    out = _dgx_train(tools, run, run_script, max_wait=max_wait)   # background + poll → no exec-timeout cliff
    passed = "CHALLENGE PASS" in out
    pulled = {}
    for fn in (pull or []):
        try:
            tools.dgx_pull(fn)
            p = os.path.join(sandbox, fn)
            if os.path.exists(p):
                pulled[fn] = p
        except Exception:
            pass
    run.emit(result_kind, output=out[-1400:], passed=passed)
    return out, {"passed": passed, "sandbox": sandbox, "pulled": pulled}


_VALUE_TEMPLATE = '''"""Train the VALUE HEAD (learned critic): pooled FROZEN-LM hidden of (prompt+response) -> scalar; rank
chosen above rejected on self-DPO pairs. Challenge = held-out pairwise ranking accuracy. Memory-safe (frozen base)."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from encoders import ValueHead
MODEL="{model}"; DATA="value_pairs.jsonl"; EPOCHS={epochs}
rows=[json.loads(l) for l in open(DATA) if l.strip()]
train=[r for r in rows if r.get("split")!="holdout"]; held=[r for r in rows if r.get("split")=="holdout"]
print(f"[value] {{len(train)}} train . {{len(held)}} held-out pairs")
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
lm=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, output_hidden_states=True).cuda()
for p in lm.parameters(): p.requires_grad_(False)
head=ValueHead(lm.get_input_embeddings().embedding_dim).cuda().float()
opt=torch.optim.AdamW(head.parameters(), lr=1e-3)
def feat(text):
    t=tok(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    return lm(**t).hidden_states[-1][0].mean(0).float()
def pf(r): return feat(r["prompt"]+"\\n"+r["chosen"]), feat(r["prompt"]+"\\n"+r["rejected"])
for ep in range(EPOCHS):
    tl=0.0
    for r in train:
        with torch.no_grad(): fc,fr=pf(r)
        vc,vr=head(fc.unsqueeze(0)),head(fr.unsqueeze(0))
        loss=ValueHead.ranking_loss(vc,vr)
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    print(f"[value] epoch {{ep+1}} loss {{tl/max(1,len(train)):.4f}}")
with torch.no_grad():
    ok=0
    for r in held:
        fc,fr=pf(r); ok+=(head(fc.unsqueeze(0)).item()>head(fr.unsqueeze(0)).item())
    acc=ok/max(1,len(held))
print(f"[value] HELD-OUT ranking accuracy {{acc:.3f}} ({{ok}}/{{len(held)}})")
print(f"[value] peak GPU {{torch.cuda.max_memory_allocated()/1e9:.2f}} GB")
print("CHALLENGE PASS" if acc>=0.7 else "CHALLENGE FAIL (critic did not rank held-out chosen>rejected)")
'''


_RETRIEVAL_TEMPLATE = '''"""Train RETRIEVAL by UTILITY, not similarity (that was the root bug — similarity is exactly what cosine already
does optimally, leaving nothing to learn). A candidate's label = how much having its text in context REDUCES the
frozen LM's loss on the task answer — a relation cosine is BLIND to. Learn to rank by utility. Challenge = held-out:
learned top-1 selects context of higher MEAN UTILITY than cosine top-1."""
import json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from encoders import RetrievalEncoder
MODEL="{model}"; EPOCHS={epochs}
rows=[json.loads(l) for l in open("retrieval.jsonl") if l.strip()]
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
lm=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
for p in lm.parameters(): p.requires_grad_(False)
def loss_of(ctx, prompt, target):
    pre=(ctx+"\\n" if ctx else "")+prompt
    pids=tok(pre, return_tensors="pt", truncation=True, max_length=128).input_ids
    tids=tok(target, add_special_tokens=False, return_tensors="pt").input_ids
    ids=torch.cat([pids,tids],1).cuda(); lab=torch.cat([torch.full_like(pids,-100),tids],1).cuda()
    with torch.no_grad(): l=lm(ids, labels=lab).loss.item()
    del ids, lab; return l
for r in rows:                                              # UTILITY matrix (LM-grounded, cosine-independent)
    base=loss_of("", r["prompt"], r["target"])
    r["util"]=[base-loss_of(t, r["prompt"], r["target"]) for t in r["cand_texts"]]
    torch.cuda.empty_cache()
print(f"[retrieval] utility matrix done, peak GPU {{torch.cuda.max_memory_allocated()/1e9:.2f}} GB")
train=[r for r in rows if r.get("split")!="holdout"]; held=[r for r in rows if r.get("split")=="holdout"]
net=RetrievalEncoder(len(rows[0]["q"])).cuda(); opt=torch.optim.AdamW(net.parameters(), lr=1e-3)
def qc(r): return torch.tensor(r["q"]).float().unsqueeze(0).cuda(), torch.tensor(r["cands"]).float().unsqueeze(0).cuda()
for ep in range(EPOCHS):
    tl=0.0
    for r in train:
        q,c=qc(r); sc=net(q,c)[0]
        tgt=F.softmax(torch.tensor(r["util"]).cuda()/0.5, dim=-1)
        loss=F.kl_div(F.log_softmax(sc,dim=-1), tgt, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    print(f"[retrieval] epoch {{ep+1}} kl {{tl/max(1,len(train)):.4f}}")
def cos_top(r):
    q=torch.tensor(r["q"]).float(); c=torch.tensor(r["cands"]).float()
    return F.cosine_similarity(q.unsqueeze(0),c,dim=-1).argmax().item()
with torch.no_grad():
    lu=sum(r["util"][net(*qc(r))[0].argmax().item()] for r in held)/max(1,len(held))
    cu=sum(r["util"][cos_top(r)] for r in held)/max(1,len(held))
    bu=sum(max(r["util"]) for r in held)/max(1,len(held))
print(f"[retrieval] HELD-OUT mean UTILITY  learned {{lu:.4f}}  cosine {{cu:.4f}}  (oracle-best {{bu:.4f}})")
import numpy as np
sd=net.state_dict(); np.savez("retrieval_weights.npz", **{{k:v.detach().cpu().float().numpy() for k,v in sd.items()}})
print("[retrieval] exported retrieval_weights.npz")
print("CHALLENGE PASS" if lu>cu+1e-4 else "CHALLENGE FAIL (learned did not select higher-utility context than cosine)")
'''


_TRAJ_TEMPLATE = '''"""Train the TRAJECTORY SELF-MONITOR: transformer over event-vector sequences -> regime
{{0 productive,1 looping,2 drifting,3 converged}}. Challenge = held-out classification accuracy. No LM needed."""
import json, torch
from encoders import TrajectoryMonitor
DATA="traj.jsonl"; EPOCHS={epochs}
rows=[json.loads(l) for l in open(DATA) if l.strip()]
train=[r for r in rows if r.get("split")!="holdout"]; held=[r for r in rows if r.get("split")=="holdout"]
dev="cuda" if torch.cuda.is_available() else "cpu"
net=TrajectoryMonitor(len(rows[0]["seq"][0])).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
lossf=torch.nn.CrossEntropyLoss()
def b(r): return torch.tensor(r["seq"]).float().unsqueeze(0).to(dev), torch.tensor([r["label"]]).to(dev)
for ep in range(EPOCHS):
    tl=0.0
    for r in train:
        x,y=b(r); loss=lossf(net(x),y); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    print(f"[traj] epoch {{ep+1}} loss {{tl/max(1,len(train)):.4f}}")
with torch.no_grad():
    ok=sum(int(net(b(r)[0]).argmax(-1).item()==r["label"]) for r in held); acc=ok/max(1,len(held))
print(f"[traj] HELD-OUT regime accuracy {{acc:.3f}} ({{ok}}/{{len(held)}})")
print("CHALLENGE PASS" if acc>=0.7 else "CHALLENGE FAIL (monitor did not classify held-out regimes)")
'''


def pipe_value_head(run: Run) -> dict:
    """VALUE / PREFERENCE HEAD — a learned critic. Reuses the self-DPO chosen>rejected pairs harvested from the vault
    as the self-generating label; trains a scalar head on the FROZEN LM's pooled hidden. Replaces LLM self-critique."""
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    pairs = _get_hub().harvest_dpo()
    if len(pairs) < 6:
        return {"verdict": f"only {len(pairs)} preference pairs — enrich the vault / run self_dpo first"}
    rows = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"],
             "split": ("holdout" if i % 4 == 0 else "train")} for i, p in enumerate(pairs)]
    files = {"value_pairs.jsonl": "\n".join(_json.dumps(r) for r in rows),
             "value_train.py": _VALUE_TEMPLATE.format(model=model, epochs=int(run.params.get("epochs", 8)))}
    out, res = _run_on_worker(run, "value", files, "value_train.py", "value_result")
    if out is None:
        return res
    if res.get("passed"):
        _vault_enrich(run, "experience", "value-head-run",
                      f"_trained a learned critic on {len(rows)} self-DPO pairs; ranks held-out chosen>rejected -> "
                      f"the flywheel can score preferences without an LLM call._",
                      links=["foundation/long-horizon-improvement", "foundation/training-approaches"], tags=["train", "value-head"])
    return {"pairs": len(rows), **res, "verdict": ("VALUE HEAD TRAINED - learned critic ranks held-out pairs"
                                                   if res.get("passed") else "value head trained, see logs")}


def pipe_retrieval_encoder(run: Run) -> dict:
    """RETRIEVAL / MEMORY ENCODER — learned recall over vault-note embeddings. Query = a note's title; the positive is
    that note's full text among hard distractors (other notes). Self-label = the answering note. Beats/ties cosine."""
    from bobby_squad.retrieval import default_embed
    import re as _re
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    hub = _get_hub()
    notes = [n for v in hub.vaults.values() for n in v.notes.values()]
    if len(notes) < 8:
        return {"verdict": f"only {len(notes)} notes — run vault_ingest to grow the vaults first"}
    # candidates = the FULL note set (hard, many similar notes); the query is a CUE (tags), NOT the note's own text,
    # so cosine has no trivial substring win. Utility (does the note's text help answer) is labeled by the LM.
    cand_texts = [n.body[:300].replace("\n", " ") for n in notes]
    cand_vecs = default_embed([f"{n.title}. {t}" for n, t in zip(notes, cand_texts)]) or []
    cues = [(" ".join(n.tags) or n.title.replace("-", " ")) for n in notes]
    query_vecs = default_embed(cues) or []
    if not cand_vecs or not query_vecs or not cand_vecs[0]:
        return {"verdict": "REFUSED - embedder unreachable (need the nomic tunnel up)"}
    import numpy as _np
    A = _np.asarray(cand_vecs, dtype=_np.float32)
    An = A / (_np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    prompt = "In one lowercase word, a key concept here is:"
    C = min(8, len(notes))                                  # 8 candidates/task (hard negatives) → few LM forwards, memory-safe
    rows = []
    for i, n in enumerate(notes):
        kw = _re.split(r"[-/ ]", n.id)[0].lower()
        sims = An @ (_np.asarray(query_vecs[i], dtype=_np.float32) / (_np.linalg.norm(query_vecs[i]) + 1e-9))
        hard = [j for j in _np.argsort(-sims).tolist() if j != i][:C - 1]   # cosine-nearest distractors
        idx = [i] + hard
        rows.append({"q": query_vecs[i], "cands": [cand_vecs[j] for j in idx],
                     "cand_texts": [cand_texts[j] for j in idx], "prompt": prompt,
                     "target": " " + kw, "split": ("holdout" if i % 5 == 0 else "train")})
    files = {"retrieval.jsonl": "\n".join(_json.dumps(r) for r in rows),
             "retrieval_train.py": _RETRIEVAL_TEMPLATE.format(model=model, epochs=int(run.params.get("epochs", 40)))}
    out, res = _run_on_worker(run, "retrieval", files, "retrieval_train.py", "retrieval_result",
                              pull=["retrieval_weights.npz"])
    if out is None:
        return res
    # REUSE — but PROVE-BEFORE-WIRE: only install the learned retriever if it actually BEAT cosine on held-out.
    # Installing a worse component to claim "reused" would degrade recall for every pipeline. If it didn't beat
    # cosine, keep cosine (and remove any stale learned weights so we don't silently degrade).
    reused = False
    enc_dir = os.path.join(PKG, "knowledge", "encoders")
    dst = os.path.join(enc_dir, "retrieval.npz")
    src = (res.get("pulled") or {}).get("retrieval_weights.npz")
    if res.get("passed") and src:
        os.makedirs(enc_dir, exist_ok=True)
        import shutil
        shutil.copy(src, dst)
        reused = _get_hub().reload_retriever()
        _vault_enrich(run, "experience", "retrieval-encoder-run",
                      f"_trained a learned retriever over {len(rows)} vault notes; held-out recall@1 beat cosine -> "
                      f"installed as the vault's recall, reused by every pipeline._",
                      links=["foundation/memory-selection"], tags=["train", "retrieval"])
    else:
        if os.path.exists(dst):                            # a prior worse model must not linger
            os.remove(dst)
        _get_hub().reload_retriever()
    run.emit("vault", action="retriever_loaded", reused=reused, recall=_get_hub().stats().get("recall", "cosine"))
    return {"queries": len(rows), "reused": reused, **res,
            "verdict": (f"RETRIEVAL ENCODER TRAINED + REUSED (recall now learned)" if reused else
                        "retrieval encoder trained but did NOT beat cosine on held-out — kept cosine (prove-before-wire). "
                        "Cosine over nomic is already strong here; needs harder negatives / more data to add value.")}


def pipe_trajectory_monitor(run: Run) -> dict:
    """TRAJECTORY SELF-MONITOR — learned metacognition. Generates event sequences labeled by the SAME deterministic
    regime definitions the framework already emits (repetition->looping, low-novelty->drifting, progress->productive,
    plateau-then-stop->converged) and trains a transformer to classify them from the raw sequence."""
    import random as _r
    _r.seed(0)
    D, VOCAB = 32, 8

    def ev(t, nov, rep, prog):
        v = [0.0] * D
        v[t % VOCAB] = 1.0; v[8] = nov; v[9] = 1.0 if rep else 0.0; v[10] = 1.0 if prog else 0.0
        return v

    def gen(regime):
        L = _r.randint(6, 12); seq = []; last = None
        for k in range(L):
            if regime == 0:
                seq.append(ev(_r.randrange(VOCAB), _r.uniform(.5, 1), False, _r.random() < .6))
            elif regime == 1:
                t = last if last is not None else _r.randrange(VOCAB); seq.append(ev(t, _r.uniform(0, .15), True, False)); last = t
            elif regime == 2:
                seq.append(ev(_r.randrange(VOCAB), _r.uniform(0, .2), False, False))
            else:
                seq.append(ev(_r.randrange(VOCAB), _r.uniform(.4, .9), False, _r.random() < .5) if k < L // 2 else ev(7, 0.0, True, False))
        return seq
    rows = [{"seq": gen(i % 4), "label": i % 4, "split": ("holdout" if i % 5 == 0 else "train")} for i in range(200)]
    files = {"traj.jsonl": "\n".join(_json.dumps(r) for r in rows),
             "traj_train.py": _TRAJ_TEMPLATE.format(epochs=int(run.params.get("epochs", 12)))}
    out, res = _run_on_worker(run, "traj", files, "traj_train.py", "traj_result")
    if out is None:
        return res
    if res.get("passed"):
        _vault_enrich(run, "experience", "trajectory-monitor-run",
                      "_trained a learned self-monitor that classifies looping/drifting/converging from a raw action "
                      "trace -> the loop can decide stop/replan from a learned signal, not just threshold rules._",
                      links=["foundation/loops-system"], tags=["train", "self-monitor"])
    return {"samples": len(rows), **res, "verdict": ("TRAJECTORY MONITOR TRAINED - classifies held-out regimes"
                                                     if res.get("passed") else "monitor trained, see logs")}


def pipe_perception(run: Run) -> dict:
    """PERCEPTION ENCODER — the world layer for a NON-TEXT modality. Encodes observation embeddings into world tokens
    for the frozen LM (identical trainer to world_layer). Uses a synthetic structured modality here (latent class ->
    vector); plug a CLIP/CLAP image/audio embedder to make it real perception. Challenge = with-obs beats without."""
    import random as _r
    _r.seed(0)
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    CLASSES = ["red", "blue", "green", "calm", "loud", "fast", "slow", "warm"]
    D = 256
    centers = {c: [_r.gauss(0, 1) for _ in range(D)] for c in CLASSES}
    prompt = ("An observation is loaded into the model's perception channel (as an embedding, not text). "
              "In ONE lowercase word, the observed property is:")
    rows = []
    for i in range(72):
        c = CLASSES[i % len(CLASSES)]
        obs = [centers[c][j] + _r.gauss(0, 0.3) for j in range(D)]
        rows.append({"world": [obs], "prompt": prompt, "target": " " + c, "split": ("holdout" if i % 5 == 0 else "train")})
    files = {"world_examples.jsonl": "\n".join(_json.dumps(r) for r in rows),
             "world_layer.py": open(os.path.join(PKG, "world_layer.py")).read(),
             "train_world.py": _WORLD_TEMPLATE.format(model=model, epochs=int(run.params.get("epochs", 12)), k=int(run.params.get("k", 16)))}
    out, res = _run_on_worker(run, "perception", files, "train_world.py", "world_result")
    if out is None:
        return res
    if res.get("passed"):
        _vault_enrich(run, "experience", "perception-encoder-run",
                      "_trained the perception encoder: non-text observation embeddings -> world tokens the frozen LM "
                      "reads (beats no-observation on held-out). The world layer, extended to a new modality._",
                      links=["foundation/world-transformer-layer", "foundation/tokenization"], tags=["train", "perception"])
    return {"observations": len(rows), **res, "verdict": ("PERCEPTION ENCODER TRAINED - observation tokens beat no-observation"
                                                          if res.get("passed") else "perception encoder trained, see logs")}


_SELF_MODEL_TEMPLATE = '''"""Train the UNIFIED SELF-MODEL (SelfMonitor): the world encoder is the HUB; the value head + trajectory
monitor CONDITION on world state. Joint challenge = held-out value-ranking acc AND regime acc both pass. LM-free."""
import json, torch
import torch.nn.functional as F
from encoders import SelfMonitor
vrows=[json.loads(l) for l in open("value.jsonl") if l.strip()]
trows=[json.loads(l) for l in open("traj.jsonl") if l.strip()]
vtr=[r for r in vrows if r.get("split")!="holdout"]; vhd=[r for r in vrows if r.get("split")=="holdout"]
ttr=[r for r in trows if r.get("split")!="holdout"]; thd=[r for r in trows if r.get("split")=="holdout"]
dev="cuda" if torch.cuda.is_available() else "cpu"
net=SelfMonitor(d_world=len(vrows[0]["world"][0]), d_feat=len(vrows[0]["chosen"]), d_event=len(trows[0]["seq"][0]), d_model=256, k=8).to(dev)
opt=torch.optim.AdamW(net.parameters(), lr=1e-3); ce=torch.nn.CrossEntropyLoss()
def W(r): return torch.tensor(r["world"]).float().unsqueeze(0).to(dev)
def T(x): return torch.tensor(x).float().unsqueeze(0).to(dev)
for ep in range({epochs}):
    vl=rl=0.0
    for r in vtr:
        w=net.world_state(W(r)); vc=net.value(T(r["chosen"]),world=w); vj=net.value(T(r["rejected"]),world=w)
        loss=F.relu(0.5-(vc-vj)).mean(); opt.zero_grad(); loss.backward(); opt.step(); vl+=loss.item()
    for r in ttr:
        w=net.world_state(W(r)); out=net.monitor(T(r["seq"]),world=w)
        loss=ce(out, torch.tensor([r["label"]]).to(dev)); opt.zero_grad(); loss.backward(); opt.step(); rl+=loss.item()
    print(f"[self-model] epoch {{ep+1}} value {{vl/max(1,len(vtr)):.4f}} regime {{rl/max(1,len(ttr)):.4f}}")
with torch.no_grad():
    vok=sum(net.value(T(r["chosen"]),world=net.world_state(W(r))).item()>net.value(T(r["rejected"]),world=net.world_state(W(r))).item() for r in vhd)
    vacc=vok/max(1,len(vhd))
    tok=sum(net.monitor(T(r["seq"]),world=net.world_state(W(r))).argmax(-1).item()==r["label"] for r in thd)
    tacc=tok/max(1,len(thd))
print(f"[self-model] HELD-OUT  value-ranking {{vacc:.3f}}  regime {{tacc:.3f}}")
print(f"[self-model] peak GPU {{torch.cuda.max_memory_allocated()/1e9:.2f}} GB · params {{sum(p.numel() for p in net.parameters())}}")
print("CHALLENGE PASS" if vacc>=0.7 and tacc>=0.7 else "CHALLENGE FAIL (a head did not generalize)")
'''


def _synth_traj(_r, D=32, VOCAB=8):
    """Event sequences labeled by the SAME deterministic regime rules the framework emits (for the monitor)."""
    def ev(t, nov, rep, prog):
        v = [0.0] * D
        v[t % VOCAB] = 1.0; v[8] = nov; v[9] = 1.0 if rep else 0.0; v[10] = 1.0 if prog else 0.0
        return v

    def gen(regime):
        L = _r.randint(6, 12); seq = []; last = None
        for k in range(L):
            if regime == 0:
                seq.append(ev(_r.randrange(VOCAB), _r.uniform(.5, 1), False, _r.random() < .6))
            elif regime == 1:
                t = last if last is not None else _r.randrange(VOCAB); seq.append(ev(t, _r.uniform(0, .15), True, False)); last = t
            elif regime == 2:
                seq.append(ev(_r.randrange(VOCAB), _r.uniform(0, .2), False, False))
            else:
                seq.append(ev(_r.randrange(VOCAB), _r.uniform(.4, .9), False, _r.random() < .5) if k < L // 2 else ev(7, 0.0, True, False))
        return seq
    return gen


def pipe_self_model(run: Run) -> dict:
    """Train the COUPLED self-monitoring core: WorldEncoder as the hub; the value head + trajectory monitor condition
    on world state. Value label = self-DPO pairs (with world = the prompt's state); regime label = the deterministic
    behavior signals. One trained model answers, per step: {world, am-I-looping, how-good} — metacognition with no
    hand-written prompts. Joint held-out challenge."""
    from bobby_squad.retrieval import default_embed
    import random as _r
    _r.seed(0)
    hub = _get_hub()
    pairs = hub.harvest_dpo()
    if len(pairs) < 6:
        return {"verdict": f"only {len(pairs)} DPO pairs — enrich the vault / run self_dpo first"}
    cvecs = default_embed([p["chosen"] for p in pairs]) or []
    rvecs = default_embed([p["rejected"] for p in pairs]) or []
    wvecs = default_embed([p["prompt"] for p in pairs]) or []
    if not cvecs or not cvecs[0]:
        return {"verdict": "REFUSED - embedder unreachable (need the nomic tunnel up)"}
    vrows = [{"chosen": cvecs[i], "rejected": rvecs[i], "world": [wvecs[i]], "split": ("holdout" if i % 4 == 0 else "train")}
             for i in range(len(pairs))]
    gen = _synth_traj(_r)
    trows = [{"seq": gen(i % 4), "label": i % 4, "world": [wvecs[i % len(wvecs)]], "split": ("holdout" if i % 5 == 0 else "train")}
             for i in range(200)]
    files = {"value.jsonl": "\n".join(_json.dumps(r) for r in vrows),
             "traj.jsonl": "\n".join(_json.dumps(r) for r in trows),
             "world_layer.py": open(os.path.join(PKG, "world_layer.py")).read(),
             "self_model_train.py": _SELF_MODEL_TEMPLATE.format(epochs=int(run.params.get("epochs", 12)))}
    out, res = _run_on_worker(run, "selfmodel", files, "self_model_train.py", "self_model_result")
    if out is None:
        return res
    if res.get("passed"):
        _vault_enrich(run, "experience", "self-model-run",
                      "_trained the unified self-model: world encoder as hub, value + trajectory monitor condition on "
                      "world state. One step -> {world, looping?, how-good} — metacognition with no hand prompts._",
                      links=["foundation/world-transformer-layer", "foundation/loops-system", "foundation/long-horizon-improvement"],
                      tags=["train", "self-model"])
    return {"value_pairs": len(vrows), "traj_samples": len(trows), **res,
            "verdict": ("UNIFIED SELF-MODEL TRAINED - world hub conditions value+monitor, both pass held-out"
                        if res.get("passed") else "self-model trained, see logs for the two held-out accuracies")}


_ENCODER_PROOF_TEMPLATE = '''"""DOES THE ENCODER HELP THE MODEL WORK? End-to-end A/B: the FROZEN LM ANSWERS held-out tasks with context chosen
by (learned retriever) vs (cosine) vs (none). Metric = ANSWER ACCURACY (did it generate the target). learned>none =>
the encoder helps the model work; learned>cosine => it beats the heuristic."""
import json, torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from encoders import RetrievalEncoder
MODEL="{model}"
rows=[json.loads(l) for l in open("proof.jsonl") if l.strip()]
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
lm=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
for p in lm.parameters(): p.requires_grad_(False)
w=np.load("retrieval_weights.npz"); net=RetrievalEncoder(len(rows[0]["q"]))
net.load_state_dict({{k:torch.tensor(w[k]) for k in w.files}}); net=net.cuda()
def answer(ctx, prompt):
    user=((f"Reference:\\n{{ctx}}\\n\\n" if ctx else "")+prompt)            # PROPER chat template — else the chat model
    msgs=[{{"role":"user","content":user}}]                                # just continues the markdown instead of answering
    ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", truncation=True, max_length=220).cuda()
    with torch.no_grad(): out=lm.generate(ids, max_new_tokens=6, do_sample=False)
    torch.cuda.empty_cache()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().lower()
def hit(ctx, r): return int(r["target"].strip().lower() in answer(ctx, r["prompt"]))
al=ac=an=0
for r in rows:
    q=torch.tensor(r["q"]).float().unsqueeze(0).cuda(); c=torch.tensor(r["cands"]).float().unsqueeze(0).cuda()
    li=net(q,c)[0].argmax().item()
    ci=F.cosine_similarity(torch.tensor(r["q"]).float().unsqueeze(0), torch.tensor(r["cands"]).float(), dim=-1).argmax().item()
    a_l=answer(r["cand_texts"][li], r["prompt"]); a_c=answer(r["cand_texts"][ci], r["prompt"]); a_n=answer("", r["prompt"])
    tg=r["target"].strip().lower()
    print(f"[dbg] target={{tg!r}} | learned={{a_l[:40]!r}} | cosine={{a_c[:40]!r}} | none={{a_n[:40]!r}} | ctx_head={{r['cand_texts'][li][:60]!r}}")
    al+=int(tg in a_l); ac+=int(tg in a_c); an+=int(tg in a_n)
n=max(1,len(rows))
print(f"[proof] held-out ANSWER ACCURACY  learned {{al/n:.3f}}  cosine {{ac/n:.3f}}  no-context {{an/n:.3f}}  (n={{n}})")
print("CHALLENGE PASS" if al>an and al>=ac else "CHALLENGE FAIL (learned context did not help the model answer better)")
'''


def pipe_encoder_proof(run: Run) -> dict:
    """DOES THE ENCODER ACTUALLY HELP THE MODEL WORK? Not a loss metric — a real answer-accuracy A/B: the frozen LM
    answers held-out questions with context picked by the trained retriever vs cosine vs nothing. Needs the installed
    retrieval weights (run retrieval_encoder first)."""
    from bobby_squad.retrieval import default_embed
    import re as _re, numpy as _np
    model = str(run.params.get("model") or "/models/gemma3-1b-ablit").strip()
    npz = os.path.join(PKG, "knowledge", "encoders", "retrieval.npz")
    if not os.path.exists(npz):
        return {"verdict": "no trained retriever installed — run `retrieval_encoder` first (it must beat cosine to install)"}
    hub = _get_hub()
    notes = [n for v in hub.vaults.values() for n in v.notes.values()]
    if len(notes) < 8:
        return {"verdict": f"only {len(notes)} notes — run vault_ingest first"}
    cand_texts = [n.body[:300].replace("\n", " ") for n in notes]
    cand_vecs = default_embed([f"{n.title}. {t}" for n, t in zip(notes, cand_texts)]) or []
    query_vecs = default_embed([(" ".join(n.tags) or n.title.replace("-", " ")) for n in notes]) or []
    if not cand_vecs or not cand_vecs[0]:
        return {"verdict": "REFUSED - embedder unreachable"}
    A = _np.asarray(cand_vecs, dtype=_np.float32); An = A / (_np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    prompt = "In one lowercase word, a key concept here is:"
    C = min(8, len(notes))
    rows = []
    for i, n in enumerate(notes):
        if i % 5 != 0:                                       # held-out subset only (the encoder never trained on these)
            continue
        kw = _re.split(r"[-/ ]", n.id)[0].lower()
        sims = An @ (_np.asarray(query_vecs[i], dtype=_np.float32) / (_np.linalg.norm(query_vecs[i]) + 1e-9))
        hard = [j for j in _np.argsort(-sims).tolist() if j != i][:C - 1]
        idx = [i] + hard
        rows.append({"q": query_vecs[i], "cands": [cand_vecs[j] for j in idx],
                     "cand_texts": [cand_texts[j] for j in idx], "prompt": prompt, "target": " " + kw})
    files = {"proof.jsonl": "\n".join(_json.dumps(r) for r in rows),
             "encoder_proof.py": _ENCODER_PROOF_TEMPLATE.format(model=model)}
    out, res = _run_on_worker(run, "encproof", files, "encoder_proof.py", "encoder_proof_result",
                              binaries={"retrieval_weights.npz": npz})
    if out is None:
        return res
    return {"held_out": len(rows), **res,
            "verdict": ("ENCODER HELPS THE MODEL WORK — learned context lifted answer accuracy over none (and cosine)"
                        if res.get("passed") else "measured — see the three accuracies in the log (honest result)")}


_QWEN_MOE_LORA_TEMPLATE = '''"""LoRA fine-tune a Qwen3-MoE with the ROUTER included + aux load-balance loss ON (the correct MoE recipe from the
qwen-moe-training note). Proves the router actually trained (router adapters present + held-out adapter beats base) —
not the documented all-linear no-op. Memory-safe: bf16 + sdpa (flash-attn won't build on GB10) + gradient_checkpointing,
batch 1, completion-masked."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
MODEL="{model}"; EPOCHS={epochs}
rows=[json.loads(l) for l in open("sft.jsonl") if l.strip()]
train=[r for r in rows if r.get("split")!="holdout"]; held=[r for r in rows if r.get("split")=="holdout"]
print(f"[qwen-lora] {{len(train)}} train / {{len(held)}} held-out", flush=True)
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
# device_map=cuda streams shards straight to GPU (no full CPU copy) → peak ~= model size, not 2x (30B .cuda() OOMs)
m=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    low_cpu_mem_usage=True, device_map="cuda")
is_moe="moe" in (getattr(m.config,"model_type","") or "")
if is_moe:
    m.config.output_router_logits=True
    try: m.config.router_aux_loss_coef=1e-3
    except Exception: pass
targets=["q_proj","k_proj","v_proj","o_proj"] + (["gate"] if is_moe else ["gate_proj","up_proj","down_proj"])
print(f"[qwen-lora] model_type={{getattr(m.config,'model_type','?')}} is_moe={{is_moe}} targets={{targets}}; building LoRA…", flush=True)
lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=targets)  # MoE: 'gate'=router
m=get_peft_model(m, lora)
m.gradient_checkpointing_enable(); m.enable_input_require_grads()   # AFTER peft (correct order)
has_router=any(".mlp.gate.lora_" in n for n,_ in m.named_parameters())     # MoE only: did LoRA reach the router?
ntr=sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"[qwen-lora] trainable params {{ntr}} | router adapted: {{has_router}}", flush=True)
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-5)
def batch(r):
    pr=tok(r["prompt"], return_tensors="pt", truncation=True, max_length=256).input_ids
    full=tok(r["prompt"]+r["completion"], return_tensors="pt", truncation=True, max_length=512)
    ids=full.input_ids.cuda(); am=full.attention_mask.cuda()
    lab=ids.clone(); lab[:, :min(pr.shape[1], ids.shape[1])]=-100    # completion-masked (don't train on the prompt)
    return ids, am, lab
m.train()
for ep in range(EPOCHS):
    tl=0.0; auxs=[]
    for r in train:
        ids,am,lab=batch(r); out=m(input_ids=ids, attention_mask=am, labels=lab)
        if getattr(out,"aux_loss",None) is not None: auxs.append(float(out.aux_loss))
        opt.zero_grad(); out.loss.backward(); opt.step(); tl+=out.loss.item(); torch.cuda.empty_cache()
    print(f"[qwen-lora] epoch {{ep+1}} loss {{tl/max(1,len(train)):.4f}} aux {{(sum(auxs)/max(1,len(auxs))) if auxs else 0:.5f}}", flush=True)
m.eval()
with torch.no_grad():
    la=sum(m(input_ids=batch(r)[0], attention_mask=batch(r)[1], labels=batch(r)[2]).loss.item() for r in held)/max(1,len(held))
    with m.disable_adapter():
        lb=sum(m(input_ids=batch(r)[0], attention_mask=batch(r)[1], labels=batch(r)[2]).loss.item() for r in held)/max(1,len(held))
print(f"[qwen-lora] HELD-OUT loss  adapter {{la:.4f}}  vs  base {{lb:.4f}}  (delta {{lb-la:+.4f}}) | router {{has_router}}", flush=True)
print(f"[qwen-lora] peak GPU {{torch.cuda.max_memory_allocated()/1e9:.2f}} GB", flush=True)
router_ok = has_router if is_moe else True
print("CHALLENGE PASS" if (la < lb and router_ok) else "CHALLENGE FAIL (adapter not better than base, or MoE router not adapted)")
'''


def pipe_qwen_moe_lora(run: Run) -> dict:
    """LoRA fine-tune the downloaded bf16 **Qwen3-MoE** with the router included + aux load-balance loss on (per
    [[qwen/qwen-moe-training]]). SFT targets = the vault's harvested `chosen` responses (completion-masked). Challenge:
    held-out adapter loss < base AND the router got LoRA (proving it's not the all-linear no-op). Needs the leader
    STOPPED for memory (30B bf16 ≈ 61GB). Background+poll."""
    model = str(run.params.get("model") or "/models/qwen3-30b-a3b-instruct").strip()
    pairs = _get_hub().harvest_dpo()
    if len(pairs) < 6:
        return {"verdict": f"only {len(pairs)} pairs — enrich the vault first"}
    rows = [{"prompt": p["prompt"].rstrip() + "\n", "completion": " " + p["chosen"].strip(),
             "split": ("holdout" if i % 4 == 0 else "train")} for i, p in enumerate(pairs)]
    files = {"sft.jsonl": "\n".join(_json.dumps(r) for r in rows),
             "qwen_lora.py": _QWEN_MOE_LORA_TEMPLATE.format(model=model, epochs=int(run.params.get("epochs", 3)))}
    out, res = _run_on_worker(run, "qwenlora", files, "qwen_lora.py", "qwen_lora_result",
                              max_wait=int(run.params.get("max_wait", 1500)))
    if out is None:
        return res
    if res.get("passed"):
        _vault_enrich(run, "experience", "qwen-moe-lora-run",
                      f"_LoRA fine-tuned Qwen3-MoE (router + experts + attn, aux-loss on) on {len(rows)} vault targets; "
                      f"held-out adapter beat base AND the router was adapted (not the all-linear no-op)._",
                      links=["qwen/qwen-moe-training", "qwen/qwen-optimization"], tags=["train", "qwen", "moe", "lora"])
    is_moe = "moe" in os.path.basename(model).lower() or "30b" in model.lower()
    return {"targets": len(rows), "moe": is_moe, **res,
            "verdict": ((f"QWEN LoRA TRAINED — held-out adapter beat base" + (" + router adapted" if is_moe else " (dense; no router)"))
                        if res.get("passed") else "qwen LoRA ran, see logs (held-out delta / router)")}


def pipe_vault_ingest(run: Run) -> dict:
    """Grow the vaults from EXTERNAL sources so agents can GO FURTHER: the framework's OWN code (so the swarm can read
    how it itself works), any paths YOU supply (`params.paths`, or the `VAULT_INGEST_PATHS` env — os.pathsep-separated),
    and — optionally — a local clone of the native Gemma foundation repo (google-deepmind/gemma) whose source becomes
    navigable notes. Files land in the `repos` vault, cross-linked to the `foundation` concepts. Idempotent."""
    hub = _get_hub()
    before = hub.stats()
    # default sources = the framework's own (public) code, so the swarm can navigate how it works
    files = [(os.path.join(PKG, f), f"framework:{os.path.splitext(f)[0]}")
             for f in ("vault.py", "core.py", "agent_tools.py") if os.path.exists(os.path.join(PKG, f))]
    runner_py = os.path.join(PKG, "studio", "backend", "runner.py")
    if os.path.exists(runner_py):
        files.append((runner_py, "framework:runner"))
    # bring-your-own sources: params.paths or the VAULT_INGEST_PATHS env var (no private paths are hardcoded)
    extra = list(run.params.get("paths") or []) + [p for p in os.environ.get("VAULT_INGEST_PATHS", "").split(os.pathsep) if p]
    files += [(p, f"path:{os.path.basename(str(p))}") for p in extra if p]
    ingested, missed = [], []
    for path, source in files:
        if not run.gate():
            break
        if os.path.exists(path):
            nid = hub.ingest_file("repos", path, source=source)
            if nid:
                ingested.append(f"repos/{nid}")
                run.emit("vault", action="ingest", vault="repos", note=nid, source=source, notes=hub.stats()["notes"])
            else:
                run.emit("vault", action="skip", source=source)
        else:
            missed.append(source)
    # the NATIVE gemma foundation repo — read its actual source if a local clone exists
    gemma_repo = run.params.get("gemma_repo") or os.environ.get("GEMMA_REPO") or os.path.join(PKG, "knowledge", "repos", "gemma")
    if os.path.isdir(gemma_repo):
        got = hub.ingest_dir("gemma", os.path.join(gemma_repo, "gemma") if os.path.isdir(os.path.join(gemma_repo, "gemma")) else gemma_repo,
                             source="gemma_repo", patterns=(".py", ".md"), max_files=60)
        for nid in got:
            ingested.append(f"gemma/{nid}")
        run.emit("vault", action="ingest_dir", vault="gemma", n=len(got), source=gemma_repo, notes=hub.stats()["notes"])
    else:
        run.emit("vault", action="missing_repo", source=gemma_repo,
                 hint=f"clone it to read the native source: git clone https://github.com/google-deepmind/gemma {gemma_repo}")
        missed.append(f"gemma_repo (clone: git clone https://github.com/google-deepmind/gemma {gemma_repo})")
    after = hub.stats()
    run.emit("result", ingested=len(ingested), vaults=after["names"], notes=after["notes"], edges=after["edges"], missed=missed)
    return {"ingested": ingested, "missed": missed, "notes_before": before["notes"], "notes_after": after["notes"],
            "vaults": after["names"], "edges": after["edges"],
            "verdict": f"vaults grew {before['notes']}→{after['notes']} notes across {after['names']} from "
                       f"{len(ingested)} sources" + (f"; missing: {missed}" if missed else "")}


def pipe_multi_day_service(run: Run) -> dict:
    """The multi-day ops world: an agent works a support desk over days, learning a per-workflow playbook (no KB).
    Streams a 'day' event per day with the clean-resolution rate so the UI can chart the learning curve."""
    from bobby_squad import Agent, SelfCore, OpsWorld, WORKFLOWS, operate
    from bobby_squad.llm import LLM
    llm = LLM(temperature=0.5, timeout=120)
    days = int(run.params.get("days", 4))
    role = "an Acme support specialist with back-office tool access"
    goal = "resolve each case by operating the tools correctly, without over-compensating"
    learner = Agent(SelfCore(role, goal), llm=llm, window=4, pinned=True, name="learner", observer=run.observe)
    run.agents = {"learner": learner}
    world = OpsWorld(seed=1, restock=1)
    tk = str(run.params.get("tickets", "")).strip()
    tickets = [t.strip() for t in tk.split(",") if t.strip() in set(WORKFLOWS)] if tk \
        else list(WORKFLOWS) + ["lost_package", "wrong_item"]     # composable caseload (define your own workflow mix)
    daily = []
    for d in range(days):
        if not run.gate():
            break
        for s in run.take_steer():
            learner.observe(f"OPERATOR DIRECTIVE: {s}")
        world.advance_day()
        clean = 0
        for wf in tickets:
            facts = world.open_ticket(wf)
            tag = f"LESSON[{wf}]:"
            pb = "\n".join(p[len(tag):] for p in learner.ctx.progress if p.startswith(tag)) or "(none yet)"
            brief = (f"You are {role}. Goal: {goal}.\nLearned for '{wf}':\n{pb}\n\nCase facts: {facts}\n"
                     "Investigate and take the correct action(s). Nothing is fixed unless you call a tool.")
            _reply, trace = operate(llm, world, brief, max_rounds=7)
            ok = world.resolved()
            clean += int(ok)
            lesson = (learner.act(f"Case '{wf}' outcome {'CLEAN' if ok else 'NOT resolved'}; QA: "
                                  f"\"{world.feedback()}\". One rule for next '{wf}' case.", max_tokens=50) or "").strip()
            if lesson:
                learner.record(f"LESSON[{wf}]: {lesson}")
                _store_knowledge(run, lesson, {"workflow": wf, "domain": "customer-ops", "resolved": ok})
            # WORLD state — the real back-office the agent operated (rich enough to RENDER, not just list)
            o = world.orders.get(world.cur, {})
            cust = world.customers.get(o.get("cust_id"), {})
            ship = world.shipments.get(world.cur, {})
            run.emit("world", workflow=wf, resolved=ok, feedback=world.feedback(), day=d + 1,
                     actions=[a.split("(")[0] for a in world.log[-6:]], inventory=dict(world.inventory),
                     customer=o.get("cust_id"), order=o.get("id"), item=o.get("item"), amount=o.get("amount"),
                     shipment=ship.get("status"), refunded=o.get("refunded", 0), reshipped=o.get("reshipped", False),
                     account_locked=cust.get("account_locked", False), credit=cust.get("credit", 0))
            learner.compact()
        rate = clean / len(tickets)
        daily.append(rate)
        run.emit("day", day=d + 1, clean=clean, total=len(tickets), rate=round(rate, 3))
        # per-workflow playbook the agent authored from experience (no KB) — surfaced for the Memory panel
        pb = [p for p in learner.ctx.progress if p.startswith("LESSON[")]
        run.emit("playbook", day=d + 1, n=len(pb), lessons=[p[:140] for p in pb[-8:]])
    lift = round((daily[-1] - daily[0]), 3) if len(daily) > 1 else 0.0
    run.emit("verdict", metric="clean-resolution", control_flat=True, learner_lift=lift,
             verdict="WIRE" if lift > 0.1 else "INCONCLUSIVE",
             detail=f"{daily[0]:.0%}→{daily[-1]:.0%} over {days} days")
    return {"days": days, "daily_rate": daily, "lift": lift,
            "verdict": f"clean-resolution {daily[0]:.0%}→{daily[-1]:.0%} over {days} days"}


# ---------------------------------------------------------------------------------------------------------------
# NATIVE pipelines stream STRUCTURED events (they wire Agent(observer=run.observe)) → the rich tabs (Squad / Board /
# Memory / World / Proofs). Everything else in examples/ is auto-discovered and runnable as a SCRIPT pipeline whose
# stdout is streamed live — so the WHOLE capability surface is exposed, not a hand-picked few.
# ---------------------------------------------------------------------------------------------------------------
import ast          # noqa: E402
import glob         # noqa: E402
import subprocess   # noqa: E402

EXAMPLES = os.path.join(PKG, "examples")

def _load_units(run: Run):
    """Turn ANY customer data type into a list of work UNITS for the long-running agents to process.
    Supports: pasted text / code / markdown, CSV & JSON (row-wise), PDF (pypdf), and a URL or file path."""
    import io
    import json as _json
    import urllib.parse
    import urllib.request
    kind = str(run.params.get("kind", "auto")).lower()
    data = run.params.get("data")
    source = str(run.params.get("source", "")).strip()
    raw, name = "", "input"

    # HuggingFace dataset — pull rows via the public datasets-server API (no deps, no auth for public datasets)
    if kind == "hf" and source:
        def _get(u):
            return _json.load(urllib.request.urlopen(u, timeout=25))
        cfg, split = "default", "train"
        try:
            sp = (_get("https://datasets-server.huggingface.co/splits?dataset=" + urllib.parse.quote(source)).get("splits") or [])
            if sp:
                cfg, split = sp[0].get("config", "default"), sp[0].get("split", "train")
        except Exception as e:
            run.emit("log", line=f"hf splits lookup failed: {e}")
        length = min(int(run.params.get("max_units", 60)), 100)
        rows = []
        try:
            r = _get(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(source)}"
                     f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}&offset=0&length={length}")
            for it in r.get("rows", []):
                row = it.get("row", {})
                txt = "; ".join(f"{k}={str(v)[:300]}" for k, v in row.items() if v not in (None, ""))
                if txt.strip():
                    rows.append(txt[:1400])
        except Exception as e:
            run.emit("log", line=f"hf rows fetch failed for '{source}': {e}")
        run.emit("log", line=f"HF dataset '{source}' ({cfg}/{split}): {len(rows)} rows")
        return rows, "hf", f"{source}:{split}"

    if source.startswith("http"):
        try:
            raw = urllib.request.urlopen(source, timeout=20).read().decode("utf-8", "ignore"); name = source
            if kind == "auto":
                kind = "pdf" if source.lower().endswith(".pdf") else "text"
        except Exception as e:
            run.emit("log", line=f"fetch failed: {e}")
    elif source and os.path.isfile(source):
        name = os.path.basename(source)
        if kind == "auto":
            ext = source.lower().rsplit(".", 1)[-1]
            kind = {"csv": "csv", "json": "json", "pdf": "pdf"}.get(ext, "code" if ext in ("py", "js", "ts", "go", "rs", "java") else "text")
        if kind == "pdf":
            try:
                import pypdf
                raw = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(source).pages)
            except Exception as e:
                run.emit("log", line=f"pdf read failed (pip install pypdf): {e}")
        else:
            raw = open(source, errors="ignore").read()
    else:
        raw = data if isinstance(data, str) else _json.dumps(data) if data is not None else ""
        if kind == "auto":
            kind = "json" if raw.strip()[:1] in "[{" else "text"

    units = []
    if kind == "csv":
        import csv as _csv
        rows = list(_csv.reader(io.StringIO(raw)))
        header = rows[0] if rows else []
        for r in rows[1:]:
            units.append("record: " + "; ".join(f"{h}={v}" for h, v in zip(header, r)))
    elif kind == "json":
        try:
            obj = _json.loads(raw)
            items = obj if isinstance(obj, list) else [obj]
            units = [_json.dumps(it)[:1400] for it in items]
        except Exception:
            units = [raw[i:i + 1400] for i in range(0, len(raw), 1400)]
    else:
        units = [raw[i:i + 1400] for i in range(0, len(raw), 1400)]   # text / code / markdown / pdf-text

    units = [u for u in units if u.strip()]
    cap = int(run.params.get("max_units", 60))
    if len(units) > cap:
        run.emit("log", line=f"{len(units)} units → capping to {cap} (raise max_units to process more)")
        units = units[:cap]
    return units, kind, name


def pipe_process_data(run: Run) -> dict:
    """UNLIMITED DATA PROCESSING — long-running persistent-self agents read a customer's data (any type) end-to-end,
    accumulating a knowledge map in the PINNED tier over the whole stream (goal never lost across the horizon), and
    plateau when the shared board drains. This is the customer's core job; it exposes the real long-horizon engine."""
    from bobby_squad import Agent, SelfCore, squad_solve
    from bobby_squad.llm import LLM
    from bobby_squad.dedup import near_dup
    llm = LLM(temperature=0.4, timeout=150)
    goal = str(run.params.get("goal") or "read this data end-to-end and build a clear structured knowledge map of it")
    units, kind, name = _load_units(run)
    if not units:
        return {"verdict": "no data — provide `data` (text) or a `source` (path/url)"}
    # SELF-CALCULATED headcount — the swarm sizes itself to the workload it found (one reader per few sections),
    # bounded by a runaway cap. Not a fixed `agents=N` knob.
    n = max(2, min(len(units), int(run.params.get("max_agents", 6))))
    run.emit("log", line=f"processing '{name}' as {kind}: {len(units)} units · swarm self-scaled to {n} readers")
    _recall = _vault_recall(run)
    agents = [Agent(SelfCore(
        identity="a long-running analyst that reads material section by section and accumulates a durable knowledge map",
        goal=goal, constraints=["ground every note in the section you were given", "don't repeat earlier notes"]),
        llm=llm, window=3, pinned=True, name=f"reader{i}", observer=run.observe, recall=_recall) for i in range(n)]
    total0 = len(units)
    state = {"done": 0, "note_of": {}, "splits": 0}

    def work(agent, unit):
        if not run.gate():
            return set()
        for s in run.take_steer():
            agent.observe(f"OPERATOR DIRECTIVE: {s}")
        agent.observe(f"SECTION ({len(unit)} chars):\n{unit}")     # perception, not an instruction
        res = agent.autonomous_cycle(max_steps=1)                  # ENGINE self-directs (select_target→plan→execute); no prompt
        note = " ".join((r.get("result") or "").strip() for r in res.get("results", [])).strip()
        state["note_of"][unit] = note
        if note:                                                   # research_cycle already deduped + recorded to the pinned tier
            _store_knowledge(run, note, {"domain": "customer-data", "source": name, "dtype": kind})
        state["done"] += 1
        pinned_tok = sum(len(p) for p in agent.ctx.progress) // 4
        run.emit("section", i=state["done"], agent=agent.name, chars=len(unit), pinned_items=len(agent.ctx.progress),
                 pinned_tokens=pinned_tok, note=note[:200])
        agent.compact()                                            # roll the working window; the pinned map stays
        return {note} if note else set()

    # RECURSIVE COVERAGE (the squad-pipeline pattern): a dense section that wasn't well covered is SPLIT into halves
    # and re-queued at finer granularity, so the squad drills down until every piece is covered or atomic.
    def verify(unit, _acc):
        note = state["note_of"].get(unit, "")
        return len(unit) < 900 or len(note) >= 60          # small enough, or produced a substantive note → covered

    def split(unit):
        if len(unit) < 900:
            return None                                    # atomic — accept the partial
        state["splits"] += 1
        mid = len(unit) // 2
        run.emit("split", chars=len(unit), into=2)
        return [unit[:mid], unit[mid:]]

    out = squad_solve(agents, list(units), work, verify=verify, split=split, accumulated=set())
    notes = [x for x in out["result"] if x]
    # DELIVERABLE = the squad's own accumulated knowledge map (no authored synthesis prompt); the UI composes it.
    run.emit("result", dtype=kind, source=name, units=total0, passes=out["passes"], splits=state["splits"],
             notes=len(notes), summary="\n".join(f"- {x}" for x in notes[:80]))
    return {"units": total0, "passes": out["passes"], "splits": state["splits"], "dtype": kind, "notes": len(notes),
            "verdict": f"read {total0} sections ({out['passes']} passes, {state['splits']} recursive splits) of {kind}"}


def pipe_goal(run: Run) -> dict:
    """GOAL-DRIVEN generative squad: a goal is decomposed into acceptance-CRITERIA and a living BOARD of work-CARDS;
    self-organizing agents work the board (self-select a target + plan, produce OUTCOME EVIDENCE), the board GENERATES
    and recursively SPLITS cards, and it converges when every criterion verifies — else it ESCALATES to the human.
    Emits: goal · criteria · card (state transitions) · split · (swarm move events) · criterion · converged/escalate."""
    from bobby_squad import Agent, SelfCore, squad_solve
    from bobby_squad.llm import LLM
    from bobby_squad.dedup import near_dup
    llm = LLM(temperature=0.5, timeout=150)
    goal = str(run.params.get("goal") or "").strip() or "produce a clear, verified result for the provided material"
    run.emit("goal", goal=goal)

    def lines(prompt, n=6, mx=260):
        out = llm([{"role": "user", "content": prompt}], max_tokens=mx) or ""
        got = [ln.strip(" -*0123456789.").strip() for ln in out.splitlines() if ln.strip()]
        return [g for g in got if len(g) > 6][:n]

    # 1) acceptance criteria — the truthful progress ledger (verify-by-outcome, no fake %)
    criteria = [{"id": f"AC{i+1}", "text": t, "met": False}
                for i, t in enumerate(lines(f"Goal: {goal}\nList 3-5 atomic, checkable ACCEPTANCE CRITERIA that must all "
                                             "be true for this goal to be DONE. One per line, no numbering.", 5))]
    run.emit("criteria", criteria=criteria)

    # 2) WORKLOAD — the swarm SELF-DECOMPOSES the goal into work items via the ENGINE (make_plan, neutral frame), or
    #    reads provided data records. No hand-authored "break this into work items" prompt.
    data_units, _kind, _name = _load_units(run) if (run.params.get("data") or run.params.get("source")) else ([], "", "")
    is_data = bool(data_units)
    if data_units:
        units = data_units
    else:
        planner = Agent(SelfCore(identity="a member of a self-organizing squad pursuing a shared goal", goal=goal),
                        llm=llm, window=3, pinned=True, name="planner", observer=run.observe, recall=_vault_recall(run))
        units = [s.get("intention", "").strip() for s in planner.make_plan(goal, max_steps=6) if s.get("intention")] or [goal]

    # SELF-CALCULATED headcount — the recursive swarm decides how many Bobby copies it needs from the workload it
    # discovered (one per top-level workstream), bounded by a runaway cap. Not a fixed `agents=N` knob.
    n = max(2, min(len(units), int(run.params.get("max_agents", 6))))
    run.emit("log", line=f"swarm self-scaled to {n} agents for {len(units)} workstreams")
    _recall = _vault_recall(run)
    agents = [Agent(SelfCore(identity="a member of a self-organizing squad pursuing a shared goal",
                             goal=goal, constraints=["produce concrete outcome evidence, not prose", "don't repeat prior work"]),
                    llm=llm, window=3, pinned=True, name=f"agent{i}", observer=run.observe, recall=_recall) for i in range(n)]
    # REUSE — seed the squad with a prior EXPERT's accumulated knowledge (cross-run transfer)
    reuse = str(run.params.get("reuse_expert", "")).strip()
    if reuse:
        exp = next((e for e in get_store().list("experts", limit=500) if e.get("id") == reuse), None)
        if exp and exp.get("knowledge"):
            for a in agents:
                for k in exp["knowledge"][:40]:
                    a.record(f"[reused from {exp.get('specialty', 'prior expert')}] {k}")
            run.emit("log", line=f"reusing expert '{exp.get('specialty')}' ({len(exp['knowledge'])} knowledge items)")
    run.emit("teams", members=[a.name for a in agents], workstreams=len(units))
    findings, state = [], {"cid": 0, "splits": 0, "note_of": {}, "depth": {}, "seen": list(units)}

    def work(agent, unit):
        if not run.gate():
            return set()
        for s in run.take_steer():
            agent.observe(f"OPERATOR DIRECTIVE: {s}")
        state["cid"] += 1
        cid = f"#{state['cid']:03d}"
        title = unit.strip().split("\n")[0][:80]
        run.emit("card", id=cid, title=title, cstate="in_progress", owner=agent.name)
        agent.observe(f"WORK ITEM (toward the goal): {unit}")      # perception, not an instruction
        res = agent.autonomous_cycle(max_steps=1)                 # ENGINE self-directs; DEPTH comes from RECURSION, not steps
        evidence = " ".join((r.get("result") or "").strip() for r in res.get("results", [])).strip()
        move = (res.get("results") or [{}])[0].get("type", "") if res.get("results") else ""
        state["note_of"][unit] = evidence
        if evidence and evidence not in findings:                 # research_cycle already recorded to the pinned tier
            findings.append(evidence)
            _store_knowledge(run, evidence, {"domain": "goal", "goal": goal[:60]})
        run.emit("card", id=cid, title=title, cstate="verified", owner=agent.name, plan=str(move)[:120],
                 evidence=evidence[:220])
        agent.compact()
        return {evidence} if evidence else set()

    def verify(unit, _acc):
        # covered = the work produced substantive outcome evidence for this item; otherwise it is SPLIT below.
        return len(state["note_of"].get(unit, "")) >= 40

    def split(unit):
        # NATIVE recursive split — DEPTH is emergent, not a param. A long/data record is CHUNKED; an under-covered
        # TASK is DECOMPOSED by an agent via the engine's make_plan (neutral frame) into sub-items and re-queued.
        # Recursion bottoms out when a record is atomic or make_plan yields nothing new (backstopped at depth 3).
        d = state["depth"].get(unit, 0)
        if d >= 3:
            return None
        if is_data or len(unit) >= 1200:                          # data record / long material → chunk (or atomic)
            if len(unit) < 900:
                return None
            state["splits"] += 1
            mid = len(unit) // 2
            subs = [unit[:mid], unit[mid:]]
        else:                                                     # a broad TASK → an agent DECOMPOSES it (generative)
            a = agents[state["cid"] % len(agents)]
            subs = [s.get("intention", "").strip() for s in a.make_plan(unit, max_steps=4) if s.get("intention")]
            subs = [s for s in subs if s and s != unit and not near_dup(s, state["seen"])]
            if not subs:
                return None                                       # atomic — accept the partial
            state["splits"] += 1
        for s in subs:
            state["depth"][s] = d + 1
            state["seen"].append(s)
        run.emit("split", item=unit[:50], into=len(subs), depth=d + 1)
        return subs

    # 3) WORK THE BOARD IN WAVES — self-managed: keep going until every criterion verifies (CONVERGED) or the board
    # stops producing new work (PLATEAU → escalate). Each wave: drain the board, re-verify criteria by OUTCOME, and
    # if any are unmet the board GENERATES fresh cards aimed at the exact gaps. Not a fixed step count.
    # The stops are by OUTCOME (all criteria verified → CONVERGED) and by PLATEAU (a wave that adds no genuinely-new
    # work → fixed point → escalate). No patience knob. GUARD is a pure runaway backstop, like squad_solve's max_passes.
    GUARD = int(run.params.get("max_waves", 12))
    seen_units = list(units)
    total_passes = met = wave = 0
    converged = False
    while wave < GUARD:
        if not run.gate():
            break
        wave += 1
        out = squad_solve(agents, list(units), work, verify=verify, split=split, accumulated=set())
        total_passes += out["passes"]
        # verify each acceptance criterion against ALL accumulated evidence (verify-by-outcome)
        ev = "\n".join(f"- {f}" for f in findings[:120]) or "(no evidence)"
        met = 0
        for c in criteria:
            ans = (llm([{"role": "user", "content": f"Evidence:\n{ev}\n\nIs this criterion satisfied by the evidence? "
                         f"Criterion: {c['text']}\nAnswer ONLY YES or NO."}], max_tokens=5) or "").strip().upper()
            c["met"] = ans.startswith("YES"); met += int(c["met"])
            run.emit("criterion", id=c["id"], text=c["text"], met=c["met"])
        converged = met == len(criteria) and criteria != []
        run.emit("wave", n=wave, met=met, total=len(criteria), findings=len(findings), converged=converged)
        if converged:
            break
        # CONTINUE without a hand-authored expansion prompt (golden rule: no scripted behavior). The next work units
        # ARE the unmet acceptance criteria themselves — re-queued verbatim for the squad to pursue directly. Nothing
        # here scripts WHAT to invent; the agents self-select how to satisfy each criterion via their neutral frame.
        unmet = [c["text"] for c in criteria if not c["met"]]
        fresh = [u for u in unmet if not near_dup(u, seen_units)]
        if not fresh:                                   # no genuinely-new work → plateau (fixed point) → escalate
            break
        seen_units += fresh
        units = fresh
    # DELIVERABLE = the squad's own accumulated evidence (no authored synthesis prompt); the UI composes it.
    summary = "\n".join(f"- {f}" for f in findings[:80]) or "(no evidence)"
    # 4) EXPERTS — each agent crystallized into a specialist; capture it + its knowledge so it can be REUSED later
    for a in agents:
        spec = (a.reflect("In ONE line, name the SPECIALTY you became from this work — a title + a short phrase.") or "").strip()
        know = [p for p in a.ctx.progress if not p.startswith("[reused")]
        eid = f"{run.id}:{a.name}"
        try:
            get_store().upsert("experts", eid, {"id": eid, "run_id": run.id, "name": a.name, "specialty": spec[:160],
                               "goal": goal[:160], "n_knowledge": len(know), "knowledge": know, "ts": now()},
                               text=spec + " " + " ".join(know[:20]))
        except Exception:
            pass
        run.emit("expert", id=eid, name=a.name, specialty=spec[:160], n_knowledge=len(know))

    run.emit("result", goal=goal, criteria_met=met, criteria_total=len(criteria), converged=converged,
             waves=wave, passes=total_passes, splits=state["splits"], cards=state["cid"], summary=summary)
    run.emit("converged" if converged else "escalate", met=met, total=len(criteria),
             reason="all acceptance criteria verified" if converged else "squad plateaued with some criteria unmet — human review")
    return {"cards": state["cid"], "waves": wave, "criteria_met": f"{met}/{len(criteria)}", "converged": converged,
            "verdict": ("CONVERGED — " if converged else "ESCALATE — ") + f"{met}/{len(criteria)} criteria met over {wave} waves"}


def pipe_persona(run: Run) -> dict:
    """A CONSTANT generative persona you converse with — a persistent agent that stays in character across the whole
    session, remembering the conversation in its pinned tier. You talk to it by STEERING; it replies live."""
    from bobby_squad import Agent, SelfCore
    from bobby_squad.llm import LLM
    llm = LLM(temperature=0.75, timeout=120)
    persona = str(run.params.get("persona") or run.params.get("goal") or "a warm, curious generative persona").strip()
    a = Agent(SelfCore(identity=persona, goal="stay fully in character; converse and act as this persona",
                       constraints=["stay in character", "remember what was said"]),
              llm=llm, window=10, pinned=True, name="persona", observer=run.observe)
    run.emit("agent", name="persona", persona=persona[:180])
    opening = (a.act("Introduce yourself in character, 1-2 sentences.", max_tokens=120) or "").strip()
    run.emit("say", who="persona", persona=persona[:80], text=opening)
    idle = 0
    while idle < 240:                                  # constant during the session; Stop ends it
        if not run.gate():
            break
        msgs = run.take_steer()
        if msgs:
            idle = 0
            for m in msgs:
                a.observe(f"User: {m}")
                reply = (a.act("Respond in character.", max_tokens=200) or "").strip()
                run.emit("say", who="persona", persona=persona[:80], text=reply)
                _store_knowledge(run, reply, {"domain": "persona", "persona": persona[:60]})
        else:
            idle += 1
            time.sleep(0.5)
    return {"verdict": f"persona session — {persona[:50]}"}


def pipe_world(run: Run) -> dict:
    """A VIRTUAL WORLD with CONSTANT agents — persistent personas that live in a themed world and interact over
    rounds, each remembering the world in its pinned tier. You can inject events by STEERING. (world-simulation style.)"""
    from bobby_squad import Agent, SelfCore
    from bobby_squad.llm import LLM
    import json as _json
    llm = LLM(temperature=0.85, timeout=120)
    theme = str(run.params.get("world") or run.params.get("goal") or "a lively town square where neighbours meet").strip()
    pf = os.path.join(PKG, "examples", "data", "personas.jsonl")
    pool = []
    if os.path.isfile(pf):
        try:
            pool = [_json.loads(l)["persona"] for l in open(pf) if l.strip()]
        except Exception:
            pool = []
    k = int(run.params.get("agents", 4))
    chosen = (pool[:k] if pool else [f"resident {i}" for i in range(k)])
    agents = [Agent(SelfCore(identity=p, goal=f"live in this world and interact naturally: {theme}",
                             constraints=["stay in character", "react to what others said"]),
                    llm=llm, window=6, pinned=True, name=f"npc{i}", observer=run.observe) for i, p in enumerate(chosen)]
    for i, a in enumerate(agents):
        run.emit("agent", name=a.name, persona=chosen[i][:120])
    run.emit("world_start", theme=theme, agents=len(agents))
    rounds = int(run.params.get("rounds", 14))
    heard = ""
    for r in range(rounds):
        if not run.gate():
            break
        for m in run.take_steer():
            for a in agents:
                a.observe(f"[a new event happens] {m}")
            run.emit("say", who="event", text=m, round=r + 1)
        a = agents[r % len(agents)]
        if heard:
            a.observe(f"You just heard: {heard}")
        say = (a.act(f"Round {r+1} in: {theme}. Say or do ONE short thing, in character.", max_tokens=120) or "").strip()
        heard = f"{a.name}: {say}"
        run.emit("say", who=a.name, persona=chosen[r % len(agents)][:80], text=say, round=r + 1)
        _store_knowledge(run, say, {"domain": "world", "theme": theme[:60]})
    run.emit("result", theme=theme, rounds=rounds, agents=len(agents), summary=f"{rounds} rounds of life in: {theme}")
    return {"rounds": rounds, "agents": len(agents), "verdict": f"world sim — {theme[:50]}"}


NATIVE: Dict[str, dict] = {
    "goal": {"fn": pipe_goal, "title": "Goal-driven squad", "kind": "native",
             "desc": "State a goal → the squad decomposes it into acceptance-criteria + a living board, works it, and converges or escalates.",
             "params": {"goal": "", "data": "", "source": ""}},
    "persona": {"fn": pipe_persona, "title": "Generative persona", "kind": "native",
                "desc": "A constant persona you converse with — a persistent agent that stays in character across the session.",
                "params": {"persona": ""}},
    "world": {"fn": pipe_world, "title": "Virtual world sim", "kind": "native",
              "desc": "A themed world with constant persona-agents that live and interact over rounds; inject events by steering.",
              "params": {"world": "", "agents": 4, "rounds": 14}},
    "process_data": {"fn": pipe_process_data, "title": "Process data (any type)", "kind": "native",
                     "desc": "Long-running agents read your data (text/code/CSV/JSON/PDF/URL) end-to-end into a knowledge map.",
                     "params": {"kind": "auto", "goal": "", "data": "", "source": ""}},
    "engine_trace": {"fn": pipe_engine_trace, "title": "Engine trace", "kind": "native",
                     "desc": "A squad reads this repo — every engine layer fires live (rich panels).",
                     "params": {"agents": 2, "rounds": 2}},
    "idea_board": {"fn": pipe_idea_board, "title": "Idea board (repulsion)", "kind": "native",
                   "desc": "Agents mine ideas; the board repels near-dups and surfaces the most-spread (rich Board tab).",
                   "params": {"agents": 2}},
    "research": {"fn": pipe_research, "title": "Domain research lab", "kind": "native",
                 "desc": "Point the idea lab at ANY topic (math · sociology · company-org · proxy · …), not just code: "
                         "invent → develop → red-team → portfolio, domain-free.",
                 "params": {"topic": "", "agents": 3}},
    "multi_day_service": {"fn": pipe_multi_day_service, "title": "Multi-day service desk", "kind": "native",
                          "desc": "An agent works a support desk over days, learning a per-workflow playbook, no KB.",
                          "params": {"days": 4}},
    "code_dev": {"fn": pipe_code_dev, "title": "Code-dev from scratch (self-challenged)", "kind": "native",
                 "desc": "A swarm builds real code from scratch (default: Gemma3 transformer), WRITES ITS OWN "
                         "acceptance challenge, and adversarially reviews its own plateau/failures — the verify layer "
                         "is generative, run-decides.",
                 "params": {"goal": "", "agents": 3}},
    "vault_ingest": {"fn": (lambda run: pipe_vault_ingest(run)), "title": "Ingest sources into the knowledge vault",
                     "kind": "native",
                     "desc": "Pull EXTERNAL sources into the navigable vault as linked notes: the framework's own code, "
                             "the HF gemma-challenge pipeline, and any paths you pass — so the swarm enters wiser and "
                             "the graph grows from every source, not just its own runs.",
                     "params": {"paths": []}},
    "train": {"fn": pipe_train, "title": "Train a foundation model (self-challenged)", "kind": "native",
              "desc": "A swarm trains a REAL foundation model on the GPU worker (LoRA/DPO, weights under /models, no "
                      "download), writes its OWN metric challenge, launches on the DGX (background), polls logs, and "
                      "iterates until the challenge passes — memory-gated + observable.",
              "params": {"model": "/models/gemma3-1b-ablit", "method": "LoRA", "goal": "", "agents": 2}},
    "self_dpo": {"fn": pipe_self_dpo, "title": "Self-DPO flywheel (meta-cognition → DPO)", "kind": "native",
                 "desc": "The agent LEARNS FROM ITS OWN SELF-ANALYSIS: meta-cognition (pattern · critique · alternative) "
                         "manufactures preference pairs (chosen≻rejected), which DPO-train the foundation model on the "
                         "GPU worker — iterative self-DPO, no external labels.",
                 "params": {"model": "/models/gemma3-1b-ablit"}},
    "world_layer": {"fn": pipe_world_layer, "title": "Train the world transformer layer (avoid chat)", "kind": "native",
                    "desc": "Trains the NEW world-transformer layer on the GPU worker: a small encoder turns real "
                            "vault-note EMBEDDINGS into world tokens prepended to a FROZEN LM — feeding world-state as "
                            "vectors, not chat. Challenge = held-out with-world loss beats without-world.",
                    "params": {"model": "/models/gemma3-1b-ablit", "epochs": 10, "k": 16}},
    "qwen_moe_lora": {"fn": (lambda run: pipe_qwen_moe_lora(run)), "title": "LoRA fine-tune Qwen3-MoE (router + aux-loss)",
                      "kind": "native",
                      "desc": "Real MoE training: LoRA on the downloaded bf16 Qwen3-30B-A3B — attn + experts + ROUTER, "
                              "aux load-balance loss on, completion-masked. Challenge: held-out adapter beats base AND "
                              "the router was adapted (not the all-linear no-op). Stop the leader first (needs ~61GB).",
                      "params": {"model": "/models/qwen3-30b-a3b-instruct", "epochs": 3}},
    "value_head": {"fn": pipe_value_head, "title": "Train the value head (learned critic)", "kind": "native",
                   "desc": "A learned critic scored from the self-DPO chosen>rejected pairs — replaces the LLM "
                           "self-critique with a cheap deterministic quality head. Challenge = held-out ranking acc.",
                   "params": {"model": "/models/gemma3-1b-ablit", "epochs": 8}},
    "retrieval_encoder": {"fn": pipe_retrieval_encoder, "title": "Train the retrieval/memory encoder", "kind": "native",
                          "desc": "Learned recall over vault-note embeddings (which item to LOAD, beyond cosine). "
                                  "Challenge = held-out recall@1 matches/beats raw cosine.",
                          "params": {"epochs": 30}},
    "trajectory_monitor": {"fn": pipe_trajectory_monitor, "title": "Train the trajectory self-monitor", "kind": "native",
                           "desc": "Learned metacognition: classifies looping/drifting/converging from a raw action "
                                   "trace (labels = the deterministic behavior signals). Challenge = held-out accuracy.",
                           "params": {"epochs": 12}},
    "perception": {"fn": pipe_perception, "title": "Train the perception encoder (multimodal)", "kind": "native",
                   "desc": "The world layer for a NON-TEXT modality: observation embeddings -> world tokens for the "
                           "frozen LM. Plug a CLIP/CLAP embedder for real image/audio. Challenge = with-obs beats without.",
                   "params": {"model": "/models/gemma3-1b-ablit", "epochs": 12, "k": 16}},
    "encoder_proof": {"fn": (lambda run: pipe_encoder_proof(run)), "title": "Prove the encoder helps the model answer",
                      "kind": "native",
                      "desc": "End-to-end A/B: the frozen LM ANSWERS held-out questions with context picked by the "
                              "trained retriever vs cosine vs none. Metric = real answer accuracy, not loss.",
                      "params": {"model": "/models/gemma3-1b-ablit"}},
    "self_model": {"fn": (lambda run: pipe_self_model(run)), "title": "Train the unified self-model (world+monitor+value)",
                   "kind": "native",
                   "desc": "The COUPLED self-monitoring core: world encoder as the hub; trajectory monitor + value head "
                           "CONDITION on world state. One step -> {world, am-I-looping, how-good}. Trained jointly; "
                           "challenge = all three heads pass on held-out.",
                   "params": {"model": "/models/gemma3-1b-ablit", "epochs": 12, "k": 16}},
}


def _make_script_pipeline(name: str) -> Callable[[Run], dict]:
    """Run an example as a live subprocess, streaming its stdout as `log` events. Params become UPPERCASE env vars
    the example already reads (e.g. AGENTS, DAYS, EPISODES, HORIZON_APPS)."""
    def _run(run: Run) -> dict:
        env = dict(os.environ)
        env.setdefault("GA_EXTRA_BODY", '{"chat_template_kwargs":{"enable_thinking":false}}')
        for k, v in (run.params or {}).items():
            env[str(k).upper()] = str(v)
        run.emit("log", line=f"$ python examples/{name}.py")
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(EXAMPLES, f"{name}.py")], cwd=PKG, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        run.proc = proc
        for raw in iter(proc.stdout.readline, ""):
            if run.controls.get("stop"):
                proc.terminate()
                run.emit("log", line="■ stopped by operator")
                break
            if not run.gate():
                proc.terminate(); break
            line = raw.rstrip()
            if line:
                run.emit("log", line=line[:400])
                low = line.lower()
                if any(w in low for w in ("verdict", "wire", "resolved", "→", "learned", "proven", "gain")):
                    _store_knowledge(run, line[:300], {"domain": name, "kind": "script-finding"})
        proc.wait()
        return {"exit": proc.returncode, "verdict": f"{name} finished (exit {proc.returncode})"}
    return _run


import specs   # noqa: E402  — declarative use-case pipelines (add a DataSpec, get a pipeline)


def _discover(reserved) -> Dict[str, dict]:
    """Auto-register every runnable example as a script pipeline — the full framework capability surface."""
    meta: Dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "*.py"))):
        name = os.path.basename(path)[:-3]
        if name.startswith("_") or name in reserved:
            continue
        desc = ""
        try:
            desc = (ast.get_docstring(ast.parse(open(path).read())) or "").strip().split("\n")[0]
        except Exception:
            pass
        meta[name] = {"fn": _make_script_pipeline(name), "title": name.replace("_", " ").title(),
                      "kind": "script", "desc": desc[:160], "params": {}}
    return meta


# Core = hand-tuned NATIVE pipelines + the declarative use-case SPECS; then every example auto-discovered on top.
CORE: Dict[str, dict] = {**NATIVE, **specs.REGISTRY}
REGISTRY: Dict[str, dict] = {**CORE, **_discover(reserved=set(CORE))}
PIPELINES: Dict[str, Callable[[Run], dict]] = {k: v["fn"] for k, v in REGISTRY.items()}
CUSTOM_IDS: List[str] = []


def register_spec(d: dict, persist: bool = True) -> dict:
    """Register a NEW use-case pipeline at runtime from a DataSpec dict (id/title/desc/identity/goal/domain).
    It's the SAME prompt-free, engine-directed factory as the built-in specs — the UI just supplies the SELF."""
    sid = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(d["id"]).strip().lower()) or "use_case"
    spec = specs.DataSpec(id=sid, title=(d.get("title") or sid), desc=d.get("desc", ""),
                          identity=d["identity"], goal=d["goal"], domain=d.get("domain", "data"))
    meta = {"fn": specs.build(spec), "title": spec.title, "kind": "native", "desc": spec.desc,
            "params": {"kind": "auto", "goal": "", "data": "", "source": ""}}
    REGISTRY[sid], PIPELINES[sid] = meta, meta["fn"]
    if sid not in CUSTOM_IDS:
        CUSTOM_IDS.append(sid)
    if persist:
        rec = {"id": sid, "title": spec.title, "desc": spec.desc, "identity": spec.identity, "goal": spec.goal, "domain": spec.domain}
        specs.save_custom([x for x in specs.load_custom() if x.get("id") != sid] + [rec])
    return {"id": sid, "title": spec.title, "kind": "native", "desc": spec.desc, "params": meta["params"]}


def delete_spec(sid: str) -> bool:
    """Remove a user-created pipeline (built-ins are protected)."""
    if sid not in CUSTOM_IDS:
        return False
    REGISTRY.pop(sid, None)
    PIPELINES.pop(sid, None)
    CUSTOM_IDS.remove(sid)
    specs.save_custom([x for x in specs.load_custom() if x.get("id") != sid])
    return True


for _d in specs.load_custom():                         # restore user-created pipelines across restarts
    try:
        register_spec(_d, persist=False)
    except Exception:
        pass


def pipeline_info() -> list:
    """Everything launchable (native + declarative + user-created use-cases first, then the example suite)."""
    head = list(CORE) + [c for c in CUSTOM_IDS if c not in CORE]
    order = head + sorted(k for k in REGISTRY if k not in head)
    return [{"id": k, "title": REGISTRY[k]["title"], "desc": REGISTRY[k]["desc"], "kind": REGISTRY[k]["kind"],
             "params": REGISTRY[k]["params"], "custom": k in CUSTOM_IDS} for k in order]
