"""app — Bobby Studio backend (FastAPI). Keeps the whole engine in Python and exposes it over a clean HTTP + SSE API
that the tRPC frontend consumes for real live rendering.

Endpoints
  GET  /health                      — backend + store status
  GET  /pipelines                   — the runnable squad pipelines
  POST /runs           {pipeline,params} — launch a run (returns run_id)
  GET  /runs                        — list past runs (from the vector store)
  GET  /runs/{id}                   — one run's status + summary + events so far
  GET  /runs/{id}/stream            — SSE live event stream (target/plan/move/tool/cycle/day/done)
  GET  /search?q=&collection=       — semantic search over stored knowledge / runs (the vector DB payoff)

Live rendering uses Server-Sent Events; the tRPC subscription on the frontend attaches to /runs/{id}/stream, so the
browser gets a typed live feed without us hand-rolling a socket protocol.
"""
import asyncio
import json
import os
import queue
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastapi import FastAPI                          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware   # noqa: E402
from fastapi.responses import StreamingResponse      # noqa: E402
from pydantic import BaseModel                        # noqa: E402

from store import get_store                           # noqa: E402
import runner                                         # noqa: E402

app = FastAPI(title="Bobby Studio", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LaunchReq(BaseModel):
    pipeline: str
    params: dict = {}


@app.get("/health")
def health():
    return {"ok": True, "store": get_store().backend, "pipelines": len(runner.PIPELINES)}


@app.get("/vault/stats")
def vault_stats():
    """All vaults at a glance — per-vault note/edge counts + totals + the sources they've grown from."""
    return runner._get_hub().stats()


@app.get("/vault/list")
def vault_list():
    """The vault names — the swarm creates new ones dynamically as it learns new domains."""
    return {"vaults": runner._get_hub().names()}


@app.post("/vault/reload")
@app.get("/vault/reload")
def vault_reload():
    """Hot-reload the vaults from disk — pick up hand-edited / externally-added notes without a backend restart.
    (Reads also auto-reload on a throttle; this forces it immediately.)"""
    return runner._get_hub().reload()


@app.get("/vault/graph")
def vault_graph():
    """Every vault as one navigable graph: nodes (vault·id·title·tags·source) + directed [[vault/note]] edges (incl.
    CROSS-VAULT links) — for the UI to render and let you hop across vaults."""
    return runner._get_hub().graph()


@app.get("/vault/note/{note_id:path}")
def vault_note(note_id: str):
    """One note's full markdown + its cross-vault neighbours (links ∪ backlinks from any vault). id = `vault/note`."""
    n = runner._get_hub().note(note_id)
    return n or {"error": "not found", "id": note_id}


@app.get("/vault/navigate")
def vault_navigate(q: str, k: int = 3, hops: int = 1):
    """Preview what an agent would recall for a query — cross-vault semantic entry + link-hop subgraph, as injected."""
    hub = runner._get_hub()
    return {"query": q, "entry": hub.search(q, k=k), "block": hub.navigate(q, per_vault_k=k, hops=hops)}


@app.get("/dgx/health")
def dgx_health():
    """Realtime DGX snapshot — GPU (util/VRAM/temp/power + procs) · CPU/RAM · disk · running docker sessions."""
    from dgx_monitor import get_monitor
    return get_monitor().snapshot()


@app.get("/dgx/safe")
def dgx_safe():
    """Pre-train GATE: is the DGX safe to train on right now (enough GPU VRAM + headroom)?"""
    from dgx_monitor import get_monitor
    return get_monitor().is_safe()


@app.get("/dgx/stream")
async def dgx_stream():
    """SSE realtime feed of the DGX snapshot (poll every ~3s) — watch GPU/CPU/disk/docker live before/while training."""
    from dgx_monitor import get_monitor
    import asyncio
    import json as _j

    async def gen():
        mon = get_monitor()
        while True:
            yield f"data: {_j.dumps(mon.snapshot(max_age=2))}\n\n"
            await asyncio.sleep(3)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/pipelines")
def pipelines():
    return runner.pipeline_info()   # native rich-panel pipelines + the entire auto-discovered example suite


class SpecReq(BaseModel):
    id: str
    title: str = ""
    desc: str = ""
    identity: str            # the SELF — a generic role
    goal: str                # the user's task; the engine self-directs (no prompt)
    domain: str = "data"


@app.post("/pipelines/spec")
def create_spec(req: SpecReq):
    """Create a NEW use-case pipeline from the UI — pure SELF (role + goal), the engine-directed factory does the rest."""
    try:
        if not req.identity.strip() or not req.goal.strip():
            return {"ok": False, "error": "identity and goal are required"}
        return {"ok": True, "pipeline": runner.register_spec(req.dict())}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.delete("/pipelines/{pid}")
def delete_pipeline(pid: str):
    """Remove a user-created pipeline (built-ins are protected)."""
    return {"ok": runner.delete_spec(pid)}


CONFIG_FILE = os.path.join(HERE, "config.json")


class ConfigReq(BaseModel):
    agents: int = 3
    patience: int = 2
    max_units: int = 60


@app.get("/config")
def get_config():
    try:
        return json.load(open(CONFIG_FILE))
    except Exception:
        return ConfigReq().dict()


@app.post("/config")
def set_config(req: ConfigReq):
    try:
        json.dump(req.dict(), open(CONFIG_FILE, "w"))
    except Exception:
        pass
    return {"ok": True, "config": req.dict()}


@app.post("/runs")
def launch(req: LaunchReq):
    if req.pipeline not in runner.PIPELINES:
        return {"error": f"unknown pipeline {req.pipeline}", "have": list(runner.PIPELINES)}
    r = runner.launch(req.pipeline, req.params)
    return {"run_id": r.id, "pipeline": r.pipeline, "status": r.status}


@app.get("/runs")
def list_runs():
    live = [{"run_id": r.id, "pipeline": r.pipeline, "status": r.status, "n_events": r.seq,
             "summary": r.summary, "ts": r.started} for r in runner.RUNS.values()]
    stored = get_store().list("runs", limit=200)
    seen = {r["run_id"] for r in live}
    return {"runs": live + [s for s in stored if s.get("run_id") not in seen]}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    r = runner.RUNS.get(run_id)
    if r is not None:
        return {"run_id": r.id, "pipeline": r.pipeline, "status": r.status, "summary": r.summary,
                "events": r.events}
    stored = get_store().get_run(run_id)
    events = get_store().list("events", limit=2000, run_id=run_id)
    return {"run_id": run_id, "status": (stored or {}).get("status", "unknown"),
            "summary": stored or {}, "events": events}


@app.get("/runs/{run_id}/stream")
async def stream(run_id: str):
    """SSE: replay events already produced, then follow live until the run ends."""
    r = runner.RUNS.get(run_id)

    async def gen():
        sent = 0
        if r is None:                                   # finished run → replay from the store, then close
            for ev in get_store().list("events", limit=5000, run_id=run_id):
                yield f"data: {json.dumps(ev)}\n\n"
            yield "event: end\ndata: {}\n\n"
            return
        while True:
            # flush backlog first
            while sent < len(r.events):
                yield f"data: {json.dumps(r.events[sent])}\n\n"
                sent += 1
            try:
                ev = r.q.get(timeout=0.5)
                if ev.get("kind") == "done":
                    yield f"data: {json.dumps(ev)}\n\n"
                    yield "event: end\ndata: {}\n\n"
                    return
            except queue.Empty:
                if r.status in ("done", "error"):
                    yield "event: end\ndata: {}\n\n"
                    return
                yield ": keepalive\n\n"                  # comment frame keeps the connection warm
            await asyncio.sleep(0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class ControlReq(BaseModel):
    action: str            # stop | pause | resume | steer
    text: str = ""


@app.post("/runs/{run_id}/control")
def control(run_id: str, req: ControlReq):
    """Steer a LIVE squad: stop it, pause/resume it, or inject a human directive its agents pick up next step."""
    r = runner.RUNS.get(run_id)
    if r is None:
        return {"error": "no such live run"}
    if req.action == "stop":
        r.controls["stop"] = True
        if getattr(r, "proc", None) is not None:
            try:
                r.proc.terminate()          # kill a script pipeline's subprocess immediately
            except Exception:
                pass
    elif req.action == "pause":
        r.controls["pause"] = True
    elif req.action == "resume":
        r.controls["pause"] = False
    elif req.action == "steer" and req.text.strip():
        r.steer.append(req.text.strip())
    else:
        return {"error": f"unknown action {req.action}"}
    return {"ok": True, "status": r.status, "controls": r.controls, "pending_steer": len(r.steer)}


class BoardReq(BaseModel):
    label: str             # match an idea by label prefix
    state: str             # the emergent state to assign


@app.post("/runs/{run_id}/board")
def manage_board(run_id: str, req: BoardReq):
    """Manage the IdeaLedger of a live idea_board run — assign an emergent state to an idea (the human curates)."""
    r = runner.RUNS.get(run_id)
    if r is None or r.ledger is None:
        return {"error": "no live board for this run"}
    hit = next((it for it in r.ledger.ideas if it["label"].startswith(req.label[:40])), None)
    if not hit:
        return {"error": "no matching idea"}
    r.ledger.set_state(hit, req.state, by="operator")
    return {"ok": True, "label": hit["label"], "state": hit["status"]}


@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    runner.RUNS.pop(run_id, None)
    get_store().delete_run(run_id)
    return {"ok": True, "deleted": run_id}


class KnowledgeDelReq(BaseModel):
    key: str


@app.post("/knowledge/delete")
def delete_knowledge(req: KnowledgeDelReq):
    return {"ok": get_store().delete("knowledge", req.key)}


@app.get("/search")
def search(q: str, collection: str = "knowledge", limit: int = 10):
    return {"query": q, "collection": collection, "hits": get_store().search(collection, q, limit=limit)}


@app.get("/primitives")
def primitives():
    """The self-extending primitive library: a category tree + per-primitive proof metadata (the distilled cognitive
    stdlib the ACR flywheel builds — reused across runs, never re-learned)."""
    return runner.primitive_registry()


@app.get("/primitives/recall")
def primitives_recall(q: str, k: int = 5):
    """Find a primitive BACK from a task description (semantic memory) — what the engine consults before re-distilling."""
    return runner.primitive_recall(q, k=k)


@app.get("/stats")
def stats():
    """FRAMEWORK-LEVEL state — aggregate across ALL runs (so the console is never empty, per-run or not)."""
    s = get_store()
    runs = s.list("runs", limit=2000)
    know = s.list("knowledge", limit=5000)
    by_pipeline: dict = {}
    for r in runs:
        by_pipeline[r.get("pipeline", "?")] = by_pipeline.get(r.get("pipeline", "?"), 0) + 1
    by_domain: dict = {}
    for k in know:
        dm = k.get("domain") or k.get("workflow") or k.get("pipeline") or "misc"
        by_domain[dm] = by_domain.get(dm, 0) + 1
    return {"store": s.backend, "runs": len(runs), "knowledge": len(know), "pipelines": len(runner.PIPELINES),
            "by_pipeline": by_pipeline, "by_domain": by_domain}


@app.get("/memory/policy")
def memory_policy():
    """The latest EVOLVED memory-policy snapshot (SemanticMemory policy='value'): what the self-governing store keeps
    vs evicts, value-ranked. Surfaces the +25%-retention mechanism the pipelines now run on."""
    rows = get_store().list("mem_policy", limit=50)
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return rows[0] if rows else {}


@app.get("/experts")
def experts(limit: int = 100):
    """Every crystallized specialist the squads produced — the reusable experts (persona-from-data)."""
    rows = get_store().list("experts", limit=limit)
    for r in rows:
        r.pop("knowledge", None)                       # list view: metadata only
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return {"experts": rows}


@app.get("/experts/{expert_id:path}")
def expert(expert_id: str):
    """One expert with its full accumulated knowledge — inspect / exploit it."""
    e = next((x for x in get_store().list("experts", limit=500) if x.get("id") == expert_id), None)
    return e or {"error": "no such expert"}


@app.get("/knowledge/scatter")
def knowledge_scatter(limit: int = 400, collection: str = "knowledge"):
    """2D projection of the stored embeddings — the vector-memory map (geometry only, no model/prompt)."""
    return {"collection": collection, "points": get_store().scatter(collection, limit=limit)}


@app.get("/proofs")
def proofs(run: bool = False):
    """The framework's REAL gain-proofs (WIRE/MARGINAL/DELETE + negative-control + CI), run via the prove primitive.
    ?run=true triggers a background run; poll without it to read cached verdicts."""
    if run:
        runner.run_proofs()
    return runner.proofs_state()


@app.get("/notebook")
def notebook():
    """The cross-run research notebook — the rd_lab RD_TRACE file if present, plus the accumulated findings corpus.
    Read-only; surfaces what the engine already wrote (no generation)."""
    md = None
    try:
        nb = os.path.join(runner.PKG, "out", "RD_LAB_NOTEBOOK.md")
        if os.path.isfile(nb):
            md = open(nb, errors="ignore").read()[:40000]
    except Exception:
        md = None
    items = get_store().list("knowledge", limit=2000)
    items.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return {"markdown": md, "items": items}


@app.get("/knowledge")
def knowledge(limit: int = 200, domain: str = ""):
    """Browse ALL learned knowledge/lessons persisted across every run (the reusability substrate)."""
    rows = get_store().list("knowledge", limit=limit)
    if domain:
        rows = [r for r in rows if (r.get("domain") or r.get("workflow") or r.get("pipeline")) == domain]
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return {"items": rows}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
