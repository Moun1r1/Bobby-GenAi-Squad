"use client";
import { useEffect, useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { chName } from "@/lib/events";
import { AVATAR_FG, hashIdx } from "@/lib/avatars";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

// A small pipeline diagram per workflow — the squad_solve shape (or the generic engine loop).
const STAGES: Record<string, string[]> = {
  goal: ["Goal", "Criteria", "Board · squad", "Verify", "Converge / Escalate"],
  idea_board: ["Mine", "Admit / repel", "Lifecycle", "Plateau"],
  process_data: ["Load units", "Read · squad", "Split dense", "Knowledge map"],
  persona: ["Persona", "Converse", "Remember"],
  world: ["World", "Agents", "Interact", "Events"],
  engine_trace: ["Self-select", "Plan", "Tool-grounded", "Record"],
  multi_day_service: ["Open ticket", "Operate tools", "Resolve", "Learn playbook"],
};
const stagesFor = (id: string) => STAGES[id] || ["Select target", "Plan", "Carry out", "Record", "Plateau"];

function Diagram({ id }: { id: string }) {
  const s = stagesFor(id);
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {s.map((n, i) => (
        <div key={i} className="flex items-center gap-1">
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[12px] text-slate-700">{n}</div>
          {i < s.length - 1 && <span className="text-slate-300">→</span>}
        </div>
      ))}
      <span className="text-slate-300 ml-1">↺</span>
    </div>
  );
}

function NewUseCase({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const create = trpc.createPipeline.useMutation({ onSuccess: (r: any) => { if (r?.ok) { onCreated(); onClose(); } } });
  const [f, setF] = useState({ id: "", title: "", identity: "", goal: "", domain: "data" });
  const set = (k: string, v: string) => setF({ ...f, [k]: v });
  const err = (create.data as any)?.ok === false ? (create.data as any).error : "";
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-900/30" onClick={onClose}>
      <div className="w-[520px] max-w-[92vw] bg-white rounded-2xl shadow-xl p-5" onClick={(e) => e.stopPropagation()}>
        <div className="text-[16px] font-semibold mb-1">New use-case pipeline</div>
        <div className="text-[12px] text-slate-500 mb-3">Define only the <b>SELF</b> — a role and a goal. The generative engine self-directs how (no prompt is written).</div>
        <div className="flex flex-col gap-2.5">
          <label className="text-[12px] text-slate-500">Name<input className="inp mt-1 text-[14px]" value={f.title} onChange={(e) => { set("title", e.target.value); if (!f.id) set("id", e.target.value.toLowerCase().replace(/[^a-z0-9]+/g, "_")); }} placeholder="Grade essays" /></label>
          <label className="text-[12px] text-slate-500">Identity (role)<input className="inp mt-1 text-[14px]" value={f.identity} onChange={(e) => set("identity", e.target.value)} placeholder="a fair grading assistant" /></label>
          <label className="text-[12px] text-slate-500">Goal (the task)<textarea className="inp mt-1 h-16 resize-none text-[14px]" value={f.goal} onChange={(e) => set("goal", e.target.value)} placeholder="read each essay and grade it 1-10 with a one-line rationale" /></label>
          <label className="text-[12px] text-slate-500">Domain<input className="inp mt-1 text-[14px]" value={f.domain} onChange={(e) => set("domain", e.target.value)} placeholder="grading" /></label>
        </div>
        {err && <div className="text-[12px] text-red-600 mt-2">{err}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" disabled={create.isPending || !f.identity.trim() || !f.goal.trim()} onClick={() => create.mutate({ ...f, id: f.id || f.title, desc: "" })}>{create.isPending ? "Creating…" : "Create pipeline"}</button>
        </div>
      </div>
    </div>
  );
}

export function WorkflowsPage({ onLaunched }: { onLaunched: (runId: string) => void }) {
  const pipelines = trpc.pipelines.useQuery();
  const launch = trpc.launch.useMutation({ onSuccess: (r: any) => onLaunched(r.run_id) });
  const del = trpc.deletePipeline.useMutation({ onSuccess: () => pipelines.refetch() });
  const [creating, setCreating] = useState(false);
  const all = (pipelines.data as any[]) || [];
  const native = all.filter((p) => p.kind === "native");
  const scripts = all.filter((p) => p.kind !== "native");
  const [sel, setSel] = useState<string>("");
  const chosen = all.find((p) => p.id === sel) || native[0];
  const [params, setParams] = useState<Record<string, any>>({});
  const [presets, setPresets] = useState<any[]>([]);
  useEffect(() => { if (chosen) { setSel(chosen.id); setParams(chosen.params || {}); } }, [chosen?.id]); // eslint-disable-line
  useEffect(() => { try { setPresets(JSON.parse(localStorage.getItem("bobby_presets") || "[]")); } catch { /* ignore */ } }, []);
  const savePreset = () => { const next = [{ name: chName(chosen.id), pipeline: chosen.id, params }, ...presets].slice(0, 12); setPresets(next); localStorage.setItem("bobby_presets", JSON.stringify(next)); };

  const Item = ({ p }: { p: any }) => (
    <button onClick={() => setSel(p.id)} className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-[13px] ${sel === p.id ? "bg-blue-50 text-blue-700" : "hover:bg-slate-50 text-slate-700"}`}>
      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: AVATAR_FG[hashIdx(p.id, AVATAR_FG.length)] }} />
      <span className="truncate flex-1">{chName(p.id)}</span>
      {p.kind === "native" && <Badge variant="outline">rich</Badge>}
    </button>
  );

  return (
    <div className="flex-1 flex min-w-0">
      {creating && <NewUseCase onClose={() => setCreating(false)} onCreated={() => pipelines.refetch()} />}
      <div className="w-[280px] shrink-0 bg-white border-r border-slate-200 flex flex-col p-3 overflow-auto">
        <div className="flex items-center gap-2 mb-1 px-1"><div className="text-[17px] font-bold flex-1">Workflows</div>
          <button className="btn btn-primary text-[12px] py-1" onClick={() => setCreating(true)}>＋ New</button></div>
        <div className="text-[12px] text-slate-400 mb-3 px-1">{all.length} capabilities · configure &amp; launch</div>
        <div className="text-[11px] font-semibold uppercase text-slate-400 px-1 mb-1">Native</div>
        {native.map((p) => <Item key={p.id} p={p} />)}
        <div className="text-[11px] font-semibold uppercase text-slate-400 px-1 mt-3 mb-1">Examples · {scripts.length}</div>
        {scripts.map((p) => <Item key={p.id} p={p} />)}
      </div>

      <div className="flex-1 overflow-auto bg-slate-50 p-6">
        {chosen ? (
          <div className="max-w-[820px] mx-auto flex flex-col gap-4">
            <div>
              <div className="flex items-center gap-2"><h1 className="text-[22px] font-bold">{chName(chosen.id)}</h1>{chosen.kind === "native" && <Badge variant="active">rich pipeline</Badge>}{chosen.custom && <Badge variant="outline">custom</Badge>}
              {chosen.custom && <button className="btn text-[12px] py-1 text-red-600 ml-auto" onClick={() => { if (confirm(`Delete pipeline "${chosen.title}"?`)) del.mutate({ id: chosen.id }); }}>Delete</button>}</div>
              <p className="text-[14px] text-slate-500 mt-1">{chosen.desc || "A runnable squad workflow."}</p>
            </div>
            <Card>
              <CardHeader><CardTitle>Pipeline</CardTitle></CardHeader>
              <CardContent className="overflow-auto"><Diagram id={chosen.id} /></CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Parameters</CardTitle></CardHeader>
              <CardContent>
                {Object.keys(params).length ? (
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(params).map(([k, v]) => (
                      <label key={k} className="text-[12px] text-slate-500">{k}
                        <input className="inp mt-1 text-[13px]" value={String(v)} onChange={(e) => setParams({ ...params, [k]: e.target.value })} />
                      </label>
                    ))}
                  </div>
                ) : <div className="text-[13px] text-slate-400">No parameters — launch as-is.</div>}
                <Separator className="my-4" />
                <div className="flex gap-2">
                  <button className="btn btn-primary" disabled={launch.isPending} onClick={() => launch.mutate({ pipeline: chosen.id, params })}>{launch.isPending ? "Starting…" : "▶ Launch workflow"}</button>
                  <button className="btn" onClick={savePreset}>Save preset</button>
                </div>
              </CardContent>
            </Card>
            {presets.length > 0 && (
              <Card>
                <CardHeader><CardTitle>Saved presets</CardTitle></CardHeader>
                <CardContent className="flex flex-wrap gap-2">
                  {presets.map((p, i) => <button key={i} className="btn text-[12px]" onClick={() => launch.mutate({ pipeline: p.pipeline, params: p.params })}>▶ {p.name}</button>)}
                </CardContent>
              </Card>
            )}
          </div>
        ) : <div className="grid place-items-center h-full text-slate-400">Select a workflow</div>}
      </div>
    </div>
  );
}
