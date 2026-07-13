"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { useRunEvents } from "@/hooks/useRunEvents";
import { PageShell } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

function Bar({ used, total, unit = "MB", warn = 0.85 }: { used?: number; total?: number; unit?: string; warn?: number }) {
  if (used == null || total == null || !total) return <span className="text-[12px] text-slate-400">—</span>;
  const pct = Math.min(100, Math.round((used / total) * 100));
  const col = pct >= warn * 100 ? "bg-rose-500" : pct >= 60 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div>
      <div className="h-2 rounded-full bg-slate-100 overflow-hidden"><div className={`h-full ${col}`} style={{ width: `${pct}%` }} /></div>
      <div className="text-[11px] text-slate-500 mt-0.5 tabular-nums">{used.toLocaleString()} / {total.toLocaleString()} {unit} · {pct}%</div>
    </div>
  );
}

function DgxMonitor() {
  const q = trpc.dgxHealth.useQuery(undefined, { refetchInterval: 3000 });
  const safe = trpc.dgxSafe.useQuery(undefined, { refetchInterval: 5000 });
  const d = q.data as any;
  const s = safe.data as any;
  if (!d) return <Card className="mb-4"><CardContent className="p-4 text-[13px] text-slate-400">Connecting to DGX…</CardContent></Card>;
  if (!d.ok) return <Card className="mb-4"><CardContent className="p-4 text-[13px] text-rose-500">DGX unreachable: {d.error}</CardContent></Card>;
  const g = d.gpu || {}, ram = d.ram || {}, cpu = d.cpu || {}, disk = d.disk || {};
  return (
    <Card className="mb-4">
      <CardHeader className="flex-row items-center gap-2">
        <CardTitle className="flex-1">DGX · realtime</CardTitle>
        {s && <Badge variant={s.safe ? "proven" : "error"}>{s.safe ? "safe to train" : `blocked: ${s.reason}`}</Badge>}
        <span className="text-[11px] text-slate-400">refresh 3s</span>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <div className="text-[11px] font-semibold uppercase text-slate-400 mb-1">GPU · {g.name || "?"}</div>
            <div className="text-2xl font-bold tabular-nums">{g.util_pct ?? "—"}<span className="text-[13px] text-slate-400">% util</span></div>
            <div className="text-[11px] text-slate-500">{g.temp_c ?? "—"}°C · {g.power_w ?? "—"}W{g.power_limit_w ? ` / ${g.power_limit_w}W` : ""}</div>
          </div>
          <div>
            <div className="text-[11px] font-semibold uppercase text-slate-400 mb-1">Memory {d.gpu?.mem_total_mb ? "(VRAM)" : "(unified)"}</div>
            <Bar used={ram.used_mb ?? g.mem_used_mb} total={ram.total_mb ?? g.mem_total_mb} />
          </div>
          <div>
            <div className="text-[11px] font-semibold uppercase text-slate-400 mb-1">CPU · {cpu.cores ?? "?"} cores</div>
            <div className="text-2xl font-bold tabular-nums">{cpu.load_pct ?? "—"}<span className="text-[13px] text-slate-400">%</span></div>
            <div className="text-[11px] text-slate-500">load {cpu.load1 ?? "—"} / {cpu.load5 ?? "—"}</div>
          </div>
          <div>
            <div className="text-[11px] font-semibold uppercase text-slate-400 mb-1">Disk</div>
            <div className="text-2xl font-bold tabular-nums">{disk.use_pct ?? "—"}<span className="text-[13px] text-slate-400">%</span></div>
            <div className="text-[11px] text-slate-500">{disk.avail || "—"} free / {disk.size || "—"}</div>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] font-semibold uppercase text-slate-400 mr-1">Sessions</span>
          {(d.docker || []).map((c: any) => (
            <span key={c.name} className={`text-[11px] px-2 py-0.5 rounded-full border ${c.name === "ga_worker" ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-500"}`}>{c.name}</span>
          ))}
          {(d.gpu_procs || []).length > 0 && <span className="text-[11px] text-slate-400 ml-2">GPU procs: {(d.gpu_procs).map((p: any) => `${p.name}(${p.mem_mb}MB)`).join(", ")}</span>}
        </div>
      </CardContent>
    </Card>
  );
}

export function ComputePage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 3000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);

  const dgx = useMemo(() => events.filter((e) => e.kind === "dgx"), [events]);
  const waves = useMemo(() => events.filter((e) => e.kind === "wave"), [events]);
  const introspects = useMemo(() => events.filter((e) => e.kind === "introspect"), [events]);
  const dpoPairs = useMemo(() => events.filter((e) => e.kind === "dpo_pair"), [events]);
  const dpoResult = useMemo(() => [...events].reverse().find((e) => e.kind === "dpo_result"), [events]);
  const worldResult = useMemo(() => [...events].reverse().find((e) => e.kind === "world_result"), [events]);
  const encoderResult = useMemo(() => [...events].reverse().find((e) => ["self_model_result", "value_result", "retrieval_result", "traj_result"].includes(e.kind as string)), [events]);
  const dpoHarvest = useMemo(() => [...events].reverse().find((e) => e.kind === "dpo_harvest" && e.source === "trajectory"), [events]);
  const challenge = useMemo(() => [...events].reverse().find((e) => e.kind === "challenge"), [events]);
  const gate = useMemo(() => [...events].reverse().find((e) => e.kind === "dgx_gate"), [events]);
  const lastTree = useMemo(() => { const e = [...events].reverse().find((x) => x.kind === "dev" && x.tree); return (e?.tree as string) || ""; }, [events]);
  const dgxCounts = useMemo(() => dgx.reduce((m: Record<string, number>, e) => ((m[e.action as string] = (m[e.action as string] || 0) + 1), m), {}), [dgx]);

  return (
    <PageShell title="Compute" desc="The GPU worker (isolated, memory-capped) in realtime — foundation training + code-dev, watched live." wide
      right={<RunSelect runList={runList} value={runId} onChange={setRunId} only={["code_dev", "train", "self_dpo", "world_layer", "value_head", "retrieval_encoder", "trajectory_monitor", "perception", "self_model"]} />}>
      <DgxMonitor />

      {encoderResult && (
        <Card className="mb-4">
          <CardHeader className="flex-row items-center gap-2">
            <CardTitle className="flex-1">Encoder bank · world hub conditions value + self-monitor</CardTitle>
            <Badge variant={encoderResult.passed ? "proven" : "warn"}>{encoderResult.passed ? "held-out passed ✓" : "trained"}</Badge>
            {dpoHarvest && <Badge variant="outline">auto-DPO: {dpoHarvest.n as number} pairs ({(dpoHarvest.reasons as string[] || []).join(", ")})</Badge>}
          </CardHeader>
          <CardContent>
            <pre className="text-[11px] font-mono bg-slate-50 rounded-lg p-3 max-h-56 overflow-auto whitespace-pre-wrap text-slate-700">{(encoderResult.output as string || "").slice(-1000)}</pre>
          </CardContent>
        </Card>
      )}

      {worldResult && (
        <Card className="mb-4">
          <CardHeader className="flex-row items-center gap-2">
            <CardTitle className="flex-1">World transformer layer · feed state as embeddings, not chat</CardTitle>
            <Badge variant={worldResult.passed ? "proven" : "warn"}>{worldResult.passed ? "world tokens beat no-world ✓" : "trained"}</Badge>
            <span className="text-[11px] text-slate-400">{worldResult.n as number} vault examples</span>
          </CardHeader>
          <CardContent>
            <pre className="text-[11px] font-mono bg-slate-50 rounded-lg p-3 max-h-56 overflow-auto whitespace-pre-wrap text-slate-700">{(worldResult.output as string || "").slice(-1200)}</pre>
          </CardContent>
        </Card>
      )}

      {dpoPairs.length > 0 && (
        <Card className="mb-4">
          <CardHeader className="flex-row items-center gap-2">
            <CardTitle className="flex-1">Self-DPO flywheel · meta-cognition → preference pairs</CardTitle>
            {dpoResult && <Badge variant={dpoResult.passed ? "proven" : "warn"}>{dpoResult.passed ? "DPO loss dropped ✓" : "DPO ran"}</Badge>}
            <span className="text-[11px] text-slate-400">{dpoPairs.filter((p) => p.improved).length}/{dpoPairs.length} improvable</span>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 max-h-96 overflow-auto">
            {dpoPairs.slice(-10).reverse().map((p, i) => (
              <div key={i} className="rounded-lg border border-slate-200 p-2.5">
                <div className="flex items-center gap-2 mb-1"><Badge variant="outline">{p.pattern as string}</Badge><span className="text-[12px] text-slate-500 truncate">{p.task as string}</span></div>
                <div className="text-[11px] text-slate-600 mb-1.5"><b>critique:</b> {(p.critique as string || "").slice(0, 240)}</div>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="rounded bg-emerald-50/50 border border-emerald-200 p-1.5"><div className="text-emerald-700 font-semibold mb-0.5">chosen ≻</div><div className="text-slate-700 whitespace-pre-wrap line-clamp-4">{(p.chosen as string || "").slice(0, 200)}</div></div>
                  <div className="rounded bg-rose-50/40 border border-rose-200 p-1.5"><div className="text-rose-600 font-semibold mb-0.5">rejected</div><div className="text-slate-600 whitespace-pre-wrap line-clamp-4">{(p.rejected as string || "").slice(0, 200)}</div></div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {!runId ? (
        <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-[13px] text-slate-500">
          Pick a <b>code_dev</b> or <b>train</b> run to watch its live progress — the swarm writes code/training, pushes to the worker, runs it, and self-challenges on a real metric.
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          <div className="col-span-2 flex flex-col gap-3">
            <Card>
              <CardHeader className="flex-row items-center gap-2">
                <CardTitle className="flex-1">Live progress</CardTitle>
                {gate && <Badge variant={gate.safe ? "proven" : "error"}>gate: {gate.safe ? "safe" : gate.reason}</Badge>}
                {challenge && <Badge variant={challenge.passed ? "proven" : "warn"}>{challenge.passed ? "CHALLENGE PASS" : "challenge failing"}</Badge>}
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4 text-[12px] text-slate-600 mb-2">
                  <span>{waves.length} waves</span>
                  <span>· dgx actions {Object.entries(dgxCounts).map(([k, v]) => `${k}:${v}`).join("  ") || "—"}</span>
                </div>
                <div className="text-[11px] font-semibold uppercase text-slate-400 mb-1">Worker sandbox (files)</div>
                <pre className="text-[11px] font-mono bg-slate-50 rounded-lg p-2.5 max-h-40 overflow-auto whitespace-pre-wrap">{lastTree || "(empty)"}</pre>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>DGX actions · live</CardTitle></CardHeader>
              <CardContent className="flex flex-col gap-1 max-h-72 overflow-auto">
                {dgx.slice(-30).reverse().map((e, i) => (
                  <div key={i} className="flex items-center gap-2 text-[12px] py-1 border-b border-slate-50 last:border-0">
                    <Badge variant={e.exit === 0 ? "proven" : e.exit == null ? "outline" : "error"}>{e.action}</Badge>
                    <span className="flex-1 truncate font-mono text-slate-600">{(e.cmd || e.path || e.job || e.image || "") as string}</span>
                    {e.exit != null && <span className={`text-[11px] ${e.exit === 0 ? "text-emerald-600" : "text-rose-500"}`}>exit {e.exit}</span>}
                  </div>
                ))}
                {!dgx.length && <div className="text-[12px] text-slate-400">no GPU actions yet — the swarm is preparing.</div>}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>Self-understanding <span className="text-[11px] font-normal text-slate-400">introspect</span></CardTitle></CardHeader>
            <CardContent className="flex flex-col gap-2 max-h-[520px] overflow-auto">
              {introspects.slice(-8).reverse().map((e, i) => (
                <div key={i} className="rounded-lg border border-violet-200 bg-violet-50/40 p-2.5">
                  <div className="text-[11px] font-medium text-violet-700 mb-0.5">{e.agent as string}</div>
                  <div className="text-[12px] text-slate-700 whitespace-pre-wrap">{(e.understanding as string || "").slice(0, 500)}</div>
                </div>
              ))}
              {!introspects.length && <div className="text-[12px] text-slate-400">The agent introspects its own behavior when it stalls — its understanding streams here.</div>}
            </CardContent>
          </Card>
        </div>
      )}
    </PageShell>
  );
}
