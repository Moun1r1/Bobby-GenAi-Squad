"use client";
import { useEffect, useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { chName, isOver, type Ev } from "@/lib/events";
import { useRunEvents } from "@/hooks/useRunEvents";
import { Avatar } from "@/components/ui/Avatar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

function agentsOf(events: Ev[]) {
  const m = new Map<string, any>();
  for (const e of events) {
    const a = e.agent as string | undefined;
    if (!a) continue;
    const c = m.get(a) || { name: a, tools: 0, moves: 0 };
    if (e.kind === "target") c.target = e.target;
    if (e.kind === "move_start") { c.move = e.move; c.intention = e.intention; c.moves++; }
    if (e.kind === "move_end") c.move = undefined;
    if (e.kind === "tool") c.tools++;
    if (e.kind === "memory") { c.pinned = e.pinned_items; c.pinnedTok = e.pinned_tokens; }
    m.set(a, c);
  }
  return [...m.values()];
}

export function SquadPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 3000 });
  const experts = trpc.experts.useQuery(undefined, { refetchInterval: 8000 });
  const launch = trpc.launch.useMutation({ onSuccess: () => runsQ.refetch() });
  const runList = (runsQ.data as any[]) || [];
  const liveRuns = runList.filter((r) => r.status !== "done" && r.status !== "error");
  const [runId, setRunId] = useState<string | null>(null);
  useEffect(() => { if (!runId && liveRuns[0]) setRunId(liveRuns[0].run_id); }, [liveRuns.length]); // eslint-disable-line
  const events = useRunEvents(runId);
  const roster = useMemo(() => agentsOf(events), [events]);
  const live = !!runId && !isOver(events);
  const exp = (experts.data as any[]) || [];

  const [goal, setGoal] = useState("");
  const [agents, setAgents] = useState(3);

  return (
    <div className="flex-1 overflow-auto bg-slate-50">
      <div className="max-w-[1000px] mx-auto p-6 flex flex-col gap-4">
        <div><h1 className="text-[22px] font-bold">Squad</h1><p className="text-[14px] text-slate-500">Assemble a self-organizing squad, watch it work, and reuse the specialists it becomes.</p></div>

        <Card>
          <CardHeader><CardTitle>Assemble a squad</CardTitle></CardHeader>
          <CardContent className="flex flex-col gap-3">
            <textarea className="inp h-20 resize-none text-[14px]" value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="Shared goal for the squad — e.g. Map this contract's risks and cite each…" />
            <div className="flex items-center gap-3">
              <label className="text-[13px] text-slate-600 flex items-center gap-2" title="The swarm self-scales its headcount to the workload; this only caps it.">Max agents (cap)
                <input type="range" min={2} max={8} value={agents} onChange={(e) => setAgents(+e.target.value)} className="accent-blue-600" />
                <span className="font-semibold w-4">{agents}</span>
              </label>
              <span className="text-[11px] text-slate-400">the squad self-scales to the goal — this only bounds it</span>
              <button className="btn btn-primary ml-auto" disabled={launch.isPending || !goal.trim()} onClick={() => launch.mutate({ pipeline: "goal", params: { goal, max_agents: agents } })}>Assemble &amp; run</button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center gap-2">
            <CardTitle className="flex-1">Live roster</CardTitle>
            <select className="inp py-1 w-[220px] text-[12px]" value={runId || ""} onChange={(e) => setRunId(e.target.value)}>
              <option value="">— pick a run —</option>
              {runList.slice(0, 20).map((r) => <option key={r.run_id} value={r.run_id}>{chName(r.pipeline)} · {r.run_id.slice(0, 6)} · {r.status}</option>)}
            </select>
            {runId && <Badge variant={live ? "proven" : "dead"}>{live ? "live" : "done"}</Badge>}
          </CardHeader>
          <CardContent>
            {roster.length ? (
              <div className="grid grid-cols-2 gap-3">
                {roster.map((a) => (
                  <div key={a.name} className="rounded-lg border border-slate-200 p-3">
                    <div className="flex items-center gap-2">
                      <Avatar name={a.name} size={32} />
                      <div className="flex-1"><div className="text-[13px] font-medium">{a.name}</div><div className="text-[11px]" style={{ color: a.move ? "#2563eb" : "#94a3b8" }}>{a.move ? `◆ ${a.move}` : "idle"}</div></div>
                      <Badge variant="outline">{a.moves} moves · {a.tools}t</Badge>
                    </div>
                    {a.target && <div className="text-[11px] text-slate-500 mt-1.5 truncate">🎯 {a.target}</div>}
                    {a.pinned != null && <div className="text-[11px] text-slate-400 mt-0.5">pinned memory · {a.pinned} items ({a.pinnedTok} tok)</div>}
                  </div>
                ))}
              </div>
            ) : <div className="text-[13px] text-slate-400">{runId ? "no agents in this run yet…" : "pick a run to inspect its live squad."}</div>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Specialist library · {exp.length}</CardTitle></CardHeader>
          <CardContent>
            {exp.length ? (
              <div className="grid grid-cols-3 gap-2">
                {exp.map((e) => <div key={e.id} className="rounded-lg border border-slate-200 p-3"><div className="flex items-center gap-1.5"><div className="text-[13px] font-medium flex-1 truncate">{e.specialty || e.name}</div>{e.kind === "area" && <span className="text-[10px] px-1.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200">area</span>}</div><div className="text-[11px] text-slate-400">{e.area ? `${e.area} · ` : ""}{e.n_knowledge} knowledge</div></div>)}
              </div>
            ) : <div className="text-[13px] text-slate-400">No specialists yet — a completed goal squad crystallizes reusable experts here.</div>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
