"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { boardIdeas } from "@/lib/events";
import { useRunEvents } from "@/hooks/useRunEvents";
import { PageShell } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";

const STATE_COLOR: Record<string, string> = { open: "#64748b", saturated: "#d97706", proven: "#16a34a", verified: "#16a34a", contested: "#e11d48", dead: "#94a3b8", merged: "#64748b", promising: "#d97706", ready: "#16a34a", blocked: "#e11d48" };
const col = (s?: string) => STATE_COLOR[(s || "open").toLowerCase()] || "#64748b";

export function IdeaBoardPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 4000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);
  const ideas = useMemo(() => boardIdeas(events), [events]);
  const byState = useMemo(() => { const m: Record<string, typeof ideas> = {}; for (const it of ideas) (m[it.state] ||= []).push(it); return m; }, [ideas]);
  const unexplored = useMemo(() => { const e = [...events].reverse().find((x) => x.kind === "board" && x.areas_unexplored); return (e?.areas_unexplored as string[]) || []; }, [events]);
  const untested = useMemo(() => { const e = [...events].reverse().find((x) => x.kind === "board" && x.untested != null); return (e?.untested as number) ?? 0; }, [events]);
  const unchallenged = useMemo(() => { const e = [...events].reverse().find((x) => x.kind === "board" && x.unchallenged != null); return (e?.unchallenged as number) ?? 0; }, [events]);

  return (
    <PageShell title="Idea board" desc="The IdeaLedger: findings cluster into ideas with an emergent lifecycle; near-dups are repelled by the identity floor." wide
      right={<RunSelect runList={runList} value={runId} onChange={setRunId} only={["idea_board", "goal"]} />}>
      {unexplored.length > 0 && (
        <div className="mb-3 rounded-lg border border-slate-200 bg-white px-3 py-2 flex items-center gap-2 flex-wrap">
          <span className="text-[11px] font-semibold uppercase text-slate-400">Area frontier · unexplored</span>
          {unexplored.map((a) => <span key={a} className="pill px-2 py-0.5 text-[11px] rounded-full border border-dashed border-slate-300 text-slate-500">◌ {a}</span>)}
          <span className="ml-auto flex items-center gap-3">
            {untested > 0 && <span className="text-[11px] text-amber-600 font-medium">🧪 {untested} without a test</span>}
            {unchallenged > 0 && <span className="text-[11px] text-rose-600 font-medium">⚔ {unchallenged} not red-teamed</span>}
          </span>
        </div>
      )}
      {unexplored.length === 0 && (untested > 0 || unchallenged > 0) && (
        <div className="mb-3 flex items-center gap-3">
          {untested > 0 && <span className="text-[11px] text-amber-600 font-medium">🧪 {untested} idea{untested > 1 ? "s" : ""} without a falsifiable test</span>}
          {unchallenged > 0 && <span className="text-[11px] text-rose-600 font-medium">⚔ {unchallenged} idea{unchallenged > 1 ? "s" : ""} not red-teamed</span>}
        </div>
      )}
      {ideas.length ? (
        <div className="flex gap-3 overflow-x-auto pb-2">
          {Object.entries(byState).map(([state, arr]) => (
            <div key={state} className="w-[260px] shrink-0">
              <div className="flex items-center gap-2 mb-2"><span className="w-2.5 h-2.5 rounded-full" style={{ background: col(state) }} /><span className="text-[13px] font-semibold capitalize">{state}</span><span className="text-[11px] text-slate-400">{arr.length}</span></div>
              <div className="flex flex-col gap-2">
                {arr.map((it, i) => (
                  <div key={i} className="rounded-lg border border-slate-200 bg-white p-2.5" style={{ borderLeftColor: col(state), borderLeftWidth: 3 }}>
                    <div className="text-[12.5px] text-slate-800">{it.label}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">{it.area || "—"}{it.touched ? ` · repelled ×${it.touched}` : ""}{it.variants ? ` · v${it.variants}` : ""}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-[13px] text-slate-500">No ideas yet — run an <b>Idea board</b> or <b>Goal</b> workflow and its board streams here live.</div>}
    </PageShell>
  );
}
