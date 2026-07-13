"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { trpc } from "@/lib/trpc";
import { chName, isOver } from "@/lib/events";
import { useRunEvents } from "@/hooks/useRunEvents";
import { Avatar } from "@/components/ui/Avatar";

export function WorldPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 3000 });
  const pipelines = trpc.pipelines.useQuery();
  const control = trpc.control.useMutation();
  const launch = trpc.launch.useMutation({ onSuccess: (r: any) => { setRunId(r.run_id); runsQ.refetch(); } });
  const [runId, setRunId] = useState<string | null>(null);
  const [theme, setTheme] = useState("");
  const worldRuns = ((runsQ.data as any[]) || []).filter((r) => r.pipeline === "world" || r.pipeline === "persona");
  const events = useRunEvents(runId);
  const live = !!runId && !isOver(events);

  const agents = useMemo(() => { const m = new Map<string, any>(); for (const e of events) if (e.kind === "agent") m.set(e.name, e); return [...m.values()]; }, [events]);
  const says = useMemo(() => events.filter((e) => e.kind === "say"), [events]);
  const lastSay = useMemo(() => { const m: Record<string, string> = {}; for (const s of says) if (s.who && s.who !== "event") m[s.who] = s.text; return m; }, [says]);
  const feedRef = useRef<HTMLDivElement>(null);
  useEffect(() => { feedRef.current?.scrollTo({ top: 1e9, behavior: "smooth" }); }, [says.length]);

  const [inject, setInject] = useState("");
  const doInject = () => { if (inject.trim() && runId) { control.mutate({ runId, action: "steer", text: inject }); setInject(""); } };

  const W = 640, H = 380, cx = W / 2, cy = H / 2, R = 120;
  return (
    <div className="flex-1 flex min-w-0">
      <div className="w-[260px] shrink-0 bg-white border-r border-slate-200 flex flex-col p-3">
        <div className="text-[17px] font-bold mb-1 px-1">World sims</div>
        <div className="text-[12px] text-slate-400 mb-3 px-1">persistent agents living in a world</div>
        <div className="rounded-lg border border-slate-200 p-2 mb-3">
          <input className="w-full text-[13px] outline-none mb-2 placeholder:text-slate-400" value={theme} onChange={(e) => setTheme(e.target.value)} placeholder="A medieval tavern at dusk…" />
          <button className="btn btn-primary w-full text-[13px] py-1.5" disabled={launch.isPending || !theme.trim()} onClick={() => launch.mutate({ pipeline: "world", params: { world: theme, agents: 5, rounds: 16 } })}>＋ New world</button>
        </div>
        <div className="overflow-auto flex-1">
          {worldRuns.map((r) => {
            const l = r.status !== "done" && r.status !== "error";
            return (
              <button key={r.run_id} onClick={() => setRunId(r.run_id)} className={`w-full flex items-center gap-2 px-2 py-2 rounded-lg text-left ${runId === r.run_id ? "bg-blue-50" : "hover:bg-slate-50"}`}>
                <Avatar name={r.pipeline} size={28} />
                <div className="min-w-0 flex-1"><div className="text-[13px] truncate">{chName(r.pipeline)}</div><div className="text-[10px] text-slate-400">{r.run_id.slice(0, 8)}</div></div>
                {l && <span className="w-1.5 h-1.5 rounded-full bg-green-500 livedot" />}
              </button>
            );
          })}
          {!worldRuns.length && <div className="text-[12px] text-slate-400 px-2">No worlds yet — start one above.</div>}
        </div>
      </div>

      <div className="flex-1 flex flex-col bg-slate-50 min-w-0">
        {!runId ? <div className="flex-1 grid place-items-center text-slate-400"><div className="text-center"><div className="text-[15px]">Select or start a world</div></div></div> : <>
          <div className="h-[52px] shrink-0 bg-white border-b border-slate-200 flex items-center px-5 gap-2">
            <div className="text-[15px] font-semibold flex-1">World · {runId?.slice(0, 8)}</div>
            <span className={`px-2 py-0.5 rounded text-[11px] ${live ? "bg-green-100 text-green-700" : "bg-slate-100 text-slate-500"}`}>{live ? "● live" : "ended"}</span>
            {live && <button className="btn btn-ghost text-[12px] py-1 text-red-600" onClick={() => control.mutate({ runId: runId!, action: "stop" })}>Stop</button>}
          </div>
          {/* canvas */}
          <div className="p-4">
            <div className="bg-white rounded-xl border border-slate-200 grid place-items-center">
              <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 380 }}>
                <circle cx={cx} cy={cy} r={R + 30} fill="#f8fafc" stroke="#e2e8f0" />
                {agents.map((a, i) => {
                  const ang = (i / Math.max(1, agents.length)) * Math.PI * 2 - Math.PI / 2;
                  const x = cx + Math.cos(ang) * R, y = cy + Math.sin(ang) * R;
                  const said = lastSay[a.name];
                  return (
                    <g key={a.name}>
                      <circle cx={x} cy={y} r={22} fill="#dbeafe" stroke="#2563eb" strokeWidth={1.2} />
                      <text x={x} y={y + 4} textAnchor="middle" fontSize="10" fill="#1e40af" fontWeight="600">{(a.persona || a.name).slice(0, 2).toUpperCase()}</text>
                      <text x={x} y={y + 34} textAnchor="middle" fontSize="8" fill="#64748b">{a.name}</text>
                      {said && <foreignObject x={x - 70} y={y - 62} width="140" height="42"><div className="text-[9px] bg-white border border-slate-200 rounded-lg px-1.5 py-1 text-slate-600 leading-tight overflow-hidden" style={{ maxHeight: 40 }}>{said.slice(0, 90)}</div></foreignObject>}
                    </g>
                  );
                })}
                {!agents.length && <text x={cx} y={cy} textAnchor="middle" fontSize="12" fill="#94a3b8">the world is waking up…</text>}
              </svg>
            </div>
          </div>
          {/* interaction feed */}
          <div ref={feedRef} className="flex-1 overflow-auto px-6 pb-3 flex flex-col gap-2">
            {says.map((s, i) => s.who === "event"
              ? <div key={i} className="text-center text-[11px] text-amber-600">— {s.text} —</div>
              : <div key={i} className="flex gap-2 rise"><Avatar name={s.who} size={26} /><div><span className="text-[12px] font-medium">{s.who}</span><div className="text-[13px] text-slate-700">{s.text}</div></div></div>)}
            {!says.length && <div className="text-center text-[13px] text-slate-400 mt-6">no interactions yet…</div>}
          </div>
          <div className="shrink-0 bg-white border-t border-slate-200 px-5 py-3 flex gap-2">
            <input className="inp text-[14px]" value={inject} onChange={(e) => setInject(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doInject()} placeholder="Inject an event into the world…" disabled={!live} />
            <button className="btn btn-primary" disabled={!live || !inject.trim()} onClick={doInject}>Inject</button>
          </div>
        </>}
      </div>
    </div>
  );
}
