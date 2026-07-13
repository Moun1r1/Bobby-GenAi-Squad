"use client";
import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { useRunEvents } from "@/hooks/useRunEvents";
import { toMsg, type Msg } from "@/lib/events";
import { PageShell } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { Bubble } from "@/components/inbox/Bubble";

// Time-travel replay — scrub a run's event stream; the thread reconstructs at any tick.
export function TimelinePage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 4000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);
  const [cursor, setCursor] = useState<number | null>(null);
  const at = cursor == null ? events.length : cursor;
  const msgs = events.slice(0, at).map(toMsg).filter(Boolean) as Msg[];
  const last = events[at - 1];

  return (
    <PageShell title="Replay" desc="Scrub a run's event stream — reconstruct the conversation at any tick." wide
      right={<RunSelect runList={runList} value={runId} onChange={(id) => { setRunId(id); setCursor(null); }} />}>
      {runId ? (
        <>
          <div className="rounded-xl border border-slate-200 bg-white p-3 mb-3 flex items-center gap-3">
            <button className={`btn text-[12px] py-1 ${cursor == null ? "btn-primary" : ""}`} onClick={() => setCursor(null)}>{cursor == null ? "● live" : "⏵ live"}</button>
            <input type="range" min={0} max={events.length} value={at} className="flex-1 accent-blue-600" onChange={(e) => { const v = +e.target.value; setCursor(v >= events.length ? null : v); }} />
            <span className="text-[12px] text-slate-500 w-40 text-right">step {at}/{events.length}{last ? ` · ${last.kind}` : ""}</span>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-5 flex flex-col gap-2.5 min-h-[400px]">
            {msgs.map((m) => <Bubble key={m.seq} m={m} />)}
            {!msgs.length && <div className="text-center text-[13px] text-slate-400 mt-6">scrub to replay…</div>}
          </div>
        </>
      ) : <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-[13px] text-slate-500">Pick a run to replay it event-by-event.</div>}
    </PageShell>
  );
}
