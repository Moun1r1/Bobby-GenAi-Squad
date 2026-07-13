"use client";
import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { useRunEvents } from "@/hooks/useRunEvents";
import { PageShell } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { RunReport } from "@/components/report/RunReport";
import { chName } from "@/lib/events";

export function ReportsPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 4000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);
  const current = runList.find((r) => r.run_id === runId);

  return (
    <PageShell title="Reports" desc="Turn any run into a formatted, exportable report." wide
      right={<RunSelect runList={runList} value={runId} onChange={setRunId} />}>
      {runId ? (
        <div className="h-[calc(100dvh-160px)] rounded-xl border border-slate-200 bg-white overflow-hidden relative">
          {/* reuse the report renderer inline (as a full panel) */}
          <div className="absolute inset-0"><RunReport runId={runId} pipeline={current?.pipeline || ""} events={events} onClose={() => setRunId(null)} embedded /></div>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {runList.slice(0, 18).map((r) => (
            <button key={r.run_id} onClick={() => setRunId(r.run_id)} className="rounded-xl border border-slate-200 bg-white p-4 text-left hover:border-blue-300">
              <div className="text-[14px] font-semibold">{chName(r.pipeline)}</div>
              <div className="text-[11px] text-slate-400 mono">{r.run_id.slice(0, 8)} · {r.status}</div>
              <div className="text-[12px] text-blue-600 mt-2">Generate report →</div>
            </button>
          ))}
          {!runList.length && <div className="text-[13px] text-slate-400">no runs yet</div>}
        </div>
      )}
    </PageShell>
  );
}
