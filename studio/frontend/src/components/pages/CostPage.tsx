"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { useRunEvents } from "@/hooks/useRunEvents";
import { PageShell, Soon } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { Card, CardContent } from "@/components/ui/card";

// Token/cost estimate from the persistent-self memory events (pinned vs naive keep-everything) — a proxy until
// real per-call token accounting is emitted by the gateway.
export function CostPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 5000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);
  const est = useMemo(() => {
    let pinned = 0, naive = 0, tools = 0;
    for (const e of events) { if (e.kind === "memory") { pinned = Math.max(pinned, e.pinned_tokens || 0); naive = Math.max(naive, e.naive_tokens || 0); } if (e.kind === "tool") tools++; }
    return { pinned, naive, tools, saved: naive - pinned };
  }, [events]);

  return (
    <PageShell title="Cost &amp; budget" desc="Token economy and spend." wide right={<RunSelect runList={runList} value={runId} onChange={setRunId} />}>
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[["pinned tokens", est.pinned], ["naive would-be", est.naive], ["tokens saved", est.saved], ["tool calls", est.tools]].map(([l, v]) => (
          <Card key={l as string}><CardContent className="p-4"><div className="text-2xl font-bold tabular-nums text-blue-600">{(v as number).toLocaleString()}</div><div className="text-[12px] text-slate-500">{l as string}</div></CardContent></Card>
        ))}
      </div>
      {est.naive > 0 && <Card className="mb-4"><CardContent className="p-4">
        <div className="text-[13px] font-medium mb-2">Persistent-self vs naive keep-everything</div>
        <div className="h-3 rounded bg-blue-500 mb-1" style={{ width: `${(est.pinned / est.naive) * 100}%` }} /><div className="text-[11px] text-slate-500">pinned {est.pinned.toLocaleString()} tok</div>
        <div className="h-3 rounded bg-slate-300 mt-2" style={{ width: "100%" }} /><div className="text-[11px] text-slate-500">naive ~{est.naive.toLocaleString()} tok</div>
      </CardContent></Card>}
      <Soon what="Real per-call token & dollar accounting, budget caps, and teacher-student routing (frontier-calls-per-good-output) come from the gateway's cost telemetry." />
    </PageShell>
  );
}
