"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { useRunEvents } from "@/hooks/useRunEvents";
import { chName, reduceRun } from "@/lib/events";
import { PageShell } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { Badge } from "@/components/ui/badge";

function Side({ runId, runList }: { runId: string | null; runList: any[] }) {
  const events = useRunEvents(runId);
  const m = useMemo(() => reduceRun(events), [events]);
  const run = runList.find((r) => r.run_id === runId);
  if (!runId) return <div className="grid place-items-center h-full text-slate-400 text-[13px]">pick a run</div>;
  return (
    <div className="p-4 overflow-auto h-full">
      <div className="text-[14px] font-semibold">{chName(run?.pipeline || "")}</div>
      <div className="text-[11px] text-slate-400 mb-3 mono">{runId.slice(0, 8)} · {run?.status}</div>
      <div className="flex gap-2 mb-3">
        {m.verdict && <Badge variant={m.verdict.verdict === "WIRE" ? "proven" : "warn"}>{m.verdict.verdict}</Badge>}
        {m.converged && <Badge variant="proven">converged</Badge>}
        {m.escalated && <Badge variant="warn">escalated</Badge>}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4 text-center">
        {[["agents", m.participants.length], ["findings", m.counts.messages], ["tools", m.counts.tools]].map(([l, v]) => (
          <div key={l as string} className="rounded-lg border border-slate-200 p-2"><div className="text-xl font-bold tabular-nums">{v as number}</div><div className="text-[10px] text-slate-500">{l as string}</div></div>
        ))}
      </div>
      {m.result?.summary && <div className="text-[13px] text-slate-600 mb-3">{m.result.summary.slice(0, 300)}</div>}
      <div className="text-[11px] uppercase text-slate-400 mb-1">findings</div>
      <div className="flex flex-col gap-1">{m.findings.slice(0, 20).map((f, i) => <div key={i} className="text-[12px] text-slate-600"><b className="text-slate-500">{f.who}</b> {f.text.slice(0, 120)}</div>)}</div>
    </div>
  );
}

export function ComparePage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 5000 });
  const runList = (runsQ.data as any[]) || [];
  const [a, setA] = useState<string | null>(null);
  const [b, setB] = useState<string | null>(null);
  return (
    <PageShell title="Compare runs" desc="Two runs side by side — e.g. with vs without a reused expert, or different parameters." wide
      right={<div className="flex gap-2"><RunSelect runList={runList} value={a} onChange={setA} /><RunSelect runList={runList} value={b} onChange={setB} /></div>}>
      <div className="grid grid-cols-2 gap-3 h-[calc(100dvh-180px)]">
        <div className="rounded-xl border border-slate-200 bg-white"><Side runId={a} runList={runList} /></div>
        <div className="rounded-xl border border-slate-200 bg-white"><Side runId={b} runList={runList} /></div>
      </div>
    </PageShell>
  );
}
