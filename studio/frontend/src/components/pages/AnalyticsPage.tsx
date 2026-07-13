"use client";
import { trpc } from "@/lib/trpc";
import { PageShell, Soon } from "@/components/ui/PageShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function Bars({ data }: { data: [string, number][] }) {
  const max = Math.max(1, ...data.map((d) => d[1]));
  return (
    <div className="flex flex-col gap-1.5">
      {data.map(([k, v]) => (
        <div key={k} className="flex items-center gap-2 text-[12px]">
          <span className="w-40 truncate text-slate-500">{k}</span>
          <div className="flex-1 h-4 bg-slate-100 rounded"><div className="h-4 rounded bg-blue-500/70" style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="w-8 text-right tabular-nums text-slate-600">{v}</span>
        </div>
      ))}
      {!data.length && <div className="text-[12px] text-slate-400">no data yet</div>}
    </div>
  );
}

export function AnalyticsPage() {
  const stats = trpc.stats.useQuery(undefined, { refetchInterval: 6000 });
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 6000 });
  const st = stats.data as any;
  const runList = (runsQ.data as any[]) || [];
  const nLive = runList.filter((r) => r.status !== "done" && r.status !== "error").length;
  const byPipeline: [string, number][] = st ? Object.entries(st.by_pipeline || {}) : [];
  const byDomain: [string, number][] = st ? Object.entries(st.by_domain || {}) : [];

  return (
    <PageShell title="Analytics" desc="Cross-run aggregates from the vector store." wide>
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[["runs", st?.runs ?? runList.length], ["live", nLive], ["knowledge", st?.knowledge ?? 0], ["capabilities", st?.pipelines ?? 0]].map(([l, v]) => (
          <Card key={l as string}><CardContent className="p-4"><div className="text-3xl font-bold text-blue-600 tabular-nums">{v as number}</div><div className="text-[12px] text-slate-500">{l as string}</div></CardContent></Card>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <Card><CardHeader><CardTitle>Runs by capability</CardTitle></CardHeader><CardContent><Bars data={byPipeline} /></CardContent></Card>
        <Card><CardHeader><CardTitle>Knowledge by domain</CardTitle></CardHeader><CardContent><Bars data={byDomain} /></CardContent></Card>
      </div>
      <Soon what="Time-series (runs/day, tokens & cost over time, move-diversity & plateau trends, success rate) needs run timing + cost telemetry from the backend." />
    </PageShell>
  );
}
