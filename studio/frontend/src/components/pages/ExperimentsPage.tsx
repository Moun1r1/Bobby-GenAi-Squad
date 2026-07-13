"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { boardIdeas, reduceRun } from "@/lib/events";
import { useRunEvents } from "@/hooks/useRunEvents";
import { PageShell } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const grp = (s: string) => ["proven", "verified"].includes(s) ? "proven" : s === "dead" ? "dead" : ["contested", "blocked"].includes(s) ? "contested" : "open";
const vBadge = (v: string) => v === "WIRE" ? "proven" : v === "DELETE" ? "dead" : v === "INVALID" ? "error" : v === "MARGINAL" || v === "INCONCLUSIVE" || v === "DEFER" ? "warn" : "outline";
const pct = (r?: number) => r == null ? "" : `${r >= 0 ? "+" : ""}${Math.round(r * 100)}%`;

function ProofBench() {
  const utils = trpc.useUtils();
  const proofsQ = trpc.proofs.useQuery({ run: false }, { refetchInterval: (q) => ((q.state.data as any)?.state === "running" ? 2500 : false) });
  const d = proofsQ.data as any;
  const results: any[] = d?.results || [];
  const running = d?.state === "running";
  const wire = results.filter((r) => r.verdict === "WIRE").length;
  const del = results.filter((r) => r.verdict === "DELETE").length;
  const run = async () => { await utils.proofs.fetch({ run: true }); proofsQ.refetch(); };

  return (
    <Card className="mb-4">
      <CardHeader className="flex-row items-center gap-2">
        <CardTitle className="flex-1">Gain-proof bench · confirm_gain / prove</CardTitle>
        {results.length > 0 && <span className="text-[12px] text-slate-500">kill wall <span className="text-green-600 font-semibold">{wire} WIRE</span> : {del} DELETE</span>}
        <button className="btn btn-primary text-[12px] py-1" disabled={running} onClick={run}>{running ? `running ${d?.ran}/${d?.total}…` : "Run proofs"}</button>
      </CardHeader>
      <CardContent>
        {results.length ? (
          <div className="flex flex-col">
            {results.map((r, i) => (
              <div key={i} className="flex items-center gap-3 py-2 border-b border-slate-50 last:border-0">
                <span className="text-[13px] font-medium w-44 truncate">{r.name || r._source}</span>
                <span className="text-[12px] text-slate-500 mono flex-1">
                  {r.control != null && r.treatment != null ? `${r.control} → ${r.treatment}` : ""}
                  {r.rel_gain != null && <span className={r.rel_gain >= 0.1 ? "text-green-600 font-semibold ml-2" : "text-slate-400 ml-2"}>{pct(r.rel_gain)}</span>}
                  {r.neg_control_rel != null && <span className="text-slate-400 ml-2">· neg {pct(r.neg_control_rel)}</span>}
                  {r.seeds != null && <span className="text-slate-400 ml-2">· {r.seeds} seeds</span>}
                </span>
                <Badge variant={vBadge(r.verdict) as any}>{r.verdict}</Badge>
              </div>
            ))}
          </div>
        ) : running ? <div className="text-[13px] text-slate-500">running the deterministic gain-proofs ({d?.ran}/{d?.total})… real A/Bs with negative-control + CI.</div>
          : <div className="text-[13px] text-slate-500">Run the framework's gain-proofs — each is a real A/B via the <span className="mono">prove</span> primitive (verify-by-outcome, negative-control, replication CI). No agent is scripted.</div>}
      </CardContent>
    </Card>
  );
}

export function ExperimentsPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 4000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);
  const ideas = useMemo(() => boardIdeas(events), [events]);
  const model = useMemo(() => reduceRun(events), [events]);
  const proveQueue = useMemo(() => { const e = [...events].reverse().find((x) => x.kind === "prove_queue"); return (e?.items as any[]) || []; }, [events]);
  const portfolio = useMemo(() => { const e = [...events].reverse().find((x) => x.kind === "portfolio"); return (e?.items as any[]) || []; }, [events]);
  const proven = ideas.filter((i) => grp(i.state) === "proven");
  const dead = ideas.filter((i) => grp(i.state) === "dead");
  const contested = ideas.filter((i) => grp(i.state) === "contested");

  return (
    <PageShell title="Experiments · Proof bench" desc="Green means a real A/B passed. Run the framework's gain-proofs, or inspect a live run's idea verdicts."
      right={<RunSelect runList={runList} value={runId} onChange={setRunId} />}>
      <ProofBench />

      {portfolio.length > 0 && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Portfolio · quick-wins / core-bets / moonshots</CardTitle></CardHeader>
          <CardContent>
            <div className="text-[12px] text-slate-500 mb-3">Ranked deterministically from the swarm’s own signals — development × feasibility (has a test) × moat (survived red-team) × novelty. No taste rubric.</div>
            <div className="grid grid-cols-3 gap-3">
              {([["quick-win", "Quick wins", "rounded-lg border border-emerald-200 bg-emerald-50/40 p-2.5", "text-[12px] font-semibold text-emerald-700 mb-1.5"], ["core-bet", "Core bets", "rounded-lg border border-blue-200 bg-blue-50/40 p-2.5", "text-[12px] font-semibold text-blue-700 mb-1.5"], ["moonshot", "Moonshots", "rounded-lg border border-violet-200 bg-violet-50/40 p-2.5", "text-[12px] font-semibold text-violet-700 mb-1.5"]] as const).map(([b, label, boxCls, headCls]) => {
                const items = portfolio.filter((p) => p.bucket === b);
                return (
                  <div key={b} className={boxCls}>
                    <div className={headCls}>{label} <span className="text-slate-400">{items.length}</span></div>
                    <div className="flex flex-col gap-1.5">
                      {items.slice(0, 8).map((p, i) => (
                        <div key={i} className="rounded border border-slate-200 bg-white p-1.5">
                          <div className="flex items-center gap-1"><span className="flex-1 truncate text-[12px]">{p.label}</span><span className="text-[10px] text-slate-400 tabular-nums">{p.score}</span></div>
                          <div className="text-[10px] text-slate-400">{p.area} · dev {p.development}{p.feasible ? " · test" : ""}{p.viability === "survived" ? " · survives" : ""}</div>
                        </div>
                      ))}
                      {!items.length && <div className="text-[11px] text-slate-400">—</div>}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {proveQueue.length > 0 && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Prove-queue · exported by the swarm ({proveQueue.length})</CardTitle></CardHeader>
          <CardContent>
            <div className="text-[12px] text-slate-500 mb-2">The local swarm mined + ranked these; each carries its own <b>falsifiable test</b> (hypothesis · probe · threshold) so the strong model / gain-proofs prove them (teaching-flywheel).</div>
            <div className="flex flex-col gap-1.5">
              {proveQueue.map((it: any, i: number) => (
                <div key={i} className="flex flex-col gap-0.5 py-1.5 border-b border-slate-50 last:border-0">
                  <div className="flex items-center gap-2">
                    <span className="w-5 text-[11px] text-slate-400 tabular-nums">{i + 1}</span>
                    <span className="flex-1 truncate text-[13px]">{it.label}</span>
                    <span className="text-[11px] text-slate-400">{it.area}</span>
                    {it.viability === "survived" && <Badge variant="proven">survives red-team</Badge>}
                    {it.viability === "at-risk" && <Badge variant="warn">at-risk</Badge>}
                    {!it.viability && <Badge variant="outline">un-red-teamed</Badge>}
                    <Badge variant="outline">{it.status} · v{it.variants}</Badge>
                  </div>
                  {it.has_test
                    ? <div className="ml-7 text-[11px] text-slate-500 mono truncate" title={it.test}>🧪 {it.test}</div>
                    : <div className="ml-7 text-[11px] text-amber-600">⚠ no falsifiable test — flag before proving</div>}
                  {it.redteam && <div className="ml-7 text-[11px] text-slate-500 truncate" title={it.redteam}>⚔ {it.redteam}</div>}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="text-[13px] font-semibold text-slate-500 mb-2">Live run · idea verdicts</div>
      <div className="grid grid-cols-4 gap-3 mb-4">
        <Card><CardContent className="p-4"><div className="text-3xl font-bold text-green-600 tabular-nums">{proven.length}</div><div className="text-[12px] text-slate-500">proven</div></CardContent></Card>
        <Card><CardContent className="p-4"><div className="text-3xl font-bold text-slate-400 tabular-nums">{dead.length}</div><div className="text-[12px] text-slate-500">deleted</div></CardContent></Card>
        <Card><CardContent className="p-4"><div className="text-3xl font-bold text-rose-500 tabular-nums">{contested.length}</div><div className="text-[12px] text-slate-500">contested</div></CardContent></Card>
        <Card><CardContent className="p-4"><div className="text-3xl font-bold tabular-nums">{proven.length}:{dead.length}</div><div className="text-[12px] text-slate-500">kill ratio</div></CardContent></Card>
      </div>
      {model.verdict && <Card className="mb-4"><CardContent className="p-4 flex items-center gap-3"><Badge variant={model.verdict.verdict === "WIRE" ? "proven" : "warn"}>{model.verdict.verdict}</Badge><span className="text-[13px] text-slate-600">{model.verdict.metric} · {model.verdict.detail}</span></CardContent></Card>}
      {ideas.length ? (
        <Card>
          <CardHeader><CardTitle>Verdict ledger</CardTitle></CardHeader>
          <CardContent className="flex flex-col gap-1">
            {ideas.map((it, i) => (
              <div key={i} className="flex items-center gap-2 py-1.5 border-b border-slate-50 last:border-0">
                <span className="flex-1 truncate text-[13px]">{it.label}</span>
                <span className="text-[11px] text-slate-400">{it.area}</span>
                <Badge variant={grp(it.state) === "proven" ? "proven" : grp(it.state) === "dead" ? "dead" : grp(it.state) === "contested" ? "error" : "outline"}>{it.state}</Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      ) : <div className="text-[13px] text-slate-400">{runId ? "no idea board in this run." : "pick an Idea board / Goal run to see its live idea verdicts."}</div>}
    </PageShell>
  );
}
