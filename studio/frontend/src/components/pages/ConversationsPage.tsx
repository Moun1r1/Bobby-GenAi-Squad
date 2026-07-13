"use client";
import { useEffect, useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { isOver, reduceRun, type Note } from "@/lib/events";
import { useRunEvents } from "@/hooks/useRunEvents";
import { useNotes } from "@/hooks/useNotes";
import { NavColumn } from "@/components/inbox/NavColumn";
import { RunList } from "@/components/inbox/RunList";
import { Thread } from "@/components/inbox/Thread";
import { ContextPanel } from "@/components/inbox/ContextPanel";
import { ComposeModal } from "@/components/inbox/ComposeModal";
import { RunReport } from "@/components/report/RunReport";

type Filter = "live" | "done" | "all";

export function ConversationsPage({ jumpRun, onConsumed }: { jumpRun?: string | null; onConsumed?: () => void }) {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 3000 });
  const pipelines = trpc.pipelines.useQuery();
  const [runId, setRunId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [compose, setCompose] = useState(false);
  const [report, setReport] = useState(false);
  const [context, setContext] = useState(true);
  const notes = useNotes();

  useEffect(() => { if (jumpRun) { setRunId(jumpRun); onConsumed?.(); } }, [jumpRun]); // eslint-disable-line

  const events = useRunEvents(runId);
  const control = trpc.control.useMutation();
  const launch = trpc.launch.useMutation({ onSuccess: (r: any) => { setRunId(r.run_id); setCompose(false); runsQ.refetch(); } });

  const runList: any[] = (runsQ.data as any[]) || [];
  const live = !!runId && !isOver(events);
  const current = runList.find((r) => r.run_id === runId);
  const nLive = runList.filter((r) => r.status !== "done" && r.status !== "error").length;
  const model = useMemo(() => reduceRun(events), [events]);

  const shown = runList.filter((r) => {
    const isLive = r.status !== "done" && r.status !== "error";
    if (filter === "live" && !isLive) return false;
    if (filter === "done" && isLive) return false;
    return !q || (r.pipeline || "").toLowerCase().includes(q.toLowerCase()) || r.run_id.includes(q);
  });

  return (
    <>
      <NavColumn runList={runList} nLive={nLive} pipelines={(pipelines.data as any[]) || []} filter={filter} active
        onFilter={setFilter} onCompose={() => setCompose(true)} onLaunch={(id) => launch.mutate({ pipeline: id, params: {} })} />
      <RunList runs={shown} all={runList} nLive={nLive} runId={runId} filter={filter} q={q} onFilter={setFilter} onQuery={setQ} onPick={setRunId} />
      {runId ? <>
        <Thread runId={runId} pipeline={current?.pipeline || ""} events={events} live={live} control={control}
          onClose={() => setRunId(null)} onReport={() => setReport(true)} onNote={(t) => notes.add(runId, t)} onToggleContext={() => setContext((v) => !v)} />
        {context && <ContextPanel runId={runId} pipeline={current?.pipeline || ""} status={live ? "live" : (current?.status || "done")}
          participants={model.participants} verdict={model.verdict} notes={(notes.map[runId] || []) as Note[]} onNote={(t) => notes.add(runId, t)} onClose={() => setContext(false)} />}
      </> : <div className="flex-1 grid place-items-center text-slate-400"><div className="text-center"><div className="text-[15px] mb-1">Select a run</div><div className="text-[13px]">or start one with ＋ New run</div></div></div>}

      {compose && <ComposeModal pipelines={pipelines.data as any[]} launching={launch.isPending} onClose={() => setCompose(false)} onLaunch={(p, params) => launch.mutate({ pipeline: p, params })} />}
      {report && runId && <RunReport runId={runId} pipeline={current?.pipeline || ""} events={events} onClose={() => setReport(false)} />}
    </>
  );
}
