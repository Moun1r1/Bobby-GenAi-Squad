"use client";
import { useEffect, useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { useRunEvents } from "@/hooks/useRunEvents";
import { chName } from "@/lib/events";
import { PageShell, Soon } from "@/components/ui/PageShell";
import { RunSelect } from "@/components/ui/RunSelect";
import { Badge } from "@/components/ui/badge";

// Human-in-the-loop queue — escalations surfaced from the selected run. Acks are local until a backend gate API lands.
export function ApprovalsPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 4000 });
  const runList = (runsQ.data as any[]) || [];
  const [runId, setRunId] = useState<string | null>(null);
  const events = useRunEvents(runId);
  const escalations = useMemo(() => events.filter((e) => e.kind === "escalate"), [events]);
  const [acked, setAcked] = useState<Record<number, string>>({});
  useEffect(() => { setAcked({}); }, [runId]);

  return (
    <PageShell title="Approvals · HITL" desc="Review the squad's escalations and human-review requests." wide
      right={<RunSelect runList={runList} value={runId} onChange={setRunId} />}>
      <div className="mb-4"><Soon what="A real approval queue (approve/reject writes back to the run and unblocks the gate) needs a backend gate API. Below are the live escalations for the selected run." /></div>
      {escalations.length ? (
        <div className="flex flex-col gap-2">
          {escalations.map((e, i) => (
            <div key={i} className="rounded-xl border border-amber-200 bg-amber-50 p-4 flex items-center gap-3">
              <Badge variant="warn">escalated</Badge>
              <div className="flex-1"><div className="text-[13px] text-slate-700">{e.reason || "human review requested"}</div><div className="text-[11px] text-slate-400">{chName(runList.find((r) => r.run_id === runId)?.pipeline || "")} · {e.met != null ? `${e.met}/${e.total} criteria` : ""}</div></div>
              {acked[i] ? <Badge variant={acked[i] === "approve" ? "proven" : "dead"}>{acked[i]}d</Badge> : <div className="flex gap-1">
                <button className="btn btn-primary text-[12px] py-1" onClick={() => setAcked({ ...acked, [i]: "approve" })}>Approve</button>
                <button className="btn text-[12px] py-1" onClick={() => setAcked({ ...acked, [i]: "reject" })}>Reject</button>
              </div>}
            </div>
          ))}
        </div>
      ) : <div className="text-[13px] text-slate-400">{runId ? "no escalations in this run — the squad converged on its own." : "pick a run to see its escalations."}</div>}
    </PageShell>
  );
}
