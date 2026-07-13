import { Avatar } from "@/components/ui/Avatar";
import { chName } from "@/lib/events";

type Filter = "live" | "done" | "all";

function RunRow({ r, active, onClick }: { r: any; active: boolean; onClick: () => void }) {
  const isLive = r.status !== "done" && r.status !== "error";
  const verdict = r.summary_verdict || r.summary?.verdict || "";
  return (
    <button onClick={onClick} className={`w-full flex gap-3 px-4 py-3 text-left border-l-2 ${active ? "bg-blue-50/60 border-blue-600" : "border-transparent hover:bg-slate-50"}`}>
      <Avatar name={r.pipeline} />
      <div className="min-w-0 flex-1">
        <div className="text-[10px] text-slate-400 truncate">{chName(r.pipeline)}</div>
        <div className="flex items-center gap-2">
          <span className="text-[13.5px] font-semibold truncate">{r.run_id.slice(0, 8)}</span>
          {isLive && <span className="w-1.5 h-1.5 rounded-full bg-green-500 livedot shrink-0" />}
        </div>
        <div className="text-[12px] text-slate-500 truncate">{verdict || (isLive ? "running…" : r.status)}</div>
      </div>
    </button>
  );
}

export function RunList({ runs, all, nLive, runId, filter, q, onFilter, onQuery, onPick }:
  { runs: any[]; all: any[]; nLive: number; runId: string | null; filter: Filter; q: string; onFilter: (f: Filter) => void; onQuery: (s: string) => void; onPick: (id: string) => void }) {
  return (
    <div className="w-[360px] shrink-0 bg-white border-r border-slate-200 flex flex-col">
      <div className="px-4 pt-3 pb-2 border-b border-slate-100">
        <div className="flex items-center gap-2 mb-2">
          <svg viewBox="0 0 24 24" className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7" /><path d="M21 21l-4-4" /></svg>
          <input className="flex-1 text-[13px] outline-none placeholder:text-slate-400" value={q} onChange={(e) => onQuery(e.target.value)} placeholder="Search runs" />
        </div>
        <div className="flex items-center justify-between">
          <div className="text-[19px] font-bold">Conversations</div>
          <span className="text-[11px] text-slate-400">{runs.length}</span>
        </div>
        <div className="flex gap-4 mt-1.5 text-[13px]">
          {(["live", "all", "done"] as const).map((f) => (
            <button key={f} onClick={() => onFilter(f)} className={`pb-1 border-b-2 ${filter === f ? "border-blue-600 text-slate-900 font-medium" : "border-transparent text-slate-400"}`}>
              {f === "live" ? "Live" : f === "all" ? "All" : "Done"} <span className="text-slate-400">{f === "live" ? nLive : f === "all" ? all.length : all.length - nLive}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-auto">
        {runs.map((r) => <RunRow key={r.run_id} r={r} active={r.run_id === runId} onClick={() => onPick(r.run_id)} />)}
        {!runs.length && <div className="p-6 text-center text-[13px] text-slate-400">No runs — click <b>＋ New run</b>.</div>}
      </div>
    </div>
  );
}
