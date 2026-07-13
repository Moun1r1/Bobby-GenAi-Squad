import { AVATAR_FG, hashIdx } from "@/lib/avatars";
import { chName } from "@/lib/events";

type Filter = "live" | "done" | "all";

function NavItem({ label, count, dot, active, onClick }: { label: string; count?: number; dot?: string; active?: boolean; onClick?: () => void }) {
  return (
    <button onClick={onClick} className={`w-full flex items-center gap-2 px-4 py-1.5 text-left text-[13px] ${active ? "bg-blue-50 text-blue-700 font-medium" : "text-slate-600 hover:bg-slate-50"}`}>
      {dot && <span className="w-2 h-2 rounded-full" style={{ background: dot }} />}
      <span className="flex-1">{label}</span>
      {count != null && <span className="text-[11px] text-slate-400 tabular-nums">{count}</span>}
    </button>
  );
}
function NavSection({ title }: { title: string }) {
  return <div className="px-4 mt-4 mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">{title}</div>;
}

const LABELS: [string, string][] = [["proven", "#16a34a"], ["active", "#2563eb"], ["escalate", "#dc2626"], ["dead", "#94a3b8"]];

export function NavColumn({ runList, nLive, pipelines, filter, active, onFilter, onCompose, onLaunch }:
  { runList: any[]; nLive: number; pipelines: any[]; filter: Filter; active: boolean; onFilter: (f: Filter) => void; onCompose: () => void; onLaunch: (id: string) => void }) {
  return (
    <div className="w-[214px] shrink-0 bg-white border-r border-slate-200 flex flex-col py-3">
      <div className="px-4 mb-3"><div className="text-[15px] font-semibold">Bobby</div><div className="text-[11px] text-slate-400">multi-agent squad</div></div>
      <button onClick={onCompose} className="mx-3 mb-3 btn btn-primary text-[13px] py-2">＋ New run</button>
      <NavItem label="All runs" count={runList.length} active={active && filter === "all"} onClick={() => onFilter("all")} />
      <NavItem label="Live" count={nLive} dot="#22c55e" active={active && filter === "live"} onClick={() => onFilter("live")} />
      <NavItem label="Completed" active={active && filter === "done"} onClick={() => onFilter("done")} />
      <NavSection title="Capabilities" />
      <div className="overflow-auto flex-1">
        {pipelines.filter((p) => p.kind === "native").map((p) => (
          <button key={p.id} onClick={() => onLaunch(p.id)} className="w-full flex items-center gap-2 px-4 py-1.5 text-left hover:bg-slate-50 text-[13px] text-slate-600">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: AVATAR_FG[hashIdx(p.id, AVATAR_FG.length)] }} />
            <span className="truncate">{chName(p.id)}</span>
          </button>
        ))}
      </div>
      <NavSection title="Labels" />
      {LABELS.map(([l, c]) => <div key={l} className="flex items-center gap-2 px-4 py-1 text-[12px] text-slate-500"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />{l}</div>)}
    </div>
  );
}
