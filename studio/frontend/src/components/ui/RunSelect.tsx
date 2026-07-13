import { chName } from "@/lib/events";

export function RunSelect({ runList, value, onChange, only }:
  { runList: any[]; value: string | null; onChange: (id: string) => void; only?: string[] }) {
  const rows = only ? runList.filter((r) => only.includes(r.pipeline)) : runList;
  return (
    <select className="inp py-1 w-[240px] text-[12px]" value={value || ""} onChange={(e) => onChange(e.target.value)}>
      <option value="">— pick a run —</option>
      {rows.slice(0, 40).map((r) => <option key={r.run_id} value={r.run_id}>{chName(r.pipeline)} · {r.run_id.slice(0, 6)} · {r.status}</option>)}
    </select>
  );
}
