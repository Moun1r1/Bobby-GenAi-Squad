"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { AVATAR_FG, hashIdx } from "@/lib/avatars";
import { PageShell } from "@/components/ui/PageShell";

// A structured knowledge map: domains → items, as a two-level tree (the recursive knowledge map, MVP).
export function KnowledgeMapPage() {
  const know = trpc.knowledgeAll.useQuery({ limit: 500, domain: "" });
  const items = (know.data as any[]) || [];
  const groups = useMemo(() => {
    const m: Record<string, any[]> = {};
    for (const it of items) { const d = it.domain || it.workflow || it.pipeline || "misc"; (m[d] ||= []).push(it); }
    return Object.entries(m).sort((a, b) => b[1].length - a[1].length);
  }, [items]);
  const [open, setOpen] = useState<string | null>(groups[0]?.[0] || null);

  return (
    <PageShell title="Knowledge map" desc={`${items.length} nodes across ${groups.length} domains — the squads' accumulated knowledge, organized.`} wide>
      {items.length ? (
        <div className="grid grid-cols-[280px_1fr] gap-4">
          <div className="flex flex-col gap-1">
            {groups.map(([d, arr]) => (
              <button key={d} onClick={() => setOpen(d)} className={`flex items-center gap-2 px-3 py-2 rounded-lg text-left text-[13px] ${open === d ? "bg-blue-50 text-blue-700" : "hover:bg-slate-50 text-slate-700"}`}>
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: AVATAR_FG[hashIdx(d, AVATAR_FG.length)] }} />
                <span className="flex-1 truncate">{d}</span><span className="text-[11px] text-slate-400">{arr.length}</span>
              </button>
            ))}
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="text-[14px] font-semibold mb-3 flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full" style={{ background: AVATAR_FG[hashIdx(open || "", AVATAR_FG.length)] }} />{open}</div>
            <div className="flex flex-col gap-2">
              {(groups.find(([d]) => d === open)?.[1] || []).map((it, i) => (
                <div key={i} className="border-l-2 border-slate-200 pl-3 py-0.5 text-[13px] text-slate-700">{it.text}</div>
              ))}
            </div>
          </div>
        </div>
      ) : <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-[13px] text-slate-500">No knowledge yet — run a workflow and its findings map here by domain.</div>}
    </PageShell>
  );
}
