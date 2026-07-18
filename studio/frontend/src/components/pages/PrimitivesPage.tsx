"use client";
import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { PageShell } from "@/components/ui/PageShell";

// The self-extending primitive library — the distilled cognitive stdlib the ACR flywheel builds. Browse it by
// category, and FIND a primitive back by task description (the semantic memory the engine consults before re-distilling).
export function PrimitivesPage() {
  const lib = trpc.primitives.useQuery(undefined, { refetchInterval: 15000 });
  const data = lib.data as any;
  const tree = (data?.tree as Record<string, string[]>) || {};
  const prims = (data?.primitives as Record<string, any>) || {};
  const [q, setQ] = useState("");
  const recall = trpc.primitivesRecall.useQuery({ q, k: 5 }, { enabled: q.trim().length > 2 });
  const hits = ((recall.data as any)?.hits as any[]) || [];
  const hitNames = new Set(hits.map((h) => h.name));

  const card = (name: string, extra?: string) => {
    const m = prims[name] || {};
    return (
      <div key={name} className={`rounded-xl border p-3.5 ${hitNames.has(name) ? "border-blue-300 bg-blue-50/40" : "border-slate-200 bg-white"}`}>
        <div className="flex items-center justify-between">
          <div className="font-mono text-[13.5px] font-semibold text-slate-800">{name}</div>
          {extra && <div className="text-[11px] text-blue-600 font-medium">{extra}</div>}
        </div>
        <div className="font-mono text-[12px] text-slate-500 mt-0.5">{m.signature}</div>
        {m.description && <div className="text-[12.5px] text-slate-600 mt-1.5">{m.description}</div>}
        <div className="flex flex-wrap gap-1 mt-2">
          {(m.passed_domains || []).map((d: string) => (
            <span key={d} className="text-[10.5px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">{d}</span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <PageShell title="Primitives" desc="The self-extending, cross-domain-proven cognitive stdlib the ACR flywheel distills — reused, never re-learned."
      right={<input className="inp py-1.5 w-64 text-[13px]" value={q} onChange={(e) => setQ(e.target.value)}
        placeholder="find a primitive by task… (e.g. sum the integers)" />}>
      {q.trim().length > 2 && (
        <div className="mb-5 rounded-xl border border-blue-200 bg-white p-4 max-w-[820px]">
          <div className="text-[11px] uppercase tracking-wide text-blue-500 mb-2">Semantic recall — found back from memory</div>
          {hits.length ? <div className="grid grid-cols-1 gap-2">{hits.map((h) => card(h.name, `${Math.round((h.score || 0) * 100)}% match`))}</div>
            : <div className="text-[13px] text-slate-400">no primitive matches — the engine would distill a new one</div>}
        </div>
      )}
      {Object.keys(tree).length ? (
        <div className="flex flex-col gap-6 max-w-[820px]">
          {Object.entries(tree).map(([cat, names]) => (
            <section key={cat}>
              <div className="text-[11px] uppercase tracking-wide text-slate-400 border-b border-slate-100 pb-1 mb-2.5">
                {cat} · {names.length}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">{names.map((n) => card(n))}</div>
            </section>
          ))}
        </div>
      ) : <div className="text-[13px] text-slate-400">library is empty — run an ACR Burn-In to distill primitives</div>}
    </PageShell>
  );
}
