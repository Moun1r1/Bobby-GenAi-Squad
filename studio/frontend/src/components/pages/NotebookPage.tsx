"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { PageShell } from "@/components/ui/PageShell";
import { Markdown } from "@/components/ui/Markdown";

// Cross-run research notebook — the rd_lab RD_TRACE notebook (nested-lab, self-written) when present, else the
// accumulated findings corpus. Served read-only by /notebook.
export function NotebookPage() {
  const nb = trpc.notebook.useQuery(undefined, { refetchInterval: 10000 });
  const data = nb.data as any;
  const md: string | null = data?.markdown || null;
  const items = (data?.items as any[]) || [];
  const [q, setQ] = useState("");
  const [tab, setTab] = useState<"notebook" | "corpus">(md ? "notebook" : "corpus");
  const byPipeline = useMemo(() => {
    const shown = items.filter((it) => !q || JSON.stringify(it).toLowerCase().includes(q.toLowerCase()));
    const m: Record<string, any[]> = {};
    for (const it of shown) (m[it.pipeline || it.domain || "run"] ||= []).push(it);
    return Object.entries(m);
  }, [items, q]);

  return (
    <PageShell title="Notebook" desc="The lab's self-written research ledger + accumulated findings across every run."
      right={<div className="flex items-center gap-2">
        {md && <div className="inline-flex rounded-lg bg-slate-100 p-0.5 text-[12px]">
          <button onClick={() => setTab("notebook")} className={`px-3 py-1 rounded-md ${tab === "notebook" ? "bg-white shadow-sm font-medium" : "text-slate-500"}`}>Notebook</button>
          <button onClick={() => setTab("corpus")} className={`px-3 py-1 rounded-md ${tab === "corpus" ? "bg-white shadow-sm font-medium" : "text-slate-500"}`}>Corpus</button>
        </div>}
        <input className="inp py-1.5 w-56 text-[13px]" value={q} onChange={(e) => setQ(e.target.value)} placeholder="search…" />
      </div>}>
      {tab === "notebook" && md ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 max-w-[820px]">
          <Markdown>{q ? md.split("\n").filter((l) => l.toLowerCase().includes(q.toLowerCase())).join("\n") || "_no matches_" : md}</Markdown>
        </div>
      ) : byPipeline.length ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 max-w-[760px]">
          {byPipeline.map(([p, arr]) => (
            <section key={p} className="mb-5">
              <div className="text-[11px] uppercase tracking-wide text-slate-400 border-b border-slate-100 pb-1 mb-2">{p} · {arr.length} findings</div>
              <div className="flex flex-col gap-1.5">{arr.slice(0, 40).map((it, i) => <div key={i} className="text-[13.5px] text-slate-700"><Markdown compact>{`- ${it.text}`}</Markdown></div>)}</div>
            </section>
          ))}
        </div>
      ) : <div className="text-[13px] text-slate-400">no findings yet</div>}
    </PageShell>
  );
}
