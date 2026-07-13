"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { AVATAR_FG, hashIdx } from "@/lib/avatars";

function VectorMap({ points }: { points: any[] }) {
  const domains = useMemo(() => [...new Set(points.map((p) => p.domain || p.workflow || p.pipeline || "misc"))], [points]);
  const [pick, setPick] = useState<any>(null);
  if (!points.length) return <div className="text-[13px] text-slate-400">No embedded knowledge yet — run a workflow and its findings map here.</div>;
  const W = 640, H = 360, pad = 20;
  const X = (x: number) => pad + ((x + 1) / 2) * (W - 2 * pad);
  const Y = (y: number) => pad + ((y + 1) / 2) * (H - 2 * pad);
  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-2">{domains.slice(0, 10).map((d) => <span key={d} className="inline-flex items-center gap-1 text-[11px] text-slate-500"><span className="w-2 h-2 rounded-full" style={{ background: AVATAR_FG[hashIdx(d, AVATAR_FG.length)] }} />{d}</span>)}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-slate-200 bg-white" style={{ maxHeight: 360 }}>
        {points.map((p, i) => { const d = p.domain || p.workflow || p.pipeline || "misc"; return <circle key={i} cx={X(p.x)} cy={Y(p.y)} r={pick === p ? 6 : 4} fill={AVATAR_FG[hashIdx(d, AVATAR_FG.length)]} opacity={0.75} onClick={() => setPick(p)} className="cursor-pointer" />; })}
      </svg>
      {pick && <div className="mt-2 rounded-lg border border-slate-200 bg-white p-2.5 text-[12px]"><span className="text-blue-600">{pick.domain || pick.workflow || pick.pipeline || "misc"}</span> · {pick.text}</div>}
      <div className="text-[11px] text-slate-400 mt-1">PCA projection of the stored embeddings (semantic neighbors sit close).</div>
    </div>
  );
}

function MemPolicyPanel() {
  const q = trpc.memoryPolicy.useQuery(undefined, { refetchInterval: 8000 });
  const d = q.data as any;
  if (!d || !d.policy) return null;
  const kept = d.stored ?? 0, evicted = d.evicted ?? 0, seen = d.seen ?? 0;
  const top: any[] = d.top || [];
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 mb-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="text-[14px] font-semibold flex-1">Evolved memory policy · self-governing retention</div>
        <span className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">policy: {d.policy}</span>
      </div>
      <div className="text-[12px] text-slate-500 mb-3">Bounded store (cap {d.capacity ?? "∞"}) that self-governs: retrieval raises an item&rsquo;s usage-value; overflow evicts the lowest. Keeps what proved <b>useful</b>, not just recent (+25% retention, negative-control clean).</div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <div className="rounded-lg border border-slate-200 p-2.5"><div className="text-2xl font-bold tabular-nums">{seen}</div><div className="text-[11px] text-slate-500">seen</div></div>
        <div className="rounded-lg border border-slate-200 p-2.5"><div className="text-2xl font-bold text-emerald-600 tabular-nums">{kept}</div><div className="text-[11px] text-slate-500">kept (value-ranked)</div></div>
        <div className="rounded-lg border border-slate-200 p-2.5"><div className="text-2xl font-bold text-slate-400 tabular-nums">{evicted}</div><div className="text-[11px] text-slate-500">evicted (low value)</div></div>
      </div>
      <div className="flex flex-col gap-1">
        {top.slice(0, 8).map((t, i) => (
          <div key={i} className="flex items-center gap-2 text-[12px] py-1 border-b border-slate-50 last:border-0">
            <span className="w-7 text-[11px] text-slate-400 tabular-nums">v{t.value}</span>
            {t.critical && <span className="text-[10px] px-1.5 rounded bg-blue-50 text-blue-600 border border-blue-200">pinned</span>}
            <span className="flex-1 truncate text-slate-700">{t.text}</span>
          </div>
        ))}
        {!top.length && <div className="text-[12px] text-slate-400">Run an idea-board workflow &mdash; its value-governed store surfaces here.</div>}
      </div>
    </div>
  );
}

export function MemoryPane() {
  const experts = trpc.experts.useQuery(undefined, { refetchInterval: 8000 });
  const stats = trpc.stats.useQuery();
  const scatter = trpc.knowledgeScatter.useQuery({ limit: 400, collection: "knowledge" }, { refetchInterval: 10000 });
  const [tab, setTab] = useState<"list" | "map">("list");
  const [q, setQ] = useState(""), [sub, setSub] = useState("");
  const search = trpc.search.useQuery({ q: sub, collection: "knowledge", limit: 30 }, { enabled: !!sub });
  const know = trpc.knowledgeAll.useQuery({ limit: 300, domain: "" });
  const st = stats.data as any;
  const rows: any[] = sub ? ((search.data as any)?.hits || []) : ((know.data as any[]) || []);
  const exp = (experts.data as any[]) || [];
  const points = ((scatter.data as any)?.points as any[]) || [];
  return (
    <div className="flex-1 bg-slate-50 overflow-auto">
      <div className="max-w-[900px] mx-auto p-6">
        <div className="flex items-center gap-3 mb-1">
          <div className="text-[19px] font-bold">Memory</div>
          <div className="inline-flex rounded-lg bg-slate-100 p-0.5 text-[12px] ml-auto">
            <button onClick={() => setTab("list")} className={`px-3 py-1 rounded-md ${tab === "list" ? "bg-white shadow-sm font-medium" : "text-slate-500"}`}>List</button>
            <button onClick={() => setTab("map")} className={`px-3 py-1 rounded-md ${tab === "map" ? "bg-white shadow-sm font-medium" : "text-slate-500"}`}>Vector map</button>
          </div>
        </div>
        <div className="text-[13px] text-slate-500 mb-4">Everything the squads learned, persisted across runs · store: {st?.store || "…"} · {st?.knowledge ?? 0} items · {exp.length} experts</div>
        {tab === "map" && <div className="bg-white rounded-xl border border-slate-200 p-4 mb-4"><VectorMap points={points} /></div>}
        <MemPolicyPanel />
        <div className="bg-white rounded-xl border border-slate-200 p-4 mb-4">
          <div className="text-[14px] font-semibold mb-2">Experts · reusable specialists <span className="text-[11px] font-normal text-slate-400">(per code-area + per-agent)</span></div>
          <div className="grid grid-cols-2 gap-2">
            {exp.map((e) => <div key={e.id} className="border border-slate-200 rounded-lg p-3"><div className="flex items-center gap-1.5"><div className="text-[13px] font-medium flex-1 truncate">{e.specialty || e.name}</div>{e.kind === "area" && <span className="text-[10px] px-1.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200">area</span>}</div><div className="text-[11px] text-slate-400">{e.area ? `${e.area} · ` : ""}{e.n_knowledge} knowledge</div></div>)}
            {!exp.length && <div className="text-[13px] text-slate-400">No experts yet.</div>}
          </div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="text-[14px] font-semibold flex-1">Knowledge</div>
            <form onSubmit={(e) => { e.preventDefault(); setSub(q); }} className="flex gap-2">
              <input className="inp py-1.5 w-64 text-[13px]" value={q} onChange={(e) => setQ(e.target.value)} placeholder="search everything learned…" />
              <button className="btn btn-primary text-[13px]" type="submit">Search</button>
            </form>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {rows.map((h: any, i: number) => (
              <div key={i} className="border border-slate-200 rounded-lg p-2.5">
                <div className="text-[10px] text-blue-600 mb-0.5">{h.score ? `${h.score} · ` : ""}{h.domain || h.workflow || h.pipeline || "misc"}{h.agent ? ` · ${h.agent}` : ""}{h.move ? ` · ${h.move}` : ""}{h.chars ? ` · ${h.chars} chars` : ""}</div>
                <div className="text-[13px] text-slate-700 whitespace-pre-wrap">{h.text}</div>
                {Array.isArray(h.source) && h.source.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {h.source.map((s: string, j: number) => <span key={j} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 border border-slate-200">↪ {s}</span>)}
                  </div>
                )}
              </div>
            ))}
            {!rows.length && <div className="text-[13px] text-slate-400">{sub ? "no matches" : "nothing yet"}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
