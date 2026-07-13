"use client";
import { useMemo, useState } from "react";
import { trpc } from "@/lib/trpc";
import { PageShell } from "@/components/ui/PageShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Node = { id: string; vault: string; title: string; tags: string[]; source: string; chars: number; links: string[] };
type Edge = { source: string; target: string };

const VAULT_COLOR: Record<string, string> = {
  foundation: "#2563eb", repos: "#475569", gemma: "#059669", experience: "#7c3aed", behavior: "#db2777",
};
const colorOf = (vault: string) => VAULT_COLOR[vault] || "#64748b";
const noteName = (id: string) => (id.includes("/") ? id.split("/").slice(1).join("/") : id);

// deterministic radial layout — no physics lib; nodes on a ring, angle by index (stable across renders).
function layout(nodes: Node[], w: number, h: number) {
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 90;
  const pos: Record<string, { x: number; y: number }> = {};
  nodes.forEach((n, i) => {
    const a = (i / Math.max(1, nodes.length)) * Math.PI * 2 - Math.PI / 2;
    pos[n.id] = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  });
  return pos;
}

function Graph({ nodes, edges, sel, onSel }: { nodes: Node[]; edges: Edge[]; sel: string | null; onSel: (id: string) => void }) {
  const W = 1000, H = 620;
  const pos = useMemo(() => layout(nodes, W, H), [nodes]);
  const near = useMemo(() => {
    if (!sel) return new Set<string>();
    const s = new Set<string>([sel]);
    edges.forEach((e) => { if (e.source === sel) s.add(e.target); if (e.target === sel) s.add(e.source); });
    return s;
  }, [sel, edges]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[620px]">
      {edges.map((e, i) => {
        const a = pos[e.source], b = pos[e.target];
        if (!a || !b) return null;
        const hot = sel && (e.source === sel || e.target === sel);
        return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={hot ? "#3b82f6" : "#e2e8f0"} strokeWidth={hot ? 1.6 : 0.8} />;
      })}
      {nodes.map((n) => {
        const p = pos[n.id]; if (!p) return null;
        const dim = sel && !near.has(n.id);
        const R = 6 + Math.min(10, n.links.length * 1.4);
        return (
          <g key={n.id} transform={`translate(${p.x},${p.y})`} onClick={() => onSel(n.id)} className="cursor-pointer" opacity={dim ? 0.28 : 1}>
            <circle r={R} fill={colorOf(n.vault)} stroke={sel === n.id ? "#0a0a0a" : "white"} strokeWidth={sel === n.id ? 2.5 : 1.5} />
            <text x={0} y={R + 12} textAnchor="middle" className="text-[10px] fill-slate-600 select-none" style={{ fontFamily: "var(--font-geist-mono, monospace)" }}>{noteName(n.id)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function NavigateBox() {
  const [q, setQ] = useState("how to train a gemma4 model natively");
  const [submitted, setSubmitted] = useState(q);
  const nav = trpc.vaultNavigate.useQuery({ q: submitted, k: 3, hops: 1 }, { enabled: !!submitted });
  const d = nav.data as any;
  return (
    <Card>
      <CardHeader><CardTitle>Navigate <span className="text-[11px] font-normal text-slate-400">what an agent recalls for a step</span></CardTitle></CardHeader>
      <CardContent>
        <form onSubmit={(e) => { e.preventDefault(); setSubmitted(q); }} className="flex gap-2 mb-3">
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="a step's target…"
            className="flex-1 rounded-lg border border-slate-200 px-3 py-1.5 text-[13px] focus:outline-none focus:ring-2 focus:ring-blue-200" />
          <button className="rounded-lg bg-blue-600 text-white px-3 py-1.5 text-[13px] font-medium">Navigate</button>
        </form>
        {d && (
          <>
            <div className="flex flex-wrap gap-1.5 mb-2">
              <span className="text-[11px] font-semibold uppercase text-slate-400 mr-1">Entry</span>
              {(d.entry || []).map((id: string) => <Badge key={id} variant="proven">{id}</Badge>)}
            </div>
            <pre className="text-[11px] font-mono bg-slate-50 rounded-lg p-3 max-h-72 overflow-auto whitespace-pre-wrap text-slate-700">{d.block || "(nothing matched)"}</pre>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function VaultPage() {
  const graphQ = trpc.vaultGraph.useQuery(undefined, { refetchInterval: 8000 });
  const [sel, setSel] = useState<string | null>(null);
  const noteQ = trpc.vaultNote.useQuery({ id: sel || "" }, { enabled: !!sel });
  const g = graphQ.data as any;
  const nodes: Node[] = g?.nodes || [];
  const edges: Edge[] = g?.edges || [];
  const stats = g?.stats || {};
  const note = noteQ.data as any;
  const vaults: string[] = stats.names || Array.from(new Set(nodes.map((n) => n.vault)));

  return (
    <PageShell title="Knowledge vault" wide
      desc="The navigable, enrichable AI-knowledge graph the swarm reasons from — click a note to read it and hop its [[links]].">
      <div className="flex items-center gap-3 mb-3 text-[12px] text-slate-500">
        <Badge variant="outline">{stats.notes ?? nodes.length} notes</Badge>
        <Badge variant="outline">{stats.edges ?? edges.length} links</Badge>
        <Badge variant={stats.embed ? "proven" : "warn"}>{stats.embed ? "semantic entry" : "lexical entry"}</Badge>
        <Badge variant="outline">{vaults.length} vaults</Badge>
        <span className="flex items-center gap-3 ml-2">
          {vaults.map((s) => <span key={s} className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: colorOf(s) }} />{s} <span className="text-slate-400">{stats.vaults?.[s]?.notes ?? ""}</span></span>)}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2 flex flex-col gap-3">
          <Card><CardContent className="p-1">
            {nodes.length ? <Graph nodes={nodes} edges={edges} sel={sel} onSel={setSel} />
              : <div className="p-8 text-center text-[13px] text-slate-400">Loading the vault graph…</div>}
          </CardContent></Card>
          <NavigateBox />
        </div>
        <Card>
          <CardHeader><CardTitle className="truncate">{sel ? note?.title || sel : "Select a note"}</CardTitle></CardHeader>
          <CardContent className="flex flex-col gap-2 max-h-[720px] overflow-auto">
            {!sel && <div className="text-[12px] text-slate-400">Click a node to read the note. Larger nodes have more links. Colors = source.</div>}
            {sel && note && (
              <>
                <div className="flex flex-wrap gap-1.5">
                  {note.source && <Badge variant="outline">{note.source}</Badge>}
                  {(note.tags || []).map((t: string) => <Badge key={t} variant="outline">{t}</Badge>)}
                </div>
                {(note.neighbors || []).length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    <span className="text-[11px] font-semibold uppercase text-slate-400 mr-1">Hop</span>
                    {(note.neighbors as string[]).map((nb) => (
                      <button key={nb} onClick={() => setSel(nb)} className="text-[11px] px-2 py-0.5 rounded-full border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100">{nb}</button>
                    ))}
                  </div>
                )}
                <pre className="text-[11px] font-mono bg-slate-50 rounded-lg p-3 whitespace-pre-wrap text-slate-700 leading-relaxed">{note.body}</pre>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </PageShell>
  );
}
