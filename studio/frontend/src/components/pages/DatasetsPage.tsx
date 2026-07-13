"use client";
import { useEffect, useState } from "react";
import { trpc } from "@/lib/trpc";
import { PageShell } from "@/components/ui/PageShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const KINDS = [["hf", "HF dataset"], ["url", "URL"], ["pdf", "PDF path"], ["csv", "CSV"], ["json", "JSON"], ["text", "Text"]] as const;

export function DatasetsPage() {
  const runsQ = trpc.runs.useQuery(undefined, { refetchInterval: 4000 });
  const launch = trpc.launch.useMutation({ onSuccess: () => runsQ.refetch() });
  const [kind, setKind] = useState("hf");
  const [source, setSource] = useState("");
  const [goal, setGoal] = useState("");
  const [saved, setSaved] = useState<any[]>([]);
  useEffect(() => { try { setSaved(JSON.parse(localStorage.getItem("bobby_datasets") || "[]")); } catch { /* ignore */ } }, []);
  const usePath = kind === "hf" || kind === "url" || kind === "pdf";
  const run = () => {
    const params: any = { goal: goal || "Read this dataset end-to-end and build a structured knowledge map", kind };
    if (usePath) params.source = source; else params.data = source;
    const rec = { kind, source: source.slice(0, 60), ts: Date.now() };
    const next = [rec, ...saved].slice(0, 12); setSaved(next); localStorage.setItem("bobby_datasets", JSON.stringify(next));
    launch.mutate({ pipeline: "process_data", params });
  };
  const dataRuns = ((runsQ.data as any[]) || []).filter((r) => r.pipeline === "process_data");

  return (
    <PageShell title="Datasets" desc="Point a squad at any data source — HF datasets, URLs, PDFs, CSV/JSON, or pasted text — and read it end-to-end.">
      <Card className="mb-4">
        <CardHeader><CardTitle>New data source</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex gap-1 flex-wrap">
            {KINDS.map(([k, l]) => <button key={k} onClick={() => setKind(k)} className={`pill px-2.5 py-1 text-[12px] rounded-full border ${kind === k ? "border-blue-600 text-blue-700 bg-blue-50" : "border-slate-200 text-slate-500"}`}>{l}</button>)}
          </div>
          <input className="inp text-[14px]" value={source} onChange={(e) => setSource(e.target.value)}
            placeholder={kind === "hf" ? "stanfordnlp/imdb  ·  HuggingFaceGECLM/REDDIT_comments" : kind === "url" ? "https://…" : kind === "pdf" ? "/path/to/file.pdf" : "paste your data…"} />
          <input className="inp text-[14px]" value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="What should the squad extract? (optional goal)" />
          <button className="btn btn-primary self-start" disabled={launch.isPending || !source.trim()} onClick={run}>{launch.isPending ? "Starting…" : "▶ Read with a squad"}</button>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-3">
        <Card><CardHeader><CardTitle>Recent sources</CardTitle></CardHeader><CardContent className="flex flex-col gap-1.5">
          {saved.map((s, i) => <button key={i} onClick={() => { setKind(s.kind); setSource(s.source); }} className="flex items-center gap-2 text-left text-[12px] hover:bg-slate-50 rounded px-2 py-1"><Badge variant="outline">{s.kind}</Badge><span className="truncate">{s.source}</span></button>)}
          {!saved.length && <div className="text-[12px] text-slate-400">no sources yet</div>}
        </CardContent></Card>
        <Card><CardHeader><CardTitle>Data runs</CardTitle></CardHeader><CardContent className="flex flex-col gap-1.5">
          {dataRuns.slice(0, 10).map((r) => <div key={r.run_id} className="flex items-center gap-2 text-[12px]"><span className="font-mono text-slate-500">{r.run_id.slice(0, 8)}</span><span className="text-slate-400">{r.status}</span></div>)}
          {!dataRuns.length && <div className="text-[12px] text-slate-400">no data runs yet</div>}
        </CardContent></Card>
      </div>
    </PageShell>
  );
}
