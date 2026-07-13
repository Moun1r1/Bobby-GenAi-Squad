"use client";
import { trpc } from "@/lib/trpc";
import { PageShell, Soon } from "@/components/ui/PageShell";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const MODELS = [
  { name: "claude-opus", role: "critique · spec · teacher", tier: "frontier" },
  { name: "claude-sonnet", role: "workers", tier: "frontier" },
  { name: "qwen (MoE)", role: "local squad · best student", tier: "local" },
  { name: "gemma", role: "reasoning specialist", tier: "local" },
  { name: "nomic-embed", role: "embeddings", tier: "local" },
];

export function ModelsPage() {
  const health = trpc.health.useQuery(undefined, { refetchInterval: 8000 });
  const h = health.data as any;
  return (
    <PageShell title="Models &amp; providers" desc="The models and endpoints the squads route through.">
      <Card className="mb-4"><CardContent className="p-4 flex items-center gap-3">
        <span className={`w-2.5 h-2.5 rounded-full ${h?.ok ? "bg-green-500" : "bg-red-500"}`} />
        <span className="text-[13px]">{h?.ok ? `backend up · store ${h.store} · ${h.pipelines} pipelines` : "backend down"}</span>
      </CardContent></Card>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {MODELS.map((m) => (
          <Card key={m.name}><CardContent className="p-4 flex items-center gap-3">
            <div className="flex-1"><div className="text-[14px] font-semibold">{m.name}</div><div className="text-[12px] text-slate-500">{m.role}</div></div>
            <Badge variant={m.tier === "frontier" ? "active" : "outline"}>{m.tier}</Badge>
          </CardContent></Card>
        ))}
      </div>
      <Soon what="Editable routing policy, endpoint config (DGX / Ollama), spec-decode and the teacher-student flywheel meter need a config API. The list above reflects the framework's configured models." />
    </PageShell>
  );
}
