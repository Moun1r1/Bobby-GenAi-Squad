"use client";
import { useEffect, useState } from "react";
import { trpc } from "@/lib/trpc";
import { PageShell, Soon } from "@/components/ui/PageShell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Settings = { agents: number; patience: number; max_units: number };
const DEFAULTS: Settings = { agents: 3, patience: 2, max_units: 60 };

export function SettingsPage() {
  const cfg = trpc.config.useQuery();
  const save = trpc.setConfig.useMutation({ onSuccess: () => { cfg.refetch(); setSaved(true); setTimeout(() => setSaved(false), 1500); } });
  const [s, setS] = useState<Settings>(DEFAULTS);
  const [saved, setSaved] = useState(false);
  useEffect(() => { if (cfg.data) setS({ ...DEFAULTS, ...(cfg.data as any) }); }, [cfg.data]);
  const Field = ({ k, label, min, max }: { k: keyof Settings; label: string; min: number; max: number }) => (
    <label className="flex items-center gap-3 text-[13px] text-slate-600">
      <span className="w-40">{label}</span>
      <input type="range" min={min} max={max} value={s[k]} onChange={(e) => setS({ ...s, [k]: +e.target.value })} className="accent-blue-600 flex-1" />
      <span className="w-10 text-right font-semibold tabular-nums">{s[k]}</span>
    </label>
  );

  return (
    <PageShell title="Settings" desc="Run defaults, persisted on the backend (config.json) and shared across sessions.">
      <Card className="mb-4">
        <CardHeader><CardTitle>Run defaults</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-3">
          <Field k="agents" label="Default agents" min={1} max={6} />
          <Field k="patience" label="Plateau patience" min={1} max={5} />
          <Field k="max_units" label="Max data units" min={10} max={200} />
          <div className="flex items-center gap-3"><button className="btn btn-primary self-start" disabled={save.isPending} onClick={() => save.mutate(s)}>{save.isPending ? "Saving…" : "Save"}</button>{saved && <span className="text-[12px] text-green-600">saved ✓</span>}</div>
        </CardContent>
      </Card>
      <Soon what="Provider keys, embedding endpoint (BOBBY_EMBED_URL), budget caps and theme still live in backend env — a fuller config API exposes those next." />
    </PageShell>
  );
}
