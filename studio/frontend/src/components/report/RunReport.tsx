"use client";
import { Markdown } from "@/components/ui/Markdown";
import { chName, reduceRun, type Ev } from "@/lib/events";

function buildMarkdown(runId: string, pipeline: string, events: Ev[]): string {
  const m = reduceRun(events);
  const lines: string[] = [];
  lines.push(`# ${chName(pipeline)} — run report`);
  lines.push(`\n\`${runId}\` · ${m.participants.length} agents · ${m.counts.messages} findings · ${m.counts.tools} tool calls · ${m.counts.cycles} cycles\n`);
  if (m.verdict) lines.push(`> **Verdict: ${m.verdict.verdict}** — ${m.verdict.metric || ""} ${m.verdict.detail || ""}\n`);
  if (m.converged) lines.push(`> ✓ **Converged** — every acceptance criterion verified.\n`);
  if (m.escalated) lines.push(`> ⚑ **Escalated** to human review.\n`);
  if (m.result?.summary) { lines.push(`## Summary\n`); lines.push(m.result.summary + "\n"); }
  if (m.findings.length) {
    lines.push(`## Findings (${m.findings.length})\n`);
    for (const f of m.findings) lines.push(`- **${f.who}** — ${f.text.replace(/\n/g, " ")}${f.sub ? `\n  - ${f.sub.replace(/\n/g, " ")}` : ""}`);
  }
  return lines.join("\n");
}

export function RunReport({ runId, pipeline, events, onClose, embedded }:
  { runId: string; pipeline: string; events: Ev[]; onClose: () => void; embedded?: boolean }) {
  const md = buildMarkdown(runId, pipeline, events);
  const download = () => {
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `report-${runId.slice(0, 8)}.md`; a.click();
    URL.revokeObjectURL(url);
  };
  const body = (
    <div className={embedded ? "w-full h-full bg-white flex flex-col" : "w-[640px] max-w-full h-full bg-white shadow-xl flex flex-col"} onClick={(e) => e.stopPropagation()}>
      <div className="h-[56px] shrink-0 border-b border-slate-200 flex items-center gap-2 px-5">
        <div className="text-[15px] font-semibold flex-1">Run report</div>
        <button className="btn btn-primary text-[13px]" onClick={download}>Export .md</button>
        <button className="btn btn-ghost text-[13px] text-slate-500" onClick={onClose}>{embedded ? "Back" : "Close"}</button>
      </div>
      <div className="flex-1 overflow-auto p-6"><Markdown>{md}</Markdown></div>
    </div>
  );
  if (embedded) return body;
  return <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/30" onClick={onClose}>{body}</div>;
}
