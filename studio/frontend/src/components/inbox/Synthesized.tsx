"use client";
import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { CitedMarkdown } from "@/components/ui/CitedMarkdown";
import { Markdown } from "@/components/ui/Markdown";
import { Avatar } from "@/components/ui/Avatar";
import { chName, reduceRun, type Ev } from "@/lib/events";

const firstSentence = (t: string) => { const s = (t || "").replace(/\s+/g, " ").trim(); const m = s.match(/^(.{20,200}?[.!?])(\s|$)/); return (m ? m[1] : s.slice(0, 180)); };
const verdictVariant = (v?: string) => v === "WIRE" || v === "PROVEN" ? "proven" : v === "DELETE" ? "dead" : v ? "warn" : "outline";

// Perplexity-style synthesis: source cards on top, a cited answer, then an expandable sources list.
export function Synthesized({ pipeline, events }: { pipeline: string; events: Ev[] }) {
  const model = useMemo(() => reduceRun(events), [events]);
  const sources = model.findings;
  const snippets = sources.map((f) => `${f.who}: ${f.text}`);
  const answer = useMemo(() => {
    const lead = model.result?.summary
      || `The **${chName(pipeline)}** squad produced ${sources.length} finding${sources.length === 1 ? "" : "s"} across ${model.participants.length} agent${model.participants.length === 1 ? "" : "s"}${model.verdict ? `, concluding **${model.verdict.verdict}**` : ""}.`;
    const points = sources.slice(0, 12).map((f, i) => `- ${firstSentence(f.text)} [${i + 1}](#src-${i + 1})`).join("\n");
    return points ? `${lead}\n\n**Key findings**\n\n${points}` : lead;
  }, [model, pipeline, sources]);

  if (!sources.length && !model.result) {
    return <div className="flex-1 grid place-items-center text-[13px] text-slate-400">Nothing to synthesize yet — the squad hasn't produced findings.</div>;
  }
  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-[760px] mx-auto flex flex-col gap-5">
        {/* source cards row (Perplexity-style) */}
        {sources.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 mb-2">Sources · {sources.length}</div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {sources.slice(0, 12).map((f, i) => (
                <button key={i} onClick={() => { const el = document.getElementById(`src-${i + 1}`); el?.scrollIntoView({ behavior: "smooth", block: "center" }); el?.classList.add("ring-2", "ring-blue-400"); setTimeout(() => el?.classList.remove("ring-2", "ring-blue-400"), 1200); }}
                  className="shrink-0 w-[180px] text-left rounded-lg border border-slate-200 bg-white hover:border-blue-300 p-2.5">
                  <div className="flex items-center gap-1.5 mb-1"><span className="w-4 h-4 rounded bg-blue-100 text-blue-700 text-[10px] grid place-items-center font-semibold">{i + 1}</span><Avatar name={f.who} size={16} /><span className="text-[11px] text-slate-500 truncate">{f.who}</span></div>
                  <div className="text-[12px] text-slate-700 line-clamp-3 leading-snug">{firstSentence(f.text)}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* answer */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <h2 className="text-[18px] font-bold flex-1">Answer</h2>
            {model.verdict && <Badge variant={verdictVariant(model.verdict.verdict)}>{model.verdict.verdict}</Badge>}
            {model.converged && <Badge variant="proven">converged</Badge>}
            {model.escalated && <Badge variant="warn">escalated</Badge>}
          </div>
          <CitedMarkdown sources={snippets}>{answer}</CitedMarkdown>
        </div>

        {/* all sources, expandable */}
        {sources.length > 0 && (
          <div className="rounded-xl border border-slate-200 bg-white">
            <div className="px-4 py-2.5 text-[13px] font-semibold border-b border-slate-100">All sources</div>
            <div className="px-4">
              <Accordion type="multiple">
                {sources.map((f, i) => (
                  <AccordionItem key={i} value={`src-${i + 1}`} id={`src-${i + 1}`} className="rounded-md">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <Badge variant="outline" className="w-5 justify-center px-0">{i + 1}</Badge>
                        <span className="text-slate-500 font-normal">{f.who}</span>
                        <span className="truncate max-w-[420px]">{firstSentence(f.text)}</span>
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <Markdown compact>{f.text}</Markdown>
                      {f.sub && <div className="mt-2 rounded-md bg-slate-50 p-2 text-[12px] text-slate-500"><Markdown compact>{f.sub}</Markdown></div>}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
