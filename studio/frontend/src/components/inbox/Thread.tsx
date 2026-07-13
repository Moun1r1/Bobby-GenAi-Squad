"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Bubble } from "@/components/inbox/Bubble";
import { Synthesized } from "@/components/inbox/Synthesized";
import { chName, toMsg, type Ev, type Msg } from "@/lib/events";

export function Thread({ runId, pipeline, events, live, control, onClose, onReport, onNote, onToggleContext }:
  { runId: string; pipeline: string; events: Ev[]; live: boolean; control: any; onClose: () => void; onReport: () => void; onNote: (text: string) => void; onToggleContext: () => void }) {
  const feedRef = useRef<HTMLDivElement>(null);
  const [text, setText] = useState("");
  const [note, setNote] = useState(false);
  const [view, setView] = useState<"raw" | "synth">("raw");
  const msgs = useMemo(() => events.map(toMsg).filter(Boolean) as Msg[], [events]);
  const participants = useMemo(() => [...new Set(events.filter((e) => e.agent || (e.kind === "agent" && e.name)).map((e) => e.agent || e.name))], [events]);
  useEffect(() => { if (view === "raw") feedRef.current?.scrollTo({ top: 1e9, behavior: "smooth" }); }, [msgs.length, view]);
  const submit = () => { if (!text.trim()) return; if (note) onNote(text.trim()); else control.mutate({ runId, action: "steer", text }); setText(""); };

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-slate-50">
      {/* header */}
      <div className="h-[60px] shrink-0 bg-white border-b border-slate-200 flex items-center gap-3 px-5">
        <Avatar name={pipeline} size={38} />
        <div className="min-w-0">
          <div className="text-[15px] font-semibold flex items-center gap-2">{chName(pipeline)}
            <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${live ? "bg-green-100 text-green-700" : "bg-slate-100 text-slate-500"}`}>{live ? "● live" : "done"}</span>
          </div>
          <div className="text-[11px] text-slate-400">{runId} · {participants.length} agents</div>
        </div>
        {/* Raw ↔ Synthesized toggle */}
        <div className="ml-3 inline-flex rounded-lg bg-slate-100 p-0.5 text-[12px]">
          <button onClick={() => setView("raw")} className={`px-3 py-1 rounded-md ${view === "raw" ? "bg-white shadow-sm font-medium text-slate-900" : "text-slate-500"}`}>Raw squad log</button>
          <button onClick={() => setView("synth")} className={`px-3 py-1 rounded-md ${view === "synth" ? "bg-white shadow-sm font-medium text-slate-900" : "text-slate-500"}`}>Synthesized</button>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <button className="btn btn-ghost text-[12px] py-1" onClick={onReport}>Report</button>
          {live && <>
            <button className="btn btn-ghost text-[12px] py-1" onClick={() => control.mutate({ runId, action: "pause" })}>Pause</button>
            <button className="btn btn-ghost text-[12px] py-1" onClick={() => control.mutate({ runId, action: "resume" })}>Resume</button>
            <button className="btn btn-ghost text-[12px] py-1 text-red-600" onClick={() => control.mutate({ runId, action: "stop" })}>Stop</button>
          </>}
          <button className="btn btn-ghost text-[12px] py-1" title="Details" onClick={onToggleContext}>
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M17 20v-2a4 4 0 00-4-4H7a4 4 0 00-4 4v2M9 8a3 3 0 100-6 3 3 0 000 6z" /></svg>
          </button>
          <button className="btn btn-ghost text-[12px] py-1 text-slate-500" onClick={onClose}>Close</button>
        </div>
      </div>

      {/* body: raw feed or synthesized answer */}
      {view === "synth" ? <Synthesized pipeline={pipeline} events={events} /> : (
        <div ref={feedRef} className="flex-1 overflow-auto px-6 py-5 flex flex-col gap-2.5">
          {msgs.map((m) => <Bubble key={m.seq} m={m} />)}
          {!msgs.length && <div className="text-center text-[13px] text-slate-400 mt-10">{live ? "The squad is warming up…" : "No messages in this run."}</div>}
        </div>
      )}

      {/* composer */}
      <div className="shrink-0 bg-white border-t border-slate-200 px-5 py-3">
        <div className="flex gap-4 mb-2 text-[13px]">
          <button onClick={() => setNote(false)} className={`pb-1 border-b-2 ${!note ? "border-blue-600 text-slate-900 font-medium" : "border-transparent text-slate-400"}`}>Steer squad</button>
          <button onClick={() => setNote(true)} className={`pb-1 border-b-2 ${note ? "border-amber-500 text-slate-900 font-medium" : "border-transparent text-slate-400"}`}>Private note</button>
        </div>
        <textarea className="inp h-16 resize-none text-[14px]" value={text} onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          placeholder={note ? "Private note (saved to Details, not sent to the squad)…" : live ? "Shift + Enter for a new line. Steer the squad — inject a directive…" : "Run is done — start a new one to steer."}
          disabled={!note && !live} />
        <div className="flex items-center justify-end mt-2">
          <button className="btn btn-primary text-[13px]" disabled={!text.trim() || (!note && !live)} onClick={submit}>{note ? "Add note" : "Send"}</button>
        </div>
      </div>
    </div>
  );
}
