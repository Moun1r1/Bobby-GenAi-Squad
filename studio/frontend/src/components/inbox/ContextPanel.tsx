"use client";
import { useState } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { chName, fmtTime, type Ev } from "@/lib/events";

type Note = { text: string; ts: number };

function Row({ icon, label, value }: { icon: string; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 py-1.5">
      <svg viewBox="0 0 24 24" className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={icon} /></svg>
      <div className="min-w-0 flex-1"><div className="text-[11px] text-slate-400">{label}</div><div className="text-[13px] text-slate-700 break-words">{value}</div></div>
    </div>
  );
}

export function ContextPanel({ runId, pipeline, status, participants, verdict, notes, onNote, onClose }:
  { runId: string; pipeline: string; status: string; participants: string[]; verdict?: Ev | null; notes: Note[]; onNote: (text: string) => void; onClose: () => void }) {
  const [text, setText] = useState("");
  const add = () => { if (text.trim()) { onNote(text.trim()); setText(""); } };
  return (
    <div className="w-[300px] shrink-0 bg-white border-l border-slate-200 flex flex-col">
      <div className="h-[60px] shrink-0 flex items-center gap-2 px-4 border-b border-slate-200">
        <Avatar name={pipeline} size={34} />
        <div className="min-w-0 flex-1"><div className="text-[14px] font-semibold truncate">{chName(pipeline)}</div><div className="text-[11px] text-slate-400">{status}</div></div>
        <button className="text-slate-400 hover:text-slate-700 text-[13px]" onClick={onClose}>✕</button>
      </div>

      <div className="p-4">
        <Row icon="M4 5h16v14H4zM4 9h16" label="Channel" value={chName(pipeline)} />
        <Row icon="M4 6h16M4 12h16M4 18h10" label="Run ID" value={<span className="font-mono text-[12px]">{runId}</span>} />
        <Row icon="M20 6L9 17l-5-5" label="Status" value={<span className="capitalize">{status}</span>} />
        <Row icon="M17 20v-2a4 4 0 00-4-4H7a4 4 0 00-4 4v2M9 8a3 3 0 100-6 3 3 0 000 6z" label="Agents" value={
          <div className="flex flex-wrap gap-1 mt-0.5">{participants.length ? participants.map((p) => <span key={p} className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-1.5 py-0.5 text-[11px]"><Avatar name={p} size={16} />{p}</span>) : "—"}</div>
        } />
        {verdict && <Row icon="M9 12l2 2 4-4" label="Verdict" value={<Badge variant={verdict.verdict === "WIRE" ? "proven" : "warn"}>{verdict.verdict}</Badge>} />}
      </div>

      <Separator />
      <div className="p-4 flex-1 overflow-auto">
        <div className="text-[13px] font-semibold mb-2">Notes</div>
        <div className="rounded-lg border border-slate-200 p-2 mb-3">
          <textarea className="w-full text-[13px] outline-none resize-none placeholder:text-slate-400" rows={2} value={text} onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); add(); } }} placeholder="Write a note…" />
          <div className="flex justify-end"><button className="btn btn-primary text-[12px] py-1" disabled={!text.trim()} onClick={add}>Add note</button></div>
        </div>
        <div className="flex flex-col gap-3">
          {[...notes].reverse().map((n, i) => (
            <div key={i} className="flex gap-2">
              <Avatar name="Operator" size={26} />
              <div><div className="flex items-baseline gap-2"><span className="text-[12px] font-medium">You</span><span className="text-[10px] text-slate-400">{fmtTime(n.ts)}</span></div><div className="text-[13px] text-slate-600">{n.text}</div></div>
            </div>
          ))}
          {!notes.length && <div className="text-[12px] text-slate-400">No notes yet.</div>}
        </div>
      </div>
    </div>
  );
}
