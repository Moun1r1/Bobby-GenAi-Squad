"use client";
import { useState } from "react";
import { chName } from "@/lib/events";

export function ComposeModal({ pipelines, launching, onClose, onLaunch }:
  { pipelines?: any[]; launching: boolean; onClose: () => void; onLaunch: (p: string, params: any) => void }) {
  const [goal, setGoal] = useState("");
  const [pipe, setPipe] = useState("goal");
  const native = (pipelines || []).filter((p) => p.kind === "native");
  const isText = pipe === "goal" || pipe === "persona" || pipe === "world" || pipe === "process_data";
  const key = pipe === "persona" ? "persona" : pipe === "world" ? "world" : "goal";
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-900/30" onClick={onClose}>
      <div className="w-[480px] max-w-[92vw] bg-white rounded-2xl shadow-xl p-5" onClick={(e) => e.stopPropagation()}>
        <div className="text-[16px] font-semibold mb-3">Start a run</div>
        <label className="text-[12px] text-slate-500">Capability</label>
        <select className="inp mt-1 mb-3 text-[14px]" value={pipe} onChange={(e) => setPipe(e.target.value)}>
          {native.map((p) => <option key={p.id} value={p.id}>{chName(p.id)}</option>)}
        </select>
        {isText && <>
          <label className="text-[12px] text-slate-500">{pipe === "persona" ? "Describe the persona" : pipe === "world" ? "Describe the world" : "Goal"}</label>
          <textarea className="inp mt-1 h-24 resize-none text-[14px]" value={goal} onChange={(e) => setGoal(e.target.value)}
            placeholder={pipe === "persona" ? "A witty 1840s mathematician…" : pipe === "world" ? "A lively town square where neighbours meet…" : "Map every risk in this contract and cite each…"} />
        </>}
        <div className="flex justify-end gap-2 mt-4">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" disabled={launching || (isText && !goal.trim())} onClick={() => onLaunch(pipe, isText ? { [key]: goal } : {})}>
            {launching ? "Starting…" : "Start run"}
          </button>
        </div>
      </div>
    </div>
  );
}
