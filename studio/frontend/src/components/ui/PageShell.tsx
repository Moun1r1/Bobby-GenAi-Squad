import * as React from "react";

export function PageShell({ title, desc, right, children, wide }:
  { title: string; desc?: string; right?: React.ReactNode; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className="flex-1 overflow-auto bg-slate-50">
      <div className={`${wide ? "max-w-[1200px]" : "max-w-[1000px]"} mx-auto p-6`}>
        <div className="flex items-start justify-between mb-4 gap-4">
          <div><h1 className="text-[22px] font-bold">{title}</h1>{desc && <p className="text-[14px] text-slate-500 mt-1">{desc}</p>}</div>
          {right}
        </div>
        {children}
      </div>
    </div>
  );
}

export function Soon({ what }: { what: string }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-white p-6 text-[13px] text-slate-500">
      <span className="inline-flex items-center gap-1.5 text-amber-600 font-medium mr-2">◔ needs backend</span>{what}
    </div>
  );
}
