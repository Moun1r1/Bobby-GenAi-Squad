import { Avatar } from "@/components/ui/Avatar";
import { Markdown } from "@/components/ui/Markdown";
import { fmtTime, type Msg } from "@/lib/events";

export function Bubble({ m }: { m: Msg }) {
  if (m.side === "system") {
    const tone =
      m.tone === "error" ? "bg-red-50 text-red-600 border-red-200"
      : m.tone === "verdict" ? "bg-amber-50 text-amber-700 border-amber-200"
      : m.tone === "ok" ? "bg-green-50 text-green-700 border-green-200"
      : "bg-slate-100 text-slate-500 border-slate-200";
    return (
      <div className="flex justify-center rise">
        <div className={`text-[11.5px] px-3 py-1 rounded-full border ${tone}`}>
          {m.text.replace(/\*\*/g, "")}
          {m.ts ? <span className="opacity-50 ml-1.5">{fmtTime(m.ts)}</span> : null}
        </div>
      </div>
    );
  }
  const right = m.side === "right";
  return (
    <div className={`flex gap-2.5 rise ${right ? "flex-row-reverse" : ""}`}>
      <Avatar name={m.who || "?"} size={32} />
      <div className={`max-w-[72%] flex flex-col ${right ? "items-end" : "items-start"}`}>
        <div className="flex items-baseline gap-2 mb-0.5">
          <span className="text-[12px] font-semibold text-slate-700">{m.who}</span>
          <span className="text-[10px] text-slate-400">{fmtTime(m.ts)}</span>
        </div>
        <div className={`px-3.5 py-2.5 rounded-2xl text-[14px] leading-relaxed break-words ${right ? "bg-blue-700 text-white rounded-tr-sm" : "bg-white border border-slate-200 text-slate-800 rounded-tl-sm"}`}>
          <Markdown compact invert={right}>{m.text}</Markdown>
          {m.sub && <div className={`mt-1.5 text-[12px] rounded-lg px-2 py-1 ${right ? "bg-blue-800/50" : "bg-slate-50 text-slate-500"}`}><Markdown compact invert={right}>{m.sub}</Markdown></div>}
        </div>
      </div>
    </div>
  );
}
