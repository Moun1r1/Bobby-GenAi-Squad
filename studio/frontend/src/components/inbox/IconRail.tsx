import { Avatar } from "@/components/ui/Avatar";

function RailIcon({ path, label, active, onClick }: { path: string; label: string; active?: boolean; onClick?: () => void }) {
  return (
    <button onClick={onClick} title={label} className={`w-10 h-10 rounded-xl grid place-items-center ${active ? "bg-blue-50 text-blue-600" : "text-slate-400 hover:bg-slate-50"}`}>
      <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={path} /></svg>
    </button>
  );
}

export function IconRail({ nav, setNav }: { nav: "runs" | "memory"; setNav: (n: "runs" | "memory") => void }) {
  return (
    <div className="w-[64px] shrink-0 bg-white border-r border-slate-200 flex flex-col items-center py-4 gap-1">
      <div className="w-9 h-9 rounded-xl bg-blue-600 text-white grid place-items-center font-bold text-[15px] mb-3">B</div>
      <RailIcon active={nav === "runs"} onClick={() => setNav("runs")} label="Chats" path="M4 5h16v11H8l-4 4V5z" />
      <RailIcon active={nav === "memory"} onClick={() => setNav("memory")} label="Memory" path="M4 5a2 2 0 012-2h12v16H6a2 2 0 00-2 2V5zM8 3v14" />
      <div className="mt-auto"><Avatar name="Operator" size={30} /></div>
    </div>
  );
}
