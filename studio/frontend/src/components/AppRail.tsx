import { Avatar } from "@/components/ui/Avatar";

export type Page =
  | "chat" | "world" | "timeline" | "compare"
  | "workflows" | "datasets"
  | "squad" | "ideaboard"
  | "experiments" | "primitives"
  | "compute"
  | "memory" | "vault" | "knowmap" | "notebook"
  | "analytics" | "cost" | "models" | "approvals" | "settings";

// The UI is organized by ENGINE LAYER, not a flat page list. Each layer holds its related views as sub-tabs.
export type Layer = { id: string; label: string; note: string; path: string; pages: Page[] };

export const LAYERS: Layer[] = [
  { id: "run", label: "Run", note: "the generative loop, live", pages: ["chat", "world", "timeline", "compare"],
    path: "M4 4v6h6M20 20v-6h-6M20 8a8 8 0 00-14-2M4 16a8 8 0 0014 2" },
  { id: "compose", label: "Compose", note: "the SELF — role + goal", pages: ["workflows", "datasets"],
    path: "M12 5v14M5 12h14" },
  { id: "squad", label: "Squad", note: "coordination — squad_solve", pages: ["squad", "ideaboard"],
    path: "M17 20v-2a4 4 0 00-4-4H7a4 4 0 00-4 4v2M9 8a3 3 0 100-6 3 3 0 000 6zM21 20v-2a4 4 0 00-3-3.9" },
  { id: "prove", label: "Prove", note: "verify-by-outcome", pages: ["experiments", "primitives"],
    path: "M9 3v6l-5 9a2 2 0 002 3h12a2 2 0 002-3l-5-9V3M8 3h8M7 15h10" },
  { id: "compute", label: "Compute", note: "GPU worker · training", pages: ["compute"],
    path: "M4 4h16v12H4zM8 20h8M12 16v4M9 8h6M9 11h6" },
  { id: "memory", label: "Memory", note: "persistent-self + knowledge", pages: ["memory", "vault", "knowmap", "notebook"],
    path: "M9 4a3 3 0 00-3 3 3 3 0 000 6 3 3 0 003 3V4zM15 4a3 3 0 013 3 3 3 0 010 6 3 3 0 01-3 3V4z" },
  { id: "system", label: "System", note: "ops & config", pages: ["analytics", "cost", "models", "approvals", "settings"],
    path: "M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 13a7.5 7.5 0 000-2l2-1.5-2-3.4-2.3 1a7.5 7.5 0 00-1.7-1L15 3H9l-.4 2.6a7.5 7.5 0 00-1.7 1l-2.3-1-2 3.4L2.6 11a7.5 7.5 0 000 2l-2 1.5 2 3.4 2.3-1a7.5 7.5 0 001.7 1L9 21h6l.4-2.6a7.5 7.5 0 001.7-1l2.3 1 2-3.4-2-1.5z" },
];

export const PAGE_LABEL: Record<Page, string> = {
  chat: "Conversations", world: "World", timeline: "Replay", compare: "Compare",
  workflows: "Workflows", datasets: "Datasets", squad: "Squad", ideaboard: "Idea board",
  experiments: "Proof bench", primitives: "Primitives", compute: "Compute", memory: "Memory", vault: "Knowledge vault", knowmap: "Knowledge map", notebook: "Notebook",
  analytics: "Analytics", cost: "Cost", models: "Models", approvals: "Approvals", settings: "Settings",
};

export const layerOf = (p: Page): Layer => LAYERS.find((l) => l.pages.includes(p)) || LAYERS[0];

export function AppRail({ page, setPage }: { page: Page; setPage: (p: Page) => void }) {
  const active = layerOf(page).id;
  return (
    <div className="w-[76px] shrink-0 bg-white border-r border-slate-200 flex flex-col items-center py-4 gap-1.5">
      <div className="w-9 h-9 rounded-xl bg-blue-600 text-white grid place-items-center font-bold text-[15px] mb-2">B</div>
      {LAYERS.map((l) => (
        <button key={l.id} onClick={() => setPage(l.pages[0])} title={l.note}
          className={`w-16 py-2 rounded-xl grid place-items-center gap-1 transition ${active === l.id ? "bg-blue-50 text-blue-600" : "text-slate-400 hover:bg-slate-50"}`}>
          <svg viewBox="0 0 24 24" className="w-[19px] h-[19px]" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d={l.path} /></svg>
          <span className="text-[10px] font-medium">{l.label}</span>
        </button>
      ))}
      <div className="mt-auto"><Avatar name="Operator" size={30} /></div>
    </div>
  );
}
