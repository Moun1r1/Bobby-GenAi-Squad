"use client";
import { useState } from "react";
import { AppRail, PAGE_LABEL, layerOf, type Page } from "@/components/AppRail";
import { ConversationsPage } from "@/components/pages/ConversationsPage";
import { WorkflowsPage } from "@/components/pages/WorkflowsPage";
import { WorldPage } from "@/components/pages/WorldPage";
import { SquadPage } from "@/components/pages/SquadPage";
import { ExperimentsPage } from "@/components/pages/ExperimentsPage";
import { PrimitivesPage } from "@/components/pages/PrimitivesPage";
import { ComputePage } from "@/components/pages/ComputePage";
import { IdeaBoardPage } from "@/components/pages/IdeaBoardPage";
import { KnowledgeMapPage } from "@/components/pages/KnowledgeMapPage";
import { MemoryPage } from "@/components/pages/MemoryPage";
import { VaultPage } from "@/components/pages/VaultPage";
import { NotebookPage } from "@/components/pages/NotebookPage";
import { AnalyticsPage } from "@/components/pages/AnalyticsPage";
import { DatasetsPage } from "@/components/pages/DatasetsPage";
import { TimelinePage } from "@/components/pages/TimelinePage";
import { ComparePage } from "@/components/pages/ComparePage";
import { ApprovalsPage } from "@/components/pages/ApprovalsPage";
import { CostPage } from "@/components/pages/CostPage";
import { ModelsPage } from "@/components/pages/ModelsPage";
import { SettingsPage } from "@/components/pages/SettingsPage";

export function App() {
  const [page, setPage] = useState<Page>("chat");
  const [jumpRun, setJumpRun] = useState<string | null>(null);
  const goChat = (runId: string) => { setJumpRun(runId); setPage("chat"); };

  const NODE: Record<Page, React.ReactNode> = {
    chat: <ConversationsPage jumpRun={jumpRun} onConsumed={() => setJumpRun(null)} />,
    world: <WorldPage />, timeline: <TimelinePage />, compare: <ComparePage />,
    workflows: <WorkflowsPage onLaunched={goChat} />, datasets: <DatasetsPage />,
    squad: <SquadPage />, ideaboard: <IdeaBoardPage />,
    experiments: <ExperimentsPage />, primitives: <PrimitivesPage />,
    compute: <ComputePage />,
    memory: <MemoryPage />, vault: <VaultPage />, knowmap: <KnowledgeMapPage />, notebook: <NotebookPage />,
    analytics: <AnalyticsPage />, cost: <CostPage />, models: <ModelsPage />, approvals: <ApprovalsPage />, settings: <SettingsPage />,
  };

  const layer = layerOf(page);
  const subs = layer.pages;

  return (
    <div className="h-[100dvh] flex bg-slate-100 text-slate-800 overflow-hidden">
      <AppRail page={page} setPage={setPage} />
      <div className="flex-1 min-w-0 flex flex-col">
        {subs.length > 1 && (
          <div className="h-11 shrink-0 bg-white border-b border-slate-200 flex items-center gap-1 px-3">
            <span className="text-[12px] font-semibold text-slate-400 mr-2">{layer.label}</span>
            {subs.map((p) => (
              <button key={p} onClick={() => setPage(p)}
                className={`px-3 py-1.5 rounded-lg text-[13px] ${page === p ? "bg-blue-50 text-blue-700 font-medium" : "text-slate-500 hover:bg-slate-50"}`}>
                {PAGE_LABEL[p]}
              </button>
            ))}
          </div>
        )}
        <div className="flex-1 min-h-0 flex">{NODE[page]}</div>
      </div>
    </div>
  );
}
