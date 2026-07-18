// The observer event stream + its projection into chat messages and a run model. Pure, testable, no React.

export type Ev = { seq: number; kind: string; [k: string]: any };
export type Note = { text: string; ts: number };

// One chat message. side: left = squad · right = you (operator) · system = inline note.
export type Msg = {
  side: "left" | "right" | "system";
  who?: string;
  text: string;
  sub?: string;
  tone?: "verdict" | "error" | "ok";
  ts?: number;
  seq: number;
};

// pipeline id → friendly channel name
export const CHANNEL: Record<string, string> = {
  goal: "Goal squad", idea_board: "Idea board", persona: "Persona", world: "World sim", process_data: "Data reader",
  engine_trace: "Engine trace", multi_day_service: "Service desk", rd_lab: "R&D lab", strategy_squad: "Strategy squad",
};
export const chName = (p: string): string => CHANNEL[p] || (p || "run").replace(/_/g, " ");

export const isOver = (events: Ev[]): boolean => events.some((e) => e.kind === "done");

export function fmtTime(ts?: number): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return isNaN(d.getTime()) ? "" : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Project one event to a chat message (or null to drop it from the thread).
export function toMsg(e: Ev): Msg | null {
  const base = { ts: e.ts, seq: e.seq };
  const L = (who: string, text: string, sub?: string): Msg => ({ side: "left", who, text, sub, ...base });
  const S = (text: string, tone?: Msg["tone"]): Msg => ({ side: "system", text, tone, ...base });
  switch (e.kind) {
    case "say": return e.who === "event" ? S(e.text) : L(e.who || "agent", e.text);
    case "target": return e.target ? L(e.agent || "agent", `Picked a target — ${e.target}`) : null;
    case "move_start": return L(e.agent || "agent", `${e.move ? `**[${e.move}]** ` : ""}${e.intention || "working…"}`);
    case "card": return e.cstate === "verified" ? L(e.owner || "agent", `✓ ${e.title || e.id}`, e.evidence) : null;
    case "section": return e.note ? L(e.agent || "reader", e.note) : null;
    case "result": return e.summary ? L("squad", e.summary) : null;
    case "log": return e.line ? L("squad", e.line) : null;
    case "steer": return { side: "right", who: "You", text: e.text || "", ...base };
    case "verdict": return { side: "system", text: `**${e.verdict}** — ${e.metric || ""} ${e.detail || ""}`.trim(), tone: "verdict", ...base };
    case "converged": return S("✓ Converged — every acceptance criterion verified.", "ok");
    case "escalate": return S(`⚑ Escalated to human — ${e.reason || "some criteria unmet"}.`, "verdict");
    case "criteria": return S(`Acceptance criteria set (${(e.criteria || []).length}).`);
    case "criterion": return e.met ? S(`✓ criterion met: ${e.text}`, "ok") : null;
    case "wave": return S(`Wave ${e.n}${e.met != null ? ` · ${e.met}/${e.total} criteria` : e.new != null ? ` · +${e.new} new` : ""}${e.dry ? ` · plateau ${e.dry}` : ""}`);
    case "day": return S(`Day ${e.day} — ${e.clean}/${e.total} clean (${Math.round((e.rate || 0) * 100)}%)`);
    case "board": return S(`Board — ${e.n_ideas} ideas${e.repelled ? ` · ${e.repelled} repelled` : ""}`);
    case "burn": return e.phase === "acr"
      ? S(`Burn-In ${e.i}/${e.n} — local ${Math.round((e.local || 0) * 100)}% · ${e.serve} serve tok · ${e.promotions} frozen`)
      : S(`No-ACR control ${e.i}/${e.n} — ${e.serve} tok (all LLM)`);
    case "error": return S(e.message || "error", "error");
    default: return null; // tool/tool_done/move_end/cycle/signal/flags/memory/playbook/agent/goal/teams/done stay out of the thread
  }
}

// A light run model for the report/stats views.
export type RunModel = {
  participants: string[];
  findings: { who: string; text: string; sub?: string }[];
  verdict: Ev | null;
  result: Ev | null;
  converged: boolean;
  escalated: boolean;
  counts: { messages: number; tools: number; cycles: number };
};

// Ideas from the latest board snapshot (idea_board / goal), flattened with their emergent state.
export type Idea = { label: string; state: string; area?: string; touched?: number; variants?: number };
export function boardIdeas(events: Ev[]): Idea[] {
  let board: any = null;
  for (const e of events) if (e.kind === "board") board = e;
  const out: Idea[] = [];
  if (board?.states) for (const [st, arr] of Object.entries(board.states)) for (const it of (arr as any[])) out.push({ ...it, state: st });
  return out;
}

export function reduceRun(events: Ev[]): RunModel {
  const participants = new Set<string>();
  const findings: RunModel["findings"] = [];
  let verdict: Ev | null = null, result: Ev | null = null, converged = false, escalated = false;
  let tools = 0, cycles = 0, messages = 0;
  for (const e of events) {
    if (e.agent) participants.add(e.agent);
    if (e.kind === "agent" && e.name) participants.add(e.name);
    if (e.kind === "tool") tools++;
    if (e.kind === "cycle") cycles++;
    if (e.kind === "verdict") verdict = e;
    if (e.kind === "result") result = e;
    if (e.kind === "converged") converged = true;
    if (e.kind === "escalate") escalated = true;
    const m = toMsg(e);
    if (m && m.side === "left") { messages++; findings.push({ who: m.who || "squad", text: m.text, sub: m.sub }); }
  }
  return { participants: [...participants], findings, verdict, result, converged, escalated, counts: { messages, tools, cycles } };
}
