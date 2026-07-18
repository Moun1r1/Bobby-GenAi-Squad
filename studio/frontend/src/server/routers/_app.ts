// _app.ts — the tRPC app router. Queries/mutations proxy to the Python backend; `events` is a live SUBSCRIPTION
// that streams the squad's observer events (target/plan/move/tool/cycle/day/done) to the browser in real time.
import { z } from "zod";
import { router, publicProcedure } from "../trpc";
import { api, streamEvents, type SquadEvent } from "../backend";

export const appRouter = router({
  health: publicProcedure.query(() => api("/health")),

  pipelines: publicProcedure.query(() => api("/pipelines")),

  runs: publicProcedure.query(async () => (await api<{ runs: any[] }>("/runs")).runs),

  run: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(({ input }) => api(`/runs/${input.id}`)),

  launch: publicProcedure
    .input(z.object({ pipeline: z.string(), params: z.record(z.any()).default({}) }))
    .mutation(({ input }) =>
      api("/runs", { method: "POST", body: JSON.stringify(input) })
    ),

  // create a NEW use-case pipeline from the UI — pure SELF (role + goal); no prompt, engine-directed.
  createPipeline: publicProcedure
    .input(z.object({ id: z.string(), title: z.string().default(""), desc: z.string().default(""), identity: z.string(), goal: z.string(), domain: z.string().default("data") }))
    .mutation(({ input }) => api("/pipelines/spec", { method: "POST", body: JSON.stringify(input) })),
  deletePipeline: publicProcedure.input(z.object({ id: z.string() })).mutation(({ input }) => api(`/pipelines/${encodeURIComponent(input.id)}`, { method: "DELETE" })),
  config: publicProcedure.query(() => api("/config")),
  setConfig: publicProcedure.input(z.object({ agents: z.number(), patience: z.number(), max_units: z.number() })).mutation(({ input }) => api("/config", { method: "POST", body: JSON.stringify(input) })),

  search: publicProcedure
    .input(z.object({ q: z.string(), collection: z.string().default("knowledge"), limit: z.number().default(10) }))
    .query(({ input }) =>
      api(`/search?q=${encodeURIComponent(input.q)}&collection=${input.collection}&limit=${input.limit}`)
    ),

  // --- management: steer, curate, delete ---
  control: publicProcedure
    .input(z.object({ runId: z.string(), action: z.enum(["stop", "pause", "resume", "steer"]), text: z.string().default("") }))
    .mutation(({ input }) =>
      api(`/runs/${input.runId}/control`, { method: "POST", body: JSON.stringify({ action: input.action, text: input.text }) })
    ),

  setBoardState: publicProcedure
    .input(z.object({ runId: z.string(), label: z.string(), state: z.string() }))
    .mutation(({ input }) =>
      api(`/runs/${input.runId}/board`, { method: "POST", body: JSON.stringify({ label: input.label, state: input.state }) })
    ),

  deleteRun: publicProcedure
    .input(z.object({ runId: z.string() }))
    .mutation(({ input }) => api(`/runs/${input.runId}`, { method: "DELETE" })),

  // --- persistent substrate: cross-run stats, experts & knowledge (the compounding memory) ---
  stats: publicProcedure.query(() => api("/stats")),
  primitives: publicProcedure.query(() => api("/primitives")),
  primitivesRecall: publicProcedure.input(z.object({ q: z.string(), k: z.number().default(5) }))
    .query(({ input }) => api(`/primitives/recall?q=${encodeURIComponent(input.q)}&k=${input.k}`)),
  knowledgeScatter: publicProcedure
    .input(z.object({ limit: z.number().default(400), collection: z.string().default("knowledge") }))
    .query(({ input }) => api(`/knowledge/scatter?limit=${input.limit}&collection=${input.collection}`)),
  notebook: publicProcedure.query(() => api("/notebook")),
  proofs: publicProcedure.input(z.object({ run: z.boolean().default(false) })).query(({ input }) => api(`/proofs?run=${input.run}`)),
  experts: publicProcedure.query(async () => (await api<{ experts: any[] }>("/experts")).experts),
  expert: publicProcedure.input(z.object({ id: z.string() })).query(({ input }) => api(`/experts/${encodeURIComponent(input.id)}`)),
  memoryPolicy: publicProcedure.query(() => api("/memory/policy")),
  dgxHealth: publicProcedure.query(() => api("/dgx/health")),
  dgxSafe: publicProcedure.query(() => api("/dgx/safe")),
  // --- knowledge vault: the navigable, enrichable AI-knowledge graph the swarm reasons from ---
  vaultGraph: publicProcedure.query(() => api("/vault/graph")),
  vaultNote: publicProcedure.input(z.object({ id: z.string() })).query(({ input }) => api(`/vault/note/${encodeURIComponent(input.id)}`)),
  vaultNavigate: publicProcedure
    .input(z.object({ q: z.string(), k: z.number().default(3), hops: z.number().default(1) }))
    .query(({ input }) => api(`/vault/navigate?q=${encodeURIComponent(input.q)}&k=${input.k}&hops=${input.hops}`)),
  knowledgeAll: publicProcedure
    .input(z.object({ limit: z.number().default(200), domain: z.string().default("") }))
    .query(async ({ input }) => (await api<{ items: any[] }>(`/knowledge?limit=${input.limit}&domain=${encodeURIComponent(input.domain)}`)).items),

  // real live rendering — a tRPC subscription (SSE) that yields each engine event as it happens.
  events: publicProcedure
    .input(z.object({ runId: z.string() }))
    .subscription(async function* ({ input, signal }) {
      for await (const ev of streamEvents(input.runId, signal)) {
        yield ev as SquadEvent;
      }
    }),
});

export type AppRouter = typeof appRouter;
