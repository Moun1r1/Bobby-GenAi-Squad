// backend.ts — thin proxy from the tRPC server (Node) to the Python FastAPI backend.
// Keeps ALL engine logic in Python; the frontend only speaks tRPC to this proxy.
const BASE = process.env.BACKEND_URL || "http://localhost:8080";

export async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`backend ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export type SquadEvent = {
  run_id: string;
  seq: number;
  ts: number;
  kind: string;
  [k: string]: any;
};

// streamEvents — connect to the backend's SSE endpoint and yield parsed events. This is what the tRPC
// subscription resolver iterates, giving the browser a typed live feed over a real protocol (SSE under tRPC).
export async function* streamEvents(runId: string, signal?: AbortSignal): AsyncGenerator<SquadEvent> {
  const res = await fetch(`${BASE}/runs/${runId}/stream`, {
    headers: { accept: "text/event-stream" },
    signal,
  });
  if (!res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop() || "";
    for (const frame of frames) {
      if (frame.startsWith("event: end")) return;
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const json = line.slice(5).trim();
      if (!json) continue;
      try {
        yield JSON.parse(json) as SquadEvent;
      } catch {
        /* skip keepalive / malformed */
      }
    }
  }
}
