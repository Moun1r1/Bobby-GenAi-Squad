"use client";
import { useEffect, useRef, useState } from "react";
import { trpc } from "@/lib/trpc";
import type { Ev } from "@/lib/events";

// Live event stream for one run over the tRPC SSE subscription, with seq-dedup so a reconnect (which replays the
// backlog) can't double events and safely backfills anything missed.
export function useRunEvents(runId: string | null): Ev[] {
  const [events, setEvents] = useState<Ev[]>([]);
  const seen = useRef<Set<number>>(new Set());
  useEffect(() => { seen.current = new Set(); setEvents([]); }, [runId]);
  trpc.events.useSubscription(
    { runId: runId || "" },
    {
      enabled: !!runId,
      onData: (ev: any) => {
        const e = ev as Ev;
        if (typeof e.seq === "number") { if (seen.current.has(e.seq)) return; seen.current.add(e.seq); }
        setEvents((p) => [...p, e]);
      },
      onError: () => { /* transport drop — the SSE link retries; backlog replay + seq-dedup restores state */ },
    }
  );
  return events;
}
