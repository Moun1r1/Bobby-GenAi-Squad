"use client";
import { useEffect, useState } from "react";
import type { Note } from "@/lib/events";

// Operator notes persisted locally, keyed by run — shared by the composer's "Private note" tab and the Details panel.
export function useNotes() {
  const [map, setMap] = useState<Record<string, Note[]>>({});
  useEffect(() => { try { setMap(JSON.parse(localStorage.getItem("bobby_notes") || "{}")); } catch { /* ignore */ } }, []);
  const add = (runId: string, text: string) => setMap((prev) => {
    const next = { ...prev, [runId]: [...(prev[runId] || []), { text, ts: Date.now() / 1000 }] };
    try { localStorage.setItem("bobby_notes", JSON.stringify(next)); } catch { /* ignore */ }
    return next;
  });
  return { map, add };
}
