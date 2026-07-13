// trpc.ts (client) — the typed React hooks, wired with a link SPLIT: subscriptions go over SSE
// (httpSubscriptionLink) for real live rendering; everything else batches over HTTP.
"use client";
import { createTRPCReact } from "@trpc/react-query";
import { httpBatchLink, httpSubscriptionLink, splitLink } from "@trpc/client";
import superjson from "superjson";
import type { AppRouter } from "@/server/routers/_app";

export const trpc = createTRPCReact<AppRouter>();

export function makeClient() {
  return trpc.createClient({
    links: [
      splitLink({
        condition: (op) => op.type === "subscription",
        true: httpSubscriptionLink({ url: "/api/trpc", transformer: superjson }),
        false: httpBatchLink({ url: "/api/trpc", transformer: superjson }),
      }),
    ],
  });
}
