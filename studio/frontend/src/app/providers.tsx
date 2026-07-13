"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { trpc, makeClient } from "@/lib/trpc";

export function Providers({ children }: { children: React.ReactNode }) {
  // Query cache defaults — best-practice caching so lists/stats aren't refetched needlessly.
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: { queries: { staleTime: 5_000, gcTime: 5 * 60_000, refetchOnWindowFocus: false, retry: 1 } },
  }));
  const [trpcClient] = useState(() => makeClient());
  return (
    <trpc.Provider client={trpcClient} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </trpc.Provider>
  );
}
