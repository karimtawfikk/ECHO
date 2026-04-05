"use client";

import { useEffect } from "react";
import PageShell from "@/components/feature/PageShell";

export default function ChatError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[Chat Error]", error);
  }, [error]);

  return (
    <PageShell>
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 px-4 text-center">
        <div className="text-5xl">𓂀</div>
        <h2
          className="text-2xl font-bold"
          style={{ fontFamily: "var(--font-playfair)", color: "#E6B23C" }}
        >
          The Ancient Scrolls Are Momentarily Veiled
        </h2>
        <p className="max-w-md text-sm" style={{ color: "#A08E70" }}>
          An unexpected disturbance has interrupted our connection to the past.
          Please try again.
        </p>
        <button
          onClick={reset}
          className="mt-2 rounded-lg px-6 py-2.5 text-sm font-semibold transition-colors duration-200"
          style={{
            background: "#E6B23C",
            color: "#0D0A07",
          }}
        >
          Try Again
        </button>
      </div>
    </PageShell>
  );
}
