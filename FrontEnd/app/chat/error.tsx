"use client";

import { useEffect } from "react";
import PageShell from "../../components/layout/PageShell";
import { useLanguage } from "../../context/LanguageContext";

export default function ChatError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const { t } = useLanguage();

  useEffect(() => {
    console.error("[Chat Error]", error);
  }, [error]);

  return (
    <PageShell>
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 px-4 text-center">
        <div className="text-5xl">𓂀</div>
        <h2
          className="text-2xl font-bold"
          style={{ fontFamily: "var(--font-cormorant), serif", color: "#E6B23C" }}
        >
          {t("error.title")}
        </h2>
        <p className="max-w-md text-sm" style={{ color: "#A08E70" }}>
          {t("error.desc")}
        </p>
        <button
          onClick={reset}
          className="mt-2 rounded-lg px-6 py-2.5 text-sm font-semibold transition-colors duration-200"
          style={{
            background: "#E6B23C",
            color: "#0D0A07",
          }}
        >
          {t("error.button")}
        </button>
      </div>
    </PageShell>
  );
}
