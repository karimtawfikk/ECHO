import PageShell from "@/components/layout/PageShell";

export default function ChatLoading() {
  return (
    <PageShell>
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 pt-24">
        {/* Header skeleton */}
        <div className="flex items-center gap-3 pb-4">
          <div
            className="h-12 w-12 animate-pulse rounded-full"
            style={{ background: "rgba(230,178,60,0.12)" }}
          />
          <div className="flex flex-col gap-2">
            <div
              className="h-4 w-32 animate-pulse rounded"
              style={{ background: "rgba(230,178,60,0.12)" }}
            />
            <div
              className="h-3 w-20 animate-pulse rounded"
              style={{ background: "rgba(230,178,60,0.06)" }}
            />
          </div>
        </div>
        {/* Message skeletons */}
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="animate-pulse rounded-2xl p-4"
            style={{
              background: "rgba(230,178,60,0.06)",
              width: i % 2 === 0 ? "60%" : "80%",
              marginLeft: i % 2 === 0 ? "auto" : "0",
              height: i === 1 ? "80px" : "48px",
              animationDelay: `${i * 150}ms`,
            }}
          />
        ))}
        {/* Input skeleton */}
        <div className="mt-auto pt-8">
          <div
            className="h-12 w-full animate-pulse rounded-full"
            style={{ background: "rgba(230,178,60,0.08)" }}
          />
        </div>
      </div>
    </PageShell>
  );
}
