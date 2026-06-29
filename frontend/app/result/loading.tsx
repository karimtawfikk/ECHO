import PageShell from "@/components/layout/PageShell";

export default function ResultLoading() {
  return (
    <PageShell>
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-8 px-4 pt-24">
        <div className="self-start">
          <div
            className="h-8 w-24 animate-pulse rounded-lg"
            style={{ background: "rgba(230,178,60,0.10)" }}
          />
        </div>
        <div
          className="aspect-[16/9] w-full max-w-2xl animate-pulse rounded-2xl"
          style={{ background: "rgba(230,178,60,0.08)" }}
        />
        <div
          className="h-8 w-64 animate-pulse rounded"
          style={{ background: "rgba(230,178,60,0.12)" }}
        />
        <div className="flex w-full max-w-xl flex-col gap-2">
          <div
            className="h-4 w-full animate-pulse rounded"
            style={{ background: "rgba(230,178,60,0.06)" }}
          />
          <div
            className="h-4 w-4/5 animate-pulse rounded"
            style={{ background: "rgba(230,178,60,0.06)" }}
          />
          <div
            className="h-4 w-3/5 animate-pulse rounded"
            style={{ background: "rgba(230,178,60,0.06)" }}
          />
        </div>
        <div className="flex gap-4">
          <div
            className="h-10 w-32 animate-pulse rounded-lg"
            style={{ background: "rgba(230,178,60,0.10)" }}
          />
          <div
            className="h-10 w-32 animate-pulse rounded-lg"
            style={{ background: "rgba(230,178,60,0.10)" }}
          />
        </div>
      </div>
    </PageShell>
  );
}
