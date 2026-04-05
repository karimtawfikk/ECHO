import PageShell from "@/components/feature/PageShell";

export default function UploadLoading() {
  return (
    <PageShell>
      <div className="mx-auto flex max-w-2xl flex-col items-center gap-8 px-4 pt-24">
        {/* Title skeleton */}
        <div
          className="h-8 w-48 animate-pulse rounded"
          style={{ background: "rgba(230,178,60,0.12)" }}
        />
        {/* Upload zone skeleton */}
        <div
          className="flex aspect-[4/3] w-full animate-pulse flex-col items-center justify-center rounded-2xl border-2 border-dashed"
          style={{
            background: "rgba(230,178,60,0.04)",
            borderColor: "rgba(230,178,60,0.12)",
          }}
        >
          <div
            className="h-16 w-16 animate-pulse rounded-full"
            style={{ background: "rgba(230,178,60,0.10)" }}
          />
          <div
            className="mt-4 h-4 w-40 animate-pulse rounded"
            style={{ background: "rgba(230,178,60,0.08)" }}
          />
        </div>
        {/* Button skeleton */}
        <div
          className="h-12 w-40 animate-pulse rounded-full"
          style={{ background: "rgba(230,178,60,0.10)" }}
        />
      </div>
    </PageShell>
  );
}
