"use client";

import { motion } from "framer-motion";
import { useRouter, useSearchParams } from "next/navigation";
import PageShell from "@/components/feature/PageShell";
import { Button } from "@/components/ui/button";
import { Video, MessageSquare, ChevronLeft, Scroll, Crown, MapPin, Sparkles, Hourglass } from "lucide-react";
import Link from "next/link";
import { Suspense, useState, useEffect, useMemo } from "react";
import { PHARAOHS, LANDMARKS } from "@/lib/mock-trending";
import { loadResultFromSession } from "@/lib/recognition";
import { formatTitle } from "@/lib/recognition";
import type { RecognitionResult, SubEntity } from "@/lib/types";

/* ── Manual / Quick-link flow (from home/trending cards) ────────────────── */
function findMockDescription(type: string | null, name: string): string {
  if (type === "pharaoh" || !type) {
    const p = PHARAOHS.find((x) => x.name.toLowerCase() === name.toLowerCase());
    if (p) return p.description;
  }
  if (type === "landmark" || !type) {
    const l = LANDMARKS.find((x) => x.name.toLowerCase() === name.toLowerCase());
    if (l) return l.description;
  }
  return "";
}

/* ── Main component ─────────────────────────────────────────────────────── */
function ResultContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // ── 1. Check for session-stored recognition result (upload flow) ──────
  const [sessionResult, setSessionResult] = useState<RecognitionResult | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const payload = loadResultFromSession();
    if (payload) {
      setSessionResult(payload.result);
      if (payload.imageDataUrl) setUploadedImageUrl(payload.imageDataUrl);
    }
  }, []);

  // ── 2. Check for URL params (quick-link / home card flow) ─────────────
  const entityTypeParam = searchParams.get("type");
  const entityNameParam = searchParams.get("entity") || searchParams.get("name");

  // ── 3. Derive display data from whichever source we have ─────────────
  const isApiFlow = !!sessionResult;
  const isQuickLink = !isApiFlow && !!entityNameParam;

  const mockMatch = useMemo(() => {
    if (isApiFlow || !entityNameParam) return null;
    const cleanName = entityNameParam.trim().toLowerCase();
    if (entityTypeParam === "pharaoh" || !entityTypeParam) {
      const p = PHARAOHS.find(x => x.name.toLowerCase() === cleanName);
      if (p) return { item: p, type: "pharaoh" as const };
    }
    if (entityTypeParam === "landmark" || !entityTypeParam) {
      const l = LANDMARKS.find(x => x.name.toLowerCase() === cleanName);
      if (l) return { item: l, type: "landmark" as const };
    }
    return null;
  }, [isApiFlow, entityNameParam, entityTypeParam]);

  // ── Display values ────────────────────────────────────────────────────
  const displayType: "pharaoh" | "landmark" = isApiFlow
    ? (sessionResult?.type === "pharaoh" ? "pharaoh" : "landmark")
    : (mockMatch?.type ?? entityTypeParam === "pharaoh" ? "pharaoh" : "landmark");

  // Prefer entity.name (DB name), fall back to formatted model label
  const displayName: string = isApiFlow
    ? (sessionResult?.entity?.name ?? formatTitle(sessionResult?.name ?? ""))
    : (mockMatch?.item.name ?? formatTitle(entityNameParam ?? "Unknown"));

  const cleanDisplayName = displayName.includes("(") ? displayName.split("(")[0].trim() : displayName;

  const displayDescription: string = isApiFlow
    ? (sessionResult?.entity?.description ?? "No description available.")
    : (mockMatch?.item.description ?? findMockDescription(entityTypeParam, entityNameParam ?? ""));

  // Metadata — only shown when non-null
  const dynasty: string | null = isApiFlow ? (sessionResult?.entity?.dynasty ?? null) : null;
  const period: string | null = isApiFlow ? (sessionResult?.entity?.period ?? null) : null;
  const location: string | null = isApiFlow ? (sessionResult?.entity?.location ?? null) : null;
  const rawType = isApiFlow ? (sessionResult?.entity?.type ?? null) : (mockMatch?.type === "pharaoh" && 'type' in mockMatch.item ? (mockMatch.item as any).type : null);
  const dbType: string = rawType || "Unknown";

  // ── Composite entity data (with per-entity metadata from DB) ──────────
  const compositeEntitiesData: SubEntity[] = useMemo(() => {
    if (!isApiFlow) return [];
    return sessionResult?.entity?.composite_entities_data ?? [];
  }, [isApiFlow, sessionResult]);

  const typeLabel = displayType === "pharaoh" ? "PHARAOH" : "LANDMARK";

  const getAssumedImageUrl = (name: string, isPharaoh: boolean) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";
    if (isPharaoh) {
      if (name === "Akhenaton") return `${baseUrl}/static/images/pharaohs/Akhenaton.JPG`;
      if (name === "Cleopatra VII Philopator") return `${baseUrl}/static/images/pharaohs/Cleopatra%20VII%20Philopator.jpg`;
      if (name === "Hatshepsut") return `${baseUrl}/static/images/pharaohs/Hatshepsut.JPG`;
      if (name === "Ramesses II") return `${baseUrl}/static/images/pharaohs/Ramesses%20II.jpg`;
      if (name === "Tutankhamun") return `${baseUrl}/static/images/pharaohs/Tutankhamun.jpg`;
    } else {
      if (name === "Pyramids of Giza") return `${baseUrl}/static/images/landmarks/Pyramids%20of%20Giza.webp`;
      if (name === "Sphinx") return `${baseUrl}/static/images/landmarks/Sphinx.jpg`;
      if (name === "Temple of Karnak") return `${baseUrl}/static/images/landmarks/Temple%20of%20Karnak.jpg`;
      if (name === "Temple of Luxor") return `${baseUrl}/static/images/landmarks/Temple%20of%20Luxor.jpg`;
      if (name === "The Great Temple of Ramesses II at Abu Simbel") return `${baseUrl}/static/images/landmarks/The%20Great%20Temple%20of%20Ramesses%20II%20at%20Abu%20Simbel.webp`;
    }
    return null;
  };

  const assumedUrl = getAssumedImageUrl(cleanDisplayName, displayType === "pharaoh");
  let finalImageUrl: string | null = null;
  if (uploadedImageUrl) {
    finalImageUrl = uploadedImageUrl;
  } else if (assumedUrl) {
    finalImageUrl = assumedUrl;
  } else if (sessionResult?.entity?.images && sessionResult.entity.images.length > 0 && sessionResult.entity.images[0].url) {
    finalImageUrl = sessionResult.entity.images[0].url.startsWith("/static")
      ? `${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010"}${sessionResult.entity.images[0].url}`
      : sessionResult.entity.images[0].url;
  }

  if (!mounted) {
    return <div className="min-h-screen" style={{ background: "#0D0A07" }} />;
  }

  return (
    <PageShell>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-6xl mx-auto">
        {/* Breadcrumb */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
          <Link href="/upload" className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors">
            <span className="group-hover:-translate-x-1 transition-transform">←</span>
            Return
          </Link>
        </motion.div>

        <div className="grid lg:grid-cols-[0.75fr_1.25fr] gap-12 items-start">

          {/* ── Left: Image card ──────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.7 }}
            className="relative group lg:max-w-sm"
          >
            <div className={`aspect-[4/5] rounded-3xl overflow-hidden border relative shadow-[0_20px_60px_rgba(0,0,0,0.5)]
              ${displayType === "pharaoh" ? "border-[#E6B23C]/10 bg-[#1E160E]" : "border-[#A08E70]/10 bg-[#12150E]"}`}>

              <div className="absolute inset-0 bg-gradient-to-t from-[#0D0A07] via-[#0D0A07]/10 to-transparent z-10" />

              {finalImageUrl ? (
                <img
                  src={finalImageUrl}
                  alt={cleanDisplayName}
                  className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.opacity = '0';
                  }}
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <motion.div
                    animate={{ rotate: [0, 5, -5, 0], scale: [1, 1.05, 1] }}
                    transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                    className="flex flex-col items-center gap-4"
                  >
                    {displayType === "pharaoh" ? (
                      <Crown size={100} className="text-[#E6B23C]/20" />
                    ) : (
                      <Scroll size={100} className="text-[#A08E70]/20" />
                    )}
                  </motion.div>
                </div>
              )}

              {/* Type badge — dynamic */}
              <div className="absolute top-5 left-5 z-20 flex flex-col gap-2">
                <div className="px-3 py-1.5 bg-gradient-to-r from-[#E6B23C] to-[#D4A030] rounded-full text-[10px] font-bold tracking-[0.2em] text-[#0D0A07] uppercase shadow-[0_4px_15px_rgba(230,178,60,0.3)] flex items-center gap-1.5">
                  {displayType === "pharaoh" ? <Crown size={10} /> : <MapPin size={10} />}
                  {typeLabel}
                </div>
                {isQuickLink && (
                  <div className="px-3 py-1 bg-[#E6B23C]/10 border border-[#E6B23C]/15 backdrop-blur-md rounded-full text-[9px] font-bold tracking-[0.15em] text-[#E6B23C] uppercase flex items-center gap-1.5 w-fit">
                    <Sparkles size={8} /> Neural Quick-Link
                  </div>
                )}
              </div>

              {/* Title overlay on card */}
              <div className="absolute bottom-10 left-8 z-20 right-8">
                <motion.h1
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="font-heading text-3xl font-bold text-white tracking-wide drop-shadow-lg"
                >
                  {cleanDisplayName}
                </motion.h1>
              </div>
            </div>
          </motion.div>

          {/* ── Right: Papyrus panel + Actions ───────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.7 }}
            className="flex flex-col justify-center gap-10"
          >
            {/* Papyrus card */}
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5, type: "spring", damping: 20 }}
              className="papyrus-paper relative group"
            >
              <div className="text-[10px] font-bold tracking-[0.3em] text-[#1A1005]/50 mb-4 uppercase flex items-center gap-2">
                <Scroll size={11} /> Eternal Record
              </div>

              {/* Entity name heading */}
              <h2
                className="text-3xl md:text-4xl font-bold text-[#1A1005] uppercase tracking-[0.06em] mb-5 border-b border-[#1A1005]/10 pb-4"
                style={{ fontFamily: "var(--font-cormorant), serif" }}
              >
                {cleanDisplayName}
              </h2>

              {/* Description */}
              <p
                className="text-[#1A1005] leading-[1.8] text-lg font-medium text-justify"
                style={{ fontFamily: "var(--font-cormorant), serif" }}
              >
                &quot;{displayDescription}&quot;
              </p>

              {/* Metadata rows — Only rendered when values are non-null */}
              {(dynasty || period || location || displayType === "pharaoh") && (
                <div className="mt-6 pt-4 border-t border-[#1A1005]/10 flex flex-col gap-3 font-cormorant">
                  {displayType === "pharaoh" && (
                    <div className="flex items-center gap-3 text-base text-[#1A1005]/70">
                      <Crown size={15} className="text-[#B8860B] shrink-0" />
                      <span className="font-semibold uppercase tracking-wide text-xs text-[#1A1005]/50 w-20">Type</span>
                      <span className="font-medium capitalize">{dbType}</span>
                    </div>
                  )}
                  {dynasty && (
                    <div className="flex items-center gap-3 text-base text-[#1A1005]/70">
                      <Scroll size={15} className="text-[#B8860B] shrink-0" />
                      <span className="font-semibold uppercase tracking-wide text-xs text-[#1A1005]/50 w-20">Dynasty</span>
                      <span className="font-medium">{dynasty}</span>
                    </div>
                  )}
                  {period && (
                    <div className="flex items-center gap-3 text-base text-[#1A1005]/70">
                      <Hourglass size={15} className="text-[#B8860B] shrink-0" />
                      <span className="font-semibold uppercase tracking-wide text-xs text-[#1A1005]/50 w-20">Period</span>
                      <span className="font-medium">{period}</span>
                    </div>
                  )}
                  {location && (
                    <div className="flex items-center gap-3 text-base text-[#1A1005]/70">
                      <MapPin size={15} className="text-[#B8860B] shrink-0" />
                      <span className="font-semibold uppercase tracking-wide text-xs text-[#1A1005]/50 w-20">Location</span>
                      <span className="font-medium">{location}</span>
                    </div>
                  )}
                </div>
              )}

              <div className="mt-8 pt-5 border-t border-[#1A1005]/8 flex justify-between items-center opacity-40">
                <div className="text-[9px] font-bold tracking-[0.2em] text-[#1A1005] uppercase">Origin Verified</div>
                <div className="text-[9px] font-bold tracking-[0.2em] text-[#1A1005] uppercase">E.C.H.O Archive</div>
              </div>
            </motion.div>

            {/* Actions: Video & Chat — per sub-entity when composite */}
            <div className="flex flex-col gap-5">
              {compositeEntitiesData.length > 0 ? (
                /* ── Composite: one row per sub-entity ────────── */
                <div className="flex flex-col gap-4">
                  {compositeEntitiesData.map((sub) => (
                    <div key={sub.name} className="grid sm:grid-cols-2 gap-4">
                      <Button
                        onClick={() => router.push(`/video?entity=${encodeURIComponent(sub.name)}&type=${displayType}&dynasty=${encodeURIComponent(sub.dynasty || '')}&period=${encodeURIComponent(sub.period || '')}&dbType=${encodeURIComponent(sub.type || '')}&location=${encodeURIComponent(location || '')}`)}
                        className="h-14 rounded-2xl bg-[#E6B23C]/10 border border-[#E6B23C]/20 hover:bg-[#E6B23C]/20 text-[#E6B23C] font-bold text-base transition-all hover:scale-[1.02]"
                      >
                        <Video className="mr-3" size={20} />
                        Generate {sub.name} Video
                      </Button>
                      <Button
                        onClick={() => router.push(`/chat?entity=${encodeURIComponent(sub.name)}&type=${displayType}`)}
                        variant="outline"
                        className="h-14 rounded-2xl border-[#E6B23C]/12 bg-[#E6B23C]/[0.04] hover:bg-[#E6B23C]/[0.08] text-[#F5E6D0] font-semibold text-base transition-all hover:scale-[1.02]"
                      >
                        <MessageSquare className="mr-3" size={20} />
                        Chat with {sub.name}
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                /* ── Normal: single row ───────────────────────── */
                <div className="grid sm:grid-cols-2 gap-5">
                  <Button
                    onClick={() => router.push(`/video?entity=${encodeURIComponent(displayName)}&type=${displayType}&dynasty=${encodeURIComponent(dynasty || '')}&period=${encodeURIComponent(period || '')}&dbType=${encodeURIComponent(dbType || '')}&location=${encodeURIComponent(location || '')}`)}
                    className="h-14 rounded-2xl bg-[#E6B23C]/10 border border-[#E6B23C]/20 hover:bg-[#E6B23C]/20 text-[#E6B23C] font-bold text-base transition-all hover:scale-[1.02]"
                  >
                    <Video className="mr-3" size={20} />
                    Generate Video
                  </Button>
                  <Button
                    onClick={() => router.push(`/chat?entity=${encodeURIComponent(displayName)}&type=${displayType}`)}
                    variant="outline"
                    className="h-14 rounded-2xl border-[#E6B23C]/12 bg-[#E6B23C]/[0.04] hover:bg-[#E6B23C]/[0.08] text-[#F5E6D0] font-semibold text-base transition-all hover:scale-[1.02]"
                  >
                    <MessageSquare className="mr-3" size={20} />
                    Chat with History
                  </Button>
                </div>
              )}

              <Button
                onClick={() => router.push("/upload")}
                className="h-14 rounded-2xl bg-gradient-to-r from-[#C1840A] to-[#A06A00] hover:from-[#D4A030] hover:to-[#C1840A] text-white font-bold text-base transition-all hover:scale-[1.02] shadow-[0_4px_30px_rgba(230,178,60,0.15)] flex items-center justify-center gap-2"
              >
                <Sparkles size={20} />
                Recognize Another Entity
              </Button>
            </div>

          </motion.div>

        </div>
      </motion.div>
    </PageShell>
  );
}

export default function ResultPage() {
  return (
    <Suspense fallback={<div className="min-h-screen" style={{ background: "#0D0A07" }} />}>
      <ResultContent />
    </Suspense>
  );
}
