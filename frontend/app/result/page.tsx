"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useRouter, useSearchParams } from "next/navigation";
import PageShell from "@/components/layout/PageShell";
import { Button } from "@/components/ui/button";
import { Video, MessageSquare, ChevronLeft, Scroll, Crown, MapPin, Sparkles, Hourglass, Bookmark, Check, BookmarkMinus } from "lucide-react";
import Link from "next/link";
import { Suspense, useState, useEffect, useMemo } from "react";
import { PHARAOHS, LANDMARKS } from "@/lib/mock/mock-trending";
import { useLanguage } from "@/context/LanguageContext";
import { loadResultFromSession } from "@/lib/services/recognition";
import { formatTitle } from "@/lib/services/recognition";
import { createClient } from "@/lib/supabase/client";
import { cleanEntityName } from "@/lib/utils";
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
  const { t, isRTL } = useLanguage();
  const router = useRouter();
  const searchParams = useSearchParams();
  const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";

  // ── 1. Check for session-stored recognition result (upload flow) ──────
  const [sessionResult, setSessionResult] = useState<RecognitionResult | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const supabase = createClient();

  useEffect(() => {
    setMounted(true);

    const entityTypeParam = searchParams.get("type");
    const entityNameParam = searchParams.get("entity") || searchParams.get("name");
    const imageUrlParam = searchParams.get("imageUrl");

    if (entityNameParam && entityTypeParam) {
      const fetchDetails = async () => {
        try {
          const res = await fetch(`${baseUrl}/api/v1/entities/details?name=${encodeURIComponent(entityNameParam)}&type=${entityTypeParam}`);
          if (res.ok) {
            const data = await res.json();
            if (!data.error) {
              setSessionResult(data);
              setUploadedImageUrl(imageUrlParam || null);
              return;
            }
          }
        } catch (err) {
          console.error("Failed to fetch entity details:", err);
        }
        fallbackToSession();
      };
      fetchDetails();
    } else {
      fallbackToSession();
    }

    function fallbackToSession() {
      const payload = loadResultFromSession();
      if (payload) {
        setSessionResult(payload.result);
        setUploadedImageUrl(payload.imageDataUrl || null);
      } else {
        setSessionResult(null);
        setUploadedImageUrl(null);
      }
    }
  }, [searchParams, baseUrl]);

  // ── 2. Check for URL params (quick-link / home card flow) ─────────────
  const entityTypeParam = searchParams.get("type");
  const entityNameParam = searchParams.get("entity") || searchParams.get("name");

  // ── Derive display data source ───────────────────────────────────────
  // If we have a dynamic or session-based API result, use it! Otherwise fallback to mock lookup.
  const isApiFlow = !!sessionResult;
  const isQuickLink = !isApiFlow && !!entityNameParam;
  const isFromExplore = sessionResult?.source === "explore";
  const isFromTrending = sessionResult?.source === "quick-link" || isQuickLink;
  const isFromHome = isFromTrending || isFromExplore;

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

  const cleanDisplayName = cleanEntityName(displayName);

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

  const typeLabel = displayType === "pharaoh" ? t("result.badge.pharaoh") : t("result.badge.landmark");

  const getAssumedImageUrl = (name: string, isPharaoh: boolean) => {
    if (isPharaoh) {
      if (name === "Akhenaton") return `/images/pharaohs/Akhenaton.JPG`;
      if (name === "Cleopatra VII Philopator") return `/images/pharaohs/Cleopatra%20VII%20Philopator.jpg`;
      if (name === "Hatshepsut") return `/images/pharaohs/Hatshepsut.JPG`;
      if (name === "Ramesses II") return `/images/pharaohs/Ramesses%20II.jpg`;
      if (name === "Tutankhamun") return `/images/pharaohs/Tutankhamun.jpg`;
    } else {
      if (name === "Pyramids of Giza") return `/images/landmarks/Pyramids%20of%20Giza.webp`;
      if (name === "Sphinx") return `/images/landmarks/Sphinx.jpg`;
      if (name === "Temple of Karnak") return `/images/landmarks/Temple%20of%20Karnak.jpg`;
      if (name === "Temple of Luxor") return `/images/landmarks/Temple%20of%20Luxor.jpg`;
      if (name === "The Great Temple of Ramesses II at Abu Simbel") return `/images/landmarks/The%20Great%20Temple%20of%20Ramesses%20II%20at%20Abu%20Simbel.webp`;
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
    // Database API images
    finalImageUrl = sessionResult.entity.images[0].url;
  } else if ((sessionResult?.entity as any)?.image) {
    // Support the singular 'image' field from mock-all-entities.ts
    const imgPath = (sessionResult?.entity as any)?.image;
    if (imgPath && typeof imgPath === 'string' && imgPath.startsWith("data/")) {
      // Use our new R2 proxy route on the backend for Cloudflare assets
      finalImageUrl = `${baseUrl}/api/v1/assets/r2/${imgPath}`;
    } else {
      finalImageUrl = imgPath;
    }
  }

  const hasImage = !!finalImageUrl;
  const hideMediaButtons = displayType === "pharaoh" && ["Hor (son of Ankh-Khonsu)", "Itysen (Prince, probably son of Djedefre)"].includes(displayName);

  const handleToggleFavorite = async () => {
    try {
      setIsSaving(true);
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push("/auth");
        return;
      }

      // Fetch current profile to get existing favorites
      const { data: profile } = await supabase
        .from('profiles')
        .select('favorites')
        .eq('id', user.id)
        .single();

      const existingFavorites = Array.isArray(profile?.favorites) ? profile.favorites : [];

      // Check if already favorited
      const alreadyFavorited = existingFavorites.some((f: any) => f.name === displayName);

      if (alreadyFavorited) {
        // Remove from favorites
        const updatedFavorites = existingFavorites.filter((f: any) => f.name !== displayName);
        const { error } = await supabase
          .from('profiles')
          .update({
            favorites: updatedFavorites
          })
          .eq('id', user.id);

        if (error) throw error;
        setIsSaved(false);
      } else {
        // Add to favorites
        const newFavorite = {
          name: displayName,
          type: displayType,
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
        };

        const { error } = await supabase
          .from('profiles')
          .update({
            favorites: [...existingFavorites, newFavorite]
          })
          .eq('id', user.id);

        if (error) throw error;
        setIsSaved(true);
      }
    } catch (error) {
      console.error("Error toggling favorite:", error);
    } finally {
      setIsSaving(false);
    }
  };

  // Check if item is already favorited on load
  useEffect(() => {
    const checkFavorite = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data: profile } = await supabase
        .from('profiles')
        .select('favorites')
        .eq('id', user.id)
        .single();

      const favorites = Array.isArray(profile?.favorites) ? profile.favorites : [];
      if (favorites.some((f: any) => f.name === displayName)) {
        setIsSaved(true);
      }
    };
    if (mounted && displayName) checkFavorite();
  }, [mounted, displayName]);

  if (!mounted) {
    return <div className="min-h-screen" style={{ background: "#0D0A07" }} />;
  }

  return (
    <PageShell>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-6xl mx-auto">
        {/* Breadcrumb */}
        <motion.div initial={{ opacity: 0, x: isRTL ? 20 : -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
          <button onClick={() => router.back()} className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors bg-transparent border-none p-0 outline-none">
            <span className={`transition-transform ${isRTL ? 'group-hover:translate-x-1' : 'group-hover:-translate-x-1'}`}>
              {isRTL ? '→' : '←'}
            </span>
            Back
          </button>
        </motion.div>

        <div className={`grid ${hasImage ? 'lg:grid-cols-[0.75fr_1.25fr]' : 'lg:grid-cols-1 max-w-2xl mx-auto'} gap-12 items-start`}>

          {/* ── Left: Image card ──────────────────────────────────────── */}
          {hasImage && (
            <motion.div
              initial={{ opacity: 0, x: -40 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.7 }}
              className="relative group lg:max-w-sm"
            >
              <div className={`aspect-[4/5] rounded-3xl overflow-hidden border relative shadow-[0_20px_60px_rgba(0,0,0,0.5)]
                ${displayType === "pharaoh" ? "border-[#E6B23C]/10 bg-[#1E160E]" : "border-[#A08E70]/10 bg-[#12150E]"}`}>

                <div className="absolute inset-0 bg-gradient-to-t from-[#0D0A07] via-[#0D0A07]/10 to-transparent z-10" />

                {finalImageUrl && (
                  <img
                    src={finalImageUrl}
                    alt={cleanDisplayName}
                    className={`absolute inset-0 w-full h-full object-cover ${displayType === "pharaoh" ? "object-top" : "object-center"} transition-transform duration-700 group-hover:scale-110`}
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.opacity = '0';
                    }}
                  />
                )}

                {/* Type badge — dynamic */}
                <div className="absolute top-5 left-5 z-20 flex flex-col gap-2">
                  <div className="px-3 py-1.5 bg-gradient-to-r from-[#E6B23C] to-[#D4A030] rounded-full text-[10px] font-bold tracking-[0.2em] text-[#0D0A07] uppercase shadow-[0_4px_15px_rgba(230,178,60,0.3)] flex items-center gap-1.5">
                    {displayType === "pharaoh" ? <Crown size={10} /> : <MapPin size={10} />}
                    {typeLabel}
                  </div>
                  {isQuickLink && (
                    <div className="px-3 py-1 bg-[#E6B23C]/10 border border-[#E6B23C]/15 backdrop-blur-md rounded-full text-[9px] font-bold tracking-[0.15em] text-[#E6B23C] uppercase flex items-center gap-1.5 w-fit">
                      <Sparkles size={8} /> {t("result.badge.quicklink")}
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
          )}

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


              {/* Entity name heading */}
              <h2
                className="text-2xl md:text-4xl font-bold text-[#1A1005] uppercase tracking-[0.06em] mb-4 md:mb-5 border-b border-[#1A1005]/10 pb-3 md:pb-4"
                style={{ fontFamily: "var(--font-cormorant), serif" }}
              >
                {cleanDisplayName}
              </h2>

              {/* Description */}
              <p
                className="text-[#1A1005] leading-[1.8] text-[15px] md:text-lg font-medium text-justify"
                style={{ fontFamily: "var(--font-cormorant), serif" }}
              >
                &quot;{displayDescription}&quot;
              </p>

              {/* Metadata rows — Only rendered when values are non-null */}
              {(dynasty || period || location || displayType === "pharaoh") && (
                <div className="mt-5 md:mt-6 pt-4 border-t border-[#1A1005]/10 flex flex-col gap-2.5 md:gap-3 font-cormorant">
                  {displayType === "pharaoh" && (
                    <div className="flex items-center gap-3 text-sm md:text-base text-[#1A1005]/70">
                      <Crown size={15} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#B8860B] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-[10px] md:text-xs text-[#1A1005]/50 ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.type")}</span>
                      <span className="font-medium capitalize whitespace-nowrap">{dbType}</span>
                    </div>
                  )}
                  {dynasty && (
                    <div className="flex items-center gap-3 text-sm md:text-base text-[#1A1005]/70">
                      <Scroll size={15} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#B8860B] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-[10px] md:text-xs text-[#1A1005]/50 ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.dynasty")}</span>
                      <span className="font-medium">{dynasty}</span>
                    </div>
                  )}
                  {period && (
                    <div className="flex items-center gap-3 text-sm md:text-base text-[#1A1005]/70">
                      <Hourglass size={15} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#B8860B] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-[10px] md:text-xs text-[#1A1005]/50 ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.period")}</span>
                      <span className="font-medium">{period}</span>
                    </div>
                  )}
                  {location && (
                    <div className="flex items-center gap-3 text-sm md:text-base text-[#1A1005]/70">
                      <MapPin size={15} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#B8860B] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-[10px] md:text-xs text-[#1A1005]/50 ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.location")}</span>
                      <span className="font-medium">{location}</span>
                    </div>
                  )}
                </div>
              )}

              <div className="mt-8 pt-5 border-t border-[#1A1005]/8 flex justify-center items-center opacity-40">
                <div className="text-[9px] font-bold tracking-[0.3em] text-[#1A1005] uppercase">{t("result.papyrus.archive")}</div>
              </div>
            </motion.div>

            {/* Actions: Video & Chat — per sub-entity when composite */}
            <div className="flex flex-col gap-5">
              {!hideMediaButtons && (
                <>
                  {compositeEntitiesData.length > 0 ? (
                    /* ── Composite: one row per sub-entity ────────── */
                    <div className="flex flex-col gap-4">
                      {compositeEntitiesData.map((sub) => {
                        const cleanSubName = cleanEntityName(sub.name);
                        return (
                          <div key={sub.name} className="grid sm:grid-cols-2 gap-4">
                            <Button
                              onClick={() => router.push(`/video?entity=${encodeURIComponent(sub.name)}&type=${displayType}&dynasty=${encodeURIComponent(sub.dynasty || '')}&period=${encodeURIComponent(sub.period || '')}&dbType=${encodeURIComponent(sub.type || '')}&location=${encodeURIComponent(location || '')}`)}
                              className="h-14 rounded-2xl bg-[#E6B23C]/10 border border-[#E6B23C]/20 hover:bg-[#E6B23C]/20 active:bg-[#E6B23C]/30 active:scale-[0.98] active:ring-2 active:ring-[#E6B23C]/50 text-[#E6B23C] font-bold text-base transition-all hover:scale-[1.02] flex items-center justify-center gap-3"
                            >
                              <Video size={20} />
                              {t("result.button.video_named", { name: cleanSubName })}
                            </Button>
                            <Button
                              onClick={() => router.push(`/chat?entity=${encodeURIComponent(sub.name)}&type=${displayType}`)}
                              variant="outline"
                              className="h-14 rounded-2xl border-[#E6B23C]/12 bg-[#E6B23C]/[0.04] hover:bg-[#E6B23C]/[0.08] active:bg-[#E6B23C]/20 active:scale-[0.98] active:ring-2 active:ring-[#E6B23C]/50 text-[#F5E6D0] font-semibold text-base transition-all hover:scale-[1.02] flex items-center justify-center gap-3"
                            >
                              <MessageSquare size={20} />
                              {t("result.button.chat_named", { name: cleanSubName })}
                            </Button>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    /* ── Normal: single row ───────────────────────── */
                    <div className="grid sm:grid-cols-2 gap-5">
                      <Button
                        onClick={() => router.push(`/video?entity=${encodeURIComponent(displayName)}&type=${displayType}&dynasty=${encodeURIComponent(dynasty || '')}&period=${encodeURIComponent(period || '')}&dbType=${encodeURIComponent(dbType || '')}&location=${encodeURIComponent(location || '')}`)}
                        className="h-14 rounded-2xl bg-[#E6B23C]/10 border border-[#E6B23C]/20 hover:bg-[#E6B23C]/20 active:bg-[#E6B23C]/30 active:scale-[0.98] active:ring-2 active:ring-[#E6B23C]/50 text-[#E6B23C] font-bold text-base transition-all hover:scale-[1.02] flex items-center justify-center gap-3"
                      >
                        <Video size={20} />
                        {t("result.button.video")}
                      </Button>
                      <Button
                        onClick={() => router.push(`/chat?entity=${encodeURIComponent(displayName)}&type=${displayType}`)}
                        variant="outline"
                        className="h-14 rounded-2xl border-[#E6B23C]/12 bg-[#E6B23C]/[0.04] hover:bg-[#E6B23C]/[0.08] active:bg-[#E6B23C]/20 active:scale-[0.98] active:ring-2 active:ring-[#E6B23C]/50 text-[#F5E6D0] font-semibold text-base transition-all hover:scale-[1.02] flex items-center justify-center gap-3"
                      >
                        <MessageSquare size={20} />
                        {t("result.button.chat")}
                      </Button>
                    </div>
                  )}
                </>
              )}

              <div className="flex w-full">
                <Button
                  onClick={handleToggleFavorite}
                  disabled={isSaving}
                  className="h-14 flex-1 rounded-2xl font-bold text-base transition-all shadow-[0_4px_30px_rgba(0,0,0,0.2)] flex items-center justify-center gap-3 bg-[#D8C09A] text-[#1A1005] hover:bg-[#C8B08A] hover:shadow-[0_0_30px_rgba(216,192,154,0.2)]"
                >
                  {isSaving ? (
                    <div className="h-5 w-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  ) : isSaved ? (
                    <>
                      <BookmarkMinus size={20} />
                      Remove from Favorites
                    </>
                  ) : (
                    <>
                      <Bookmark size={20} />
                      Add to Favorites
                    </>
                  )}
                </Button>
              </div>
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
