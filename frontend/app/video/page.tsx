"use client";

import { useMemo, useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { Play, Wand2, Sparkles, Film, Crown, Hourglass, Scroll, MapPin, MessageSquare, Video } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useLanguage } from "../../context/LanguageContext";
import { Suspense } from "react";

function VideoPageContent() {
  const { t, isRTL } = useLanguage();
  const sp = useSearchParams();
  const entityType = (sp.get("type") || "landmark").toLowerCase();
  const entityName = sp.get("name") || sp.get("entity") || "Great Sphinx of Giza";
  const dynasty = sp.get("dynasty");
  const period = sp.get("period");
  const dbType = sp.get("dbType");
  const location = sp.get("location");
  const label = useMemo(() => (entityType === "pharaoh" || entityType === "king") ? t("result.badge.pharaoh") : t("result.badge.landmark"), [entityType, t]);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const hasFetched = useRef(false);

  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true;
      onGenerate();
    }
  }, []);

  const onGenerate = async () => {
    try {
      setIsGenerating(true);
      const isLandmark = entityType === "landmark";
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";
      const response = await fetch(`${API_BASE_URL}/api/v1/video/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entity_name: entityName, is_landmark: isLandmark })
      });

      if (!response.ok) {
        throw new Error("Failed to start video generation.");
      }

      // Poll for status
      let ready = false;
      while (!ready) {
        await new Promise(r => setTimeout(r, 5000)); // wait 5 seconds

        const statusRes = await fetch(`${API_BASE_URL}/api/v1/video/status/${encodeURIComponent(entityName)}`);
        if (!statusRes.ok) continue;

        const statusData = await statusRes.json();
        if (statusData.status === "ready") {
          ready = true;
          setVideoUrl(`${API_BASE_URL}/api/v1/video/stream/${encodeURIComponent(entityName)}`);
        } else if (statusData.status === "failed") {
          throw new Error("Video generation failed on the server.");
        }
      }
    } catch (error) {
      alert(error instanceof Error ? error.message : "Error generating video");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <>
      <motion.div initial={{ opacity: 0, x: isRTL ? 20 : -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
        <Link href="/result" className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors">
          <span className={`transition-transform ${isRTL ? 'group-hover:translate-x-1' : 'group-hover:-translate-x-1'}`}>
            {isRTL ? '→' : '←'}
          </span>
          {t("common.return")}
        </Link>
      </motion.div>

      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
        <div className="flex items-center gap-5">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-[#E6B23C]/15 to-[#E6B23C]/5 border border-[#E6B23C]/15 flex items-center justify-center text-[#E6B23C]">
            <Video size={32} />
          </div>
          <div>
            <h1 className="font-heading text-3xl md:text-4xl font-bold text-[#F5E6D0] tracking-tight">
              {t("video.title.part1")} <span className="text-[#E6B23C] gold-glow">{t("video.title.part2")}</span>
            </h1>
          </div>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }} className="lg:col-span-2">
          <div className="group relative aspect-video rounded-3xl overflow-hidden border border-[#E6B23C]/8 bg-[#0D0A07] shadow-[0_20px_60px_rgba(0,0,0,0.6)] flex items-center justify-center">
            <div className="absolute inset-0 bg-gradient-to-t from-[#0D0A07] via-transparent to-transparent z-10 pointer-events-none" />
            <div className="absolute inset-0 bg-gradient-to-br from-[#E6B23C]/[0.03] to-transparent pointer-events-none" />

            {videoUrl ? (
              <div className="relative w-full h-full group/video cursor-pointer">
                <video
                  src={videoUrl}
                  controls
                  autoPlay
                  onPlay={() => {
                    const overlay = document.getElementById('video-play-overlay');
                    if (overlay) overlay.style.opacity = '0';
                  }}
                  onPause={() => {
                    const overlay = document.getElementById('video-play-overlay');
                    if (overlay) overlay.style.opacity = '1';
                  }}
                  className="absolute inset-0 w-full h-full object-contain z-20 bg-black"
                />
                {/* Custom Play Overlay - Matches Home Page */}
                <div
                  id="video-play-overlay"
                  className="absolute inset-0 flex items-center justify-center z-30 pointer-events-none transition-opacity duration-500 opacity-0 group-hover/video:opacity-100"
                >
                  <div className="w-16 h-16 rounded-full bg-[#0D0A07]/40 backdrop-blur-md border-2 border-[#E6B23C] flex items-center justify-center shadow-[0_0_50px_rgba(0,0,0,0.5)]">
                    <Play size={32} fill="#E6B23C" className="text-[#E6B23C] ml-1" />
                  </div>
                </div>
              </div>
            ) : isGenerating ? (
              <div className="flex flex-col items-center z-20 gap-6">
                <div className="relative w-24 h-24 flex items-center justify-center">
                  {/* Rotating Decoder Rings */}
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 8, ease: "linear" }}
                    className="absolute inset-0 border border-dashed border-[#E6B23C]/30 rounded-full"
                  />
                  <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ repeat: Infinity, duration: 12, ease: "linear" }}
                    className="absolute inset-2 border border-dotted border-[#E6B23C]/20 rounded-full"
                  />
                  {/* Central Pulsing Icon */}
                  <motion.div
                    animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="text-[#E6B23C]"
                  >
                    <Film size={32} />
                  </motion.div>
                </div>

                {/* Synthesis Metadata HUD */}
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center gap-3">
                    <motion.div
                      animate={{ width: [0, 40, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="h-[1px] bg-gradient-to-r from-transparent to-[#E6B23C]"
                    />
                    <div className="text-[#E6B23C] font-bold tracking-[0.3em] uppercase text-[10px]">{t("video.status.loading")}</div>
                    <motion.div
                      animate={{ width: [0, 40, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="h-[1px] bg-gradient-to-l from-transparent to-[#E6B23C]"
                    />
                  </div>
                  <div className="text-[#A08E70] text-xs max-w-[250px] text-center italic mt-2 opacity-80">{t("video.status.loading_desc")}</div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center z-20 gap-4">
                <div className="text-[#A08E70] text-sm text-center italic">{t("video.status.failed")}</div>
                <Button onClick={onGenerate} className="mt-4 bg-[#E6B23C] text-black hover:bg-[#FFD369]">{t("video.button.retry")}</Button>
              </div>
            )}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2, duration: 0.6 }} className="h-full">
          <div className="glass-surface rounded-3xl p-8 relative overflow-hidden h-full flex flex-col">
            <div className="absolute top-0 right-0 w-24 h-24 bg-[#E6B23C]/[0.04] blur-[60px]" />
            <div className="text-3xl font-bold text-[#F5E6D0] mb-2" style={{ fontFamily: 'var(--font-cormorant), serif' }}>{entityName}</div>
            <div className="inline-flex px-3 py-1.5 rounded-full bg-[#E6B23C]/8 border border-[#E6B23C]/15 text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-5">{label} {t("video.badge.detected")}</div>

            <div className="flex-1 flex flex-col justify-center">
              {(dynasty || period || location || (label === "PHARAOH" && dbType && dbType !== "Unknown")) && (
                <div className="border-y border-[#E6B23C]/10 py-8 flex flex-col gap-4">
                  {label === "PHARAOH" && dbType && dbType !== "Unknown" && (
                    <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                      <Crown size={16} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#E6B23C] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-xs text-[#A08E70] ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.type")}</span>
                      <span className="font-small capitalize">{dbType}</span>
                    </div>
                  )}
                  {dynasty && (
                    <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                      <Scroll size={16} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#E6B23C] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-xs text-[#A08E70] ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.dynasty")}</span>
                      <span className="font-small">{dynasty}</span>
                    </div>
                  )}
                  {period && (
                    <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                      <Hourglass size={16} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#E6B23C] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-xs text-[#A08E70] ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.period")}</span>
                      <span className="font-small">{period}</span>
                    </div>
                  )}
                  {location && (
                    <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                      <MapPin size={16} className={`${isRTL ? 'ml-0' : 'mr-0'} text-[#E6B23C] shrink-0`} />
                      <span className={`font-semibold uppercase tracking-wide text-xs text-[#A08E70] ${isRTL ? 'w-24' : 'w-20'}`}>{t("result.meta.location")}</span>
                      <span className="font-small">{location}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="mt-8 pt-6 border-t border-[#E6B23C]/5">
              <Link href={`/chat?entity=${encodeURIComponent(entityName)}&type=${encodeURIComponent(entityType)}`} className="block w-full">
                <Button className="w-full h-14 rounded-2xl bg-gradient-to-r from-[#E6B23C] to-[#D4A030] hover:from-[#FFD369] hover:to-[#E6B23C] text-[#1A1005] font-bold text-base transition-all hover:scale-[1.02] shadow-[0_4px_30px_rgba(230,178,60,0.15)] flex items-center justify-center gap-2">
                  <MessageSquare size={20} className={isRTL ? "ml-0" : "mr-0"} />
                  {t("video.button.chat")}
                </Button>
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </>
  );
}

export default function VideoPage() {
  const { t } = useLanguage();
  return (
    <PageShell>
      <Suspense fallback={<div className="flex items-center justify-center min-h-[50vh]"><div className="animate-pulse text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase">{t("video.suspense")}</div></div>}>
        <VideoPageContent />
      </Suspense>
    </PageShell>
  );
}
