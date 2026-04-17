"use client";

import { useMemo, useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { Play, Wand2, Sparkles, Film, Crown, Hourglass, Scroll, MapPin, MessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Suspense } from "react";

function VideoPageContent() {
  const sp = useSearchParams();
  const entityType = (sp.get("type") || "landmark").toLowerCase();
  const entityName = sp.get("name") || sp.get("entity") || "Great Sphinx of Giza";
  const dynasty = sp.get("dynasty");
  const period = sp.get("period");
  const dbType = sp.get("dbType");
  const location = sp.get("location");
  const label = useMemo(() => (entityType === "pharaoh" || entityType === "king") ? "PHARAOH" : "LANDMARK", [entityType]);
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
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8010";
      const response = await fetch(`${API_BASE_URL}/api/v1/video/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entity_name: entityName, is_landmark: isLandmark })
      });

      if (!response.ok) {
        throw new Error("Failed to generate video. The artifact may not have a ready script.");
      }

      const blob = await response.blob();
      setVideoUrl(URL.createObjectURL(blob));
    } catch (error) {
      alert(error instanceof Error ? error.message : "Error generating video");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <>
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
        <Link href="/result" className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors">
          <span className="group-hover:-translate-x-1 transition-transform">←</span>
          Return
        </Link>
      </motion.div>

      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
        <div className="flex items-center gap-5">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-[#E6B23C]/15 to-[#E6B23C]/5 border border-[#E6B23C]/15 flex items-center justify-center text-[#E6B23C]">
            <Film size={32} />
          </div>
          <div>
            <h1 className="font-heading text-3xl md:text-4xl font-bold text-[#F5E6D0] tracking-tight">
              Story of <span className="text-[#E6B23C] gold-glow">Origins</span>
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
              <video src={videoUrl} controls autoPlay className="absolute inset-0 w-full h-full object-contain z-20 bg-black" />
            ) : isGenerating ? (
              <div className="flex flex-col items-center z-20 gap-4">
                <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }} className="text-[#E6B23C]">
                  <Sparkles size={40} />
                </motion.div>
                <div className="text-[#E6B23C] font-bold tracking-[0.2em] uppercase text-sm animate-pulse">Synthesizing...</div>
                <div className="text-[#A08E70] text-xs max-w-[200px] text-center italic">Unfolding the story from history. This may take a moment.</div>
              </div>
            ) : (
              <div className="flex flex-col items-center z-20 gap-4">
                <div className="text-[#A08E70] text-sm text-center italic">Video generation failed or not ready.</div>
                <Button onClick={onGenerate} className="mt-4 bg-[#E6B23C] text-black hover:bg-[#FFD369]">Retry</Button>
              </div>
            )}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2, duration: 0.6 }} className="space-y-6">
          <div className="glass-surface rounded-3xl p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-[#E6B23C]/[0.04] blur-[60px]" />
            <div className="text-3xl font-bold text-[#F5E6D0] mb-2" style={{ fontFamily: 'var(--font-cormorant), serif' }}>{entityName}</div>
            <div className="inline-flex px-3 py-1.5 rounded-full bg-[#E6B23C]/8 border border-[#E6B23C]/15 text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-5">{label} Detected</div>

            {(dynasty || period || location || (label === "PHARAOH" && dbType && dbType !== "Unknown")) && (
              <div className="mt-2 pt-5 border-t border-[#E6B23C]/10 flex flex-col gap-3">
                {label === "PHARAOH" && dbType && dbType !== "Unknown" && (
                  <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                    <Crown size={16} className="text-[#E6B23C] shrink-0" />
                    <span className="font-semibold uppercase tracking-wide text-xs text-[#A08E70] w-20">Type</span>
                    <span className="font-small capitalize">{dbType}</span>
                  </div>
                )}
                {dynasty && (
                  <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                    <Scroll size={16} className="text-[#E6B23C] shrink-0" />
                    <span className="font-semibold uppercase tracking-wide text-xs text-[#A08E70] w-20">Dynasty</span>
                    <span className="font-small">{dynasty}</span>
                  </div>
                )}
                {period && (
                  <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                    <Hourglass size={16} className="text-[#E6B23C] shrink-0" />
                    <span className="font-semibold uppercase tracking-wide text-xs text-[#A08E70] w-20">Period</span>
                    <span className="font-small">{period}</span>
                  </div>
                )}
                {location && (
                  <div className="flex items-center gap-3 text-base text-[#F5E6D0]/70">
                    <MapPin size={16} className="text-[#E6B23C] shrink-0" />
                    <span className="font-semibold uppercase tracking-wide text-xs text-[#A08E70] w-20">Location</span>
                    <span className="font-small">{location}</span>
                  </div>
                )}
              </div>
            )}

            <div className="mt-8 pt-6 border-t border-[#E6B23C]/5">
              <Link href={`/chat?entity=${encodeURIComponent(entityName)}&type=${encodeURIComponent(entityType)}`} className="block w-full">
                <Button className="w-full h-14 rounded-2xl bg-gradient-to-r from-[#C1840A] to-[#A06A00] hover:from-[#D4A030] hover:to-[#C1840A] text-white font-bold text-base transition-all hover:scale-[1.02] shadow-[0_4px_30px_rgba(230,178,60,0.15)]">
                  <MessageSquare className="mr-3" size={20} />
                  Chat with {entityName}
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
  return (
    <PageShell>
      <Suspense fallback={<div className="flex items-center justify-center min-h-[50vh]"><div className="animate-pulse text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase">Initializing Neural Engine...</div></div>}>
        <VideoPageContent />
      </Suspense>
    </PageShell>
  );
}
