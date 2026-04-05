"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import PageShell from "../../components/feature/PageShell";
import { Button } from "../../components/ui/button";
import { Play, Wand2, Sparkles, Film } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Suspense } from "react";

function VideoPageContent() {
  const sp = useSearchParams();
  const entityType = (sp.get("type") || "landmark").toLowerCase();
  const entityName = sp.get("name") || sp.get("entity") || "Great Sphinx of Giza";
  const label = useMemo(() => (entityType === "pharaoh" || entityType === "king") ? "PHARAOH" : "LANDMARK", [entityType]);
  const [isGenerating, setIsGenerating] = useState(false);

  const onGenerate = () => {
    setIsGenerating(true);
    setTimeout(() => { setIsGenerating(false); alert("Neural Cinematography Engine initialized. Video generation in progress..."); }, 1500);
  };

  return (
    <>
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
        <Link href="/result" className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors">
          <span className="group-hover:-translate-x-1 transition-transform">←</span>
          Return to Discovery
        </Link>
      </motion.div>

      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
        <div className="flex items-start gap-5">
          <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-[#E6B23C]/15 to-[#E6B23C]/5 border border-[#E6B23C]/15 flex items-center justify-center text-[#E6B23C]">
            <Film size={28} />
          </div>
          <div>
            <div className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase mb-1">Cinematic Synthesis</div>
            <h1 className="font-heading text-3xl md:text-4xl font-bold text-[#F5E6D0] tracking-tight">
              AI <span className="text-[#E6B23C] gold-glow">Cinematics</span>
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase bg-[#E6B23C]/[0.06] border border-[#E6B23C]/10 px-4 py-2 rounded-full">
          <Sparkles size={12} />
          Tourist Mode
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }} className="lg:col-span-2">
          <div className="group relative aspect-video rounded-3xl overflow-hidden border border-[#E6B23C]/8 bg-[#0D0A07] shadow-[0_20px_60px_rgba(0,0,0,0.6)]">
            <div className="absolute inset-0 bg-gradient-to-t from-[#0D0A07] via-transparent to-transparent z-10" />
            {/* Warm cinema ambient */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#E6B23C]/[0.03] to-transparent" />

            <div className="absolute inset-0 flex items-center justify-center z-20">
              <motion.button whileHover={{ scale: 1.12 }} whileTap={{ scale: 0.92 }} className="relative" onClick={() => alert("Previewing...")}>
                <div className="absolute inset-0 bg-[#E6B23C]/20 rounded-full blur-3xl" />
                <motion.div animate={{ scale: [1, 1.35, 1], opacity: [0.2, 0.5, 0.2] }} transition={{ repeat: Infinity, duration: 2.5 }} className="absolute inset-[-14px] border-2 border-[#E6B23C]/25 rounded-full" />
                <div className="h-20 w-20 rounded-full bg-gradient-to-br from-[#E6B23C] to-[#D4A030] flex items-center justify-center text-[#0D0A07] shadow-[0_0_40px_rgba(230,178,60,0.4)]">
                  <Play fill="currentColor" size={32} className="ml-1" />
                </div>
              </motion.button>
            </div>

            <div className="absolute bottom-8 left-8 z-20">
              <div className="text-[10px] font-bold tracking-[0.3em] text-[#E6B23C] uppercase mb-2">Sequence Preview</div>
              <h3 className="font-heading text-3xl font-bold text-white">{entityName}</h3>
            </div>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2, duration: 0.6 }} className="space-y-6">
          <div className="glass-surface rounded-3xl p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-[#E6B23C]/[0.04] blur-[60px]" />
            <div className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase mb-5">Entity Metadata</div>
            <div className="font-heading text-2xl font-bold text-[#F5E6D0] mb-2">{entityName}</div>
            <div className="inline-flex px-3 py-1.5 rounded-full bg-[#E6B23C]/8 border border-[#E6B23C]/15 text-[10px] font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-5">{label} Detected</div>
            <p className="text-sm text-[#A08E70] leading-relaxed border-l-2 border-[#E6B23C]/15 pl-4 italic">
              AI has synthesized historical data for this cinematic perspective.
            </p>
          </div>

          <div className="glass-surface rounded-3xl p-8">
            <Button disabled={isGenerating} onClick={onGenerate}
              className="w-full h-16 rounded-2xl bg-gradient-to-r from-[#E6B23C] to-[#D4A030] hover:from-[#FFD369] hover:to-[#E6B23C] text-[#0D0A07] font-bold text-lg transition-all hover:scale-[1.03] shadow-[0_4px_30px_rgba(230,178,60,0.2)]">
              <AnimatePresence mode="wait">
                {isGenerating ? (
                  <motion.div key="gen" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center">
                    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }} className="mr-3"><Sparkles size={20} /></motion.div>
                    Synthesizing...
                  </motion.div>
                ) : (
                  <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center">
                    <Wand2 className="mr-3" size={22} /> Generate Video
                  </motion.div>
                )}
              </AnimatePresence>
            </Button>
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
