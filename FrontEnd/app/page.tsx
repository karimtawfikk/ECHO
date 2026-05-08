"use client";

import Link from "next/link";
import PageShell from "../components/layout/PageShell";
import { Button } from "../components/ui/button";
import { motion } from "framer-motion";
import { DoorOpen, Sparkles, Camera, Cpu, PlayCircle, MessageSquare, Video, Bird } from "lucide-react";
import TrendingRow from "../components/trending/TrendingRow";
import ScrollReveal from "../components/animations/ScrollReveal";
import ParallaxLayer from "../components/animations/ParallaxLayer";
import { useEffect, useState } from "react";
import type { RecognitionEntity } from "../lib/types";

// ── Minimal mock fallback (used only when the API is unreachable) ──────────
import { PHARAOHS as MOCK_PHARAOHS, LANDMARKS as MOCK_LANDMARKS } from "../lib/mock/mock-trending";

// Preserve the exact same 5 names in display order
const PHARAOH_ORDER = [
  "Akhenaton",
  "Cleopatra VII Philopator",
  "Hatshepsut",
  "Ramesses II",
  "Tutankhamun",
];
const LANDMARK_ORDER = [
  "Pyramids of Giza",
  "Sphinx",
  "Temple of Karnak",
  "Temple of Luxor",
  "The Great Temple of Ramesses II at Abu Simbel",
];

function mockToEntity(item: { name: string; description: string; dynasty?: string; period?: string; location?: string }, idx: number): RecognitionEntity {
  return {
    id: idx + 1,
    name: item.name,
    description: item.description,
    type: (item as { type?: string }).type ?? null,
    dynasty: (item as { dynasty?: string }).dynasty ?? null,
    period: (item as { period?: string }).period ?? null,
    location: (item as { location?: string }).location ?? null,
    composite_entity: null,
    composite_entities_data: null,
    images: [],
    scripts: null,
  };
}

const FALLBACK_PHARAOHS: RecognitionEntity[] = PHARAOH_ORDER
  .map((name, idx) => {
    const p = MOCK_PHARAOHS.find((x) => x.name === name);
    return p ? mockToEntity(p, idx) : null;
  })
  .filter((x): x is RecognitionEntity => x !== null);

const FALLBACK_LANDMARKS: RecognitionEntity[] = LANDMARK_ORDER
  .map((name, idx) => {
    const l = MOCK_LANDMARKS.find((x) => x.name === name);
    return l ? mockToEntity(l, idx) : null;
  })
  .filter((x): x is RecognitionEntity => x !== null);

// ── Page ─────────────────────────────────────────────────────────────────
export default function HomePage() {
  const [pharaohs] = useState<RecognitionEntity[]>(FALLBACK_PHARAOHS);
  const [landmarks] = useState<RecognitionEntity[]>(FALLBACK_LANDMARKS);
  const [isLoading] = useState(false);

  // Database fetch removed to eliminate latency as requested
  useEffect(() => {
    // No-op: we now use the hardcoded mock data directly
  }, []);

  return (
    <PageShell fullWidth>
      {/* =========== HERO =========== */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="min-h-[100vh] pt-[25vh] pb-10 flex flex-col justify-center items-center text-center relative overflow-hidden"
      >
        {/* Dynamic Background Image */}
        <div
          className="absolute inset-0 z-[-2]"
          style={{
            maskImage: "linear-gradient(to bottom, black 80%, transparent 100%)",
            WebkitMaskImage: "linear-gradient(to bottom, black 80%, transparent 100%)"
          }}
        >
          {/* Base Color Fill */}
          <div className="absolute inset-0 bg-[#0D0A07]" />

          <div
            className="w-full h-full"
            style={{
              maskImage: "linear-gradient(to bottom, black 0%, black 25%, transparent 80%)",
              WebkitMaskImage: "linear-gradient(to bottom, black 0%, black 25%, transparent 80%)"
            }}
          >
            <img
              src="/images/backgrounds/x.jpg"
              alt="Background"
              className="w-full h-auto object-top opacity-60"
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
          </div>
          {/* Darkening layer */}
          <div className="absolute inset-0 bg-[#0D0A07]/50" />
          {/* Bottom blend */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#0D0A07]" />
        </div>

        {/* Warm ambient light behind hero with subtle parallax */}
        <ParallaxLayer speed={0.15} className="absolute inset-0 z-[-1] pointer-events-none flex items-center justify-center">
          <div className="w-[700px] h-[500px] rounded-full"
            style={{ background: "radial-gradient(circle, rgba(230,178,60,0.08) 0%, rgba(200,140,30,0.03) 40%, transparent 70%)" }}
          />
        </ParallaxLayer>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#E6B23C]/[0.08] border border-[#E6B23C]/15 mb-8"
        >
          <DoorOpen size={14} className="text-[#E6B23C]" />
          <span className="text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase">Gateway to Ancient Egypt</span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.7 }}
          className="mb-8 max-w-5xl"
        >
          <h1
            className="font-display text-6xl md:text-8xl lg:text-[7rem] font-bold tracking-[0.15em] uppercase text-[#E6B23C] gold-glow mb-4"
            style={{ fontFamily: 'var(--font-cormorant), serif' }}
          >
            E.C.H.O
          </h1>
          <p
            className="font-display text-2xl md:text-3xl lg:text-4xl font-bold leading-[1.2] tracking-[0.03em] uppercase text-[#F5E6D0]"
            style={{ fontFamily: 'var(--font-cormorant), serif' }}
          >
            Every Capture Has <span className="text-[#E6B23C]">Origins</span>
          </p>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="text-[#A08E70] text-lg md:text-xl max-w-xl font-semibold leading-relaxed mb-12"
          style={{ fontFamily: 'var(--font-cormorant), serif' }}
        >
          Upload a landmark or artifact to explore its origins, context, and story through visuals, narration, and conversation.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.65, duration: 0.5 }}
          className="flex flex-col sm:flex-row flex-wrap justify-center gap-5"
        >
          <Button
            asChild
            className="h-14 w-64 rounded-2xl bg-gradient-to-r from-[#E6B23C] to-[#D4A030] hover:from-[#FFD369] hover:to-[#E6B23C] text-[#0D0A07] font-bold text-base transition-all hover:scale-105 shadow-[0_4px_30px_rgba(230,178,60,0.25)] hover:shadow-[0_4px_40px_rgba(230,178,60,0.4)]"
          >
            <Link href="/upload">
              Start Your Journey
            </Link>
          </Button>

          <Button
            asChild
            variant="outline"
            className="h-14 w-64 rounded-2xl border-[#E6B23C]/15 bg-[#E6B23C]/[0.04] hover:bg-[#E6B23C]/[0.08] text-[#F5E6D0] font-bold text-base transition-all hover:scale-105 hover:border-[#E6B23C]/25"
          >
            <Link href="#how-it-works">
              See How It Works
            </Link>
          </Button>

        </motion.div>

        {/* Decorative Egyptian line */}
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 1, duration: 0.8, ease: "easeOut" }}
          className="mt-20 w-48 h-[1px]"
          style={{ background: "linear-gradient(90deg, transparent, #E6B23C, transparent)" }}
        />
      </motion.section>



      {/* =========== HOW IT WORKS =========== */}
      <section id="how-it-works" className="mt-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-20">
        <div className="text-center mb-16">
          <span className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase">The Process</span>
          <h2 className="font-heading text-3xl md:text-4xl lg:text-5xl font-bold mt-4 text-[#F5E6D0]">
            How ECHO Brings History to Life
          </h2>
        </div>
        <div className="relative max-w-4xl mx-auto py-12">
          {/* Vertical Connecting Line */}
          <div className="absolute left-10 md:left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 bg-gradient-to-b from-transparent via-[#E6B23C]/30 to-transparent z-0" />

          <div className="space-y-24 relative z-10">
            {[
              { icon: Camera, title: "1. Capture", text: "Snap a photo of a landmark, statue, or temple wall for instant AI recognition.", isText: false },
              { icon: Video, title: "2. Entertain", text: "Experience custom generated documentary-style videos that bring ancient history to life.", isText: false },
              { icon: MessageSquare, title: "3. Engage", text: "Chat directly with pharaohs and iconic monuments.", isText: false },
              { icon: Bird, title: "4. Discover", text: "Decode and translate complex ancient hieroglyphic inscriptions into modern text.", isText: false }
            ].map((step, i) => (
              <ScrollReveal key={step.title} direction={i % 2 === 0 ? "right" : "left"} delay={0.1} className="relative flex items-center">

                {/* Timeline Center Node */}
                <div className="absolute left-10 md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-[#0D0A07] border border-[#E6B23C]/50 flex items-center justify-center shadow-[0_0_30px_rgba(230,178,60,0.15)] z-10">
                  {step.isText ? (
                    <span className="text-4xl text-[#E6B23C] leading-none -translate-y-[10px]">{step.icon as any}</span>
                  ) : (
                    // @ts-ignore
                    <step.icon size={28} className="text-[#E6B23C]" />
                  )}
                </div>

                {/* Content */}
                <div className={`w-[calc(100%-6rem)] md:w-[42%] ml-auto md:ml-0 ${i % 2 === 0 ? "md:mr-auto" : "md:ml-auto"} text-left`}>
                  <div className="inline-flex items-center px-5 py-2 rounded-full bg-[#E6B23C]/[0.08] border border-[#E6B23C]/20 mb-4 shadow-[0_0_15px_rgba(230,178,60,0.05)]">
                    <h3 className="text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase">{step.title}</h3>
                  </div>
                  <p className="text-[#A08E70] leading-relaxed text-lg">{step.text}</p>
                </div>

              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* =========== EXPLORE GALLERY =========== */}
      <section className="mt-40 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="font-heading text-3xl md:text-4xl font-bold text-[#F5E6D0]">Experience ECHO</h2>
          <p className="text-[#A08E70] mt-3 max-w-xl mx-auto text-lg">Choose an icon. Uncover its origins. Start a conversation.</p>
        </div>
        <ScrollReveal direction="up" delay={0.1}>
          <TrendingRow
            title="Icons of Ancient Egypt"
            items={pharaohs}
            type="pharaoh"
            isLoading={isLoading}
          />
        </ScrollReveal>
      </section>

      {/* =========== TRENDING: LANDMARKS =========== */}
      <ScrollReveal direction="up" delay={0.1} className="mt-16 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <TrendingRow
          title="Must-See Ancient Sites"
          items={landmarks}
          type="landmark"
          isLoading={isLoading}
        />
      </ScrollReveal>

      {/* =========== FEATURE SHOWCASES =========== */}
      <section className="mt-40 mb-20 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        {/* The Snaking Timeline Line (Desktop Only) */}
        <div className="absolute left-0 right-0 top-0 bottom-0 pointer-events-none hidden lg:block z-0">
          <svg width="100%" height="100%" viewBox="0 0 1000 1600" fill="none" preserveAspectRatio="none" className="opacity-20">
            <path
              d="M 250 100 C 600 100, 850 300, 850 550 C 850 800, 150 800, 150 1050 C 150 1300, 600 1500, 850 1500"
              stroke="#E6B23C"
              strokeWidth="2"
              strokeDasharray="12 12"
            />
            {/* Animated drawing path */}
            <motion.path
              d="M 250 100 C 600 100, 850 300, 850 550 C 850 800, 150 800, 150 1050 C 150 1300, 600 1500, 850 1500"
              stroke="#E6B23C"
              strokeWidth="4"
              initial={{ pathLength: 0 }}
              whileInView={{ pathLength: 1 }}
              transition={{ duration: 2.5, ease: "easeInOut" }}
              viewport={{ once: false, amount: 0.1 }}
              className="drop-shadow-[0_0_15px_rgba(230,178,60,0.6)]"
            />
          </svg>
        </div>

        <div className="space-y-64 relative z-10">
          {/* 1. Recognition Feature */}
          <ScrollReveal direction="up" className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="relative">
              <div className="aspect-[4/3] max-w-[480px] mx-auto rounded-3xl overflow-hidden border border-[#E6B23C]/20 bg-[#1A1208] shadow-[0_0_50px_rgba(230,178,60,0.15)] relative z-10">
                <img
                  src="/images/cards/Tutankhamun(1).jpg"
                  alt="Recognition Mockup"
                  className="w-full h-full object-cover opacity-60"
                  onError={(e) => { e.currentTarget.style.display = 'none'; }}
                />
                {/* Dynamic Scanning HUD Overlay */}
                <div className="absolute inset-0 z-20 pointer-events-none">
                  {/* The Scanning Line */}
                  <motion.div
                    animate={{ top: ["0%", "100%", "0%"] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#E6B23C] to-transparent shadow-[0_0_15px_#E6B23C] z-30"
                  />
                  
                  {/* Floating Recognition Points */}
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 1, 0] }}
                    transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                    className="absolute top-[30%] left-[40%] flex flex-col items-start gap-2"
                  >
                    <div className="w-3 h-3 rounded-full border border-[#E6B23C] bg-[#E6B23C]/20 shadow-[0_0_10px_#E6B23C]" />
                    <div className="px-2 py-1 bg-[#0D0A07]/80 backdrop-blur-md border border-[#E6B23C]/30 rounded text-[8px] font-bold text-[#E6B23C] uppercase tracking-tighter">Surface Mapping...</div>
                  </motion.div>

                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 1, 0] }}
                    transition={{ duration: 2, repeat: Infinity, delay: 1.5 }}
                    className="absolute top-[60%] left-[70%] flex flex-col items-start gap-2"
                  >
                    <div className="w-3 h-3 rounded-full border border-[#E6B23C] bg-[#E6B23C]/20 shadow-[0_0_10px_#E6B23C]" />
                    <div className="px-2 py-1 bg-[#0D0A07]/80 backdrop-blur-md border border-[#E6B23C]/30 rounded text-[8px] font-bold text-[#E6B23C] uppercase tracking-tighter">Pattern Detected</div>
                  </motion.div>

                  {/* Lens/Compass Effect */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-64 h-64 border border-[#E6B23C]/20 rounded-full animate-[spin_20s_linear_infinite]" />
                    <div className="absolute w-72 h-72 border border-[#E6B23C]/10 rounded-full animate-[spin_30s_linear_infinite_reverse]" />
                  </div>
                </div>
              </div>
              {/* Background Glow */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[#E6B23C]/10 blur-[120px] z-[-1]" />
            </div>
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase">
                <Camera size={14} /> Entity Recognition
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                Instant Recognition
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                Capture any artifact or landmark and let ECHO's advanced AI identify its historical significance in real-time. Uncover names, dates, and dynasties in an instant.
              </p>
            </div>
          </ScrollReveal>

          {/* 2. Video Feature */}
          <ScrollReveal direction="up" className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="lg:text-right space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase lg:flex-row-reverse">
                <PlayCircle size={14} /> Video Generation
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                Cinematic Histories
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                Transform static artifacts into immersive documentaries. Our engine pieces together historical records to generate narrated videos, bringing the stories of the ancients directly to your screen.
              </p>
            </div>
            <div className="relative">
              <div className="aspect-[9/16] max-w-[280px] mx-auto rounded-3xl overflow-hidden border border-[#E6B23C]/20 relative shadow-[0_0_50px_rgba(230,178,60,0.15)] bg-[#1A1208] z-10">
                {/* Mockup Screen Content */}
                <div className="absolute top-0 left-0 right-0 p-6 bg-gradient-to-b from-[#0D0A07]/90 to-transparent z-10">
                  <div className="text-[#F5E6D0] font-bold text-lg font-heading">The Sphinx</div>
                  <div className="text-[#E6B23C] text-xs tracking-widest uppercase">Documentary</div>
                </div>
                <img src="/images/landmarks/Sphinx.jpg" alt="Sphinx Video Mockup" className="w-full h-full object-cover opacity-70" />
                <div className="absolute inset-0 flex items-center justify-center z-20">
                  <div className="h-16 w-16 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/30 flex items-center justify-center backdrop-blur-md transition-transform hover:scale-110 cursor-pointer">
                    <PlayCircle size={32} className="text-[#E6B23C]" />
                  </div>
                </div>
              </div>
              {/* Background Glow */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[#E6B23C]/10 blur-[100px] z-[-1]" />
            </div>
          </ScrollReveal>

          {/* 3. Chat Feature */}
          <ScrollReveal direction="up" className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="relative">
              <div className="aspect-[4/5] max-w-[320px] mx-auto rounded-3xl overflow-hidden border border-[#E6B23C]/20 bg-[#0D0A07] shadow-[0_0_50px_rgba(230,178,60,0.15)] flex flex-col relative z-10">
                {/* Chat Header */}
                <div className="p-4 border-b border-[#E6B23C]/10 bg-[#1A1208] flex items-center gap-4">
                  <div className="h-10 w-10 rounded-full bg-gradient-to-br from-[#E6B23C]/30 to-transparent border border-[#E6B23C]/30 flex items-center justify-center">
                    <span className="text-xl">𓁹</span>
                  </div>
                  <div>
                    <div className="text-[#F5E6D0] font-bold text-sm">Ramesses II</div>
                    <div className="text-[#A08E70] text-[10px] tracking-wider uppercase">New Kingdom</div>
                  </div>
                </div>
                {/* Chat Body Mockup */}
                <div className="flex-1 p-5 space-y-4 bg-[#0D0A07] flex flex-col justify-end">
                  <div className="ml-auto w-[85%] p-3.5 rounded-2xl rounded-tr-sm bg-gradient-to-br from-[#E6B23C] to-[#C1840A] text-[#0D0A07] text-sm font-medium">
                    Tell me about your greatest victory.
                  </div>
                  <div className="w-[90%] p-3.5 rounded-2xl rounded-tl-sm bg-[#1A1208] border border-[#E6B23C]/20 text-[#F5E6D0] text-sm leading-relaxed">
                    The Battle of Kadesh was a triumph of the gods. Though the Hittites sought to ambush us, Amun gave me the strength of Montu.
                  </div>
                </div>
              </div>
              {/* Background Glow */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3/4 h-3/4 bg-[#E6B23C]/10 blur-[100px] z-[-1]" />
            </div>
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase">
                <MessageSquare size={14} /> Interactive Dialogue
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                Converse with Antiquity
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                Step back in time and speak directly with historical figures. Grounded in curated archaeological data, our interactive chat lets you explore the personal histories and reigns of pharaohs.
              </p>
            </div>
          </ScrollReveal>

          {/* 4. Translation Feature */}
          <ScrollReveal direction="up" className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="lg:text-right space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase lg:flex-row-reverse">
                <Bird size={14} /> Hieroglyphics Translation
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                Decipher the Past
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                Uncover the hidden meanings within ancient inscriptions. Our engine translates hieroglyphs directly into modern language, revealing the prayers, laws, and records of the ancients.
              </p>
            </div>
            <div className="relative">
              <div className="aspect-[4/3] max-w-[480px] mx-auto rounded-3xl overflow-hidden border border-[#E6B23C]/20 bg-[#0D0A07] shadow-[0_0_50px_rgba(230,178,60,0.15)] relative z-10 flex items-center justify-center">
                <img
                  src="/images/cards/Tutankhamun(1).jpg"
                  alt="Hieroglyphic Script"
                  className="absolute inset-0 w-full h-full object-cover opacity-40"
                  onError={(e) => { e.currentTarget.style.display = 'none'; }}
                />
                <div className="absolute inset-0 opacity-10 bg-[url('https://www.transparenttextures.com/patterns/papyros.png')]" />
                {/* Placeholder for Hieroglyph Image */}
                <div className="text-[#E6B23C]/20 text-6xl font-display uppercase tracking-[0.5em] select-none relative z-10">𓁹 𓅓 𓊵</div>
                {/* Translation Result Popover */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 p-5 bg-[#0D0A07]/90 backdrop-blur-md border border-[#E6B23C]/30 rounded-2xl shadow-2xl z-20">
                  <div className="text-[#E6B23C] text-[10px] font-bold uppercase tracking-widest mb-2">Translation Result</div>
                  <div className="text-[#F5E6D0] text-sm italic font-heading">"To live, prosper, and be in health..."</div>
                </div>
              </div>
              {/* Background Glow */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[#E6B23C]/10 blur-[120px] z-[-1]" />
            </div>
          </ScrollReveal>
        </div>
      </section>
    </PageShell>
  );
}