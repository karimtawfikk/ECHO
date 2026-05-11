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
import { useLanguage } from "../context/LanguageContext";
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
  const { t } = useLanguage();
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
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.7 }}
          className="mb-8 max-w-5xl"
        >
          <h1
            className="font-display text-7xl md:text-9xl lg:text-[9rem] font-bold tracking-[0.15em] uppercase text-[#E6B23C] gold-glow mb-4"
            style={{ fontFamily: 'var(--font-cormorant), serif' }}
          >
            {t("home.hero.title")}
          </h1>
          <p
            className="font-display text-2xl md:text-4xl lg:text-5xl font-bold leading-[1.2] tracking-[0.03em] uppercase text-[#F5E6D0]"
            style={{ fontFamily: 'var(--font-cormorant), serif' }}
          >
            {t("home.hero.subtitle").split(" ").map((word, i, arr) => {
              const isOrigins = word.toLowerCase().includes("origins") || word.includes("أصول");
              return (
                <span key={i} className={isOrigins ? "text-[#E6B23C]" : ""}>
                  {word}{i < arr.length - 1 ? " " : ""}
                </span>
              );
            })}
          </p>
        </motion.div>

        <p
          className="text-[#A08E70] text-lg md:text-xl max-w-xl font-semibold leading-relaxed mb-12"
          style={{ fontFamily: 'var(--font-cormorant), serif' }}
        >
          {t("home.hero.description")}
        </p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.65, duration: 0.5 }}
          className="flex flex-col sm:flex-row flex-wrap justify-center gap-5"
        >
          <Button
            asChild
            className="h-14 w-64 rounded-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 text-[#E6B23C] hover:bg-[#E6B23C]/10 font-bold text-sm uppercase tracking-widest transition-all hover:scale-105 shadow-[0_10px_30px_rgba(230,178,60,0.1)]"
          >
            <Link href="/upload">
              {t("home.hero.cta.start")}
            </Link>
          </Button>

          <Button
            asChild
            className="h-14 w-64 rounded-full bg-transparent border border-[#E6B23C]/20 text-[#F5E6D0] hover:bg-[#E6B23C]/5 font-bold text-sm uppercase tracking-widest transition-all hover:scale-105"
          >
            <Link href="#how-it-works">
              {t("home.hero.cta.how_it_works")}
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
          <span className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase">{t("home.process.badge")}</span>
          <h2 className="font-heading text-3xl md:text-4xl lg:text-5xl font-bold mt-4 text-[#F5E6D0]">
            {t("home.process.title")}
          </h2>
        </div>
        <div className="relative max-w-4xl mx-auto py-12">
          {/* Vertical Connecting Line */}
          <div className="absolute left-10 md:left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 bg-gradient-to-b from-transparent via-[#E6B23C]/30 to-transparent z-0" />

          <div className="space-y-24 relative z-10">
            {[
              { icon: Camera, title: t("home.process.step1.title"), text: t("home.process.step1.desc"), isText: false },
              { icon: Video, title: t("home.process.step2.title"), text: t("home.process.step2.desc"), isText: false },
              { icon: MessageSquare, title: t("home.process.step3.title"), text: t("home.process.step3.desc"), isText: false },
              { icon: Bird, title: t("home.process.step4.title"), text: t("home.process.step4.desc"), isText: false }
            ].map((step, i) => (
              <ScrollReveal key={step.title} direction={i % 2 === 0 ? "right" : "left"} delay={0.1} className="relative flex items-center">

                {/* Timeline Center Node */}
                <div className="absolute left-10 md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-[#0D0A07] border border-[#E6B23C]/50 flex items-center justify-center shadow-[0_0_30px_rgba(230,178,60,0.15)] z-10">
                  {i === 3 ? (
                    <svg width="34" height="34" viewBox="0 0 50 50" fill="none" stroke="#E6B23C" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" className="opacity-90">
                      {/* Accurate Minimalist Owl Hieroglyph (G17) */}
                      <path d="M25 8c-3.5 0-6 2.5-6 6v4c0 3.5 2.5 6 6 6s6-2.5 6-6v-4c0-3.5-2.5-6-6-6z" />
                      <path d="M19 12l-2-2M31 12l2-2" />
                      <path d="M22 14h.01M28 14h.01" strokeWidth="2.5" />
                      <path d="M25 18l-1 2h2l-1-2z" fill="#E6B23C" />
                      <path d="M19 18c-6 0-11 4-11 14 0 10 5 18 11 18s8-4 13-4 13 4 13 4c0-14-5-28-13-28s-7 6-13 6z" />
                      <path d="M21 48v2M29 48v2" />
                    </svg>
                  ) : step.isText ? (
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
          <h2 className="font-heading text-3xl md:text-4xl font-bold text-[#F5E6D0]">{t("home.experience.title")}</h2>
          <p className="text-[#A08E70] mt-3 max-w-xl mx-auto text-lg">{t("home.experience.subtitle")}</p>
        </div>
        <TrendingRow
          title={t("home.experience.pharaohs")}
          items={pharaohs}
          type="pharaoh"
          isLoading={isLoading}
        />
      </section>

      {/* =========== TRENDING: LANDMARKS =========== */}
      <section className="mt-16 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <TrendingRow
          title={t("home.experience.landmarks")}
          items={landmarks}
          type="landmark"
          isLoading={isLoading}
        />
      </section>

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
                <Camera size={14} /> {t("home.feature1.badge")}
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                {t("home.feature1.title")}
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                {t("home.feature1.desc")}
              </p>
            </div>
          </ScrollReveal>

          {/* 2. Video Feature */}
          <ScrollReveal direction="up" className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="lg:text-right space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase lg:flex-row-reverse">
                <PlayCircle size={14} /> {t("home.feature2.badge")}
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                {t("home.feature2.title")}
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                {t("home.feature2.desc")}
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
                {/* Chat Atmosphere Background */}
                <div className="absolute inset-0 opacity-40 pointer-events-none">
                  <div className="absolute inset-0 bg-[url('/images/patterns/egyptian-pattern.png')] opacity-20" />
                  <div className="absolute inset-0 bg-gradient-to-b from-[#E6B23C]/5 via-transparent to-transparent" />
                </div>

                {/* Chat Header (Simplified Identity Bar) */}
                <div className="pt-8 pb-4 border-b border-[#E6B23C]/10 bg-[#0D0A07]/50 backdrop-blur-md flex flex-col items-center gap-2 relative z-10">
                  <div className="h-12 w-12 rounded-full bg-gradient-to-br from-[#E6B23C] to-[#D4A030] p-[1.5px]">
                    <div className="h-full w-full rounded-full bg-[#0D0A07] overflow-hidden flex items-center justify-center">
                      <img src="/images/pharaohs/Ramesses II.jpg" alt="Ramesses II" className="w-full h-full object-cover scale-110" onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-[#F5E6D0] font-bold text-xs tracking-wide">Ramesses II</div>
                    <div className="text-[#E6B23C] text-[8px] font-bold tracking-[0.2em] uppercase opacity-60">New Kingdom</div>
                  </div>
                </div>

                {/* Chat Body Mockup */}
                <div className="flex-1 p-5 space-y-6 flex flex-col justify-start relative z-10">
                  {/* User Message Bubble */}
                  <div className="ml-auto px-4 py-2 rounded-[18px] bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-[11px] shadow-[0_4px_15px_rgba(230,178,60,0.05)]">
                    Tell me about your greatest victory.
                  </div>
                  {/* Assistant Message (Clean Text) */}
                  <div className="w-full text-[#F5E6D0] text-[12px] leading-relaxed font-light tracking-wide">
                    The Battle of Kadesh was a triumph of the gods. Though the Hittites sought to ambush us, Amun gave me the strength of Montu.
                  </div>

                  {/* Input Bar Mockup - Updated with Voice Icon */}
                  <div className="mt-auto flex gap-2 items-center">
                    <div className="flex-1 px-4 py-2.5 rounded-full bg-[#1A1208]/80 border border-[#E6B23C]/10 flex items-center shadow-inner">
                      <span className="text-[#A08E70]/40 text-[10px] font-medium">Ask Ramesses II...</span>
                    </div>
                    <div className="h-8 w-8 rounded-full bg-[#0D0A07] border border-[#E6B23C]/30 flex items-center justify-center shadow-[0_0_10px_rgba(230,178,60,0.15)]">
                      <svg width="14" height="12" viewBox="0 0 24 20" fill="none" stroke="#E6B23C" strokeWidth="2.5" strokeLinecap="round">
                        <path d="M2 9v2" /><path d="M6 5v10" /><path d="M10 2v16" /><path d="M14 5v10" /><path d="M18 9v2" /><path d="M22 7v6" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
              {/* Background Glow */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3/4 h-3/4 bg-[#E6B23C]/10 blur-[100px] z-[-1]" />
            </div>
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase">
                <MessageSquare size={14} /> {t("home.feature3.badge")}
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                {t("home.feature3.title")}
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                {t("home.feature3.desc")}
              </p>
            </div>
          </ScrollReveal>

          {/* 4. Translation Feature */}
          <ScrollReveal direction="up" className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="lg:text-right space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold tracking-widest uppercase lg:flex-row-reverse">
                <svg width="18" height="18" viewBox="0 0 50 50" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M25 8c-3.5 0-6 2.5-6 6v4c0 3.5 2.5 6 6 6s6-2.5 6-6v-4c0-3.5-2.5-6-6-6z" />
                  <path d="M19 12l-2-2M31 12l2-2" />
                  <path d="M22 14h.01M28 14h.01" strokeWidth="2.5" />
                  <path d="M25 18l-1 2h2l-1-2z" fill="currentColor" />
                  <path d="M19 18c-6 0-11 4-11 14 0 10 5 18 11 18s8-4 13-4 13 4 13 4c0-14-5-28-13-28s-7 6-13 6z" />
                  <path d="M21 48v2M29 48v2" />
                </svg> {t("home.feature4.badge")}
              </div>
              <h2 className="font-heading text-4xl md:text-5xl font-bold text-[#F5E6D0]">
                {t("home.feature4.title")}
              </h2>
              <p className="text-[#A08E70] text-lg leading-relaxed">
                {t("home.feature4.desc")}
              </p>
            </div>
            <div className="relative">
              <div className="aspect-[4/3] max-w-[480px] mx-auto rounded-3xl overflow-hidden border border-[#E6B23C]/20 bg-[#0D0A07] shadow-[0_0_50px_rgba(230,178,60,0.15)] relative z-10 flex items-center justify-center">
                <img
                  src="/images/cards/hieroglyphs.jpg"
                  alt="Hieroglyphic Script"
                  className="absolute inset-0 w-full h-full object-cover opacity-40"
                  onError={(e) => { e.currentTarget.src = "/images/cards/Tutankhamun(1).jpg"; }}
                />
                <div className="absolute inset-0 opacity-10 bg-[url('https://www.transparenttextures.com/patterns/papyros.png')]" />
                {/* Placeholder for Hieroglyph Image */}
                <div className="text-[#E6B23C]/20 text-6xl font-display uppercase tracking-[0.5em] select-none relative z-10">𓁹 𓅓 𓊵</div>
                {/* Mystical Reveal HUD - Elegant & Magical */}
                <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden">
                  {/* The 'Mystical Lens' Reveal Effect */}
                  <motion.div
                    animate={{
                      x: ["-20%", "60%", "10%"],
                      y: ["-10%", "30%", "0%"]
                    }}
                    transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
                    className="absolute w-80 h-80 rounded-full z-10"
                    style={{
                      background: "radial-gradient(circle, rgba(230,178,60,0.15) 0%, transparent 70%)",
                      boxShadow: "0 0 100px rgba(230,178,60,0.1) inset"
                    }}
                  >
                    {/* Inner Lens Glow */}
                    <div className="absolute inset-0 rounded-full border border-[#E6B23C]/20 shadow-[0_0_30px_rgba(230,178,60,0.1)]" />
                  </motion.div>

                  {/* Ambient Particles */}
                  <div className="absolute inset-0 opacity-40">
                    {[...Array(6)].map((_, i) => (
                      <motion.div
                        key={i}
                        animate={{
                          y: [0, -100],
                          opacity: [0, 1, 0],
                          x: [0, (i % 2 === 0 ? 50 : -50)]
                        }}
                        transition={{ duration: 5 + i, repeat: Infinity, delay: i * 0.5 }}
                        className="absolute w-1 h-1 bg-[#E6B23C] rounded-full blur-[1px]"
                        style={{ bottom: "10%", left: `${20 + i * 15}%` }}
                      />
                    ))}
                  </div>

                  {/* Final Deciphered Result - Floating Spirit-like Text */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.div
                      initial={{ opacity: 0 }}
                      whileInView={{ opacity: 1 }}
                      transition={{ duration: 1.5 }}
                      className="text-center px-12 relative"
                    >
                      {/* Decorative Ancient Accents */}
                      <div className="absolute -top-8 left-1/2 -translate-x-1/2 flex items-center gap-4 opacity-40">
                        <div className="h-[1px] w-12 bg-gradient-to-r from-transparent to-[#E6B23C]" />
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#E6B23C" strokeWidth="1.5">
                          <path d="M12 2L2 12l10 10 10-10L12 2z" />
                          <path d="M12 6l-6 6 6 6 6-6-6-6z" />
                        </svg>
                        <div className="h-[1px] w-12 bg-gradient-to-l from-transparent to-[#E6B23C]" />
                      </div>

                      <div className="text-[#E6B23C] text-[11px] font-bold uppercase tracking-[0.6em] mb-4 opacity-60">Ancient Script Decoded</div>
                      <h3 className="text-[#F5E6D0] text-2xl md:text-4xl italic font-heading gold-glow leading-tight drop-shadow-[0_15px_30px_rgba(0,0,0,0.9)]">
                        "To live, prosper, and be in health..."
                      </h3>
                    </motion.div>
                  </div>
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