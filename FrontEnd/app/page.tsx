"use client";

import Link from "next/link";
import PageShell from "../components/layout/PageShell";
import { Button } from "../components/ui/button";
import { motion } from "framer-motion";
import { ArrowRight, DoorOpen, Sparkles } from "lucide-react";
import TrendingRow from "../components/trending/TrendingRow";
import ScrollReveal from "../components/animations/ScrollReveal";
import ParallaxLayer from "../components/animations/ParallaxLayer";
import { useEffect, useState } from "react";
import { fetchTrendingEntities } from "../lib/services/entities";
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
  const [pharaohs, setPharaohs] = useState<RecognitionEntity[]>([]);
  const [landmarks, setLandmarks] = useState<RecognitionEntity[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchTrendingEntities()
      .then((data) => {
        setPharaohs(data.pharaohs.length > 0 ? data.pharaohs : FALLBACK_PHARAOHS);
        setLandmarks(data.landmarks.length > 0 ? data.landmarks : FALLBACK_LANDMARKS);
      })
      .catch(() => {
        // API unreachable – fall back to mock data silently
        setPharaohs(FALLBACK_PHARAOHS);
        setLandmarks(FALLBACK_LANDMARKS);
      })
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <PageShell>
      {/* =========== HERO =========== */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="min-h-[75vh] flex flex-col justify-center items-center text-center relative"
      >
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
          className="flex flex-col sm:flex-row gap-5"
        >
          <Button
            asChild
            className="h-14 px-10 rounded-2xl bg-gradient-to-r from-[#E6B23C] to-[#D4A030] hover:from-[#FFD369] hover:to-[#E6B23C] text-[#0D0A07] font-bold text-base transition-all hover:scale-105 shadow-[0_4px_30px_rgba(230,178,60,0.25)] hover:shadow-[0_4px_40px_rgba(230,178,60,0.4)]"
          >
            <Link href="/upload">
              Recognize Entities
              <ArrowRight className="ml-3" size={18} />
            </Link>
          </Button>

          <Button
            asChild
            variant="outline"
            className="h-14 px-10 rounded-2xl border-[#E6B23C]/15 bg-[#E6B23C]/[0.04] hover:bg-[#E6B23C]/[0.08] text-[#F5E6D0] font-semibold text-base transition-all hover:scale-105 hover:border-[#E6B23C]/25"
          >
            <Link href="/translate">
              Translate Hieroglyphs
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

      {/* =========== TRENDING: PHARAOHS =========== */}
      <ScrollReveal direction="up" delay={0.1} className="mt-24">
        <TrendingRow
          title="Icons of Ancient Egypt"
          items={pharaohs}
          type="pharaoh"
          isLoading={isLoading}
        />
      </ScrollReveal>

      {/* =========== TRENDING: LANDMARKS =========== */}
      <ScrollReveal direction="up" delay={0.1} className="mt-16">
        <TrendingRow
          title="Must-See Ancient Sites"
          items={landmarks}
          type="landmark"
          isLoading={isLoading}
        />
      </ScrollReveal>

      {/* =========== ABOUT =========== */}
      <ScrollReveal
        direction="up"
        margin="-80px"
        distance={50}
        duration={0.7}
        className="mt-40 grid lg:grid-cols-2 gap-16 items-start"
      >
        <div>
          <div className="flex items-center gap-3 mb-4">
            <div className="h-[1px] w-8 bg-[#E6B23C]/40" />
            <span className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase">About</span>
          </div>
          <h2 className="font-heading text-xl md:text-2xl lg:text-3xl font-bold leading-[1.2] tracking-[0.03em] uppercase text-[#F5E6D0]">
            What is E.C.H.O?
          </h2>

          <div
            className="mt-2 mb-4 h-[1px] w-24"
          />
          <div className="space-y-6 text-[#A08E70] leading-relaxed text-lg md:text-xl" style={{ fontFamily: 'var(--font-cormorant), serif' }}>
            <p>
              E.C.H.O stands for <strong className="text-[#F5E6D0]">Every Capture Has Origins</strong>.
              It is an AI-powered archaeological portal that transforms tourist photos into
              immersive historical experiences.
            </p>
            <p>
              Point your camera at a sphinx, a temple wall, or a pharaoh&apos;s statue —
              our engine will identify the artifact, generate a cinematic documentary,
              and let you speak directly with the historical figure it represents.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          {[
            { label: "Recognition", text: "Recognizes landmarks and artifacts from images, using models trained on curated archaeological data." },
            { label: "Video", text: "Generates documentary-style videos with historically accurate narration." },
            { label: "Dialogue", text: "Enables interactive conversations grounded in historical context, allowing you to explore each entity in depth." },
            { label: "Translation", text: "Translates ancient Egyptian hieroglyphic inscriptions into modern language using OCR and language models." },
          ].map((item, i) => (
            <ScrollReveal
              key={item.label}
              direction="left"
              distance={20}
              delay={i * 0.1}
              className="glass-surface rounded-2xl p-6 group hover:border-[#E6B23C]/15 transition-all"
            >
              <div className="text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-2">{item.label}</div>
              <p className="text-sm text-[#A08E70] leading-relaxed">{item.text}</p>
            </ScrollReveal>
          ))}
        </div>
      </ScrollReveal>
    </PageShell>
  );
}