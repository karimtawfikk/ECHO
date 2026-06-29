"use client";

import PageShell from "../../components/layout/PageShell";
import ScrollReveal from "../../components/animations/ScrollReveal";
import { Camera, MessageSquare, Video, Bird } from "lucide-react";
import { useLanguage } from "../../context/LanguageContext";
import { motion } from "framer-motion";

export default function AboutPage() {
  const { t } = useLanguage();

  return (
    <PageShell>
      <section className="relative pt-32 pb-20 overflow-hidden flex flex-col items-center justify-center min-h-[50vh]">
        <div className="absolute inset-0 z-0 pointer-events-none flex items-center justify-center">
          <div className="w-[600px] h-[400px] rounded-full"
            style={{ background: "radial-gradient(circle, rgba(230,178,60,0.08) 0%, rgba(200,140,30,0.03) 40%, transparent 70%)" }}
          />
        </div>

        <div className="max-w-10xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="font-heading text-3xl md:text-4xl lg:text-5xl font-bold text-[#F5E6D0] mb-8 drop-shadow-[0_0_15px_rgba(230,178,60,0.3)]">
              About <span className="text-[#E6B23C]">E.C.H.O</span>
            </h1>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="space-y-8"
          >
            <p className="text-[#A08E70] text-lg md:text-2xl tracking-wide leading-loose max-w-6xl mx-auto">
              E.C.H.O stands for <strong className="text-[#E6B23C]">Every Capture Has Origins</strong>.
              It is an AI-powered archaeological portal that transforms tourist photos into immersive historical experiences.
            </p>
            <p className="text-[#A08E70] text-lg md:text-2xl tracking-wide leading-loose max-w-6xl mx-auto">
              Point your camera at a sphinx, a temple wall, or a pharaoh&apos;s statue —
              our engine will identify the artifact, generate a cinematic documentary,
              and let you speak directly with the historical figure it represents.
            </p>
          </motion.div>
        </div>
      </section>

      <motion.div
        initial={{ opacity: 0, scaleX: 0 }}
        animate={{ opacity: 1, scaleX: 1 }}
        transition={{ delay: 0.6, duration: 1 }}
        className="w-full flex justify-center py-8"
      >
        <div className="w-64 h-[1px]" style={{ background: "linear-gradient(90deg, transparent, rgba(230,178,60,0.5), transparent)" }} />
      </motion.div>

      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-32">
        <div className="text-center mb-16">
          <span className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase">{t("home.process.badge")}</span>
          <h2 className="font-heading text-3xl md:text-4xl lg:text-5xl font-bold mt-4 text-[#F5E6D0]">
            {t("home.process.title")}
          </h2>
        </div>
        <div className="relative max-w-4xl mx-auto py-12">
          <div className="absolute left-10 md:left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 bg-gradient-to-b from-transparent via-[#E6B23C]/30 to-transparent z-0" />

          <div className="space-y-24 relative z-10">
            {[
              { icon: Camera, title: t("home.process.step1.title"), text: t("home.process.step1.desc"), isText: false },
              { icon: Video, title: t("home.process.step2.title"), text: t("home.process.step2.desc"), isText: false },
              { icon: MessageSquare, title: t("home.process.step3.title"), text: t("home.process.step3.desc"), isText: false },
              { icon: Bird, title: t("home.process.step4.title"), text: t("home.process.step4.desc"), isText: false }
            ].map((step, i) => (
              <ScrollReveal key={step.title} direction={i % 2 === 0 ? "right" : "left"} delay={0.1} className="relative flex items-center">

                <div className="absolute left-10 md:left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-[#0D0A07] border border-[#E6B23C]/50 flex items-center justify-center shadow-[0_0_30px_rgba(230,178,60,0.15)] z-10">
                  {i === 3 ? (
                    <span className="text-4xl text-[#E6B23C] leading-none -translate-y-1">𓅓</span>
                  ) : step.isText ? (
                    <span className="text-4xl text-[#E6B23C] leading-none -translate-y-[10px]">{step.icon as any}</span>
                  ) : (
                    <step.icon size={28} className="text-[#E6B23C]" />
                  )}
                </div>

                <div className={`w-[calc(100%-6rem)] md:w-[42%] ml-auto md:ml-0 ${i % 2 === 0 ? "md:mr-auto" : "md:ml-auto"} text-left`}>
                  <div className="inline-flex items-center px-5 py-2 rounded-full bg-[#E6B23C]/[0.08] border border-[#E6B23C]/20 mb-4 shadow-[0_0_15px_rgba(230,178,60,0.05)]">
                    <h3 className="text-base md:text-lg font-bold tracking-[0.2em] text-[#E6B23C] uppercase">{step.title}</h3>
                  </div>
                  <p className="text-[#A08E70] leading-relaxed text-xl">{step.text}</p>
                </div>

              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>
    </PageShell>
  );
}
