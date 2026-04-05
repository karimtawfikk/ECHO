"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import RouteTransition from "../animations/RouteTransition";
import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

import Footer from "./Footer";

export default function PageShell({ children, fullScreen = false }: { children: ReactNode, fullScreen?: boolean }) {
  const pathname = usePathname();

  const navLinks = [
    { name: "Recognize", href: "/upload" },
    { name: "Video", href: "/video" },
    { name: "Chat", href: "/chat" },
    { name: "Translate", href: "/translate" },
  ];

  return (
    <main className="min-h-screen relative">
      {/* Rich Animated Background */}
      <div className="cinematic-bg">
        <div className="egyptian-pattern" />
        <div className="golden-atmosphere" />
        <div className="warm-vignette" />
      </div>
      <div className="film-grain" />

      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-[#E6B23C]/[0.06]"
        style={{
          background: "linear-gradient(180deg, rgba(13,10,7,0.92) 0%, rgba(13,10,7,0.75) 100%)",
          backdropFilter: "blur(20px)",
        }}
      >
        <div className="mx-auto max-w-7xl h-20 px-6 flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group relative">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-[#E6B23C] to-[#D4A030] flex items-center justify-center shadow-[0_0_25px_rgba(230,178,60,0.3)] group-hover:shadow-[0_0_35px_rgba(230,178,60,0.5)] transition-shadow">
              <Sparkles size={18} className="text-[#0D0A07]" />
            </div>
            <span
              className="text-xl font-bold tracking-[0.25em] text-[#E6B23C] gold-glow group-hover:text-[#FFD369] transition-colors"
              style={{ fontFamily: "var(--font-cinzel-dec), serif" }}
            >
              ECHO
            </span>
          </Link>

          {/* Nav Links */}
          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`text-xs font-semibold tracking-[0.15em] uppercase transition-all relative py-2 ${isActive
                    ? "text-[#E6B23C]"
                    : "text-[#A08E70] hover:text-[#F5E6D0]"
                    }`}
                >
                  {link.name}
                  {isActive && (
                    <motion.div
                      layoutId="nav-active"
                      className="absolute -bottom-0.5 left-0 right-0 h-[2px] rounded-full"
                      style={{
                        background: "linear-gradient(90deg, transparent, #E6B23C, transparent)",
                        boxShadow: "0 0 12px rgba(230,178,60,0.5)",
                      }}
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </Link>
              );
            })}
          </div>

          {/* Status */}
          <div className="flex items-center gap-2 bg-[#E6B23C]/[0.06] border border-[#E6B23C]/10 rounded-full px-4 py-2">
            <div className="h-2 w-2 rounded-full bg-[#E6B23C] animate-pulse shadow-[0_0_8px_rgba(230,178,60,0.6)]" />
            <span className="text-[10px] font-bold tracking-[0.2em] text-[#E6B23C] uppercase hidden sm:inline">Neural Active</span>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className={fullScreen ? "relative z-10 pt-20 h-screen w-full flex flex-col overflow-hidden" : "relative z-10 pt-32 pb-20 px-6 lg:px-12 max-w-7xl mx-auto"}>
        <RouteTransition fullScreen={fullScreen}>{children}</RouteTransition>
      </div>

      {!fullScreen && <Footer />}

      {/* SVG Filter for Papyrus */}
      <svg className="hidden" aria-hidden="true">
        <filter id="rough-edge">
          <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="5" seed="5" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="18" />
        </filter>
      </svg>
    </main>
  );
}
