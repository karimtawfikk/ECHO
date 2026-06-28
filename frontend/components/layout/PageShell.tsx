"use client";

import { ReactNode, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import RouteTransition from "../animations/RouteTransition";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, User, Globe, ChevronDown, Menu, X } from "lucide-react";
import { useLanguage } from "../../context/LanguageContext";
import type { Language } from "../../lib/i18n/dictionaries";
import { useEffect } from "react";
import { createClient } from "../../lib/supabase/client";
import { LogOut, Settings } from "lucide-react";

import Footer from "./Footer";
import ProfileSidebar from "../profile/ProfileSidebar";

export default function PageShell({
  children,
  fullScreen = false,
  fullWidth = false,
  headerExtension,
  minimal = false
}: {
  children: ReactNode,
  fullScreen?: boolean,
  fullWidth?: boolean,
  headerExtension?: ReactNode,
  minimal?: boolean
}) {
  const pathname = usePathname();
  const { language, setLanguage, t, isRTL } = useLanguage();
  const [menuOpen, setMenuOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const [userOpen, setUserOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [user, setUser] = useState<any>(null);
  const supabase = createClient();

  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
    };
    getUser();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, [supabase]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setUserOpen(false);
  };

  const languages: { code: Language; name: string }[] = [
    { code: "EN", name: "English" },
    { code: "AR", name: "Arabic" },
    { code: "FR", name: "Français" },
  ];

  const navLinks = [
    { name: t("nav.home"), href: "/" },
    { name: t("nav.explore"), href: "/explore" },
    { name: t("nav.recognize"), href: "/upload" },
    { name: t("nav.translate"), href: "/translate" },
  ];

  return (
    <main className={fullScreen ? "fixed inset-0 md:relative md:min-h-screen overflow-hidden md:overflow-visible" : "min-h-screen relative"}>
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
        <div className="w-full px-4 md:px-8 py-3 md:py-0 md:h-20 flex flex-col md:grid md:grid-cols-3 items-center relative gap-3 md:gap-0">

          {/* Mobile Top Row: Logo & Controls */}
          <div className="w-full flex justify-between items-center md:contents relative">

            {/* Mobile Hamburger Menu & Inline Links */}
            <div className="md:hidden flex items-center flex-1 justify-start">
              {!minimal && (
                <button onClick={() => setMenuOpen(!menuOpen)} className="p-2 -ml-2 text-[#E6B23C] hover:bg-[#E6B23C]/10 rounded-full transition-colors z-50">
                  {menuOpen ? <X size={20} /> : <Menu size={20} />}
                </button>
              )}

              <AnimatePresence>
                {menuOpen && !minimal && (
                  <motion.div
                    initial={{ opacity: 0, x: -10, width: 0 }}
                    animate={{ opacity: 1, x: 0, width: "auto" }}
                    exit={{ opacity: 0, x: -10, width: 0 }}
                    className="flex items-center gap-2.5 overflow-hidden whitespace-nowrap ml-1"
                  >
                    {navLinks.filter(l => l.href !== "/").map((link) => {
                      const isActive = pathname === link.href || (link.href === "/upload" && pathname.startsWith("/result"));
                      return (
                        <Link
                          key={link.name}
                          href={link.href}
                          onClick={() => setMenuOpen(false)}
                          className={`text-[8px] font-bold tracking-widest uppercase transition-colors ${isActive ? "text-[#E6B23C]" : "text-[#A08E70] hover:text-[#F5E6D0]"
                            }`}
                        >
                          {link.name}
                        </Link>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Center Column: Logo */}
            <div className={`flex justify-center md:col-start-2 md:row-start-1 shrink-0 transition-opacity duration-300 ${menuOpen ? 'opacity-0 pointer-events-none md:opacity-100 md:pointer-events-auto absolute left-1/2 -translate-x-1/2 md:static md:translate-x-0' : 'absolute left-1/2 -translate-x-1/2 md:static md:translate-x-0'}`}>
              <Link href="/" className="group">
                <span
                  className="text-3xl font-bold tracking-[0.35em] text-[#E6B23C] gold-glow group-hover:text-[#FFD369] transition-colors"
                  style={{ fontFamily: 'var(--font-cormorant), serif' }}
                >
                  ECHO
                </span>
              </Link>
            </div>

            {/* Right Column: Language & User */}
            <div className="flex justify-end items-center gap-2 md:gap-4 md:col-start-3 md:row-start-1 flex-1">
              {!minimal && (
                <>
                  <AnimatePresence>
                    {menuOpen && (
                      <motion.div
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        className="md:hidden mr-1"
                      >
                        <Link href="/" className="group" onClick={() => setMenuOpen(false)}>
                          <span
                            className="text-[16px] font-bold tracking-[0.35em] text-[#E6B23C] gold-glow transition-colors"
                            style={{ fontFamily: 'var(--font-cormorant), serif' }}
                          >
                            ECHO
                          </span>
                        </Link>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Language Switcher */}
                  <div className="relative hidden md:block">
                    <button
                      onClick={() => setLangOpen(!langOpen)}
                      className="h-10 px-3 flex items-center gap-2 rounded-full bg-[#0D0A07] md:bg-[#E6B23C]/[0.04] border border-[#E6B23C]/20 md:border-[#E6B23C]/10 text-[#E6B23C] hover:bg-[#E6B23C]/10 transition-all group"
                    >
                      <Globe size={16} className="group-hover:rotate-12 transition-transform" />
                      <span className="text-[10px] font-bold tracking-widest">{language}</span>
                      <ChevronDown size={12} className={`transition-transform duration-300 ${langOpen ? 'rotate-180' : ''}`} />
                    </button>

                    <AnimatePresence>
                      {langOpen && (
                        <>
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setLangOpen(false)}
                            className="fixed inset-0 z-[-1]"
                          />
                          <motion.div
                            initial={{ opacity: 0, y: 10, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 10, scale: 0.95 }}
                            className={`absolute top-full ${isRTL ? 'left-0' : 'right-0'} mt-4 w-40 py-2 bg-[#0D0A07]/95 backdrop-blur-2xl border border-[#E6B23C]/20 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden`}
                          >
                            <div className="px-4 py-2 mb-1 border-b border-[#E6B23C]/10">
                              <span className="text-[9px] font-bold tracking-[0.2em] text-[#E6B23C]/50 uppercase">{t("nav.language")}</span>
                            </div>
                            {languages.map((lang) => (
                              <button
                                key={lang.code}
                                onClick={() => {
                                  setLanguage(lang.code);
                                  setLangOpen(false);
                                }}
                                className={`w-full flex items-center justify-between px-4 py-2.5 text-[10px] font-bold tracking-widest uppercase transition-all hover:bg-[#E6B23C]/5 ${language === lang.code ? "text-[#E6B23C]" : "text-[#A08E70]"}`}
                              >
                                {lang.name}
                                {language === lang.code && <div className="h-1 w-1 rounded-full bg-[#E6B23C] shadow-[0_0_5px_#E6B23C]" />}
                              </button>
                            ))}
                          </motion.div>
                        </>
                      )}
                    </AnimatePresence>
                  </div>

                  {/* User Profile */}
                  <div className="relative">
                    {!user ? (
                      <Link
                        href="/login"
                        className="h-10 w-10 flex items-center justify-center rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] hover:bg-[#E6B23C]/20 transition-all shadow-[0_0_15px_rgba(230,178,60,0.1)] group"
                      >
                        <User size={18} className="transition-transform group-hover:scale-110" />
                      </Link>
                    ) : (
                      <>
                        <button
                          onClick={() => setProfileOpen(true)}
                          className="h-10 w-10 flex items-center justify-center rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] hover:bg-[#E6B23C]/20 transition-all shadow-[0_0_15px_rgba(230,178,60,0.1)] group overflow-hidden"
                        >
                          {user.user_metadata?.avatar_url ? (
                            <img src={user.user_metadata.avatar_url} alt="Profile" className="w-full h-full object-cover" />
                          ) : (
                            <User size={18} className="transition-transform group-hover:scale-110" />
                          )}
                        </button>
                      </>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Left Column: Desktop Navigation */}
          <div className="hidden md:flex justify-start items-center gap-8 w-full md:w-auto md:col-start-1 md:row-start-1">
            {!minimal && navLinks.map((link) => {
              const isActive = pathname === link.href || (link.href === "/upload" && pathname.startsWith("/result"));
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`text-[11px] font-bold tracking-[0.2em] uppercase transition-all relative group py-2 ${isActive ? "text-[#E6B23C]" : "text-[#A08E70] hover:text-[#F5E6D0]"
                    }`}
                >
                  {link.name}
                  {/* Glowing Tapered Underline */}
                  <div className="absolute -bottom-1 left-0 right-0 flex justify-center pointer-events-none">
                    <motion.div
                      initial={false}
                      animate={{
                        width: isActive ? "100%" : "0%",
                        opacity: isActive ? 1 : 0
                      }}
                      className="h-[1.5px] bg-gradient-to-r from-transparent via-[#E6B23C] to-transparent shadow-[0_0_12px_rgba(230,178,60,0.6)]"
                    />
                  </div>

                  {/* Hover State: Subtle Glow Reveal */}
                  {!isActive && (
                    <div className="absolute -bottom-1 left-0 right-0 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-[#F5E6D0]/50 to-transparent" />
                    </div>
                  )}
                </Link>
              );
            })}
          </div>

          {/* Removed Mobile Dropdown Menu, now handled inline above */}
        </div>
      </nav>

      {headerExtension && (
        <div className="fixed top-0 left-0 right-0 z-[45] pointer-events-none">
          {headerExtension}
        </div>
      )}

      {/* Content */}
      <div className={fullScreen ? "relative z-10 pt-14 md:pt-20 h-[100dvh] w-full flex flex-col overflow-hidden md:overflow-y-auto md:overflow-x-hidden" : (fullWidth ? "relative z-10 w-full" : "relative z-10 pt-24 md:pt-32 pb-12 md:pb-20 px-6 lg:px-12 max-w-7xl mx-auto")}>
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

      {/* Profile Sidebar */}
      <ProfileSidebar isOpen={profileOpen} onClose={() => setProfileOpen(false)} />
    </main>
  );
}
