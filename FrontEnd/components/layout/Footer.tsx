"use client";

import Link from "next/link";
import { Github, Sparkles } from "lucide-react";

export default function Footer() {
    return (
        <footer className="mt-40 border-t border-[#E6B23C]/10 bg-[#0D0A07] relative overflow-hidden">
            {/* Subtle glow background */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] opacity-[0.03] pointer-events-none"
                style={{ background: "radial-gradient(circle, #E6B23C 0%, transparent 70%)" }}
            />

            <div className="max-w-7xl mx-auto px-6 py-16 lg:py-24">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 lg:gap-8">

                    {/* Brand Column */}
                    <div className="lg:col-span-2 space-y-6">
                        <Link href="/" className="flex items-center gap-3 group">
                            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-[#E6B23C] to-[#D4A030] flex items-center justify-center shadow-[0_0_20px_rgba(230,178,60,0.2)]">
                                <span className="text-[#0D0A07] text-2xl leading-none">☥</span>
                            </div>
                            <span
                                className="text-2xl font-bold tracking-[0.25em] text-[#E6B23C] gold-glow"
                                style={{ fontFamily: 'var(--font-cormorant), serif' }}
                            >
                                ECHO
                            </span>
                        </Link>
                        <p className="max-w-md text-[#A08E70] leading-relaxed text-sm lg:text-base">
                            Every Capture Has Origins — an AI-powered interactive
                            exploration of Ancient Egypt through captured moments, stories, and dialogue.
                        </p>
                    </div>

                    {/* Explore Column */}
                    <div className="space-y-6">
                        <h3 className="text-[#F5E6D0] font-bold tracking-[0.1em] uppercase text-sm">Explore</h3>
                        <ul className="space-y-4">
                            <li>
                                <Link href="/upload" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    Recognition
                                </Link>
                            </li>
                            <li>
                                <Link href="/video" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    Video Generation
                                </Link>
                            </li>
                            <li>
                                <Link href="/chat" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    Chatbot
                                </Link>
                            </li>
                            <li>
                                <Link href="/translate" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    Hieroglyphs
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Project Column */}
                    <div className="space-y-6">
                        <h3 className="text-[#F5E6D0] font-bold tracking-[0.1em] uppercase text-sm">Project</h3>
                        <ul className="space-y-4">
                            <li>
                                <Link href="#" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    About
                                </Link>
                            </li>
                            <li>
                                <Link href="#" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    Team
                                </Link>
                            </li>
                            <li>
                                <Link href="#" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    Contact
                                </Link>
                            </li>
                            <li>
                                <Link href="#" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors flex items-center gap-2 text-sm font-medium">
                                    <Github size={14} /> GitHub
                                </Link>
                            </li>
                        </ul>
                    </div>

                </div>

                {/* Bottom Bar */}
                <div className="mt-16 pt-8 border-t border-[#E6B23C]/5 flex flex-col sm:flex-row justify-between items-center gap-4">
                    <p className="text-[#A08E70]/50 text-[11px] font-bold tracking-[0.1em] uppercase">
                        © 2026 ECHO Project. Built with AI.
                    </p>
                    <div className="flex gap-6">
                        <Link href="#" className="text-[#A08E70]/40 hover:text-[#E6B23C] text-[10px] font-bold tracking-[0.2em] uppercase transition-colors">
                            Privacy Policy
                        </Link>
                        <Link href="#" className="text-[#A08E70]/40 hover:text-[#E6B23C] text-[10px] font-bold tracking-[0.2em] uppercase transition-colors">
                            Terms of Use
                        </Link>
                    </div>
                </div>
            </div>
        </footer>
    );
}
