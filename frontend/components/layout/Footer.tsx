"use client";

import Link from "next/link";
import { Github, Sparkles, Mail, MapPin } from "lucide-react";
import { useLanguage } from "../../context/LanguageContext";

export default function Footer() {
    const { t } = useLanguage();

    return (
        <footer className="mt-40 border-t border-[#E6B23C]/10 bg-[#0D0A07] relative overflow-hidden">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[800px] md:w-[800px] md:h-[300px] opacity-[0.03] pointer-events-none"
                style={{ background: "radial-gradient(circle, #E6B23C 0%, transparent 70%)" }}
            />

            <div className="max-w-7xl mx-auto px-6 py-16 lg:py-24">
                <div className="grid grid-cols-1 md:grid-cols-6 gap-12 lg:gap-8">

                    <div className="md:col-span-2 space-y-6">
                        <span
                            className="text-2xl font-bold tracking-[0.25em] text-[#E6B23C] gold-glow select-none"
                            style={{ fontFamily: 'var(--font-cormorant), serif' }}
                        >
                            ECHO
                        </span>
                        <p className="max-w-xs text-[#A08E70] leading-relaxed text-[13px]">
                            {t("footer.desc")}
                        </p>
                    </div>

                    <div className="md:col-span-2 space-y-6 md:ps-20 lg:ps-32">
                        <div className="relative inline-block pb-2">
                            <h3 className="text-[#F5E6D0] font-bold tracking-[0.1em] uppercase text-sm">Navigation</h3>
                            <div className="absolute bottom-0 left-0 w-1/2 h-[2px] bg-[#E6B23C]" />
                        </div>
                        <ul className="grid grid-cols-2 gap-x-4 gap-y-4 max-w-[200px]">
                            <li>
                                <Link href="/explore" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    {t("nav.explore")}
                                </Link>
                            </li>
                            <li>
                                <Link href="/upload" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    {t("nav.recognize")}
                                </Link>
                            </li>
                            <li>
                                <Link href="/translate" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    {t("nav.translate")}
                                </Link>
                            </li>
                            <li>
                                <Link href="/about" className="text-[#A08E70] hover:text-[#E6B23C] transition-colors text-sm font-medium">
                                    {t("footer.about")}
                                </Link>
                            </li>
                        </ul>
                    </div>

                    <div className="md:col-span-2 space-y-6">
                        <div className="relative inline-block pb-2">
                            <h3 className="text-[#F5E6D0] font-bold tracking-[0.1em] uppercase text-sm">Contact</h3>
                            <div className="absolute bottom-0 left-0 w-1/2 h-[2px] bg-[#E6B23C]" />
                        </div>
                        <ul className="space-y-4">
                            <li className="flex items-center gap-3 text-[#A08E70] text-sm font-medium group cursor-pointer hover:text-[#E6B23C] transition-colors">
                                <Mail size={16} className="text-[#E6B23C]" />
                                <span className="whitespace-nowrap">echo.archive.ai@gmail.com</span>
                            </li>
                            <li className="flex items-center gap-3 text-[#A08E70] text-sm font-medium group cursor-pointer hover:text-[#E6B23C] transition-colors">
                                <MapPin size={16} className="text-[#E6B23C]" />
                                <span>Cairo, Egypt</span>
                            </li>
                        </ul>
                    </div>

                </div>

                <div className="mt-16 pt-8 border-t border-[#E6B23C]/5 flex flex-col sm:flex-row justify-between items-center gap-4">
                    <p className="text-[#A08E70]/50 text-[11px] font-bold tracking-[0.1em] uppercase">
                        {t("footer.copyright")}
                    </p>
                </div>
            </div>
        </footer>
    );
}
