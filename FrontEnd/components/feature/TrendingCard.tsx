"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import {
    Crown,
    Scroll,
    Sparkles,
    Shield,
    Star,
    MapPin,
    Landmark,
    Columns3,
    Mountain,
    Navigation,
} from "lucide-react";
import type { RecognitionEntity } from "@/lib/types";
import { saveResultToSession } from "@/lib/recognition";
import type { RecognitionResult } from "@/lib/types";

/* ── Consistent icon pools (cycled by index) ──────────────────────────── */
const PHARAOH_ICONS = [Crown, Star, Sparkles, Shield, Scroll];
const LANDMARK_ICONS = [Mountain, Navigation, Landmark, Columns3, MapPin];


/* ── Props ────────────────────────────────────────────────────────────── */
interface TrendingCardProps {
    variant: "pharaoh" | "landmark";
    entity: RecognitionEntity;
    index: number;
}

export default function TrendingCard({ variant, entity, index }: TrendingCardProps) {
    const router = useRouter();
    const isPharaoh = variant === "pharaoh";

    const IconComp = isPharaoh
        ? PHARAOH_ICONS[index % PHARAOH_ICONS.length]
        : LANDMARK_ICONS[index % LANDMARK_ICONS.length];


    function handleClick() {
        // Build a RecognitionResult-shaped payload so the Result page renders identically to the recognition flow
        const fakeResult: RecognitionResult = {
            source: "quick-link",
            type: variant,
            name: entity.name,
            category: variant,
            confidence: 1.0,
            binary_confidence: 1.0,
            entity: entity,
            debug_info: null,
        };
        // No uploaded image for quick-link cards
        saveResultToSession({ result: fakeResult, imageDataUrl: null });
        router.push("/result");
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.08, duration: 0.45 }}
            className="w-full"
        >
            <motion.div
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.98 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
                onClick={handleClick}
                className={`relative h-[320px] rounded-2xl overflow-hidden border transition-shadow duration-300 cursor-pointer outline-none
                    focus-visible:ring-2 focus-visible:ring-[#E6B23C]/60
                    ${isPharaoh
                        ? "border-[#E6B23C]/10 hover:border-[#E6B23C]/25 hover:shadow-[0_0_30px_rgba(230,178,60,0.12)]"
                        : "border-[#A08E70]/10 hover:border-[#A08E70]/25 hover:shadow-[0_0_30px_rgba(160,142,112,0.1)]"
                    }`}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && handleClick()}
            >
                {/* ── Background ─────────────────────────────────────── */}
                <div className="absolute inset-0">
                    {/* Hero Image from DB */}
                    {entity.images && entity.images.length > 0 && entity.images[0].url ? (
                        <motion.img
                            src={entity.images[0].url.startsWith("/static")
                                ? `${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010"}${entity.images[0].url}`
                                : entity.images[0].url
                            }
                            alt={entity.name}
                            className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                            fetchPriority={index < 2 ? "high" : "auto"}
                            loading={index < 2 ? "eager" : "lazy"}
                        />
                    ) : (
                        isPharaoh ? (
                            <div
                                className="absolute inset-0"
                                style={{
                                    background:
                                        "linear-gradient(160deg, #1E160E 0%, #2A1D0E 30%, #1A1407 60%, #0D0A07 100%)",
                                }}
                            />
                        ) : (
                            <div
                                className="absolute inset-0"
                                style={{
                                    background:
                                        "linear-gradient(160deg, #12150E 0%, #1A1810 30%, #151210 60%, #0D0A07 100%)",
                                }}
                            />
                        )
                    )}

                    {/* SVG pattern overlay */}
                    <div
                        className="absolute inset-0 opacity-[0.035]"
                        style={{
                            backgroundImage: isPharaoh
                                ? `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23E6B23C'%3E%3Cpath d='M30 5l6 12H24l6-12z'/%3E%3Ccircle cx='30' cy='45' r='3'/%3E%3Cpath d='M10 30h4v4h-4zm36 0h4v4h-4z'/%3E%3C/g%3E%3C/svg%3E")`
                                : `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23A08E70'%3E%3Cpath d='M30 0l30 52H0L30 0z' opacity='0.4'/%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/svg%3E")`,
                            backgroundSize: "60px 60px",
                        }}
                    />

                    {/* Warm radial highlight at top */}
                    <div
                        className="absolute inset-0 pointer-events-none"
                        style={{
                            background: isPharaoh
                                ? "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(230,178,60,0.08) 0%, transparent 60%)"
                                : "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(160,142,112,0.06) 0%, transparent 60%)",
                        }}
                    />
                </div>

                {/* ── Shine sweep on hover ────────────────────────────── */}
                <div className="trending-card-shine absolute inset-0 pointer-events-none z-10" />

                {/* ── Corner icon ─────────────────────────────────────── */}
                <div
                    className={`absolute top-4 right-4 z-20 p-2 rounded-xl transition-colors duration-300
                        ${isPharaoh
                            ? "bg-[#E6B23C]/[0.07] group-hover:bg-[#E6B23C]/[0.14] text-[#E6B23C]/50"
                            : "bg-[#A08E70]/[0.07] group-hover:bg-[#A08E70]/[0.14] text-[#A08E70]/50"
                        }`}
                >
                    <IconComp size={18} />
                </div>


                {/* ── Rank number ─────────────────────────────────────── */}
                <div
                    className="absolute bottom-20 left-4 z-20 text-[60px] font-black leading-none pointer-events-none select-none"
                    style={{
                        fontFamily: "var(--font-cinzel-dec), serif",
                        color: isPharaoh
                            ? "rgba(230,178,60,0.06)"
                            : "rgba(160,142,112,0.06)",
                    }}
                >
                    {index + 1}
                </div>

                {/* ── Content area ────────────────────────────────────── */}
                <div className="absolute bottom-0 left-0 right-0 z-20 p-4">
                    {/* gradient fade above content */}
                    <div
                        className="absolute inset-x-0 bottom-0 h-40 -z-10"
                        style={{
                            background:
                                "linear-gradient(to top, rgba(13,10,7,0.95) 0%, rgba(13,10,7,0.7) 50%, transparent 100%)",
                        }}
                    />

                    {/* Entity name (from DB) */}
                    <h3 className="font-heading text-base font-bold text-[#F5E6D0] mb-1 leading-tight tracking-wide hover:text-white transition-colors line-clamp-2">
                        {entity.name.includes("(") ? entity.name.split("(")[0].trim() : entity.name}
                    </h3>

                    {/* Pharaoh: tags chip */}
                    {isPharaoh && (entity.dynasty || entity.type) && (
                        <div className="flex flex-wrap gap-1 mb-1.5 overflow-hidden">
                            {entity.type && (
                                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-[#E6B23C]/[0.08] border border-[#E6B23C]/10 text-[9px] font-semibold tracking-wider text-[#E6B23C] uppercase whitespace-nowrap">
                                    {entity.type}
                                </span>
                            )}
                            {entity.dynasty && (
                                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-[#E6B23C]/[0.08] border border-[#E6B23C]/10 text-[9px] font-semibold tracking-wider text-[#E6B23C] uppercase whitespace-nowrap">
                                    <Crown size={8} />
                                    {entity.dynasty}
                                </span>
                            )}
                        </div>
                    )}

                    {/* Landmark: location line */}
                    {!isPharaoh && entity.location && (
                        <div className="flex items-center gap-1 mb-1.5 text-[#A08E70]">
                            <MapPin size={9} className="flex-shrink-0" />
                            <span className="text-[10px] font-medium tracking-wide truncate">
                                {entity.location}
                            </span>
                        </div>
                    )}

                    {/* Teaser description */}
                    {entity.description && (
                        <p className="text-[10px] text-[#A08E70]/80 leading-relaxed line-clamp-2">
                            {entity.description}
                        </p>
                    )}
                </div>
            </motion.div>
        </motion.div>
    );
}
