"use client";

import { motion } from "framer-motion";
import TrendingCard from "./TrendingCard";
import type { RecognitionEntity } from "@/lib/types";

interface TrendingRowProps {
    title: string;
    items: RecognitionEntity[];
    type: "pharaoh" | "landmark";
    isLoading?: boolean;
}

function SkeletonCard({ isPharaoh }: { isPharaoh: boolean }) {
    return (
        <div className={`relative h-[320px] rounded-2xl overflow-hidden border animate-pulse
            ${isPharaoh ? "border-[#E6B23C]/10 bg-[#1E160E]" : "border-[#A08E70]/10 bg-[#12150E]"}`}>
            {/* shimmer overlay */}
            <div
                className="absolute inset-0 opacity-[0.04]"
                style={{
                    background: isPharaoh
                        ? "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(230,178,60,0.5) 0%, transparent 60%)"
                        : "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(160,142,112,0.5) 0%, transparent 60%)",
                }}
            />
            {/* Badge placeholder */}
            <div className={`absolute top-4 left-4 h-5 w-20 rounded-full ${isPharaoh ? "bg-[#E6B23C]/10" : "bg-[#A08E70]/10"}`} />
            {/* Content placeholders */}
            <div className="absolute bottom-4 left-4 right-4 space-y-2">
                <div className={`h-4 w-4/5 rounded ${isPharaoh ? "bg-[#E6B23C]/8" : "bg-[#A08E70]/8"}`} />
                <div className={`h-3 w-2/5 rounded ${isPharaoh ? "bg-[#E6B23C]/5" : "bg-[#A08E70]/5"}`} />
                <div className={`h-3 w-full rounded ${isPharaoh ? "bg-[#E6B23C]/4" : "bg-[#A08E70]/4"}`} />
                <div className={`h-3 w-3/4 rounded ${isPharaoh ? "bg-[#E6B23C]/4" : "bg-[#A08E70]/4"}`} />
            </div>
        </div>
    );
}

export default function TrendingRow({ title, items, type, isLoading = false }: TrendingRowProps) {
    const isPharaoh = type === "pharaoh";

    return (
        <motion.section
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6 }}
            className="relative"
        >
            {/* ── Header ──────────────────────────────────────────── */}
            <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                    <div
                        className="h-[2px] w-8"
                        style={{
                            background: isPharaoh
                                ? "linear-gradient(90deg, #E6B23C, transparent)"
                                : "linear-gradient(90deg, #A08E70, transparent)",
                        }}
                    />
                    <h2 className="font-heading text-2xl md:text-3xl font-bold text-[#F5E6D0] tracking-tight">
                        {title}
                    </h2>
                </div>
            </div>

            {/* ── 5-Column Grid ───────────────────────────────────── */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {isLoading
                    ? Array.from({ length: 5 }).map((_, i) => (
                        <SkeletonCard key={i} isPharaoh={isPharaoh} />
                    ))
                    : items.slice(0, 5).map((entity, i) => (
                        <TrendingCard key={entity.id} variant={type} entity={entity} index={i} />
                    ))
                }
            </div>
        </motion.section>
    );
}
