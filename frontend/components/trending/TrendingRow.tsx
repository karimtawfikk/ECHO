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
            <div
                className="absolute inset-0 opacity-[0.04]"
                style={{
                    background: isPharaoh
                        ? "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(230,178,60,0.5) 0%, transparent 60%)"
                        : "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(160,142,112,0.5) 0%, transparent 60%)",
                }}
            />
            <div className={`absolute top-4 left-4 h-5 w-20 rounded-full ${isPharaoh ? "bg-[#E6B23C]/10" : "bg-[#A08E70]/10"}`} />
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

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.12,
                delayChildren: 0.1,
            }
        }
    };

    return (
        <motion.section
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={containerVariants}
            className="relative"
        >
            <motion.div
                variants={{
                    hidden: { opacity: 0, x: -20 },
                    visible: { opacity: 1, x: 0, transition: { duration: 0.5 } }
                }}
                className="flex items-center justify-between mb-8"
            >
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
            </motion.div>

            <motion.div
                variants={containerVariants}
                className="flex flex-wrap justify-center gap-4 [&>*]:w-[calc(50%-0.5rem)] md:[&>*]:w-[calc(33.33%-0.67rem)] lg:[&>*]:w-[calc(20%-0.8rem)]"
            >
                {isLoading
                    ? Array.from({ length: 5 }).map((_, i) => (
                        <SkeletonCard key={i} isPharaoh={isPharaoh} />
                    ))
                    : items.slice(0, 5).map((entity, i) => (
                        <TrendingCard key={entity.id} variant={type} entity={entity} index={i} />
                    ))
                }
            </motion.div>
        </motion.section>
    );
}
