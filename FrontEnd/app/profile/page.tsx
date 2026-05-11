"use client";

import PageShell from "../../components/layout/PageShell";
import { motion, AnimatePresence } from "framer-motion";
import {
    LogOut, Settings, Camera, MessageSquare, History as HistoryIcon, Bookmark, Search, ChevronRight, User, History
} from "lucide-react";
import { ALL_PHARAOHS, ALL_LANDMARKS } from "../../lib/mock/mock-all-entities";
import { saveResultToSession } from "../../lib/services/recognition";
import type { RecognitionEntity, RecognitionResult } from "../../lib/types";
import { useRouter } from "next/navigation";
import { Button } from "../../components/ui/button";
import { useLanguage } from "../../context/LanguageContext";
import Link from "next/link";
import { useState, useEffect } from "react";
import { createClient } from "../../lib/supabase/client";

type TabType = "saved" | "chats" | "history";

export default function ProfilePage() {
    const { t, language } = useLanguage();
    const [activeTab, setActiveTab] = useState<TabType>("saved");
    const [user, setUser] = useState<any>(null);
    const [profileData, setProfileData] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();
    const supabase = createClient();

    const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";

    const getEntityImage = (name: string, type: string = "") => {
        const safeType = (type || "").toLowerCase();
        const entityType = safeType.includes("pharaoh") ? "pharaoh" : "landmark";
        const source = entityType === "pharaoh" ? ALL_PHARAOHS : ALL_LANDMARKS;
        const entity = source.find(e => e.name.toLowerCase() === name.toLowerCase());

        if (!entity || !entity.image) {
            return entityType === "pharaoh" ? "/assets/trending/pharaohs/tutankhamun.jpg" : "/assets/trending/landmarks/giza.jpg";
        }

        if (entity.image.startsWith('/') || entity.image.startsWith('http')) {
            return entity.image;
        }

        // If it's a data/ path, it needs the R2 proxy
        if (entity.image.startsWith("data/")) {
            return `${baseUrl}/api/v1/assets/r2/${encodeURI(entity.image)}`;
        }

        // Encode URI to handle spaces and special characters in the path
        return `${baseUrl}/${encodeURI(entity.image)}`;
    };

    const getEntityDescription = (name: string, type: string = "") => {
        const safeType = (type || "").toLowerCase();
        const entityType = safeType.includes("pharaoh") ? "pharaoh" : "landmark";
        const source = entityType === "pharaoh" ? ALL_PHARAOHS : ALL_LANDMARKS;
        const entity = source.find(e => e.name.toLowerCase() === name.toLowerCase());
        return entity?.description || "Explore the legacy of this ancient entity...";
    };

    const cleanName = (name: string) => name.includes("(") ? name.split("(")[0].trim() : name;

    const handleEntityClick = (name: string, type: string) => {
        const entityType = type.toLowerCase().includes("pharaoh") ? "pharaoh" : "landmark";
        const source = entityType === "pharaoh" ? ALL_PHARAOHS : ALL_LANDMARKS;
        const entity = source.find(e => e.name.toLowerCase() === name.toLowerCase()) as unknown as RecognitionEntity;

        if (entity) {
            const result: RecognitionResult = {
                source: "explore",
                type: entityType as "pharaoh" | "landmark",
                name: entity.name,
                category: entityType,
                confidence: 1.0,
                binary_confidence: 1.0,
                entity: entity,
                debug_info: null,
            };
            saveResultToSession({ result, imageDataUrl: null });
            router.push("/result");
        } else {
            router.push(`/result?name=${encodeURIComponent(name)}&type=${entityType}`);
        }
    };

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                const { data: { user } } = await supabase.auth.getUser();
                if (user) {
                    const { data, error } = await supabase
                        .from('profiles')
                        .select('*')
                        .eq('id', user.id)
                        .single();

                    if (data) setProfileData(data);
                    setUser(user);
                }
            } catch (err) {
                console.error("Error fetching profile:", err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchProfile();
    }, [supabase]);

    // Mock Data
    const userData = {
        name: profileData?.full_name || user?.user_metadata?.full_name || user?.email?.split('@')[0] || "Explorer",
        username: profileData?.username || user?.user_metadata?.user_name || user?.email?.split('@')[0],
        avatar: profileData?.avatar_url || user?.user_metadata?.avatar_url,
        title: "High Priest of Discovery",
        joined: "Joined May 2024",
        location: "Giza, Egypt",
        stats: {
            discoveries: profileData?.history?.length || 0,
            favorites: profileData?.favorites?.length || 0,
            chats: 3
        },
        favorites: ((profileData?.favorites || []) as any[]).map((fav: any) => ({
            ...fav,
            image: getEntityImage(fav.name, fav.type),
            description: getEntityDescription(fav.name, fav.type)
        })),
        chats: [
            { name: "Ramesses II", lastMsg: "The Battle of Kadesh was a triumph...", date: "2h ago", image: "/assets/trending/pharaohs/ramesses.jpg" },
            { name: "The Sphinx", lastMsg: "I have stood watch for millennia...", date: "Yesterday", image: "/assets/trending/landmarks/sphinx.jpg" },
            { name: "Hatshepsut", lastMsg: "My temple at Deir el-Bahari is unique...", date: "3 days ago", image: "/assets/trending/pharaohs/hatshepsut.jpg" }
        ],
        history: ((profileData?.history || []) as any[]).map((hist: any) => ({
            ...hist,
            image: getEntityImage(hist.name, hist.type),
            description: getEntityDescription(hist.name, hist.type)
        }))
    };

    if (isLoading) {
        return (
            <PageShell>
                <div className="min-h-[60vh] flex items-center justify-center">
                    <div className="h-12 w-12 border-4 border-[#E6B23C]/20 border-t-[#E6B23C] rounded-full animate-spin" />
                </div>
            </PageShell>
        );
    }

    return (
        <PageShell>
            <div className="max-w-2xl mx-auto pt-0 pb-20 px-4">

                {/* ── MAIN PROFILE CARD ────────────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="relative rounded-[3rem] bg-[#0D0A07]/80 backdrop-blur-xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)]"
                >
                    {/* Avatar & Action Button (Now at the very top) */}
                    <div className="px-8 flex justify-between items-end pt-8 relative z-10">
                        <div className="h-32 w-32 rounded-full border-4 border-[#0D0A07] bg-[#1A1208] overflow-hidden shadow-2xl">
                            {userData.avatar ? (
                                <img src={userData.avatar} alt="Profile" className="w-full h-full object-cover" />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-[#1A1208] to-[#0D0A07]">
                                    <User size={48} className="text-[#E6B23C]/40" />
                                </div>
                            )}
                        </div>
                        <Link href="/profile/settings">
                            <Button className="rounded-full border border-[#E6B23C]/30 bg-[#E6B23C]/5 text-[#F5E6D0] hover:bg-[#E6B23C]/15 px-6 font-bold text-xs h-10 transition-all uppercase tracking-widest">
                                Profile settings
                            </Button>
                        </Link>
                    </div>

                    {/* Simple User Info */}
                    <div className="px-8 mt-6 mb-8">
                        <h1 className="text-3xl font-bold text-[#F5E6D0]">
                            {userData.name}
                        </h1>
                    </div>

                    {/* ── TABS ─────────────────────────────────────────────── */}
                    <div className="flex bg-[#0D0A07]/20">
                        {(["saved", "chats", "history"] as TabType[]).map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab)}
                                className="flex-1 py-5 text-[11px] font-black tracking-widest uppercase relative transition-colors"
                            >
                                <span className={activeTab === tab ? "text-[#E6B23C]" : "text-[#A08E70] hover:text-[#F5E6D0]"}>
                                    {tab === "saved" ? "Saved" : tab === "chats" ? "Chats" : "History"}
                                </span>
                                {activeTab === tab && (
                                    <motion.div
                                        layoutId="activeTab"
                                        className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#E6B23C] shadow-[0_-2px_10px_#E6B23C]"
                                    />
                                )}
                            </button>
                        ))}
                    </div>

                    {/* ── TAB CONTENT ─────────────────────────────────────── */}
                    <div className="min-h-[400px]">
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={activeTab}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                transition={{ duration: 0.2 }}
                                className="divide-y divide-[#E6B23C]/5"
                            >
                                {activeTab === "saved" && userData.favorites.map((item, i) => (
                                    <div
                                        key={i}
                                        onClick={() => handleEntityClick(item.name, item.type)}
                                        className="p-4 flex gap-4 hover:bg-[#E6B23C]/[0.05] transition-colors group cursor-pointer"
                                    >
                                        <div className="h-20 w-20 rounded-xl overflow-hidden border border-[#E6B23C]/10 shrink-0">
                                            <img src={item.image} alt={item.name} className="h-full w-full object-cover group-hover:scale-110 transition-transform duration-500" />
                                        </div>
                                        <div className="flex-1 py-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-[10px] font-bold tracking-widest text-[#E6B23C] uppercase">{item.type}</span>
                                                <span className="text-[10px] text-[#A08E70]">{item.date}</span>
                                            </div>
                                            <h3 className="text-[#F5E6D0] font-bold text-lg mb-1 group-hover:text-[#E6B23C] transition-colors">
                                                {cleanName(item.name)}
                                            </h3>
                                            <p className="text-xs text-[#A08E70] line-clamp-1">{item.description}</p>
                                        </div>
                                    </div>
                                ))}

                                {activeTab === "chats" && userData.chats.map((chat, i) => (
                                    <Link key={i} href="/chat" className="p-4 flex gap-4 hover:bg-[#E6B23C]/[0.02] transition-colors group">
                                        <div className="h-14 w-14 rounded-full overflow-hidden border border-[#E6B23C]/10 shrink-0">
                                            <img src={chat.image} alt={chat.name} className="h-full w-full object-cover" />
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <h3 className="text-[#F5E6D0] font-bold">{chat.name}</h3>
                                                <span className="text-[10px] text-[#A08E70]">{chat.date}</span>
                                            </div>
                                            <p className="text-sm text-[#A08E70] line-clamp-1 group-hover:text-[#F5E6D0]/70 transition-colors">
                                                {chat.lastMsg}
                                            </p>
                                        </div>
                                    </Link>
                                ))}

                                {activeTab === "history" && userData.history.map((entry, i) => (
                                    <div
                                        key={i}
                                        onClick={() => handleEntityClick(entry.name, entry.type)}
                                        className="p-4 flex gap-4 hover:bg-[#E6B23C]/[0.05] transition-colors group cursor-pointer"
                                    >
                                        <div className="h-16 w-16 rounded-lg overflow-hidden border border-[#E6B23C]/10 shrink-0 relative">
                                            <img src={entry.image} alt={entry.name} className="h-full w-full object-cover" />
                                            <div className="absolute inset-0 bg-[#0D0A07]/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                <Search size={20} className="text-[#E6B23C]" />
                                            </div>
                                        </div>
                                        <div className="flex-1 py-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <h3 className="text-[#F5E6D0] font-bold group-hover:text-[#E6B23C] transition-colors">
                                                    {cleanName(entry.name)}
                                                </h3>
                                                <span className="text-[10px] text-[#A08E70]">{entry.date}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 h-1.5 bg-[#E6B23C]/10 rounded-full overflow-hidden">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: entry.confidence || "100%" }}
                                                        className="h-full bg-[#E6B23C]"
                                                    />
                                                </div>
                                                <span className="text-[10px] font-bold text-[#E6B23C]">{entry.confidence || "100%"} Match</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </motion.div>
                        </AnimatePresence>
                    </div>
                </motion.div>
            </div>
        </PageShell>
    );
}
