"use client";

import PageShell from "../../components/layout/PageShell";
import { motion, AnimatePresence } from "framer-motion";
import {
    LogOut, Settings, Camera, MessageSquare, History as HistoryIcon, Bookmark, Search, ChevronRight, User, History, Calendar
} from "lucide-react";

import { saveResultToSession } from "../../lib/services/recognition";
import type { RecognitionEntity, RecognitionResult } from "../../lib/types";
import { useRouter } from "next/navigation";
import { Button } from "../../components/ui/button";
import { useLanguage } from "../../context/LanguageContext";
import Link from "next/link";
import { useState, useEffect } from "react";
import { createClient } from "../../lib/supabase/client";
import { cleanEntityName } from "../../lib/utils";

type TabType = "saved" | "chats" | "history";

export default function ProfilePage() {
    const { t, language } = useLanguage();
    const [activeTab, setActiveTab] = useState<TabType>("saved");
    const [user, setUser] = useState<any>(null);
    const [profileData, setProfileData] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [historyList, setHistoryList] = useState<any[]>([]);
    const [selectedTranslation, setSelectedTranslation] = useState<string | null>(null);
    const [dbEntities, setDbEntities] = useState<{ pharaohs: any[]; landmarks: any[] } | null>(null);
    const [historyFilter, setHistoryFilter] = useState<"all" | "recognition" | "translation">("all");
    const router = useRouter();
    const supabase = createClient();

    const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";

    const getEntityImage = (name: string, type: string = "") => {
        const cleanNameStr = cleanEntityName(name);
        const normalized = cleanNameStr.toLowerCase().trim();

        // Static local folder fallback for the 10 trending entities
        const trendingImages: Record<string, string> = {
            "akhenaton": "/images/pharaohs/Akhenaton.JPG",
            "cleopatra vii philopator": "/images/pharaohs/Cleopatra%20VII%20Philopator.jpg",
            "hatshepsut": "/images/pharaohs/Hatshepsut.JPG",
            "ramesses ii": "/images/pharaohs/Ramesses%20II.jpg",
            "tutankhamun": "/images/pharaohs/Tutankhamun.jpg",
            "pyramids of giza": "/images/landmarks/Pyramids%20of%20Giza.webp",
            "sphinx": "/images/landmarks/Sphinx.jpg",
            "temple of karnak": "/images/landmarks/Temple%20of%20Karnak.jpg",
            "temple of luxor": "/images/landmarks/Temple%20of%20Luxor.jpg",
            "the great temple of ramesses ii at abu simbel": "/images/landmarks/The%20Great%20Temple%20of%20Ramesses%20II%20at%20Abu%20Simbel.webp"
        };

        if (trendingImages[normalized]) {
            return trendingImages[normalized];
        }

        const safeType = (type || "").toLowerCase();
        const entityType = safeType.includes("pharaoh") ? "pharaoh" : "landmark";

        // Try dynamic entities first to find image from DB
        if (dbEntities) {
            const list = entityType === "pharaoh" ? dbEntities.pharaohs : dbEntities.landmarks;
            const found = list.find((e: any) => e.name.toLowerCase() === name.toLowerCase());
            if (found && found.image) {
                if (found.image.startsWith('/') || found.image.startsWith('http')) return found.image;
                if (found.image.startsWith("data/")) return `${baseUrl}/api/v1/assets/r2/${encodeURI(found.image)}`;
                return `${baseUrl}/${encodeURI(found.image)}`;
            }
        }

        return entityType === "pharaoh" ? "/assets/trending/pharaohs/tutankhamun.jpg" : "/assets/trending/landmarks/giza.jpg";
    };

    const getEntityDescription = (name: string, type: string = "") => {
        const safeType = (type || "").toLowerCase();
        const entityType = safeType.includes("pharaoh") ? "pharaoh" : "landmark";

        // Try searching in the dynamically fetched dbEntities first!
        if (dbEntities) {
            const list = entityType === "pharaoh" ? dbEntities.pharaohs : dbEntities.landmarks;
            const found = list.find((e: any) => e.name.toLowerCase() === name.toLowerCase());
            if (found && found.description) {
                return found.description;
            }
        }

        return "Explore the legacy of this ancient entity...";
    };

    const cleanName = (name: string) => cleanEntityName(name);

    const handleEntityClick = (name: string, type: string, imageUrl: string | null = null) => {
        const entityType = type.toLowerCase().includes("pharaoh") ? "pharaoh" : "landmark";

        let entity = null;
        if (dbEntities) {
            const list = entityType === "pharaoh" ? dbEntities.pharaohs : dbEntities.landmarks;
            entity = list.find((e: any) => e.name.toLowerCase() === name.toLowerCase());
        }



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
            saveResultToSession({ result, imageDataUrl: imageUrl });
            router.push(`/result?t=${Date.now()}`);
        } else {
            router.push(`/result?name=${encodeURIComponent(name)}&type=${entityType}${imageUrl ? `&imageUrl=${encodeURIComponent(imageUrl)}` : ""}&t=${Date.now()}`);
        }
    };

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                // Fetch dynamic dbEntities from DB
                try {
                    const dbRes = await fetch(`${baseUrl}/api/v1/entities/all`);
                    if (dbRes.ok) {
                        const dbData = await dbRes.json();
                        setDbEntities(dbData);
                    }
                } catch (dbErr) {
                    console.error("Error loading dbEntities:", dbErr);
                }

                const { data: { user } } = await supabase.auth.getUser();
                if (user) {
                    const { data, error } = await supabase
                        .from('profiles')
                        .select('*')
                        .eq('id', user.id)
                        .single();

                    if (data) setProfileData(data);
                    setUser(user);

                    // Fetch history records from dynamic tables
                    const { data: recData } = await supabase
                        .from('recognition_history')
                        .select('*')
                        .eq('user_id', user.id);

                    const { data: transData } = await supabase
                        .from('translation_history')
                        .select('*')
                        .eq('user_id', user.id);

                    const merged = [
                        ...(recData || []).map((item: any) => ({
                            ...item,
                            history_type: "recognition"
                        })),
                        ...(transData || []).map((item: any) => ({
                            ...item,
                            history_type: "translation"
                        }))
                    ].sort((a: any, b: any) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

                    setHistoryList(merged);
                }
            } catch (err) {
                console.error("Error fetching profile:", err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchProfile();
    }, [supabase]);

    const formatJoinedDate = (dateString: string) => {
        if (!dateString) return "Joined May 2024";
        try {
            const date = new Date(dateString);
            const month = date.toLocaleString('en-US', { month: 'short' });
            const year = date.getFullYear();
            return `Joined ${month} ${year}`;
        } catch (e) {
            return "Joined May 2024";
        }
    };

    // Mock Data
    const userData = {
        name: profileData?.full_name || user?.user_metadata?.full_name || user?.email?.split('@')[0] || "Explorer",
        username: profileData?.username || user?.user_metadata?.user_name || user?.email?.split('@')[0],
        avatar: profileData?.avatar_url || user?.user_metadata?.avatar_url,
        joined: formatJoinedDate(profileData?.created_at),
        favorites: ((profileData?.favorites || []) as any[]).map((fav: any) => ({
            ...fav,
            image: getEntityImage(fav.name, fav.type),
            description: getEntityDescription(fav.name, fav.type)
        })),
        chats: [
            { id: 1, name: "Giza Pyramids Chat", lastMsg: "The Great Pyramid of Giza is the oldest...", date: "2 mins ago", image: "/assets/trending/landmarks/giza.jpg" },
            { id: 2, name: "Tutankhamun Chat", lastMsg: "Tutankhamun was an ancient Egyptian pharaoh...", date: "1 hour ago", image: "/assets/trending/pharaohs/tutankhamun.jpg" },
        ],
        history: historyList.map((entry: any) => {
            const dateObj = new Date(entry.created_at);
            const dateStr = dateObj.toLocaleDateString(language === "AR" ? "ar-EG" : "en-US", {
                month: "short",
                day: "numeric",
                year: "numeric"
            });

            const imageUrl = entry.image_path?.startsWith("http")
                ? entry.image_path
                : `${baseUrl}/api/v1/assets/r2-history/${entry.image_path}`;

            if (entry.history_type === "recognition") {
                return {
                    id: entry.id,
                    name: entry.entity_name,
                    type: entry.entity_type,
                    image: imageUrl,
                    date: dateStr,
                    history_type: "recognition",
                    confidence: "98%",
                    description: getEntityDescription(entry.entity_name, entry.entity_type)
                };
            } else {
                return {
                    id: entry.id,
                    name: "Hieroglyphic Translation",
                    type: "translation",
                    image: imageUrl,
                    date: dateStr,
                    history_type: "translation",
                    translation: entry.translation,
                    description: entry.translation
                };
            }
        })
    };

    const filteredHistory = userData.history.filter(
        (entry) => historyFilter === "all" || entry.history_type === historyFilter
    );

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
                        <div className="flex items-center gap-2 mt-2 text-[#A08E70]">
                            <Calendar size={13} className="text-[#E6B23C]/60" />
                            <span className="text-[10px] font-bold uppercase tracking-[0.2em]">
                                {userData.joined}
                            </span>
                        </div>
                    </div>

                    {/* ── TABS ─────────────────────────────────────────────── */}
                    <div className="flex bg-[#0D0A07]/20">
                        {(["saved", "chats", "history"] as TabType[]).map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab)}
                                className="flex-1 py-5 flex items-center justify-center relative transition-colors"
                            >
                                <span className={activeTab === tab ? "text-[#E6B23C]" : "text-[#A08E70] hover:text-[#F5E6D0]"}>
                                    {tab === "saved" ? <Bookmark size={20} /> : tab === "chats" ? <MessageSquare size={20} /> : <HistoryIcon size={20} />}
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

                                {activeTab === "history" && (
                                    <>
                                        {/* Dynamic Sub-Filter Buttons */}
                                        <div className="flex justify-center gap-3 px-6 py-4 bg-[#0D0A07]/40 border-b border-[#E6B23C]/5">
                                            {([
                                                { id: "all", label: language === "AR" ? "الكل" : language === "FR" ? "Tout" : "All" },
                                                { id: "recognition", label: language === "AR" ? "التعرف" : language === "FR" ? "Reconnaissances" : "Recognitions" },
                                                { id: "translation", label: language === "AR" ? "الترجمة" : language === "FR" ? "Translations" : "Translations" }
                                            ] as const).map((filter) => (
                                                <button
                                                    key={filter.id}
                                                    onClick={() => setHistoryFilter(filter.id)}
                                                    className={`px-4 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider transition-all duration-300 ${historyFilter === filter.id
                                                            ? "bg-[#E6B23C] text-[#0D0A07] shadow-[0_2px_10px_rgba(230,178,60,0.3)]"
                                                            : "bg-[#E6B23C]/5 border border-[#E6B23C]/10 text-[#A08E70] hover:text-[#F5E6D0] hover:border-[#E6B23C]/20"
                                                        }`}
                                                >
                                                    {filter.label}
                                                </button>
                                            ))}
                                        </div>

                                        {/* History items list */}
                                        {filteredHistory.length > 0 ? (
                                            filteredHistory.map((entry, i) => (
                                                <div
                                                    key={i}
                                                    onClick={() => {
                                                        if (entry.history_type === "recognition") {
                                                            handleEntityClick(entry.name, entry.type, entry.image);
                                                        } else {
                                                            sessionStorage.setItem("echo_translation_history_result", JSON.stringify({
                                                                translation: entry.translation,
                                                                imageUrl: entry.image
                                                            }));
                                                            router.push("/translate");
                                                        }
                                                    }}
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
                                                            <span className="text-[10px] font-bold tracking-widest text-[#E6B23C] uppercase">
                                                                {entry.history_type === "recognition" ? entry.type : (language === "AR" ? "ترجمة" : language === "FR" ? "Traduction" : "Translation")}
                                                            </span>
                                                            <span className="text-[10px] text-[#A08E70]">{entry.date}</span>
                                                        </div>
                                                        <h3 className="text-[#F5E6D0] font-bold text-lg mb-1 group-hover:text-[#E6B23C] transition-colors">
                                                            {entry.history_type === "recognition" ? cleanName(entry.name) : (language === "AR" ? "ترجمة هيروغليفية" : language === "FR" ? "Traduction Hiéroglyphique" : "Hieroglyphic Translation")}
                                                        </h3>

                                                        {entry.history_type === "recognition" ? (
                                                            <p className="text-xs text-[#A08E70] line-clamp-1">
                                                                {entry.description}
                                                            </p>
                                                        ) : (
                                                            <p className="text-xs text-[#A08E70] line-clamp-1 italic">
                                                                "{entry.translation}"
                                                            </p>
                                                        )}
                                                    </div>
                                                </div>
                                            ))
                                        ) : (
                                            <div className="p-20 text-center">
                                                <HistoryIcon size={32} className="mx-auto mb-4 text-[#A08E70]/20" />
                                                <p className="text-xs text-[#A08E70]">
                                                    {language === "AR" ? "لا يوجد سجل للبحث حالياً." : language === "FR" ? "Aucun historique trouvé." : "No history found."}
                                                </p>
                                            </div>
                                        )}
                                    </>
                                )}
                            </motion.div>
                        </AnimatePresence>
                    </div>
                </motion.div>

                {/* ── TRANSLATION DETAIL MODAL ────────────────────────── */}
                <AnimatePresence>
                    {selectedTranslation && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setSelectedTranslation(null)}
                            className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4 cursor-pointer"
                        >
                            <motion.div
                                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.9, y: 20 }}
                                transition={{ duration: 0.3 }}
                                onClick={(e) => e.stopPropagation()}
                                className="w-full max-w-lg bg-[#0F0C08]/95 border border-[#E6B23C]/30 rounded-[2.5rem] p-8 md:p-10 relative overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.8)] cursor-default"
                            >
                                <div className="absolute inset-0 opacity-[0.05] bg-[url('https://www.transparenttextures.com/patterns/papyros.png')] pointer-events-none" />
                                <div className="absolute -inset-1 bg-[#E6B23C]/5 rounded-[2.5rem] blur-xl pointer-events-none" />

                                <div className="relative z-10 flex flex-col items-center">
                                    <div className="text-[#E6B23C] text-2xl font-display tracking-[0.4em] mb-4 select-none">
                                        𓂀 𓅃 𓆣
                                    </div>

                                    <h3 className="font-display text-2xl font-bold text-[#F5E6D0] tracking-[0.05em] uppercase text-center mb-2" style={{ fontFamily: 'var(--font-cormorant), serif' }}>
                                        {language === "AR" ? "الترجمة الهيروغليفية" : language === "FR" ? "Traduction Hiéroglyphique" : "Hieroglyphic Translation"}
                                    </h3>

                                    <div className="w-20 h-[1px] bg-gradient-to-r from-transparent via-[#E6B23C]/30 to-transparent mb-8" />

                                    <div className="w-full bg-[#1A1208]/30 rounded-2xl p-6 border border-[#E6B23C]/10 mb-8 max-h-[250px] overflow-y-auto">
                                        <p className="text-[#F5E6D0] text-lg leading-relaxed text-center font-medium italic">
                                            "{selectedTranslation}"
                                        </p>
                                    </div>

                                    <Button
                                        onClick={() => setSelectedTranslation(null)}
                                        className="h-12 px-8 rounded-full bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] hover:bg-[#E6B23C]/20 font-bold text-xs uppercase tracking-widest transition-all"
                                    >
                                        {language === "AR" ? "إغلاق" : language === "FR" ? "Fermer" : "Close"}
                                    </Button>
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </PageShell>
    );
}
