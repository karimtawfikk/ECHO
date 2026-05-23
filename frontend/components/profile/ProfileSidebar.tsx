"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, LogOut, Settings, Camera, MessageSquare, History as HistoryIcon, Bookmark, Search, User, Calendar, ChevronRight, ArrowLeft, AtSign, Mail, Trash2, ChevronDown, Pencil, Lock, Eye, EyeOff } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { createClient } from "../../lib/supabase/client";
import { useLanguage } from "../../context/LanguageContext";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ALL_PHARAOHS, ALL_LANDMARKS } from "../../lib/mock/mock-all-entities";
import { saveResultToSession } from "../../lib/services/recognition";
import type { RecognitionEntity, RecognitionResult } from "../../lib/types";
import { Button } from "../ui/button";

type TabType = "saved" | "chats" | "history";
type ViewType = "profile" | "settings";

interface ProfileSidebarProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function ProfileSidebar({ isOpen, onClose }: ProfileSidebarProps) {
    const { t, language } = useLanguage();
    const [view, setView] = useState<ViewType>("profile");
    const [activeTab, setActiveTab] = useState<TabType>("saved");
    const [user, setUser] = useState<any>(null);
    const [profileData, setProfileData] = useState<any>(null);
    const [chatHistory, setChatHistory] = useState<any[]>([]);
    const [historyList, setHistoryList] = useState<any[]>([]);
    const [expandedEntity, setExpandedEntity] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();
    const supabase = createClient();

    // Form States
    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [isSaving, setIsSaving] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [showSuccess, setShowSuccess] = useState(false);
    const [showNoChanges, setShowNoChanges] = useState(false);
    const [selectedAvatarFile, setSelectedAvatarFile] = useState<File | null>(null);
    const [avatarPreviewUrl, setAvatarPreviewUrl] = useState<string | null>(null);
    const [settingsError, setSettingsError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Password States
    const [currentPassword, setCurrentPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [showCurrentPassword, setShowCurrentPassword] = useState(false);
    const [showNewPassword, setShowNewPassword] = useState(false);
    const [isSavingPassword, setIsSavingPassword] = useState(false);
    const [showPasswordSuccess, setShowPasswordSuccess] = useState(false);
    const [passwordError, setPasswordError] = useState<string | null>(null);
    const [dbEntities, setDbEntities] = useState<{ pharaohs: any[]; landmarks: any[] } | null>(null);
    const [historyFilter, setHistoryFilter] = useState<"all" | "recognition" | "translation">("all");

    const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";

    const getEntityImage = (name: string, type: string = "") => {
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

        const source = entityType === "pharaoh" ? ALL_PHARAOHS : ALL_LANDMARKS;
        const entity = source.find(e => e.name.toLowerCase() === name.toLowerCase());

        if (!entity || !entity.image) {
            return entityType === "pharaoh" ? "/assets/trending/pharaohs/tutankhamun.jpg" : "/assets/trending/landmarks/giza.jpg";
        }

        if (entity.image.startsWith('/') || entity.image.startsWith('http')) {
            return entity.image;
        }

        if (entity.image.startsWith("data/")) {
            return `${baseUrl}/api/v1/assets/r2/${encodeURI(entity.image)}`;
        }

        return `${baseUrl}/${encodeURI(entity.image)}`;
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

        const source = entityType === "pharaoh" ? ALL_PHARAOHS : ALL_LANDMARKS;
        const entity = source.find(e => e.name.toLowerCase() === name.toLowerCase());
        return entity?.description || "Explore the legacy of this ancient entity...";
    };

    const cleanName = (name: string) => name.includes("(") ? name.split("(")[0].trim() : name;

    const handleEntityClick = (name: string, type: string, imageUrl: string | null = null) => {
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
            saveResultToSession({ result, imageDataUrl: imageUrl });
            router.push(`/result?t=${Date.now()}`);
            onClose();
        } else {
            router.push(`/result?name=${encodeURIComponent(name)}&type=${entityType}${imageUrl ? `&imageUrl=${encodeURIComponent(imageUrl)}` : ""}&t=${Date.now()}`);
            onClose();
        }
    };

    useEffect(() => {
        if (!isOpen) return;

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

                    if (data) {
                        setProfileData(data);
                        const fullName = data.full_name || user.user_metadata?.full_name || "";
                        const nameParts = fullName.split(' ');
                        setFirstName(nameParts[0] || "");
                        setLastName(nameParts.slice(1).join(' ') || "");
                        setUsername(data.username || user.user_metadata?.user_name || user.email?.split('@')[0] || "");
                        setEmail(user.email || "");
                    }

                    // Fetch real conversations
                    const { data: convs } = await supabase
                        .from('conversations')
                        .select('*')
                        .eq('user_id', user.id)
                        .order('created_at', { ascending: false });

                    if (convs) setChatHistory(convs);

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

                    setUser(user);
                }
            } catch (err) {
                console.error("Error fetching profile:", err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchProfile();
    }, [supabase, isOpen]);

    const handleSignOut = async () => {
        await supabase.auth.signOut();
        onClose();
        router.refresh();
    };

    const handleAvatarUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file || !user) return;

        setSettingsError(null);

        // 1. Validations
        const MAX_SIZE = 2 * 1024 * 1024; // 2MB
        if (file.size > MAX_SIZE) {
            setSettingsError("File is too large. Maximum size is 2MB.");
            return;
        }

        if (!file.type.startsWith('image/')) {
            setSettingsError("Please upload an image file.");
            return;
        }

        // Generate local preview URL
        const previewUrl = URL.createObjectURL(file);
        setAvatarPreviewUrl(previewUrl);
        setSelectedAvatarFile(file);
    };

    const handleSaveSettings = async () => {
        setSettingsError(null);
        const currentFullName = `${firstName} ${lastName}`.trim();
        const initialFullName = profileData?.full_name || "";
        const initialUsername = profileData?.username || "";

        const hasChanges = currentFullName !== initialFullName || 
                            username !== initialUsername || 
                            selectedAvatarFile !== null;

        if (!hasChanges) {
            setShowNoChanges(true);
            setTimeout(() => setShowNoChanges(false), 3000);
            return;
        }

        setIsSaving(true);
        try {
            if (!user) {
                setSettingsError("You must be logged in to save changes.");
                return;
            }

            let newAvatarUrl = profileData?.avatar_url || user.user_metadata?.avatar_url || null;

            // Upload selected avatar file if changed
            if (selectedAvatarFile) {
                setIsUploading(true);
                const filePath = `${user.id}`;

                // 1. Clean up any existing files for this user (handles old extensions and avoids UPDATE permission issues)
                try {
                    const { data: existingFiles } = await supabase.storage.from('avatars').list('', { search: user.id });
                    if (existingFiles && existingFiles.length > 0) {
                        const filesToRemove = existingFiles
                            .filter(f => f.name.startsWith(user.id))
                            .map(f => f.name);
                        
                        if (filesToRemove.length > 0) {
                            const { error: removeError } = await supabase.storage.from('avatars').remove(filesToRemove);
                            if (removeError) {
                                console.warn("Failed to delete old images:", removeError.message);
                            }
                        }
                    }
                } catch (e) {
                    console.warn("Could not list/remove existing files. Proceeding with upload.", e);
                }

                // 2. Upload new file (using upsert: true)
                const { error: uploadError } = await supabase.storage
                    .from('avatars')
                    .upload(filePath, selectedAvatarFile, {
                        upsert: true,
                        cacheControl: '0'
                    });

                if (uploadError) {
                    // Fallback to update if upload fails (some Supabase setups require this for overwriting)
                    const { error: updateError } = await supabase.storage
                        .from('avatars')
                        .update(filePath, selectedAvatarFile, {
                            upsert: true,
                            cacheControl: '0'
                        });
                    if (updateError) throw updateError;
                }

                const { data: { publicUrl } } = supabase.storage
                    .from('avatars')
                    .getPublicUrl(filePath);

                // Add cache-buster to force-reload browser cache of the same static URL path
                newAvatarUrl = `${publicUrl}?t=${Date.now()}`;
                setIsUploading(false);
            }

            const fullName = currentFullName;

            const { error: authError } = await supabase.auth.updateUser({
                data: {
                    full_name: fullName,
                    user_name: username,
                    avatar_url: newAvatarUrl
                }
            });
            if (authError) throw authError;

            const { error: dbError } = await supabase
                .from('profiles')
                .update({
                    full_name: fullName,
                    username: username,
                    avatar_url: newAvatarUrl
                })
                .eq('id', user.id);

            if (dbError) throw dbError;

            // Update local state
            setProfileData({ 
                ...profileData, 
                full_name: fullName, 
                username: username,
                avatar_url: newAvatarUrl
            });
            
            setUser({
                ...user,
                user_metadata: {
                    ...user.user_metadata,
                    full_name: fullName,
                    user_name: username,
                    avatar_url: newAvatarUrl
                }
            });

            // Clear temp states
            setSelectedAvatarFile(null);
            setAvatarPreviewUrl(null);

            setShowSuccess(true);
            setTimeout(() => {
                setShowSuccess(false);
                setView("profile");
            }, 2000);
        } catch (err: any) {
            console.error("Error updating user:", err);
            setSettingsError(`Update failed: ${err.message || "Unknown error"}`);
            setIsUploading(false);
        } finally {
            setIsSaving(false);
        }
    };

    const handleUpdatePassword = async () => {
        setPasswordError(null);

        if (!currentPassword || !newPassword) {
            setPasswordError("Please fill in both current and new password fields.");
            return;
        }

        setIsSavingPassword(true);
        try {
            if (!user || !user.email) {
                setPasswordError("You must be logged in to update your password.");
                return;
            }

            // 1. Verify current password by signing in
            const { error: verifyError } = await supabase.auth.signInWithPassword({
                email: user.email,
                password: currentPassword
            });

            if (verifyError) {
                setPasswordError("Invalid current password.");
                return;
            }

            // 2. Update to new password
            const { error: updateError } = await supabase.auth.updateUser({
                password: newPassword
            });

            if (updateError) {
                throw updateError;
            }

            setShowPasswordSuccess(true);
            setCurrentPassword("");
            setNewPassword("");
            setTimeout(() => {
                setShowPasswordSuccess(false);
                setView("profile");
            }, 2500);
        } catch (err: any) {
            console.error("Error updating password:", err);
            setPasswordError(`Password update failed: ${err.message || "Unknown error"}`);
        } finally {
            setIsSavingPassword(false);
        }
    };

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

    // Scroll lock and reset view when closing
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "unset";
            setTimeout(() => setView("profile"), 300);
        }

        return () => {
            document.body.style.overflow = "unset";
        };
    }, [isOpen]);

    // Mock Data
    const userData = {
        name: (firstName || lastName) ? `${firstName} ${lastName}`.trim() : profileData?.full_name || user?.user_metadata?.full_name || user?.email?.split('@')[0] || "Explorer",
        username: username || profileData?.username || user?.user_metadata?.user_name || user?.email?.split('@')[0],
        avatar: profileData?.avatar_url || user?.user_metadata?.avatar_url,
        joined: formatJoinedDate(profileData?.created_at),
        favorites: ((profileData?.favorites || []) as any[]).map((fav: any) => ({
            ...fav,
            image: getEntityImage(fav.name, fav.type),
            description: getEntityDescription(fav.name, fav.type)
        })),
        chats: chatHistory,
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

    // Group chats by entity for the records view
    const groupedChats = userData.chats.reduce((acc, chat) => {
        const key = chat.entity_name || "Unknown";
        if (!acc[key]) acc[key] = [];
        acc[key].push(chat);
        return acc;
    }, {} as Record<string, any[]>);

    const isEmailUser = user && (
        (user.app_metadata?.provider || "").toLowerCase() === "email" ||
        user.identities?.some((identity: any) => (identity.provider || "").toLowerCase() === "email")
    );

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100]"
                    />

                    {/* Sidebar */}
                    <motion.div
                        initial={{ x: "100%" }}
                        animate={{ x: 0 }}
                        exit={{ x: "100%" }}
                        transition={{ type: "spring", damping: 25, stiffness: 200 }}
                        className="fixed top-0 right-0 h-screen w-full max-w-md bg-[#0D0A07] border-l border-[#E6B23C]/20 z-[101] shadow-[-20px_0_50px_rgba(0,0,0,0.5)] flex flex-col"
                    >
                        <AnimatePresence mode="wait">
                            {view === "profile" ? (
                                <motion.div
                                    key="profile-view"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    className="flex flex-col h-full"
                                >
                                    {/* Header Area */}
                                    <div className="pt-14 px-8 pb-6 flex justify-between items-start">
                                        <div className="flex items-center gap-6">
                                            <div className="h-20 w-20 rounded-full border-2 border-[#E6B23C]/20 bg-[#1A1208] overflow-hidden shadow-2xl">
                                                {(avatarPreviewUrl || userData.avatar) ? (
                                                    <img src={avatarPreviewUrl || userData.avatar} alt="Profile" className="w-full h-full object-cover" />
                                                ) : (
                                                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-[#1A1208] to-[#0D0A07]">
                                                        <User size={28} className="text-[#E6B23C]/40" />
                                                    </div>
                                                )}
                                            </div>
                                            <div>
                                                <h1 className="text-2xl font-bold text-[#F5E6D0] tracking-tight leading-tight">
                                                    {userData.name}
                                                </h1>
                                                <div className="flex items-center gap-2 mt-2 text-[#A08E70]">
                                                    <Calendar size={13} className="text-[#E6B23C]/60" />
                                                    <span className="text-[10px] font-bold uppercase tracking-[0.2em]">
                                                        {userData.joined}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <button
                                            onClick={onClose}
                                            className="h-10 w-10 flex items-center justify-center rounded-full bg-white/5 text-[#A08E70] hover:text-[#F5E6D0] hover:bg-white/10 transition-all mt-1"
                                        >
                                            <X size={20} />
                                        </button>
                                    </div>

                                    {/* Top Actions */}
                                    <div className="px-8 pb-5 flex gap-4">
                                        <button
                                            onClick={() => setView("settings")}
                                            className="flex-1 py-2.5 rounded-xl border border-[#E6B23C]/20 bg-[#E6B23C]/5 text-[#F5E6D0] hover:bg-[#E6B23C]/15 font-bold text-[10px] uppercase tracking-widest transition-all flex items-center justify-center gap-2"
                                        >
                                            <Settings size={14} />
                                            Profile Settings
                                        </button>
                                        <button
                                            onClick={handleSignOut}
                                            className="group flex items-center gap-0 hover:gap-2 px-4 py-2.5 rounded-xl border border-red-500/20 bg-red-500/5 text-red-400 hover:bg-red-500/15 font-bold text-[10px] uppercase tracking-widest transition-all"
                                        >
                                            <LogOut size={14} />
                                            <span className="max-w-0 overflow-hidden group-hover:max-w-[60px] transition-all duration-300">
                                                Logout
                                            </span>
                                        </button>
                                    </div>

                                    {/* Tabs */}
                                    <div className="flex px-4 bg-[#0D0A07]/40 border-b border-[#E6B23C]/10">
                                        {(["saved", "chats", "history"] as TabType[]).map((tab) => (
                                            <button
                                                key={tab}
                                                onClick={() => setActiveTab(tab)}
                                                className={`flex-1 py-5 flex flex-col items-center justify-center gap-1.5 transition-all relative ${activeTab === tab ? "text-[#E6B23C]" : "text-[#A08E70] hover:text-[#F5E6D0]"
                                                    }`}
                                            >
                                                <span className="transition-transform">
                                                    {tab === "saved" ? <Bookmark size={18} /> : tab === "chats" ? <MessageSquare size={18} /> : <HistoryIcon size={18} />}
                                                </span>
                                                <span className="text-[9px] font-bold uppercase tracking-[0.2em]">
                                                    {tab === "saved" ? "SAVED" : tab === "chats" ? "CHATS" : "HISTORY"}
                                                </span>
                                                {activeTab === tab && (
                                                    <motion.div
                                                        layoutId="activeTabSidebar"
                                                        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-0.5 bg-[#E6B23C] shadow-[0_0_10px_#E6B23C]"
                                                    />
                                                )}
                                            </button>
                                        ))}
                                    </div>

                                    {/* Tab Content - Scrollable */}
                                    <div className="flex-1 overflow-y-auto trending-scrollbar-hide">
                                        <motion.div
                                            key={activeTab}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0, y: -10 }}
                                            transition={{ duration: 0.2 }}
                                            className="divide-y divide-[#E6B23C]/5"
                                        >
                                            {isLoading ? (
                                                <div className="p-20 flex justify-center">
                                                    <div className="h-8 w-8 border-2 border-[#E6B23C]/20 border-t-[#E6B23C] rounded-full animate-spin" />
                                                </div>
                                            ) : (
                                                <>
                                                    {activeTab === "saved" && (
                                                        userData.favorites.length > 0 ? (
                                                            userData.favorites.map((item, i) => (
                                                                <div
                                                                    key={i}
                                                                    onClick={() => handleEntityClick(item.name, item.type)}
                                                                    className="p-4 flex gap-4 hover:bg-[#E6B23C]/[0.05] transition-colors group cursor-pointer"
                                                                >
                                                                    <div className="h-16 w-16 rounded-xl overflow-hidden border border-[#E6B23C]/10 shrink-0">
                                                                        <img src={item.image} alt={item.name} className="h-full w-full object-cover group-hover:scale-110 transition-transform duration-500" />
                                                                    </div>
                                                                    <div className="flex-1 py-0.5">
                                                                        <span className="text-[9px] font-bold tracking-widest text-[#E6B23C] uppercase block mb-0.5">{item.type}</span>
                                                                        <h3 className="text-[#F5E6D0] font-bold text-sm mb-0.5 group-hover:text-[#E6B23C] transition-colors">
                                                                            {cleanName(item.name)}
                                                                        </h3>
                                                                        <p className="text-[11px] text-[#A08E70] line-clamp-1">{item.description}</p>
                                                                    </div>
                                                                </div>
                                                            ))
                                                        ) : (
                                                            <div className="p-20 text-center">
                                                                <Bookmark size={32} className="mx-auto mb-4 text-[#A08E70]/20" />
                                                                <p className="text-xs text-[#A08E70]">No saved treasures yet.</p>
                                                            </div>
                                                        )
                                                    )}

                                                    {activeTab === "chats" && (
                                                        Object.keys(groupedChats).length > 0 ? (
                                                            <div className="px-8 py-6">
                                                                {Object.keys(groupedChats).map((entity) => {
                                                                    const chats = groupedChats[entity];
                                                                    return (
                                                                        <div key={entity} className="mb-2">
                                                                            <button
                                                                                onClick={() => setExpandedEntity(expandedEntity === entity ? null : entity)}
                                                                                className="w-full text-left group/header py-4"
                                                                            >
                                                                                <h3 className="text-[11px] font-bold uppercase tracking-[0.4em] text-[#E6B23C]/50 flex items-center gap-4 group-hover/header:text-[#E6B23C] transition-all">
                                                                                    <span className="min-w-fit">{entity}</span>
                                                                                    <div className="flex-1 h-[1px] bg-[#E6B23C]/10 group-hover/header:bg-[#E6B23C]/30" />
                                                                                    <span className="text-[9px] font-mono opacity-40 group-hover/header:opacity-100">
                                                                                        {chats.length} {chats.length === 1 ? 'RECORD' : 'RECORDS'}
                                                                                    </span>
                                                                                </h3>
                                                                            </button>

                                                                            <AnimatePresence>
                                                                                {expandedEntity === entity && (
                                                                                    <motion.div
                                                                                        initial={{ height: 0, opacity: 0 }}
                                                                                        animate={{ height: 'auto', opacity: 1 }}
                                                                                        exit={{ height: 0, opacity: 0 }}
                                                                                        className="overflow-hidden"
                                                                                    >
                                                                                        <div className="space-y-3 pb-6">
                                                                                            {[...chats]
                                                                                                .sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime())
                                                                                                .map((chat) => (
                                                                                                    <div
                                                                                                        key={chat.id}
                                                                                                        onClick={() => {
                                                                                                            router.push(`/chat?entity=${encodeURIComponent(chat.entity_name)}&type=${chat.entity_type || 'landmark'}&conv=${chat.id}`);
                                                                                                            onClose();
                                                                                                        }}
                                                                                                        className="group cursor-pointer border-b border-[#E6B23C]/5 py-3 px-2 rounded-xl hover:bg-[#E6B23C]/5 transition-all"
                                                                                                    >
                                                                                                        <div className="flex items-center justify-between mb-0.5">
                                                                                                            <h4 className="text-[13px] font-bold text-[#F5E6D0] group-hover:text-[#E6B23C] transition-colors truncate">
                                                                                                                {chat.title || "New Chat"}
                                                                                                            </h4>
                                                                                                            <span className="text-[9px] text-[#A08E70] font-mono whitespace-nowrap ml-4">
                                                                                                                {chat.created_at ? new Date(chat.created_at).toLocaleDateString() : ""}
                                                                                                            </span>
                                                                                                        </div>
                                                                                                    </div>
                                                                                                ))}
                                                                                        </div>
                                                                                    </motion.div>
                                                                                )}
                                                                            </AnimatePresence>
                                                                        </div>
                                                                    );
                                                                })}
                                                            </div>
                                                        ) : (
                                                            <div className="p-20 text-center">
                                                                <MessageSquare size={32} className="mx-auto mb-4 text-[#A08E70]/20" />
                                                                <p className="text-xs text-[#A08E70]">No echoes from the past.</p>
                                                            </div>
                                                        )
                                                    )}

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
                                                                        className={`px-4 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider transition-all duration-300 ${
                                                                            historyFilter === filter.id
                                                                                ? "bg-[#E6B23C] text-[#0D0A07] shadow-[0_2px_10px_rgba(230,178,60,0.3)]"
                                                                                : "bg-[#E6B23C]/5 border border-[#E6B23C]/10 text-[#A08E70] hover:text-[#F5E6D0] hover:border-[#E6B23C]/20"
                                                                        }`}
                                                                    >
                                                                        {filter.label}
                                                                    </button>
                                                                ))}
                                                            </div>

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
                                                                                 onClose();
                                                                             }
                                                                         }}
                                                                         className="p-4 flex gap-4 hover:bg-[#E6B23C]/[0.05] transition-colors group cursor-pointer"
                                                                     >
                                                                         <div className="h-14 w-14 rounded-lg overflow-hidden border border-[#E6B23C]/10 shrink-0 relative">
                                                                             <img src={entry.image} alt={entry.name} className="h-full w-full object-cover" />
                                                                             <div className="absolute inset-0 bg-[#0D0A07]/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                                                 <Search size={16} className="text-[#E6B23C]" />
                                                                             </div>
                                                                         </div>
                                                                         <div className="flex-1 py-0.5">
                                                                             <div className="flex items-center justify-between mb-0.5">
                                                                                 <span className="text-[9px] font-bold tracking-widest text-[#E6B23C] uppercase">
                                                                                     {entry.history_type === "recognition" ? entry.type : (language === "AR" ? "ترجمة" : language === "FR" ? "Traduction" : "Translation")}
                                                                                 </span>
                                                                                 <span className="text-[9px] text-[#A08E70]">{entry.date}</span>
                                                                             </div>
                                                                             <h3 className="text-[#F5E6D0] font-bold text-sm mb-0.5 group-hover:text-[#E6B23C] transition-colors">
                                                                                 {entry.history_type === "recognition" ? cleanName(entry.name) : (language === "AR" ? "ترجمة هيروغليفية" : language === "FR" ? "Traduction Hiéroglyphique" : "Hieroglyphic Translation")}
                                                                             </h3>
                                                                             
                                                                             {entry.history_type === "recognition" ? (
                                                                                 <p className="text-[11px] text-[#A08E70] line-clamp-1">
                                                                                     {entry.description}
                                                                                 </p>
                                                                             ) : (
                                                                                 <p className="text-[11px] text-[#A08E70] line-clamp-1 italic">
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
                                                </>
                                            )}
                                        </motion.div>
                                    </div>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="settings-view"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    className="flex flex-col h-full"
                                >
                                    {/* Header */}
                                    <div className="p-8 flex items-center justify-between border-b border-[#E6B23C]/10">
                                        <div className="flex items-center gap-4">
                                            <button
                                                onClick={() => setView("profile")}
                                                className="h-10 w-10 flex items-center justify-center rounded-full bg-white/5 text-[#A08E70] hover:text-[#F5E6D0] hover:bg-white/10 transition-all"
                                            >
                                                <ArrowLeft size={20} />
                                            </button>
                                            <div>
                                                <h2 className="text-xl font-bold text-[#F5E6D0]">Profile Settings</h2>
                                                <p className="text-[9px] text-[#E6B23C] font-bold tracking-widest uppercase opacity-60">
                                                    Manage account
                                                </p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={onClose}
                                            className="h-10 w-10 flex items-center justify-center rounded-full bg-white/5 text-[#A08E70] hover:text-[#F5E6D0] hover:bg-white/10 transition-all"
                                        >
                                            <X size={20} />
                                        </button>
                                    </div>

                                    {/* Form Content */}
                                    <div className="flex-1 overflow-y-auto p-8 space-y-10 trending-scrollbar-hide">
                                        {/* Profile Picture*/}
                                        <section>
                                            <div className="flex flex-col items-center justify-center gap-3">
                                                <div className="relative group">
                                                    <div className="h-32 w-32 rounded-full border-2 border-[#E6B23C]/20 p-1 group-hover:border-[#E6B23C]/50 transition-all shadow-2xl">
                                                        <div className="h-full w-full rounded-full bg-[#1A1208] flex items-center justify-center overflow-hidden relative">
                                                            {isUploading && (
                                                                <div className="absolute inset-0 bg-black/60 z-10 flex items-center justify-center">
                                                                    <div className="h-6 w-6 border-2 border-[#E6B23C]/20 border-t-[#E6B23C] rounded-full animate-spin" />
                                                                </div>
                                                            )}
                                                             {(avatarPreviewUrl || userData.avatar) ? (
                                                                 <img src={avatarPreviewUrl || userData.avatar} alt="Avatar" className="w-full h-full object-cover" />
                                                            ) : (
                                                                <User size={40} className="text-[#E6B23C]/30" />
                                                            )}
                                                        </div>
                                                    </div>
                                                    <input
                                                        type="file"
                                                        ref={fileInputRef}
                                                        onChange={handleAvatarUpload}
                                                        accept="image/*"
                                                        className="hidden"
                                                    />
                                                    <button
                                                        onClick={() => fileInputRef.current?.click()}
                                                        disabled={isUploading}
                                                        className="absolute bottom-1 right-1 h-10 w-10 rounded-full bg-[#E6B23C] flex items-center justify-center text-[#0D0A07] shadow-xl hover:scale-110 transition-transform disabled:opacity-50 disabled:hover:scale-100"
                                                    >
                                                        <Camera size={16} />
                                                    </button>
                                                </div>
                                                <AnimatePresence mode="wait">
                                                    {settingsError && (
                                                        <motion.div
                                                            initial={{ opacity: 0, y: -5 }}
                                                            animate={{ opacity: 1, y: 0 }}
                                                            exit={{ opacity: 0, y: -5 }}
                                                            className="text-xs text-red-500 font-bold text-center mt-2"
                                                        >
                                                            {settingsError}
                                                        </motion.div>
                                                    )}
                                                </AnimatePresence>
                                            </div>
                                        </section>

                                        {/* Account Info */}
                                        <section className="space-y-6">
                                            <h3 className="text-[10px] font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-4 opacity-60">Account Details</h3>

                                            <div className="space-y-5">
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div className="space-y-2">
                                                        <label className="text-[9px] font-bold text-[#A08E70] uppercase tracking-widest px-1">First Name</label>
                                                        <div className="relative group">
                                                            <User size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                                            <input
                                                                id="firstName"
                                                                type="text"
                                                                value={firstName}
                                                                onChange={(e) => setFirstName(e.target.value)}
                                                                className="w-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 rounded-xl py-3.5 pl-12 pr-12 text-sm text-[#F5E6D0] focus:border-[#E6B23C]/60 focus:shadow-[0_0_25px_rgba(230,178,60,0.15)] outline-none transition-all shadow-[0_0_15px_rgba(230,178,60,0.05)]"
                                                                placeholder="First name"
                                                            />
                                                            <button
                                                                onClick={() => document.getElementById("firstName")?.focus()}
                                                                className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A08E70]/30 group-hover:text-[#E6B23C] transition-colors"
                                                            >
                                                                <Pencil size={13} />
                                                            </button>
                                                        </div>
                                                    </div>
                                                    <div className="space-y-2">
                                                        <label className="text-[9px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Last Name</label>
                                                        <div className="relative group">
                                                            <User size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                                            <input
                                                                id="lastName"
                                                                type="text"
                                                                value={lastName}
                                                                onChange={(e) => setLastName(e.target.value)}
                                                                className="w-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 rounded-xl py-3.5 pl-12 pr-12 text-sm text-[#F5E6D0] focus:border-[#E6B23C]/60 focus:shadow-[0_0_25px_rgba(230,178,60,0.15)] outline-none transition-all shadow-[0_0_15px_rgba(230,178,60,0.05)]"
                                                                placeholder="Last name"
                                                            />
                                                            <button
                                                                onClick={() => document.getElementById("lastName")?.focus()}
                                                                className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A08E70]/30 group-hover:text-[#E6B23C] transition-colors"
                                                            >
                                                                <Pencil size={13} />
                                                            </button>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="space-y-2">
                                                    <label className="text-[9px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Username</label>
                                                    <div className="relative group">
                                                        <AtSign size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                                        <input
                                                            id="username"
                                                            type="text"
                                                            value={username}
                                                            onChange={(e) => setUsername(e.target.value)}
                                                            className="w-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 rounded-xl py-3.5 pl-12 pr-12 text-sm text-[#F5E6D0] focus:border-[#E6B23C]/60 focus:shadow-[0_0_25px_rgba(230,178,60,0.15)] outline-none transition-all shadow-[0_0_15px_rgba(230,178,60,0.05)]"
                                                            placeholder="Username"
                                                        />
                                                        <button
                                                            onClick={() => document.getElementById("username")?.focus()}
                                                            className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A08E70]/30 group-hover:text-[#E6B23C] transition-colors"
                                                        >
                                                            <Pencil size={13} />
                                                        </button>
                                                    </div>
                                                </div>

                                                <div className="space-y-2">
                                                    <label className="text-[9px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Email</label>
                                                    <div className="relative">
                                                        <Mail size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/20" />
                                                        <input
                                                            type="email"
                                                            value={email}
                                                            disabled
                                                            className="w-full bg-transparent border border-[#E6B23C]/5 rounded-xl py-3.5 pl-12 pr-4 text-sm text-[#A08E70]/40 outline-none cursor-not-allowed"
                                                        />
                                                    </div>
                                                </div>

                                               <div className="mt-4 flex flex-col items-center gap-3">
                                                   <Button
                                                       onClick={handleSaveSettings}
                                                       disabled={isSaving}
                                                       className="w-full bg-[#E6B23C] text-[#0D0A07] hover:scale-[1.02] active:scale-[0.98] font-bold py-6 rounded-xl transition-all shadow-[0_0_30px_rgba(230,178,60,0.15)] hover:shadow-[0_0_40px_rgba(230,178,60,0.3)] uppercase tracking-widest text-[12px]"
                                                   >
                                                       {isSaving ? "SAVING..." : "SAVE CHANGES"}
                                                   </Button>
                                                   <AnimatePresence mode="wait">
                                                       {showSuccess ? (
                                                           <motion.div
                                                               key="success"
                                                               initial={{ opacity: 0, y: -5 }}
                                                               animate={{ opacity: 1, y: 0 }}
                                                               exit={{ opacity: 0, y: -5 }}
                                                               className="text-[10px] font-bold text-[#A08E70] uppercase tracking-[0.2em]"
                                                           >
                                                               Changes saved
                                                            </motion.div>
                                                        ) : showNoChanges ? (
                                                            <motion.div
                                                                key="no-changes"
                                                                initial={{ opacity: 0, y: -5 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                exit={{ opacity: 0, y: -5 }}
                                                                className="text-[10px] font-bold text-[#A08E70] uppercase tracking-[0.2em]"
                                                            >
                                                                No changes detected
                                                            </motion.div>
                                                        ) : null}
                                                    </AnimatePresence>
                                               </div>
                                                {isEmailUser && (
                                                    <div className="pt-6 mt-6 border-t border-[#E6B23C]/10 space-y-4">
                                                        <h3 className="text-[10px] font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-4 opacity-60">Change Password</h3>

                                                        <div className="space-y-2">
                                                            <label className="text-[9px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Current Password</label>
                                                            <div className="relative group">
                                                                <Lock size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                                                <input
                                                                    type={showCurrentPassword ? "text" : "password"}
                                                                    value={currentPassword}
                                                                    onChange={(e) => setCurrentPassword(e.target.value)}
                                                                    autoComplete="new-password"
                                                                    className="w-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 rounded-xl py-3.5 pl-12 pr-12 text-sm text-[#F5E6D0] focus:border-[#E6B23C]/60 focus:shadow-[0_0_25px_rgba(230,178,60,0.15)] outline-none transition-all shadow-[0_0_15px_rgba(230,178,60,0.05)]"
                                                                    placeholder="Enter password"
                                                                />
                                                                <button
                                                                    type="button"
                                                                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                                                                    className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A08E70]/30 hover:text-[#E6B23C] transition-colors"
                                                                >
                                                                    {showCurrentPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                                                                </button>
                                                            </div>
                                                        </div>

                                                        <div className="space-y-2">
                                                            <label className="text-[9px] font-bold text-[#A08E70] uppercase tracking-widest px-1">New Password</label>
                                                            <div className="relative group">
                                                                <Lock size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                                                <input
                                                                    type={showNewPassword ? "text" : "password"}
                                                                    value={newPassword}
                                                                    onChange={(e) => setNewPassword(e.target.value)}
                                                                    autoComplete="new-password"
                                                                    className="w-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 rounded-xl py-3.5 pl-12 pr-12 text-sm text-[#F5E6D0] focus:border-[#E6B23C]/60 focus:shadow-[0_0_25px_rgba(230,178,60,0.15)] outline-none transition-all shadow-[0_0_15px_rgba(230,178,60,0.05)]"
                                                                    placeholder="Enter password"
                                                                />
                                                                <button
                                                                    type="button"
                                                                    onClick={() => setShowNewPassword(!showNewPassword)}
                                                                    className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A08E70]/30 hover:text-[#E6B23C] transition-colors"
                                                                >
                                                                    {showNewPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                                                                </button>
                                                            </div>
                                                        </div>

                                                        <div className="pt-2">
                                                            <Button
                                                                onClick={handleUpdatePassword}
                                                                disabled={isSavingPassword}
                                                                className="w-full bg-[#E6B23C] text-[#0D0A07] hover:scale-[1.02] active:scale-[0.98] font-bold py-6 rounded-xl transition-all shadow-[0_0_30px_rgba(230,178,60,0.15)] hover:shadow-[0_0_40px_rgba(230,178,60,0.3)] uppercase tracking-widest text-[12px]"
                                                            >
                                                                {isSavingPassword ? "SAVING..." : "UPDATE PASSWORD"}
                                                            </Button>
                                                            <AnimatePresence mode="wait">
                                                                {showPasswordSuccess ? (
                                                                    <motion.div
                                                                        key="pw-success"
                                                                        initial={{ opacity: 0, y: -5 }}
                                                                        animate={{ opacity: 1, y: 0 }}
                                                                        exit={{ opacity: 0, y: -5 }}
                                                                        className="text-[10px] font-bold text-[#E6B23C] uppercase tracking-[0.2em] text-center mt-2"
                                                                    >
                                                                        Password updated successfully
                                                                    </motion.div>
                                                                ) : passwordError ? (
                                                                    <motion.div
                                                                        key="pw-error"
                                                                        initial={{ opacity: 0, y: -5 }}
                                                                        animate={{ opacity: 1, y: 0 }}
                                                                        exit={{ opacity: 0, y: -5 }}
                                                                        className="text-xs text-red-500 font-bold text-center mt-2"
                                                                    >
                                                                        {passwordError}
                                                                    </motion.div>
                                                                ) : null}
                                                            </AnimatePresence>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>

                                        </section>

                                        {/* Danger Zone */}
                                        <section className="pt-8 border-t border-[#E6B23C]/10">
                                            <button className="w-full flex items-center justify-center gap-3 p-4 rounded-xl border border-red-500/10 bg-red-500/5 text-red-500 hover:bg-red-500/10 transition-all font-bold text-[10px] uppercase tracking-widest">
                                                <Trash2 size={16} />
                                                Delete account
                                            </button>
                                        </section>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
