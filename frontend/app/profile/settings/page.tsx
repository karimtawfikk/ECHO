"use client";

import PageShell from "../../../components/layout/PageShell";
import { motion, AnimatePresence } from "framer-motion";
import {
    ArrowLeft,
    LogOut,
    Trash2,
    User,
    Mail,
    AtSign,
    ShieldCheck,
    Camera,
    ChevronRight
} from "lucide-react";
import { Button } from "../../../components/ui/button";
import Link from "next/link";
import { useState, useEffect, useMemo } from "react";
import { createClient } from "../../../lib/supabase/client";

export default function ProfileSettingsPage() {
    const supabase = useMemo(() => createClient(), []);
    const [user, setUser] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);

    const [fullName, setFullName] = useState("");
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [isSaving, setIsSaving] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

    useEffect(() => {
        const getUser = async () => {
            try {
                const { data: { user }, error } = await supabase.auth.getUser();
                if (error) throw error;
                if (user) {
                    setUser(user);
                    setFullName(user.user_metadata?.full_name || "");
                    setUsername(user.user_metadata?.user_name || "");
                    setEmail(user.email || "");
                }
            } catch (err) {
                console.error("Error fetching user:", err);
            } finally {
                setIsLoading(false);
            }
        };
        getUser();
    }, [supabase]);

    const handleSignOut = async () => {
        await supabase.auth.signOut();
        window.location.href = "/";
    };

    const handleSave = async () => {
        setIsSaving(true);
        try {
            const { error } = await supabase.auth.updateUser({
                data: { full_name: fullName, user_name: username }
            });
            if (error) throw error;
            alert("Settings saved successfully!");
        } catch (err) {
            console.error("Error updating user:", err);
            alert("Failed to save settings.");
        } finally {
            setIsSaving(false);
        }
    };

    const handleDeleteAccount = () => {
        if (!user) return;
        setShowDeleteConfirm(true);
    };

    const confirmDeleteAccount = async () => {
        if (!user) return;

        setShowDeleteConfirm(false);
        setIsSaving(true);
        try {
            const baseUrl = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";

            try {
                const { data: existingFiles } = await supabase.storage.from('avatars').list('', { search: user.id });
                if (existingFiles && existingFiles.length > 0) {
                    const filesToRemove = existingFiles.map((x) => x.name);
                    await supabase.storage.from('avatars').remove(filesToRemove);
                }
            } catch (err) {
                console.error("Error deleting avatars:", err);
            }

            const res = await fetch(`${baseUrl}/api/v1/assets/delete-account/${user.id}`, {
                method: 'DELETE',
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => null);
                throw new Error(errData?.detail || "Failed to delete account from backend.");
            }

            await supabase.auth.signOut();
            window.location.href = "/login";
        } catch (err: any) {
            console.error("Error deleting account:", err);
            alert(`Failed to delete account: ${err.message || "Unknown error"}`);
            setIsSaving(false);
        }
    };

    if (isLoading) {
        return (
            <PageShell>
                <div className="max-w-2xl mx-auto pt-32 flex justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#E6B23C]" />
                </div>
            </PageShell>
        );
    }

    return (
        <PageShell>
            <div className="max-w-2xl mx-auto pt-10 pb-20 px-4">

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="relative rounded-[3rem] bg-[#0D0A07]/80 backdrop-blur-xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)]"
                >
                    {/* Header Navigation */}
                    <div className="px-8 py-6 flex items-center gap-6 border-b border-[#E6B23C]/10 bg-[#0D0A07]/40">
                        <Link href="/profile" className="p-2 hover:bg-[#E6B23C]/10 rounded-full transition-colors">
                            <ArrowLeft size={20} className="text-[#F5E6D0]" />
                        </Link>
                        <div>
                            <h2 className="text-xl font-bold text-[#F5E6D0] leading-tight">Settings</h2>
                            <p className="text-[10px] text-[#E6B23C] font-bold tracking-widest uppercase opacity-60">
                                Manage your credentials
                            </p>
                        </div>
                    </div>

                    <div className="p-8 md:p-12 space-y-12">

                        <section>
                            <h3 className="text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-8 opacity-60">Profile Media</h3>
                            <div className="flex items-center gap-8">
                                <div className="relative group">
                                    <div className="h-28 w-28 rounded-full border-2 border-[#E6B23C]/20 p-1 group-hover:border-[#E6B23C]/50 transition-all">
                                        <div className="h-full w-full rounded-full bg-[#1A1208] flex items-center justify-center overflow-hidden">
                                            {user?.user_metadata?.avatar_url ? (
                                                <img src={user.user_metadata.avatar_url} alt="Avatar" className="w-full h-full object-cover" />
                                            ) : (
                                                <User size={40} className="text-[#E6B23C]/30" />
                                            )}
                                        </div>
                                    </div>
                                    <button className="absolute bottom-0 right-0 h-9 w-9 rounded-full bg-[#E6B23C] flex items-center justify-center text-[#0D0A07] shadow-xl hover:scale-110 transition-transform">
                                        <Camera size={16} />
                                    </button>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-base font-bold text-[#F5E6D0]">Profile Picture</p>
                                    <p className="text-xs text-[#A08E70]">JPG, GIF or PNG. Max size 2MB.</p>
                                </div>
                            </div>
                        </section>

                        <section className="space-y-8">
                            <h3 className="text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-4 opacity-60">Account Details</h3>

                            <div className="grid gap-6">
                                <div className="space-y-2">
                                    <label className="text-[10px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Full Name</label>
                                    <div className="relative">
                                        <User size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                        <input
                                            type="text"
                                            value={fullName}
                                            onChange={(e) => setFullName(e.target.value)}
                                            className="w-full bg-[#0D0A07]/50 border border-[#E6B23C]/10 rounded-2xl py-4 pl-12 pr-4 text-[#F5E6D0] focus:border-[#E6B23C]/40 outline-none transition-all"
                                            placeholder="Enter your full name"
                                        />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-[10px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Username</label>
                                    <div className="relative">
                                        <AtSign size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" />
                                        <input
                                            type="text"
                                            value={username}
                                            onChange={(e) => setUsername(e.target.value)}
                                            className="w-full bg-[#0D0A07]/50 border border-[#E6B23C]/10 rounded-2xl py-4 pl-12 pr-4 text-[#F5E6D0] focus:border-[#E6B23C]/40 outline-none transition-all"
                                            placeholder="choose_a_nickname"
                                        />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-[10px] font-bold text-[#A08E70] uppercase tracking-widest px-1">Email Address</label>
                                    <div className="relative">
                                        <Mail size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#E6B23C]/20" />
                                        <input
                                            type="email"
                                            value={email}
                                            className="w-full bg-transparent border border-[#E6B23C]/5 rounded-2xl py-4 pl-12 pr-4 text-[#A08E70]/50 outline-none cursor-not-allowed"
                                            disabled
                                        />
                                        <div className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] font-bold text-[#E6B23C]/40 uppercase tracking-tighter">Verified</div>
                                    </div>
                                </div>
                            </div>

                            <Button
                                onClick={handleSave}
                                disabled={isSaving}
                                className="w-full bg-[#E6B23C] text-[#0D0A07] hover:bg-[#F5E6D0] font-bold py-7 rounded-2xl transition-all shadow-[0_0_30px_rgba(230,178,60,0.15)] mt-4 uppercase tracking-widest text-xs"
                            >
                                {isSaving ? "Saving changes..." : "Save settings"}
                            </Button>
                        </section>

                        <section className="space-y-6 pt-12 border-t border-[#E6B23C]/10">
                            <h3 className="text-xs font-bold tracking-[0.2em] text-red-500 uppercase mb-4 opacity-80">Danger Zone</h3>

                            <div className="grid sm:grid-cols-2 gap-4">
                                <button
                                    onClick={handleSignOut}
                                    className="flex items-center justify-center gap-3 p-4 rounded-2xl border border-red-500/10 bg-red-500/5 text-red-500 hover:bg-red-500/10 transition-all font-bold text-xs uppercase tracking-widest"
                                >
                                    <LogOut size={18} />
                                    Logout
                                </button>

                                <button
                                    onClick={handleDeleteAccount}
                                    disabled={isSaving}
                                    className="flex items-center justify-center gap-3 p-4 rounded-2xl border border-red-500/10 bg-red-500/5 text-red-500 hover:bg-red-500/10 transition-all font-bold text-xs uppercase tracking-widest disabled:opacity-50"
                                >
                                    <Trash2 size={18} />
                                    {isSaving ? "Deleting..." : "Delete account"}
                                </button>
                            </div>
                        </section>

                    </div>
                </motion.div>

                <AnimatePresence>
                    {showDeleteConfirm && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="fixed inset-0 z-[120] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
                        >
                            <motion.div
                                initial={{ scale: 0.95, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.95, opacity: 0 }}
                                className="bg-[#1A1208] border border-red-500/20 rounded-2xl p-6 w-full max-w-sm shadow-[0_0_50px_rgba(239,68,68,0.15)] relative overflow-hidden"
                            >
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500/0 via-red-500 to-red-500/0 opacity-50" />
                                <div className="flex items-center gap-3 mb-4 text-red-500">
                                    <Trash2 size={24} />
                                    <h3 className="font-bold text-lg">Delete Account</h3>
                                </div>
                                <p className="text-sm text-[#A08E70] mb-8 leading-relaxed">
                                    Are you sure you want to delete your account? This action cannot be undone and will permanently delete all your data.
                                </p>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setShowDeleteConfirm(false)}
                                        disabled={isSaving}
                                        className="flex-1 py-3.5 rounded-xl border border-[#E6B23C]/20 text-[#F5E6D0] hover:bg-[#E6B23C]/10 transition-all font-bold text-[10px] uppercase tracking-[0.2em] disabled:opacity-50"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={confirmDeleteAccount}
                                        disabled={isSaving}
                                        className="flex-1 py-3.5 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 transition-all font-bold text-[10px] uppercase tracking-[0.2em] disabled:opacity-50"
                                    >
                                        {isSaving ? "Deleting..." : "Delete"}
                                    </button>
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </PageShell>
    );
}
