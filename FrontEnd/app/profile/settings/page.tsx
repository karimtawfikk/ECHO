"use client";

import PageShell from "../../../components/layout/PageShell";
import { motion } from "framer-motion";
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
    
    // Form States
    const [fullName, setFullName] = useState("");
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [isSaving, setIsSaving] = useState(false);

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
                
                {/* ── SETTINGS CARD (Centered & Curved) ────────────────── */}
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
                        
                        {/* Profile Media */}
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

                        {/* Account Info */}
                        <section className="space-y-8">
                            <h3 className="text-xs font-bold tracking-[0.2em] text-[#E6B23C] uppercase mb-4 opacity-60">Account Details</h3>
                            
                            <div className="grid gap-6">
                                {/* Full Name */}
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

                                {/* Username */}
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

                                {/* Email */}
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

                        {/* Danger Zone */}
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
                                
                                <button className="flex items-center justify-center gap-3 p-4 rounded-2xl border border-red-500/10 bg-red-500/5 text-red-500 hover:bg-red-500/10 transition-all font-bold text-xs uppercase tracking-widest">
                                    <Trash2 size={18} />
                                    Delete account
                                </button>
                            </div>
                        </section>

                    </div>
                </motion.div>
            </div>
        </PageShell>
    );
}
