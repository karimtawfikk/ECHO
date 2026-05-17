"use client";

import { useState, useEffect } from "react";
import { createClient } from "../../../lib/supabase/client";
import { motion } from "framer-motion";
import { Input } from "../../../components/ui/input";
import { Button } from "../../../components/ui/button";
import { Label } from "../../../components/ui/label";
import { Loader2, Lock, ShieldCheck, Eye, EyeOff, CheckCircle2 } from "lucide-react";
import PageShell from "../../../components/layout/PageShell";

export default function ResetPasswordPage() {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [mounted, setMounted] = useState(false);
  const supabase = createClient();

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleUpdatePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      setLoading(false);
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }

    const { error } = await supabase.auth.updateUser({
      password: password,
    });

    if (error) {
      setError(error.message);
    } else {
      setSuccess(true);
      // Sign out to ensure they have to log in with the new password
      await supabase.auth.signOut();
      // Wait a bit then redirect to login
      setTimeout(() => {
        window.location.href = "/login";
      }, 3000);
    }
    setLoading(false);
  };

  if (!mounted) return null;

  return (
    <PageShell fullScreen minimal>
      <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden">
        {/* Background Elements */}
        <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-[#0D0A07]" />
          <div className="egyptian-pattern" />
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#E6B23C]/5 rounded-full blur-[120px]" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-[#E6B23C]/10 rounded-full blur-[150px]" />
        </div>

        <div className="relative z-10 w-full max-w-[480px] px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative p-1 rounded-[32px] bg-gradient-to-b from-[#E6B23C]/20 to-transparent backdrop-blur-xl shadow-2xl"
          >
            <div className="bg-[#0D0A07]/80 backdrop-blur-3xl rounded-[30px] p-8 md:p-10 border border-white/5">
              <div className="text-center mb-8">
                <div className="flex justify-center mb-4">
                  <div className="w-16 h-16 bg-[#E6B23C]/10 rounded-2xl flex items-center justify-center border border-[#E6B23C]/20 rotate-12 group-hover:rotate-0 transition-transform duration-500">
                    <ShieldCheck size={32} className="text-[#E6B23C]" />
                  </div>
                </div>
                <h2 className="text-3xl font-bold text-white mb-2">New Password</h2>
              </div>

              <form onSubmit={handleUpdatePassword} className="space-y-6">
                <div className="space-y-1.5">
                  <Label htmlFor="password" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">New Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-4 top-3 h-4 w-4 text-[#E6B23C]/40" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Minimum 8 characters"
                      className="h-11 pl-12 pr-12 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      disabled={success}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-2.5 text-[#A08E70]/40 hover:text-[#E6B23C] transition-colors"
                    >
                      {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="confirmPassword" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">Confirm Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-4 top-3 h-4 w-4 text-[#E6B23C]/40" />
                    <Input
                      id="confirmPassword"
                      type={showPassword ? "text" : "password"}
                      placeholder="Repeat your new password"
                      className="h-11 pl-12 pr-12 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                      disabled={success}
                    />
                  </div>
                </div>

                {error && (
                  <p className="text-red-400 text-[11px] font-medium text-center">
                    {error}
                  </p>
                )}

                <Button
                  type="submit"
                  disabled={loading || success}
                  className="w-full h-12 rounded-xl bg-transparent border-2 border-[#E6B23C] text-[#E6B23C] font-bold uppercase tracking-widest hover:bg-[#E6B23C]/10 hover:shadow-[0_0_30px_rgba(230,178,60,0.2)] transition-all flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : success ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    "Save Password"
                  )}
                </Button>

                {success && (
                  <motion.div 
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center justify-center gap-2 text-[#E6B23C] text-sm font-medium bg-[#E6B23C]/10 py-2 rounded-lg border border-[#E6B23C]/20"
                  >
                    <CheckCircle2 size={16} />
                    <span>Password updated! Redirecting...</span>
                  </motion.div>
                )}
              </form>
            </div>
          </motion.div>
        </div>
      </div>
    </PageShell>
  );
}
