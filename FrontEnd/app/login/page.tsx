"use client";

import { useState, useEffect } from "react";
import { createClient } from "../../lib/supabase/client";
import { motion, AnimatePresence } from "framer-motion";
import { Input } from "../../components/ui/input";
import { Button } from "../../components/ui/button";
import { Label } from "../../components/ui/label";
import { Loader2, Mail, Lock, LogIn, UserPlus, User, ShieldCheck, Sparkles, ArrowRight, Eye, EyeOff } from "lucide-react";
import PageShell from "../../components/layout/PageShell";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [username, setUsername] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSignUp, setIsSignUp] = useState(false);
  const [mounted, setMounted] = useState(false);
  const supabase = createClient();

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError("Please enter a valid email address with a domain (e.g., name@domain.com).");
      setLoading(false);
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      setLoading(false);
      return;
    }

    const { data, error } = isSignUp
      ? await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: `${firstName} ${lastName}`.trim(),
            user_name: username,
          }
        }
      })
      : await supabase.auth.signInWithPassword({ email, password });

    if (error) {
      setError(error.message);
      setLoading(false);
    } else if (data.session) {
      // Instant login (works if 'Confirm email' is OFF in Supabase)
      window.location.href = "/";
    } else if (isSignUp) {
      setError("Check your email for the confirmation link!");
      setLoading(false);
    } else {
      window.location.href = "/";
    }
    setLoading(false);
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });
    if (error) setError(error.message);
    setLoading(false);
  };

  return (
    <PageShell fullScreen minimal>
      <style jsx global>{`
        input:-webkit-autofill,
        input:-webkit-autofill:hover, 
        input:-webkit-autofill:focus, 
        input:-webkit-autofill:active{
            -webkit-box-shadow: 0 0 0 30px #0D0A07 inset !important;
            -webkit-text-fill-color: #F5E6D0 !important;
            transition: background-color 5000s ease-in-out 0s;
        }
      `}</style>
      <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden">
        {/* Background Elements */}
        <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-[#0D0A07]" />
          <div className="egyptian-pattern" />
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#E6B23C]/5 rounded-full blur-[120px]" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-[#E6B23C]/10 rounded-full blur-[150px]" />
          <div className="absolute inset-0 bg-[url('/bg-pattern.png')] opacity-10 mix-blend-overlay" />

          {/* Floating Particles - Only render on client to avoid hydration mismatch */}
          {mounted && [...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-[#E6B23C]/30 rounded-full"
              initial={{
                x: Math.random() * 100 + "%",
                y: Math.random() * 100 + "%",
                opacity: Math.random()
              }}
              animate={{
                y: [null, "-20%"],
                opacity: [0, 1, 0]
              }}
              transition={{
                duration: 5 + Math.random() * 10,
                repeat: Infinity,
                delay: Math.random() * 5
              }}
            />
          ))}
        </div>

        <div className="relative z-10 w-full max-w-6xl px-4 flex flex-col md:flex-row items-center gap-16 py-12">

          {/* Left Side: Branding / Story */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex-1 text-center md:text-left space-y-8"
          >
            <h1 className="text-6xl md:text-8xl font-heading font-bold text-white leading-[1.1]">
              Unveil the <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#E6B23C] to-[#B48B2D]">Past.</span>
            </h1>
            <p className="text-[#A08E70] text-xl md:text-2xl max-w-xl leading-relaxed">
              Step into the digital archives of Ancient Egypt. Every capture has an origin, and every discovery begins here.
            </p>
          </motion.div>

          {/* Right Side: The Form */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-[480px]"
          >
            <div className="relative p-1 rounded-[32px] bg-gradient-to-b from-[#E6B23C]/20 to-transparent backdrop-blur-xl shadow-2xl">
              <div className="bg-[#0D0A07]/80 backdrop-blur-3xl rounded-[30px] p-8 md:p-10 border border-white/5">
                <div className="text-center mb-6">
                  <h2 className="text-3xl font-bold text-white mb-2">
                    {isSignUp ? "Create Your Identity" : "Welcome Back"}
                  </h2>
                  <p className="text-[#A08E70] text-sm">
                    {isSignUp ? "Join the explorers of the Nile" : "Continue your journey through time"}
                  </p>
                </div>

                <div className="space-y-6">
                  <form onSubmit={handleEmailAuth} className="space-y-4">
                    <AnimatePresence mode="popLayout">
                      {isSignUp && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="space-y-4 overflow-hidden"
                        >
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1.5">
                              <Label htmlFor="firstName" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">First Name</Label>
                              <div className="relative">
                                <Input
                                  id="firstName"
                                  type="text"
                                  placeholder="Enter first name"
                                  className="h-11 px-4 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                                  value={firstName}
                                  onChange={(e) => setFirstName(e.target.value)}
                                  required
                                />
                              </div>
                            </div>
                            <div className="space-y-1.5">
                              <Label htmlFor="lastName" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">Last Name</Label>
                              <div className="relative">
                                <Input
                                  id="lastName"
                                  type="text"
                                  placeholder="Enter last name"
                                  className="h-11 px-4 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                                  value={lastName}
                                  onChange={(e) => setLastName(e.target.value)}
                                  required
                                />
                              </div>
                            </div>
                          </div>
                          <div className="space-y-1.5">
                            <Label htmlFor="username" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">Username</Label>
                            <div className="relative">
                              <Input
                                id="username"
                                type="text"
                                placeholder="Choose a username"
                                className="h-11 px-4 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                              />
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <div className="space-y-1.5">
                      <Label htmlFor="email" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">Email Address</Label>
                      <div className="relative">
                        <Mail className="absolute left-4 top-3 h-4 w-4 text-[#E6B23C]/40" />
                        <Input
                          id="email"
                          type="email"
                          placeholder="yourname@example.com"
                          className="h-11 pl-12 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          required
                        />
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <Label htmlFor="password" className="text-[#A08E70] text-[10px] uppercase font-bold tracking-widest pl-1">Password</Label>
                      <div className="relative">
                        <Lock className="absolute left-4 top-3 h-4 w-4 text-[#E6B23C]/40" />
                        <Input
                          id="password"
                          type={showPassword ? "text" : "password"}
                          placeholder={isSignUp ? "Minimum 8 characters" : "••••••••"}
                          className="h-11 pl-12 pr-12 bg-white/5 border-white/10 text-white rounded-xl focus-visible:ring-[#E6B23C]/30 focus:border-[#E6B23C]/50 transition-all"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-4 top-2.5 text-[#A08E70]/40 hover:text-[#E6B23C] transition-colors"
                          title={showPassword ? "Hide" : "Reveal"}
                        >
                          {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                      </div>
                    </div>

                    {error && (
                      <motion.p
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-red-400 text-[11px] font-medium text-center"
                      >
                        {error}
                      </motion.p>
                    )}

                    <Button
                      type="submit"
                      disabled={loading}
                      className="w-full h-12 mt-2 rounded-xl bg-transparent border-2 border-[#E6B23C] text-[#E6B23C] font-bold uppercase tracking-widest hover:bg-[#E6B23C]/10 hover:shadow-[0_0_30px_rgba(230,178,60,0.2)] transition-all flex items-center justify-center gap-2"
                    >
                      {loading ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>{isSignUp ? "Sign Up" : "Sign In"}</>
                      )}
                    </Button>
                  </form>

                  <div className="flex items-center gap-4 py-1">
                    <div className="flex-1 border-t border-white/5"></div>
                    <span className="text-[10px] uppercase font-bold tracking-[0.2em] text-[#A08E70]/40 italic">
                      OR
                    </span>
                    <div className="flex-1 border-t border-white/5"></div>
                  </div>

                  {/* Google Login */}
                  <Button
                    onClick={handleGoogleLogin}
                    variant="outline"
                    className="w-full h-11 rounded-xl bg-white/5 border-white/10 text-white hover:bg-white/10 hover:border-[#E6B23C]/30 transition-all flex items-center justify-center gap-3 text-sm group"
                  >
                    <svg className="w-4 h-4 transition-transform group-hover:scale-110" viewBox="0 0 24 24">
                      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.24.81-.6z" fill="#FBBC05" />
                      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                    </svg>
                    Continue with Google
                  </Button>

                  <div className="text-center">
                    <div className="text-[#A08E70] text-sm flex items-center justify-center gap-1.5 mx-auto">
                      <span>{isSignUp ? "Already have an account?" : "First time?"}</span>
                      <button
                        type="button"
                        onClick={() => setIsSignUp(!isSignUp)}
                        className="text-[#E6B23C] hover:text-[#FFD369] font-bold transition-colors"
                      >
                        {isSignUp ? "Sign in" : "Sign up for free"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </PageShell>
  );
}
