"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { useLanguage } from "../../context/LanguageContext";
import { Camera, Languages, Trash2, Upload, BookOpen, Search, X, Cpu, Loader2, Image as ImageIcon } from "lucide-react";
import { api, API_BASE_URL } from "../../lib/services/api";
import { createClient } from "../../lib/supabase/client";

type TranslateResponse = {
  translation: string;
  symbols?: any[];
  num_symbols_detected?: number;
  num_clusters?: number;
  annotated_image_base64?: string;
};

export default function TranslatePage() {
  const { t, isRTL } = useLanguage();
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"translation">("translation");
  const [result, setResult] = useState<TranslateResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const stepIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const pickFile = () => fileInputRef.current?.click();
  const openCamera = () => cameraInputRef.current?.click();

  useEffect(() => {
    const loadFromStorage = () => {
      try {
        const raw = sessionStorage.getItem("echo_translation_history_result");
        if (raw) {
          const payload = JSON.parse(raw);
          setResult({
            translation: payload.translation,
          });
          setPreviewUrl(payload.imageUrl);
          sessionStorage.removeItem("echo_translation_history_result");
          setCurrentStep(4);
          setFile(null);
          setFileName("");
          setIsLoading(false);
          setTimeout(() => {
            document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 200);
        }
      } catch (e) {
        console.error("Error reading translation history result:", e);
      }
    };

    loadFromStorage();
    window.addEventListener("echo_load_translation", loadFromStorage);
    return () => window.removeEventListener("echo_load_translation", loadFromStorage);
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl && !previewUrl.startsWith("http")) URL.revokeObjectURL(previewUrl);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, [previewUrl]);

  const handleDecipher = async () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();

    setIsLoading(true);
    setResult(null);
    setCurrentStep(0.1);

    setTimeout(() => {
      if (window.innerWidth < 1024) {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 200);

    const formData = new FormData();
    formData.append("image", file as File);

    try {
      const baseUrl = API_BASE_URL.replace(/\/api\/v1\/?$/, "");
      const response = await fetch(`${baseUrl}/api/v1/hieroglyphs/translate/stream`, {
        method: "POST",
        body: formData,
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error("Translation failed");

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error("No reader");

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;

          try {
            const message = JSON.parse(line.substring(6));

            if (message.type === "progress") {
              setCurrentStep(message.step);
            } else if (message.type === "result") {
              const data = message.data;
              setCurrentStep(4);

              try {
                const supabase = createClient();
                const { data: { user } } = await supabase.auth.getUser();

                const isSuccess = data.translation_text &&
                  data.translation_text.trim() !== "" &&
                  !data.translation_text.toLowerCase().includes("failed") &&
                  !data.translation_text.toLowerCase().includes("no hieroglyphs");

                if (user && file && isSuccess) {
                  const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";
                  const uploadData = new FormData();
                  uploadData.append("file", file);
                  uploadData.append("user_id", user.id);
                  uploadData.append("task_type", "hieroglyphics");

                  const uploadRes = await fetch(`${API_BASE}/api/v1/assets/upload/history`, {
                    method: "POST",
                    body: uploadData,
                  });

                  if (uploadRes.ok) {
                    const { key } = await uploadRes.json();

                    await supabase.from('translation_history').insert({
                      user_id: user.id,
                      image_path: key,
                      translation: data.translation_text
                    });
                  }
                }
              } catch (dbErr) {
                console.error("Failed to save translation history:", dbErr);
              }

              setTimeout(() => {
                setResult({
                  translation: data.translation_text,
                  symbols: data.symbols,
                  num_symbols_detected: data.num_symbols_detected,
                  num_clusters: data.num_clusters,
                  annotated_image_base64: data.annotated_image_base64,
                });
                setIsLoading(false);
              }, 1200);
            } else if (message.type === "error") {
              throw new Error(message.message);
            }
          } catch (e) {
            console.error("Error parsing stream chunk:", e);
          }
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError') return;
      console.error("Translation error:", error);
      alert("Failed to decipher the inscription. Please try again.");
      setIsLoading(false);
    } finally {
      abortControllerRef.current = null;
    }
  };

  const acceptFile = (f: File | null) => {
    setResult(null);
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setFileName(f.name);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };


  const resetAll = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (stepIntervalRef.current) clearInterval(stepIntervalRef.current);
    setResult(null);
    setFile(null);
    setFileName("");
    setIsLoading(false);
    setCurrentStep(0);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const [particles, setParticles] = useState<{ x: string; y: string; driftX: number; driftY: number; duration: number; delay: number }[]>([]);

  useEffect(() => {
    const newParticles = [...Array(20)].map(() => ({
      x: Math.random() * 100 + "%",
      y: Math.random() * 100 + "%",
      driftX: (Math.random() - 0.5) * 60,
      driftY: (Math.random() - 0.5) * 60,
      duration: 10 + Math.random() * 15,
      delay: Math.random() * 10
    }));
    setParticles(newParticles);
  }, []);

  return (
    <PageShell>
      <div className="absolute inset-0 pointer-events-none z-0 overflow-hidden">
        {particles.map((p, i) => (
          <motion.div
            key={i}
            style={{
              left: p.x,
              top: p.y,
            }}
            initial={{
              opacity: 0,
              x: 0,
              y: 0,
            }}
            animate={{
              opacity: [0, 0.4, 0],
              x: [0, p.driftX, 0],
              y: [0, p.driftY, 0],
            }}
            transition={{
              duration: p.duration,
              repeat: Infinity,
              ease: "easeInOut",
              delay: p.delay
            }}
            className="absolute w-1 h-1 bg-[#E6B23C] rounded-full blur-[1px]"
          />
        ))}
      </div>


      <div className="-mt-6 md:mt-0 min-h-[calc(100dvh-160px)] md:min-h-[calc(100dvh-120px)] flex flex-col items-center justify-center p-4 md:p-8 relative">


        <div
          className={`w-full grid gap-12 transition-all duration-1000 ease-[0.16,1,0.3,1] mx-auto items-center ${(isLoading || result) ? 'lg:grid-cols-2 max-w-7xl' : 'grid-cols-1 max-w-xl'
            }`}
        >

          <motion.div
            layout
            initial={{ opacity: 0, scale: 0.95, y: 30 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="w-full max-w-xl relative z-10"
          >
            <motion.div
              animate={{
                opacity: [0.3, 0.6, 0.3],
                scale: [1, 1.02, 1]
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="absolute -inset-1 bg-[#E6B23C]/20 rounded-[2.6rem] blur-2xl z-[-1]"
            />

            <div className={`transition-all duration-700 rounded-[2.5rem] shadow-[0_30px_100px_rgba(0,0,0,0.9)] shadow-[inset_0_1px_1px_rgba(230,178,60,0.11)] overflow-hidden relative ${previewUrl ? "bg-gradient-to-br from-[#1A140F] to-[#0D0A07] border border-[#E6B23C] shadow-[0_0_50px_rgba(230,178,60,0.2)]" : "bg-gradient-to-br from-[#120D08] to-[#0A0805]"
              }`}>

              {!previewUrl && !isLoading && (
                <div className="absolute inset-0 pointer-events-none rounded-[2.5rem] overflow-hidden">
                  <div className="absolute inset-[-100%] animate-[spin_8s_linear_infinite] opacity-40"
                    style={{
                      background: "conic-gradient(from 0deg, transparent 0%, transparent 40%, #E6B23C 50%, transparent 60%, transparent 100%)"
                    }}
                  />
                  <div className="absolute inset-[2px] bg-gradient-to-br from-[#120D08] to-[#0A0805] rounded-[2.4rem]" />
                </div>
              )}

              <div className="absolute inset-0 opacity-[0.08] bg-[url('https://www.transparenttextures.com/patterns/papyros.png')] pointer-events-none" />

              <div className="p-6 md:p-12 relative z-10 flex-1 flex flex-col justify-center">
                <div className="text-center mb-6 md:mb-10">
                  <h1 className="font-display text-3xl font-bold text-[#F5E6D0] tracking-[0.1em] uppercase mb-2 md:mb-3" style={{ fontFamily: 'var(--font-cormorant), serif' }}>
                    Hieroglyphics Decoder
                  </h1>
                  <div className="w-24 h-[1px] mx-auto mb-3 md:mb-4 bg-gradient-to-r from-transparent via-[#E6B23C]/40 to-transparent" />
                  <p className="text-[#A08E70] text-sm font-medium opacity-80 max-w-md mx-auto leading-relaxed">
                    Upload an image of carved hieroglyphs to uncover the stories hidden.
                  </p>
                </div>

                <div
                  onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={(e) => { e.preventDefault(); setDragActive(false); acceptFile(e.dataTransfer.files[0]); }}
                  className={`relative min-h-[220px] md:min-h-[340px] rounded-3xl transition-all duration-500 flex flex-col items-center justify-center p-6 md:p-8 overflow-hidden group ${dragActive ? "bg-[#E6B23C]/[0.08] scale-[1.02]" : "bg-[#E6B23C]/[0.02]"
                    }`}
                >
                  <motion.div animate={{ opacity: dragActive ? 1 : 0.4 }} className="absolute top-0 left-0 w-12 h-12 border-t-2 border-l-2 border-[#E6B23C] rounded-tl-3xl" />
                  <motion.div animate={{ opacity: dragActive ? 1 : 0.4 }} className="absolute top-0 right-0 w-12 h-12 border-t-2 border-r-2 border-[#E6B23C] rounded-tr-3xl" />
                  <motion.div animate={{ opacity: dragActive ? 1 : 0.4 }} className="absolute bottom-0 left-0 w-12 h-12 border-b-2 border-l-2 border-[#E6B23C] rounded-bl-3xl" />
                  <motion.div animate={{ opacity: dragActive ? 1 : 0.4 }} className="absolute bottom-0 right-0 w-12 h-12 border-b-2 border-r-2 border-[#E6B23C] rounded-br-3xl" />

                  <AnimatePresence mode="wait">
                    {previewUrl ? (
                      <motion.div
                        key="preview"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className="flex flex-col items-center w-full"
                      >
                        <div className="relative group/preview mb-4 md:mb-6">
                          <div className="relative rounded-2xl overflow-hidden border border-[#E6B23C]/30 shadow-[0_0_50px_rgba(230,178,60,0.15)] z-10">
                            <img
                              src={result?.annotated_image_base64 || previewUrl}
                              alt="Preview"
                              className={`max-h-[140px] md:max-h-[220px] w-auto object-contain transition-opacity duration-700 ${isLoading ? 'opacity-40' : 'opacity-100'}`}
                            />

                            <AnimatePresence>
                              {isLoading && (
                                <motion.div
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  exit={{ opacity: 0 }}
                                  className="absolute inset-0 z-20 pointer-events-none overflow-hidden"
                                >
                                  <motion.div
                                    animate={{
                                      x: ["-20%", "60%", "10%"],
                                      y: ["-10%", "30%", "0%"]
                                    }}
                                    transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
                                    className="absolute w-64 h-64 rounded-full z-10"
                                    style={{
                                      background: "radial-gradient(circle, rgba(230,178,60,0.15) 0%, transparent 70%)",
                                      boxShadow: "0 0 100px rgba(230,178,60,0.1) inset"
                                    }}
                                  >
                                    <div className="absolute inset-0 rounded-full border border-[#E6B23C]/20 shadow-[0_0_30px_rgba(230,178,60,0.1)]" />
                                  </motion.div>

                                  <div className="absolute top-2 left-2 w-4 h-4 border-t border-l border-[#E6B23C]/40" />
                                  <div className="absolute top-2 right-2 w-4 h-4 border-t border-r border-[#E6B23C]/40" />
                                  <div className="absolute bottom-2 left-2 w-4 h-4 border-b border-l border-[#E6B23C]/40" />
                                  <div className="absolute bottom-2 right-2 w-4 h-4 border-b border-r border-[#E6B23C]/40" />
                                </motion.div>
                              )}
                            </AnimatePresence>

                          </div>

                          <button
                            onClick={resetAll}
                            className="absolute -top-3 -right-3 h-8 w-8 bg-[#0D0A07] border border-[#E6B23C]/30 rounded-full flex items-center justify-center text-[#A08E70] hover:text-[#E6B23C] transition-all shadow-xl z-20"
                          >
                            <X size={16} />
                          </button>
                        </div>

                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="mb-4 md:mb-8"
                        >
                          <p className="text-[10px] font-bold text-[#A08E70] tracking-widest uppercase truncate opacity-80">
                            {fileName}
                          </p>
                        </motion.div>

                        <AnimatePresence mode="wait">
                          {!result && (
                            <motion.div
                              key="identify-action-button"
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -10 }}
                              className="w-full max-w-[280px]"
                            >
                              <Button
                                onClick={handleDecipher}
                                disabled={isLoading}
                                className="h-12 md:h-14 px-12 rounded-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 text-[#E6B23C] hover:bg-[#E6B23C]/10 font-bold text-xs md:text-sm uppercase tracking-[0.2em] transition-all hover:scale-105 shadow-[0_10px_30px_rgba(230,178,60,0.1)] w-full"
                              >
                                {isLoading ? (
                                  <Loader2 size={20} className="animate-spin" />
                                ) : (
                                  <>IDENTIFY INSCRIPTION</>
                                )}
                              </Button>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    ) : (
                      <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center">
                        <motion.div animate={{ opacity: [0.1, 0.4, 0.1] }} transition={{ duration: 4, repeat: Infinity }} className="text-[#E6B23C] text-2xl md:text-3xl font-display tracking-[0.6em] mb-4 md:mb-6 select-none">
                          𓂀 𓃭 𓅃 𓆣 𓇳
                        </motion.div>
                        <p className="text-[#F5E6D0] font-bold text-lg mb-2">Place Your Image</p>
                        <p className="text-[#A08E70] text-[10px] md:text-xs font-medium opacity-60 mb-6 md:mb-10 tracking-widest text-center">Drop an image or Use your camera</p>

                        <div className="flex flex-row w-full max-w-sm gap-2 sm:gap-4 justify-center">
                          <Button onClick={pickFile} className="flex-1 h-12 px-2 sm:px-8 rounded-xl bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] hover:bg-[#E6B23C]/20 font-bold text-[10px] sm:text-xs uppercase tracking-widest transition-all">
                            <Upload className={isRTL ? "ml-1 sm:ml-2" : "mr-1 sm:mr-2"} size={14} /> UPLOAD
                          </Button>
                          <Button variant="outline" onClick={openCamera} className="flex-1 h-12 px-2 sm:px-8 rounded-xl border-[#A08E70]/20 bg-transparent text-[#A08E70] hover:text-[#F5E6D0] hover:border-[#F5E6D0]/30 font-bold text-[10px] sm:text-xs uppercase tracking-widest transition-all">
                            <Camera className={isRTL ? "ml-1 sm:ml-2" : "mr-1 sm:mr-2"} size={14} /> CAPTURE
                          </Button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*,.heic,.heif"
                    onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
                    onClick={(e) => (e.currentTarget.value = "")}
                  />
                  <input
                    ref={cameraInputRef}
                    type="file"
                    className="hidden"
                    accept="image/*"
                    capture="environment"
                    onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
                    onClick={(e) => (e.currentTarget.value = "")}
                  />
                </div>
              </div>
            </div>
          </motion.div>

          <AnimatePresence>
            {(isLoading || result) && (
              <motion.div
                id="result-section"
                layout
                initial={{ opacity: 0, x: 100 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 100 }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                className="lg:col-span-1 scroll-mt-24"
              >
                <div>
                  <AnimatePresence mode="wait">
                    {isLoading ? (
                      <motion.div
                        key="loading"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="p-12 h-[640px] flex flex-col relative overflow-hidden"
                      >

                        <div className="text-center mb-10 relative z-10">
                          <h1 className="font-display text-3xl font-bold text-[#F5E6D0] tracking-[0.1em] uppercase mb-3" style={{ fontFamily: 'var(--font-cormorant), serif' }}>
                            Deciphering Inscription
                          </h1>
                          <div className="w-24 h-[1px] mx-auto mb-4 bg-gradient-to-r from-transparent via-[#E6B23C]/40 to-transparent" />
                        </div>

                        <div className="relative flex-1 mt-4 z-10">
                          <svg className="absolute inset-0 w-full h-full overflow-visible pointer-events-none z-10" preserveAspectRatio="none" viewBox="0 0 100 100">
                            <defs>
                              <mask id="line-mask">
                                <rect x="0" y="0" width="100" height="100" fill="white" />
                                <circle cx="10" cy="10" r="9" fill="black" />
                                <circle cx="83" cy="35" r="7.47" fill="black" />
                                <circle cx="6" cy="58" r="9" fill="black" />

                              </mask>
                            </defs>

                            <path
                              d="M 10 10 C 10 32.5, 90 12.5, 90 35 C 90 57.5, 10 37.5, 10 60 C 10 82.5, 90 62.5, 95 85"
                              stroke="#F5E6D0"
                              fill="none"
                              strokeWidth="0.6"
                              strokeDasharray="4 4"
                              className="opacity-15"
                              mask="url(#line-mask)"
                            />

                            <motion.path
                              d="M 10 10 C 10 32.5, 90 12.5, 90 35 C 90 57.5, 10 37.5, 10 60 C 10 82.5, 90 62.5, 95 85"
                              stroke="#E6B23C"
                              fill="none"
                              strokeWidth="0.4"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: currentStep / 3 }}
                              transition={{ duration: 1.5, ease: "easeInOut" }}
                              className="drop-shadow-[0_0_10px_rgba(230,178,60,0.5)]"
                              mask="url(#line-mask)"
                            />
                          </svg>

                          <div className="absolute inset-0">
                            {[
                              { title: "Scanning Inscription", icon: Search, top: "10%", left: "10%", align: "start" },
                              { title: "Determining Sequence", icon: BookOpen, top: "30%", left: "83%", align: "end" },
                              { title: "Recognizing Symbols", icon: Cpu, top: "52%", left: "4.7%", align: "start" },
                              { title: "Generating Translation", icon: "𓅓", top: "85%", left: "83%", align: "end" }
                            ].map((step, i) => (
                              <AnimatePresence key={i}>
                                {currentStep >= (i === 0 ? 0.1 : i) && (
                                  <motion.div
                                    initial={{ opacity: 1, scale: 0.8, y: 10 }}
                                    animate={{ opacity: 1, scale: 1, y: 0 }}
                                    className="absolute"
                                    style={{
                                      top: step.top,
                                      left: step.left,
                                      transform: 'translate(-50%, -50%)'
                                    }}
                                  >
                                    {/* ICON */}
                                    <motion.div
                                      animate={{
                                        boxShadow: (currentStep >= i && currentStep < i + 1) ? "0 0 30px rgba(230,178,60,0.4)" : "0 0 0px transparent",
                                        scale: (currentStep >= i && currentStep < i + 1) ? [1, 1.08, 1] : 1
                                      }}
                                      transition={{
                                        duration: (currentStep >= i && currentStep < i + 1) ? 1.5 : 0.5,
                                        repeat: (currentStep >= i && currentStep < i + 1) ? Infinity : 0,
                                        ease: "easeInOut"
                                      }}
                                      className={`w-14 h-14 rounded-full border border-[#E6B23C] bg-[#E6B23C] flex items-center justify-center relative z-20 ${(i === 0) ? 'translate-y-1' : ''}`}
                                    >
                                      {typeof step.icon === 'string' ? (
                                        <span className={`text-3xl font-black leading-none select-none -translate-y-0.5 transition-colors ${currentStep >= (i === 0 ? 0.1 : i) ? "text-[#120D08]" : "text-[#E6B23C]/40"}`}>{step.icon}</span>
                                      ) : (
                                        <step.icon size={24} strokeWidth={3} className={`transition-colors ${currentStep >= (i === 0 ? 0.1 : i) ? "text-[#120D08]" : "text-[#E6B23C]/40"}`} />
                                      )}
                                      {currentStep >= i + 1 && (
                                        <motion.div
                                          initial={{ scale: 0 }}
                                          animate={{ scale: 1 }}
                                          className="absolute -top-1 -right-1 w-5 h-5 bg-[#120D08] rounded-full flex items-center justify-center text-[#E6B23C]"
                                        >
                                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                                        </motion.div>
                                      )}
                                    </motion.div>

                                    <div
                                      className={`absolute top-1/2 -translate-y-1/2 whitespace-nowrap space-y-0.5 ${step.align === "end" ? "right-[calc(100%+32px)] text-right" : "left-[calc(100%+32px)] text-left"}`}
                                    >
                                      <div className="text-[10px] font-bold tracking-[0.3em] text-[#E6B23C]/60 uppercase font-sans">Phase 0{i + 1}</div>
                                      <div className="text-[13px] font-bold text-[#F5E6D0]/80 tracking-widest uppercase font-sans">
                                        {step.title}
                                      </div>
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    ) : result ? (
                      <motion.div
                        key="result"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="papyrus-paper h-fit flex flex-col transition-all duration-1000 !p-12 shadow-[0_30px_100px_rgba(0,0,0,0.9)]"
                      >
                        <div className="flex-1">
                          <div className="mb-12 text-center">
                            <h1 className="font-display text-3xl font-bold text-[#1A1005] tracking-[0.1em] uppercase mb-4" style={{ fontFamily: 'var(--font-cormorant), serif' }}>
                              Translation
                            </h1>
                            <div className="w-24 h-[1px] mx-auto mb-10 bg-gradient-to-r from-transparent via-[#1A1005]/20 to-transparent" />
                          </div>

                          <div className="text-left">
                            <p className="text-xl md:text-2xl font-medium text-[#1A1005]/90 leading-relaxed" style={{ fontFamily: "var(--font-cormorant), serif" }}>
                              {result.translation}
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    ) : null}
                  </AnimatePresence>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-6 md:mt-12 flex justify-center w-full"
        >
          <button
            onClick={result ? resetAll : () => router.back()}
            className="flex items-center gap-3 px-6 py-2 rounded-full bg-[#E6B23C]/5 border border-[#E6B23C]/10 text-[10px] font-bold tracking-[0.3em] text-[#A08E70]/60 hover:text-[#E6B23C] hover:border-[#E6B23C]/30 uppercase transition-all group cursor-pointer"
          >
            <motion.span
              animate={{ x: isRTL ? [0, 5, 0] : [0, -5, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              {isRTL ? "→" : "←"}
            </motion.span>
            {t("common.return")}
          </button>
        </motion.div>
      </div>
    </PageShell>
  );
}
