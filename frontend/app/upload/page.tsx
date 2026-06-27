"use client";

import Link from "next/link";
import { useRef, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { useLanguage } from "../../context/LanguageContext";
import { Image, Upload, Camera, X, ArrowRight, Loader2, AlertCircle, Sparkles } from "lucide-react";
import { recognizeImage, saveResultToSession } from "../../lib/services/recognition";
import { createClient } from "../../lib/supabase/client";

export default function UploadPage() {
  const { t, isRTL } = useLanguage();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);
  const reqIdRef = useRef(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function openFilePicker() { inputRef.current?.click(); }
  function openCamera() { cameraInputRef.current?.click(); }

  function handleFile(file: File) {
    setFileName(file.name);
    setSelectedFile(file);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    handleFile(file);
  }

  function clearFile() {
    reqIdRef.current += 1;
    setFileName("");
    setSelectedFile(null);
    setError(null);
    setIsLoading(false);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  async function handleRecognize() {
    if (!selectedFile || isLoading) return;
    setIsLoading(true);
    setError(null);

    const currentReq = ++reqIdRef.current;

    try {
      const result = await recognizeImage(selectedFile);
      if (currentReq !== reqIdRef.current) return;

      // Save to recognition_history if user is logged in and recognition was successful
      try {
        const supabase = createClient();
        const { data: { user } } = await supabase.auth.getUser();
        
        const isSuccess = result.source !== "error" && 
                          result.type !== "error" && 
                          result.name !== "recognition_failed" && 
                          result.name.toLowerCase() !== "unknown" &&
                          !result.name.toLowerCase().includes("failed");

        if (user && isSuccess) {
          const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";
          const uploadData = new FormData();
          uploadData.append("file", selectedFile);
          uploadData.append("user_id", user.id);
          uploadData.append("task_type", "recognition");

          const uploadRes = await fetch(`${API_BASE}/api/v1/assets/upload/history`, {
            method: "POST",
            body: uploadData,
          });

          if (uploadRes.ok) {
            const { key } = await uploadRes.json();
            
            await supabase.from('recognition_history').insert({
              user_id: user.id,
              image_path: key,
              entity_name: (result.raw_name || result.name).replace(/_/g, " "),
              entity_type: result.type
            });
          }
        }
      } catch (dbErr) {
        console.error("Failed to save recognition history:", dbErr);
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        if (currentReq !== reqIdRef.current) return;
        const imageDataUrl = typeof reader.result === "string" ? reader.result : null;
        saveResultToSession({ result, imageDataUrl });
        router.push("/result");
      };
      reader.onerror = () => {
        if (currentReq !== reqIdRef.current) return;
        saveResultToSession({ result, imageDataUrl: null });
        router.push("/result");
      };
      reader.readAsDataURL(selectedFile);
    } catch (err: unknown) {
      if (currentReq !== reqIdRef.current) return;
      const msg = err instanceof Error ? err.message : t("upload.error.failed");
      setError(msg);
      setIsLoading(false);
    }
  }

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
      <div className="-mt-6 md:mt-0 min-h-[calc(100dvh-160px)] md:min-h-[calc(100dvh-120px)] flex flex-col items-center justify-center p-4 md:p-8 relative">
        
        {/* Cinematic Particles */}
        <div className="absolute inset-0 pointer-events-none">
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
                y: 0
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

        {/* The Core Utility Card */}
        <motion.div
          layout
          initial={{ opacity: 0, scale: 0.95, y: 30 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="w-full max-w-xl relative z-10"
        >
          {/* External Card Glow */}
          <motion.div 
            animate={{ 
              opacity: [0.3, 0.6, 0.3],
              scale: [1, 1.02, 1]
            }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -inset-1 bg-[#E6B23C]/20 rounded-[2.6rem] blur-2xl z-[-1]" 
          />

          {/* Glowing Card Container */}
          <div className={`transition-all duration-700 rounded-[2.5rem] shadow-[0_30px_100px_rgba(0,0,0,0.9)] shadow-[inset_0_1px_1px_rgba(230,178,60,0.1)] overflow-hidden relative ${
            previewUrl ? "bg-gradient-to-br from-[#1A140F] to-[#0D0A07] border border-[#E6B23C] shadow-[0_0_50px_rgba(230,178,60,0.2)]" : "bg-gradient-to-br from-[#120D08] to-[#0A0805]"
          }`}>
            
            {/* Spinning Border Beam Animation (Pro Rotating Gradient Approach) */}
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
            
            {/* Subtle Texture Overlay */}
            <div className="absolute inset-0 opacity-[0.08] bg-[url('https://www.transparenttextures.com/patterns/papyros.png')] pointer-events-none" />

            <div className="p-6 md:p-12 relative z-10">
              
              {/* Internal Header */}
              <div className="text-center mb-6 md:mb-10">
                <motion.h1
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="font-display text-3xl md:text-4xl font-bold text-[#F5E6D0] tracking-[0.05em] uppercase mb-2 md:mb-3"
                  style={{ fontFamily: 'var(--font-cormorant), serif' }}
                >
                  {t("upload.title")}
                </motion.h1>
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: 0.6, duration: 0.8 }}
                  className="w-24 h-[1px] mx-auto mb-4 bg-gradient-to-r from-transparent via-[#E6B23C]/40 to-transparent"
                />
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="text-[#A08E70] text-sm font-medium opacity-80"
                >
                  {t("upload.subtitle")}
                </motion.p>
              </div>

              {/* Integrated Action Zone */}
              <motion.div
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
                whileHover={{ borderColor: "rgba(230,178,60,0.4)" }}
                className={`relative min-h-[220px] md:min-h-[340px] rounded-3xl transition-all duration-500 flex flex-col items-center justify-center p-6 md:p-8 overflow-hidden group ${
                  isDragging 
                    ? "bg-[#E6B23C]/[0.08] scale-[1.02]" 
                    : "bg-[#E6B23C]/[0.02]"
                }`}
              >

                {/* HUD Scanning Accents (Rounded to match corners) */}
                <motion.div animate={{ opacity: isDragging ? 1 : 0.4 }} className="absolute top-0 left-0 w-12 h-12 border-t-2 border-l-2 border-[#E6B23C] rounded-tl-3xl" />
                <motion.div animate={{ opacity: isDragging ? 1 : 0.4 }} className="absolute top-0 right-0 w-12 h-12 border-t-2 border-r-2 border-[#E6B23C] rounded-tr-3xl" />
                <motion.div animate={{ opacity: isDragging ? 1 : 0.4 }} className="absolute bottom-0 left-0 w-12 h-12 border-b-2 border-l-2 border-[#E6B23C] rounded-bl-3xl" />
                <motion.div animate={{ opacity: isDragging ? 1 : 0.4 }} className="absolute bottom-0 right-0 w-12 h-12 border-b-2 border-r-2 border-[#E6B23C] rounded-br-3xl" />

                {/* Content Switching */}
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
                        <motion.div 
                          animate={{ rotate: 360 }}
                          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                          className="absolute -inset-4 border border-[#E6B23C]/10 rounded-full"
                        />
                        
                        <div className="relative rounded-2xl overflow-hidden border border-[#E6B23C]/30 shadow-[0_0_50px_rgba(230,178,60,0.15)] z-10">
                          <img
                            src={previewUrl}
                            alt="Preview"
                            className="max-h-[140px] md:max-h-[220px] w-auto object-contain"
                          />
                          
                          {/* Constrained Scanning HUD (Now clipped to image) */}
                          <AnimatePresence>
                            {isLoading && (
                              <motion.div
                                initial={{ top: "0%" }}
                                animate={{ top: "100%" }}
                                transition={{ 
                                  duration: 2, 
                                  repeat: Infinity, 
                                  repeatType: "reverse", 
                                  ease: "linear" 
                                }}
                                className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#E6B23C] to-transparent shadow-[0_0_20px_#E6B23C] z-30"
                              />
                            )}
                          </AnimatePresence>
                        </div>

                        <button
                          onClick={clearFile}
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

                      <Button
                        onClick={handleRecognize}
                        disabled={isLoading}
                        className="h-12 md:h-14 px-12 rounded-full bg-[#E6B23C]/5 border border-[#E6B23C]/30 text-[#E6B23C] hover:bg-[#E6B23C]/10 font-bold text-xs md:text-sm uppercase tracking-[0.2em] transition-all hover:scale-105 shadow-[0_10px_30px_rgba(230,178,60,0.1)] w-full max-w-[280px]"
                      >
                        {isLoading ? (
                          <Loader2 size={20} className="animate-spin" />
                        ) : (
                          <>
                            {t("upload.button.recognize")}
                          </>
                        )}
                      </Button>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="idle"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="flex flex-col items-center"
                    >
                      {/* Hieroglyph Spirit row */}
                      <motion.div 
                        animate={{ opacity: [0.1, 0.4, 0.1] }}
                        transition={{ duration: 4, repeat: Infinity }}
                        className="text-[#E6B23C] text-2xl md:text-3xl font-display tracking-[0.6em] mb-4 md:mb-6 select-none"
                      >
                        𓂀 𓃭 𓅃 𓆣 𓇳
                      </motion.div>
                      
                      <p className="text-[#F5E6D0] font-bold text-lg mb-2">{t("upload.dropzone.title")}</p>
                      <p className="text-[#A08E70] text-[10px] md:text-xs font-medium opacity-60 mb-6 md:mb-10 text-center">{t("upload.dropzone.subtitle")}</p>

                      {/* Integrated Action Buttons */}
                      <div className="flex flex-row w-full max-w-sm gap-2 sm:gap-4 justify-center">
                        <Button
                          onClick={openFilePicker}
                          className="flex-1 h-12 px-2 sm:px-8 rounded-xl bg-[#E6B23C]/10 border border-[#E6B23C]/20 text-[#E6B23C] hover:bg-[#E6B23C]/20 font-bold text-[10px] sm:text-xs uppercase tracking-widest transition-all"
                        >
                          <Upload className={isRTL ? "ml-1 sm:ml-2" : "mr-1 sm:mr-2"} size={14} />
                          {t("upload.button.upload")}
                        </Button>

                        <Button
                          variant="outline"
                          onClick={openCamera}
                          className="flex-1 h-12 px-2 sm:px-8 rounded-xl border-[#A08E70]/20 bg-transparent text-[#A08E70] hover:text-[#F5E6D0] hover:border-[#F5E6D0]/30 font-bold text-[10px] sm:text-xs uppercase tracking-widest transition-all"
                        >
                          <Camera className={isRTL ? "ml-1 sm:ml-2" : "mr-1 sm:mr-2"} size={14} />
                          {t("upload.button.capture")}
                        </Button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

              </motion.div>

              {/* Error Feedback */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-[10px] font-bold uppercase tracking-widest text-center"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>

        {/* Global Footer Navigation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-6 md:mt-12 flex justify-center w-full"
        >
          <button 
            onClick={() => router.back()} 
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
      <input ref={inputRef} type="file" accept="image/*,.heic,.heif" className="hidden" onChange={onPickFile} />
      <input ref={cameraInputRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={onPickFile} />
    </PageShell>
  );
}
