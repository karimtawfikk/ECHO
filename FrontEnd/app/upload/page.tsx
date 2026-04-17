"use client";

import Link from "next/link";
import { useRef, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import PageShell from "../../components/feature/PageShell";
import { Button } from "../../components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { Image, Upload, Camera, X, ArrowRight, Loader2, AlertCircle } from "lucide-react";
import { recognizeImage, saveResultToSession } from "../../lib/recognition";

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function openFilePicker() { inputRef.current?.click(); }

  function handleFile(file: File) {
    setFileName(file.name);
    setSelectedFile(file);
    setError(null);

    // Revoke any previous Object URL before creating a new one
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
    setFileName("");
    setSelectedFile(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleRecognize() {
    if (!selectedFile || isLoading) return;
    setIsLoading(true);
    setError(null);

    try {
      const result = await recognizeImage(selectedFile);

      // Store image as DataURL (base64) so it survives navigation
      // We use FileReader and store in sessionStorage alongside the result
      const reader = new FileReader();
      reader.onloadend = () => {
        const imageDataUrl = typeof reader.result === "string" ? reader.result : null;
        saveResultToSession({ result, imageDataUrl });
        router.push("/result");
      };
      reader.onerror = () => {
        // Still navigate even if we can't read the image
        saveResultToSession({ result, imageDataUrl: null });
        router.push("/result");
      };
      reader.readAsDataURL(selectedFile);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Recognition failed. Please try again.";
      setError(msg);
      setIsLoading(false);
    }
  }

  return (
    <PageShell>
      {/* Breadcrumb */}
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
        <Link href="/" className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors">
          <span className="group-hover:-translate-x-1 transition-transform">←</span>
          Return
        </Link>
      </motion.div>

      {/* Main Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="relative overflow-hidden rounded-[2.5rem]"
      >
        {/* Animated background layer */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#1A1208] via-[#0D0A07] to-[#1E160E]" />
        <div className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23E6B23C'%3E%3Cpath d='M40 5l8 16H32l8-16zm0 54l8 16H32l8-16zM5 40l16-8v16L5 40zm54 0l16-8v16L59 40z'/%3E%3Ccircle cx='40' cy='40' r='3'/%3E%3C/g%3E%3C/svg%3E")`,
            backgroundSize: "80px 80px",
            animation: "patternDrift 50s linear infinite",
          }}
        />
        {/* Golden corner accents */}
        <div className="absolute top-0 left-0 w-32 h-32 border-t-2 border-l-2 border-[#E6B23C]/15 rounded-tl-[2.5rem]" />
        <div className="absolute top-0 right-0 w-32 h-32 border-t-2 border-r-2 border-[#E6B23C]/15 rounded-tr-[2.5rem]" />
        <div className="absolute bottom-0 left-0 w-32 h-32 border-b-2 border-l-2 border-[#E6B23C]/15 rounded-bl-[2.5rem]" />
        <div className="absolute bottom-0 right-0 w-32 h-32 border-b-2 border-r-2 border-[#E6B23C]/15 rounded-br-[2.5rem]" />

        {/* Ambient gold light */}
        <div className="absolute top-[-100px] left-1/2 -translate-x-1/2 w-[500px] h-[300px] bg-[#E6B23C]/[0.04] blur-[120px] pointer-events-none" />

        <div className="relative z-10 p-10 sm:p-16">
          {/* Header with Eye of Horus SVG */}
          <div className="text-center mb-16">
            {/* Animated Eye of Horus */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2, type: "spring", damping: 15 }}
              className="mx-auto mb-8 relative"
            >
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="w-28 h-28 mx-auto rounded-full border border-[#E6B23C]/10"
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[#E6B23C]/15 to-[#E6B23C]/5 border border-[#E6B23C]/20 flex items-center justify-center shadow-[0_0_40px_rgba(230,178,60,0.12)]">
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" className="text-[#E6B23C]">
                    <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M12 5v-1M12 20v-1M7 7L6 6M18 18l-1-1M5 12H4M20 12h-1" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.5" />
                  </svg>
                </div>
              </div>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="font-display text-4xl md:text-5xl font-bold text-[#F5E6D0] tracking-[0.05em] uppercase mb-4"
              style={{ fontFamily: 'var(--font-cormorant), serif' }}
            >
              Unveil the <span className="text-[#E6B23C] gold-glow">Past</span>
            </motion.h1>

            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ delay: 0.6, duration: 0.6 }}
              className="w-32 h-[1px] mx-auto mb-5"
              style={{ background: "linear-gradient(90deg, transparent, #E6B23C, transparent)" }}
            />

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="text-[#A08E70] text-base max-w-md mx-auto"
            >
              place your artifact before the Eye, and uncover its origins and the story it holds
            </motion.p>
          </div>

          {/* Dropzone */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
            className={`relative group rounded-3xl border-2 border-dashed transition-all duration-500 overflow-hidden ${isDragging
              ? "border-[#E6B23C] bg-[#E6B23C]/[0.06] scale-[1.01]"
              : "border-[#E6B23C]/12 hover:border-[#E6B23C]/30"
              }`}
            style={{
              background: isDragging
                ? "rgba(230,178,60, 0.04)"
                : "radial-gradient(ellipse at center, rgba(230,178,60,0.02) 0%, transparent 70%)",
            }}
          >
            <div className="relative z-10 py-16 md:py-24 px-8 text-center">
              <AnimatePresence mode="wait">
                {previewUrl ? (
                  <motion.div
                    key="preview"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="flex flex-col items-center"
                  >
                    <div className="relative mb-8">
                      {/* Golden ring around preview */}
                      <motion.div
                        animate={{ rotate: [0, 360] }}
                        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                        className="absolute -inset-3 rounded-3xl border border-[#E6B23C]/15"
                      />
                      <div className="absolute -inset-6 bg-[#E6B23C]/10 blur-3xl rounded-full" />
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="relative w-64 h-64 md:w-80 md:h-80 object-cover rounded-2xl border-2 border-[#E6B23C]/25 shadow-[0_20px_60px_rgba(0,0,0,0.5)]"
                      />
                      <button
                        onClick={clearFile}
                        className="absolute -top-3 -right-3 h-9 w-9 bg-[#1A1208] border border-[#E6B23C]/20 rounded-full flex items-center justify-center text-[#A08E70] hover:text-[#E6B23C] hover:border-[#E6B23C]/40 transition-all"
                      >
                        <X size={18} />
                      </button>
                    </div>

                    <h3 className="font-heading text-xl font-bold text-[#F5E6D0] mb-1">{fileName}</h3>

                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 1.5, delay: 0.3 }}
                      className="h-[2px] max-w-[200px] bg-gradient-to-r from-transparent via-[#E6B23C] to-transparent my-4"
                    />
                  </motion.div>
                ) : (
                  <motion.div
                    key="idle"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center"
                  >
                    {/* Animated floating upload icon */}
                    <motion.div
                      animate={{ y: [0, -12, 0] }}
                      transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut" }}
                      className="relative mb-8"
                    >
                      <div className="absolute inset-0 bg-[#E6B23C]/8 blur-2xl rounded-full scale-150" />
                      <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-[#E6B23C]/15 to-[#E6B23C]/5 border border-[#E6B23C]/20 flex items-center justify-center text-[#E6B23C] shadow-[0_0_30px_rgba(230,178,60,0.1)]">
                        <Image size={30} />
                      </div>
                    </motion.div>

                    <h3 className="font-heading text-2xl font-bold text-[#F5E6D0] mb-2">
                      Place Your Image
                    </h3>
                    <p className="text-sm text-[#A08E70] mb-2 max-w-sm mx-auto">
                      Drop an image or Use your camera
                    </p>

                    {/* Decorative hieroglyph row */}
                    <motion.div
                      animate={{ opacity: [0.3, 0.6, 0.3] }}
                      transition={{ duration: 4, repeat: Infinity }}
                      className="text-[#E6B23C]/20 text-2xl font-display tracking-[0.4em] mt-4 select-none"
                    >
                      𓂀 𓃭 𓅃 𓆣 𓇳
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

          {/* Error Banner */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mt-5 p-4 rounded-2xl bg-red-900/20 border border-red-500/20 flex items-center gap-3 text-red-300 text-sm"
              >
                <AlertCircle size={16} className="shrink-0" />
                <span>{error}</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Action Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mt-10"
          >
            <AnimatePresence mode="wait">
              {selectedFile ? (
                <motion.div
                  key="recognize"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                >
                  <Button
                    onClick={handleRecognize}
                    disabled={isLoading}
                    className="h-14 px-12 rounded-2xl bg-gradient-to-r from-[#C1840A] to-[#A06A00] hover:from-[#D4A030] hover:to-[#C1840A] text-white font-bold text-base transition-all hover:scale-105 shadow-[0_4px_30px_rgba(230,178,60,0.15)] disabled:opacity-70 flex items-center gap-2 w-full sm:w-auto"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        Consulting History
                      </>
                    ) : (
                      <>
                        <ArrowRight size={20} />
                        Reveal the Origin
                      </>
                    )}
                  </Button>
                </motion.div>
              ) : (
                <motion.div
                  key="upload-capture"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex flex-col sm:flex-row gap-4"
                >
                  <Button
                    onClick={openFilePicker}
                    disabled={isLoading}
                    className="h-14 px-12 rounded-2xl bg-gradient-to-r from-[#E6B23C] to-[#D4A030] hover:from-[#FFD369] hover:to-[#E6B23C] text-[#0D0A07] font-bold text-base transition-all hover:scale-105 shadow-[0_4px_30px_rgba(230,178,60,0.2)] disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Upload className="mr-3" size={20} />
                    Upload
                  </Button>

                  <Button
                    variant="outline"
                    className="h-14 px-12 rounded-2xl border-[#E6B23C]/12 bg-[#E6B23C]/[0.03] hover:bg-[#E6B23C]/[0.08] text-[#F5E6D0] font-semibold text-base transition-all hover:scale-105"
                    onClick={() => alert("Initializing Ancient Scanner...")}
                    disabled={isLoading}
                  >
                    <Camera className="mr-3" size={20} />
                    Capture
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onPickFile} />

          {/* Footer */}
          <div className="mt-14 flex flex-col md:flex-row items-center justify-between gap-4">
            <motion.div
              animate={{ opacity: [0.4, 0.7, 0.4] }}
              transition={{ duration: 5, repeat: Infinity }}
              className="text-[10px] font-semibold tracking-[0.15em] text-[#E6B23C]/30 uppercase text-center md:text-left flex items-center gap-3"
            >
            </motion.div>
            <div className="flex-1" />
          </div>
        </div>
      </motion.div>
    </PageShell>
  );
}
