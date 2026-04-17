"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import PageShell from "../../components/feature/PageShell";
import { Button } from "../../components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, Languages, Trash2, Upload, Scroll, Zap, BookOpen, Search, Image as ImageIcon } from "lucide-react";

type TranslateResponse = {
  ocr_text: string;
  transliteration: string;
  translation: string;
  explanation: string;
};

export default function TranslatePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"translation">("translation");
  const [result, setResult] = useState<TranslateResponse | null>(null);

  const pickFile = () => fileInputRef.current?.click();

  useEffect(() => {
    if (!file) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const acceptFile = (f: File | null) => {
    if (!f || !f.type.startsWith("image/")) return;
    setResult(null);
    setFile(f);
    setIsLoading(true);
    setTimeout(() => {
      setResult({
        ocr_text: "𓋹 𓎬 𓇳 𓅓 𓊵 𓏏 𓊪",
        transliteration: "Ankh Udja Seneb m Hotep",
        translation: "Life, Prosperity, Health in Peace",
        explanation: "This classic formulary invokes the triad of vital blessings upon the bearer. Found frequently on funerary stelae and royal seals of the New Kingdom.",
      });
      setIsLoading(false);
    }, 2000);
  };

  const resetAll = () => { setResult(null); setFile(null); setIsLoading(false); };

  return (
    <PageShell>
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="mb-8">
        <Link href="/" className="group inline-flex items-center gap-2 text-xs font-semibold tracking-[0.15em] uppercase text-[#A08E70] hover:text-[#E6B23C] transition-colors">
          <span className="group-hover:-translate-x-1 transition-transform">←</span>
          Return
        </Link>
      </motion.div>

      <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
        <div className="flex items-start gap-5">
          <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-[#E6B23C]/15 to-[#E6B23C]/5 border border-[#E6B23C]/15 flex items-center justify-center text-[#E6B23C] overflow-hidden">
            <span className="text-5xl leading-none -translate-y-4">𓁹</span>
          </div>
          <div>
            <div className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase mb-1">Ancient Linguistics</div>
            <h1 className="font-heading text-3xl md:text-4xl font-bold text-[#F5E6D0] tracking-tight" style={{ fontFamily: 'var(--font-cormorant), serif' }}>
              Hieroglyphs <span className="text-[#E6B23C] gold-glow" >Decoder</span>
            </h1>
          </div>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-5 items-start">
        {/* Left: Upload */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="lg:col-span-2 space-y-6">
          <div className="glass-surface rounded-3xl p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-[#E6B23C]/[0.04] blur-[60px] pointer-events-none" />

            <div
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => { e.preventDefault(); setDragActive(false); acceptFile(e.dataTransfer.files[0]); }}
              onClick={pickFile}
              className={`relative aspect-[4/3] rounded-2xl border-2 border-dashed transition-all duration-500 flex flex-col items-center justify-center cursor-pointer group overflow-hidden ${dragActive ? "border-[#E6B23C] bg-[#E6B23C]/[0.06]" : "border-[#E6B23C]/10 bg-[#E6B23C]/[0.02] hover:border-[#E6B23C]/25"
                }`}
            >
              <AnimatePresence mode="wait">
                {previewUrl ? (
                  <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0">
                    <img src={previewUrl} alt="Artifact" className="h-full w-full object-cover" />
                    <div className="absolute inset-0 bg-black/40 group-hover:bg-black/60 transition-colors flex items-center justify-center">
                      <div className="text-[10px] font-bold text-white px-3 py-1.5 bg-black/40 border border-white/10 rounded-full backdrop-blur-md uppercase tracking-[0.15em]">Update</div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center p-6 text-center">
                    <div className="w-14 h-14 rounded-full bg-[#E6B23C]/8 flex items-center justify-center text-[#E6B23C] mb-5 group-hover:scale-110 transition-transform">
                      <ImageIcon size={24} />
                    </div>
                    <div className="text-sm font-semibold text-[#F5E6D0] mb-1">Place Image</div>
                    <p className="text-[10px] text-[#A08E70]">Drop image or Capture a Photo</p>
                  </motion.div>
                )}
              </AnimatePresence>
              <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={(e) => acceptFile(e.target.files?.[0] ?? null)} />
            </div>

            <div className="mt-6 flex gap-3">
              <Button onClick={pickFile} className="flex-[2] h-12 rounded-2xl bg-gradient-to-r from-[#E6B23C] to-[#D4A030] text-[#0D0A07] font-bold text-sm hover:scale-[1.02] transition-all flex items-center justify-center">
                <Upload size={18} className="mr-2" /> Upload
              </Button>
              <Button variant="outline" className="flex-1 h-12 border-[#E6B23C]/10 hover:border-[#E6B23C]/25 rounded-2xl bg-transparent flex items-center justify-center text-[#A08E70] font-bold text-sm transition-all hover:scale-[1.02]" onClick={() => alert("Lens Active...")}>
                <Camera size={18} className="mr-2" /> Capture
              </Button>
            </div>
          </div>

          <div className="glass-surface rounded-2xl p-5">
            <div className="flex items-start gap-3">
              <div className="h-9 w-9 flex-shrink-0 bg-[#E6B23C]/8 border border-[#E6B23C]/15 rounded-xl flex items-center justify-center text-[#E6B23C]"><BookOpen size={16} /></div>
              <div>
                <h4 className="text-xs font-bold tracking-[0.15em] text-[#F5E6D0] uppercase mb-1">Status</h4>
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${isLoading ? "bg-[#E6B23C] animate-pulse shadow-[0_0_6px_rgba(230,178,60,0.8)]" : result ? "bg-[#2A7B6F]" : "bg-[#A08E70]/40"}`} />
                  <span className="text-[10px] font-semibold text-[#A08E70]">
                    {isLoading ? "Synthesizing..." : result ? "Decoding Complete" : "Standby"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Right: Result */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="lg:col-span-3 h-full">
          <div className="glass-surface rounded-3xl h-full flex flex-col overflow-hidden min-h-[500px]">
            <div className="flex border-b border-[#E6B23C]/[0.06]">
              {(["translation"] as const).map((tab) => (
                <button key={tab} onClick={() => setActiveTab(tab)}
                  className={`flex-1 py-5 text-xs font-bold tracking-[0.2em] uppercase transition-all relative ${activeTab === tab ? "text-[#E6B23C]" : "text-[#A08E70] hover:text-[#F5E6D0]"}`}>
                  {tab}
                  {activeTab === tab && (
                    <motion.div layoutId="tab-line" className="absolute bottom-0 left-0 right-0 h-[2px]"
                      style={{ background: "linear-gradient(90deg, transparent, #E6B23C, transparent)", boxShadow: "0 0 10px rgba(230,178,60,0.4)" }}
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </button>
              ))}
            </div>

            <div className="flex-1 p-8 md:p-10">
              <AnimatePresence mode="wait">
                {isLoading ? (
                  <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-6">
                    <div className="h-12 w-3/4 rounded-2xl shimmer-gold" />
                    <div className="h-4 w-full rounded-full shimmer-gold" />
                    <div className="grid grid-cols-3 gap-4 mt-8">
                      {[0, 1, 2].map(i => <div key={i} className="h-20 rounded-2xl shimmer-gold" style={{ animationDelay: `${i * 0.3}s` }} />)}
                    </div>
                  </motion.div>
                ) : result ? (
                  <motion.div key="result" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-10">
                    <div>
                      <div className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase mb-4">Raw Recognition</div>
                      <div className="text-3xl md:text-4xl font-bold text-[#F5E6D0] leading-tight font-display tracking-widest bg-[#E6B23C]/[0.03] p-7 rounded-2xl border border-[#E6B23C]/8">
                        {result.ocr_text}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-bold tracking-[0.25em] text-[#E6B23C] uppercase mb-4">
                        Historical Meaning
                      </div>
                      <div className="relative group">
                        <div className="absolute -inset-1 bg-[#E6B23C]/10 blur opacity-20 rounded-2xl group-hover:opacity-35 transition-opacity" />
                        <div className="relative bg-[#0D0A07]/50 border border-[#E6B23C]/10 p-8 rounded-2xl">
                          <p className="font-heading text-2xl font-bold text-[#F5E6D0] leading-relaxed">
                            {result.translation}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="pt-8 border-t border-[#E6B23C]/[0.06]">
                      <div className="flex items-center gap-2 mb-3">
                        <Search size={14} className="text-[#E6B23C]" />
                        <span className="text-xs font-bold tracking-[0.15em] text-[#F5E6D0] uppercase">Explanation</span>
                      </div>
                      <p className="text-sm text-[#A08E70] leading-relaxed italic border-l-2 border-[#E6B23C]/20 pl-5">{result.explanation}</p>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col items-center justify-center text-center opacity-30 select-none py-20">
                    <Languages size={56} className="mb-6 text-[#E6B23C]/50" />
                    <p className="text-xs font-bold tracking-[0.15em] text-[#A08E70] uppercase max-w-xs">Upload an inscription to translate</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {result && (
              <div className="p-5 border-t border-[#E6B23C]/[0.06] flex justify-end">
                <Button variant="ghost" className="text-[#E6B23C] hover:text-[#FFD369] hover:bg-[#E6B23C]/5 rounded-xl h-10 px-5" onClick={resetAll}>
                  <Trash2 size={14} className="mr-2" />
                  <span className="text-xs font-bold tracking-[0.15em] uppercase">Purge Session</span>
                </Button>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </PageShell>
  );
}
