"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { Send, Scroll, Mic, MicOff, Check, Copy, Menu, SquarePen, Search, MessageSquare } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Suspense } from "react";
import { useLanguage } from "../../context/LanguageContext";
import { loadResultFromSession } from "../../lib/services/recognition";

const generateId = () => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id_${Math.random().toString(36).slice(2)}_${Date.now()}`;
};

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  ts: number;
  audioUrl?: string;
  isSearching?: boolean;
}

type RecordingState = "idle" | "recording" | "processing";

const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";
const CHAT_API = `${API_BASE}/api/v1/chat/chat`;
const STT_API = `${API_BASE}/api/v1/chat/transcribe`;
const TUT_AVATAR = "/tut.png";

// Voice auto-stop config
const SILENCE_THRESHOLD = 0.05;
const SILENCE_DURATION_MS = 1200;
const MIN_DURATION_MS = 1000;

const renderMessageText = (text: string) => {
  if (!text) return null;
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="font-bold text-[#E6B23C]">{part.slice(2, -2)}</strong>;
    }
    return <span key={i}>{part}</span>;
  });
};

function ChatContent() {
  const { t, language, isRTL } = useLanguage();
  const sp = useSearchParams();
  const entityName = sp.get("entity") ?? "Ancient Spirit";
  const entityType = sp.get("type") || "pharaoh";

  const getAssumedImageUrl = (name: string, isPharaoh: boolean) => {
    if (isPharaoh) {
      if (name === "Akhenaton") return `/images/pharaohs/Akhenaton.JPG`;
      if (name === "Cleopatra VII Philopator") return `/images/pharaohs/Cleopatra VII Philopator.jpg`;
      if (name === "Hatshepsut") return `/images/pharaohs/Hatshepsut.JPG`;
      if (name === "Ramesses II") return `/images/pharaohs/Ramesses II.jpg`;
      if (name === "Tutankhamun") return `/images/pharaohs/Tutankhamun.jpg`;
    } else {
      if (name === "Pyramids of Giza") return `/images/landmarks/Pyramids of Giza.webp`;
      if (name === "Sphinx") return `/images/landmarks/Sphinx.jpg`;
      if (name === "Temple of Karnak") return `/images/landmarks/Temple of Karnak.jpg`;
      if (name === "Temple of Luxor") return `/images/landmarks/Temple of Luxor.jpg`;
      if (name === "The Great Temple of Ramesses II at Abu Simbel") return `/images/landmarks/The Great Temple of Ramesses II at Abu Simbel.webp`;
    }
    return null;
  };

  const isPharaoh = entityType === "pharaoh" || entityType === "king";
  const staticUrl = getAssumedImageUrl(entityName, isPharaoh);

  // Fallback: if no static image, try the user's uploaded image from sessionStorage
  // Load entity metadata (period/location) from session
  const [statusText, setStatusText] = useState("");
  useEffect(() => {
    const payload = loadResultFromSession();
    if (payload?.result?.entity) {
      const e = payload.result.entity;
      if (entityType === "landmark") {
        setStatusText(e.location || "Ancient Landmark");
      } else {
        setStatusText(e.period || e.dynasty || "Ancient Pharaoh");
      }
    } else {
      setStatusText(entityType === "landmark" ? "Ancient Landmark" : "Ancient Pharaoh");
    }
  }, [entityType]);

  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);

  useEffect(() => {
    const payload = loadResultFromSession();
    // Prioritize captured image if it matches the current entity
    if (payload?.imageDataUrl && payload?.result?.entity?.name === entityName) {
      setAvatarUrl(payload.imageDataUrl);
    } else if (staticUrl) {
      // Fallback to professional archive image
      setAvatarUrl(staticUrl);
    }
  }, [staticUrl, entityName]);
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "assistant", text: t("chat.welcome", { name: entityName }), ts: Date.now() },
  ]);

  // Update welcome message if language changes
  useEffect(() => {
    setMessages(prev => {
      if (prev.length === 1 && prev[0].id === "1") {
        return [{ ...prev[0], text: t("chat.welcome", { name: entityName }) }];
      }
      return prev;
    });
  }, [language, t, entityName]);

  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [threadId] = useState(() => `thread_${Math.random().toString(36).slice(2)}`);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Mock chat history
  const chatHistory = [
    { id: "h1", title: "The Curse of Tutankhamun", date: "Today" },
    { id: "h2", title: "Pyramids Engineering Secrets", date: "Yesterday" },
    { id: "h3", title: "Hieroglyphs Translation", date: "2 days ago" },
  ];

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  useEffect(() => {
    const initChat = async () => {
      try {
        await fetch(`${API_BASE}/api/v1/chat/init`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            thread_id: threadId,
            entity: entityName,
            entity_type: entityType,
          }),
        });
      } catch (e) {
        console.error("Session init failed:", e);
      }
    };
    initChat();
  }, [threadId, entityName, entityType]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const recordStartRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);
  const wasCancelledRef = useRef<boolean>(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, isTyping]);

  // ── Send message to real backend ──────────────────────────────────────
  const sendMessage = useCallback(async (text?: string, useVoice: boolean = false) => {
    const trimmed = (text ?? input).trim();
    if (!trimmed || isTyping) return;

    setMessages((m) => [...m, { id: generateId(), role: "user", text: trimmed, ts: Date.now() }]);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    setIsTyping(true);
    const assistantMsgId = generateId();
    let isStreamComplete = false;

    try {
      const res = await fetch(CHAT_API, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
          "Cache-Control": "no-store",
        },
        body: JSON.stringify({
          message: trimmed,
          thread_id: threadId,
          voice_mode: useVoice,
          entity: entityName,
          entity_type: entityType,
        }),
      });

      if (!res.ok) {
        let errDetail = t("chat.error.api");
        try {
          const errData = await res.json();
          errDetail = errData.detail || errDetail;
        } catch {
          const text = await res.text();
          if (text) errDetail = text;
        }
        throw new Error(errDetail);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("Could not read message stream.");

      const decoder = new TextDecoder();

      let fullText = "";
      let displayedText = "";
      let firstChunk = true;
      let buffer = "";

      // Typewriter drain interval - separates network speed from visual speed
      const typewriterId = setInterval(() => {
        if (displayedText.length < fullText.length) {
          // Reveal 8 characters every 5ms for an ultra-fast terminal-like blur speed
          const nextChunk = fullText.slice(displayedText.length, displayedText.length + 8);
          displayedText += nextChunk;
          setMessages((m) => m.map(msg => msg.id === assistantMsgId ? { ...msg, text: displayedText, isSearching: false } : msg));
        } else if (isStreamComplete) {
          clearInterval(typewriterId);
        }
      }, 10);

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          let newlineIdx;
          while ((newlineIdx = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, newlineIdx);
            buffer = buffer.slice(newlineIdx + 1);

            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).trim();
              if (dataStr === "[DONE]") continue;

              let data: any = null;
              try {
                data = JSON.parse(dataStr);
              } catch (e) {
                continue;
              }

              if (data.error) throw new Error(data.error);

              // ── Agentic Search Indicator ──
              if (data.tool === "tavily_search" || data.search || data.event === "on_tool_start" || data.tool_calls || data.name === "tavily_search_results_json" || data.name === "search_tool") {
                if (firstChunk) {
                  setIsTyping(false);
                  setMessages((m) => [...m, { id: assistantMsgId, role: "assistant", text: "", ts: Date.now(), isSearching: true }]);
                  firstChunk = false;
                } else {
                  setMessages((m) => m.map(msg => msg.id === assistantMsgId ? { ...msg, isSearching: true } : msg));
                }
                continue;
              }

              if (firstChunk) {
                setIsTyping(false);
                setMessages((m) => [...m, { id: assistantMsgId, role: "assistant", text: "", ts: Date.now() }]);
                firstChunk = false;
              }

              if (data.text !== undefined) {
                fullText += data.text; // Just append to fullText; setInterval handles display
              }

              if (data.audio_url) {
                const url = data.audio_url.startsWith("data:") ? data.audio_url : `${API_BASE}${data.audio_url}`;
                setMessages((m) => m.map(msg => msg.id === assistantMsgId ? { ...msg, audioUrl: url } : msg));
                if (useVoice && audioRef.current) {
                  audioRef.current.src = url;
                  audioRef.current.play().catch(() => { });
                }
              }
            }
          }
        }

        if (firstChunk) {
          setIsTyping(false);
          setMessages((m) => [...m, { id: assistantMsgId, role: "assistant", text: t("chat.error.unreachable"), ts: Date.now() }]);
        }
      } finally {
        isStreamComplete = true;
      }
    } catch (err: any) {
      isStreamComplete = true;
      console.error("[Chat] Error:", err);

      setMessages((m) => {
        const hasText = m.find(msg => msg.id === assistantMsgId)?.text;
        if (!hasText) return m.filter((msg) => msg.id !== assistantMsgId);
        return m;
      });

      const errorMsg = err.message || t("chat.error.disrupted");
      setMessages((m) => [...m, {
        id: generateId(),
        role: "assistant",
        text: `[System Error] ${errorMsg}`,
        ts: Date.now(),
      }]);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping, threadId, t, entityName, entityType]);

  // ── Voice recording ────────────────────────────────────────────────────
  const stopVAD = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    silenceStartRef.current = null;
  }, []);

  const stopRecording = useCallback(() => {
    stopVAD();
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }, [stopVAD]);

  const cancelRecording = useCallback(() => {
    wasCancelledRef.current = true;
    stopVAD();
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      mediaRecorderRef.current.stop();
      audioChunksRef.current = [];
    }
    setRecordingState("idle");
  }, [stopVAD]);

  const startRecording = useCallback(async () => {
    if (recordingState !== "idle") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      audioChunksRef.current = [];
      recordStartRef.current = Date.now();
      silenceStartRef.current = null;

      const mimeTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"];
      const mime = mimeTypes.find(t => MediaRecorder.isTypeSupported(t)) ?? "";
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());

        if (wasCancelledRef.current) {
          wasCancelledRef.current = false;
          setRecordingState("idle");
          return;
        }

        const chunks = audioChunksRef.current;
        if (chunks.length === 0) { setRecordingState("idle"); return; }

        const blob = new Blob(chunks, { type: mime || "audio/webm" });
        setRecordingState("processing");

        try {
          const form = new FormData();
          form.append("audio", blob, "recording.webm");
          const r = await fetch(STT_API, { method: "POST", body: form });
          const d = await r.json();

          // Clear "Transcribing..." immediately after receiving text
          setRecordingState("idle");

          if (d.text?.trim()) {
            sendMessage(d.text.trim(), true);
          }
        } catch {
          console.error("[STT] Transcription failed");
          setRecordingState("idle");
        }
      };

      recorder.start(200);
      setRecordingState("recording");

      // VAD loop — auto-stop on silence
      const dataArr = new Float32Array(analyser.fftSize);
      const vadLoop = () => {
        analyser.getFloatTimeDomainData(dataArr);
        const vol = Math.sqrt(dataArr.reduce((s, v) => s + v * v, 0) / dataArr.length);
        const elapsed = Date.now() - recordStartRef.current;

        if (elapsed > MIN_DURATION_MS) {
          if (vol < SILENCE_THRESHOLD) {
            if (silenceStartRef.current === null) silenceStartRef.current = Date.now();
            if (Date.now() - silenceStartRef.current >= SILENCE_DURATION_MS) {
              stopRecording();
              return;
            }
          } else {
            silenceStartRef.current = null;
          }
        }
        rafRef.current = requestAnimationFrame(vadLoop);
      };
      rafRef.current = requestAnimationFrame(vadLoop);
    } catch {
      setRecordingState("idle");
    }
  }, [recordingState, stopRecording, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const chatHeader = (
    <div className="w-full pt-[110px] relative">
      <div className="w-full max-w-5xl mx-auto relative flex flex-col items-center pb-4 px-3 md:px-4">

        <div className="flex flex-col items-center text-center gap-2 pointer-events-auto">
          <motion.div 
            animate={{ 
              scale: [1, 1.04, 1],
              boxShadow: [
                "0 0 20px rgba(230,178,60,0.2)",
                "0 0 40px rgba(230,178,60,0.4)",
                "0 0 20px rgba(230,178,60,0.2)"
              ]
            }}
            transition={{ 
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="h-14 w-14 md:h-16 md:w-16 rounded-full bg-gradient-to-br from-[#E6B23C] to-[#D4A030] p-[2px]"
          >
            <div className="h-full w-full rounded-full bg-[#0D0A07] overflow-hidden flex items-center justify-center">
              {avatarUrl ? (
                <img src={avatarUrl} alt={entityName} className="w-full h-full object-cover object-center scale-110" onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
              ) : (
                <span className="text-[#E6B23C] text-4xl leading-none">☥</span>
              )}
            </div>
          </motion.div>
          <div className="space-y-0.5">
            <h1 className="font-heading text-xl md:text-2xl font-bold text-[#F5E6D0] tracking-wide">{entityName}</h1>
            <div className="text-[9px] md:text-[10px] font-bold tracking-[0.4em] text-[#E6B23C] uppercase opacity-70">{statusText}</div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <PageShell fullScreen headerExtension={chatHeader}>
      <audio ref={audioRef} />
      <div className="flex h-full w-full bg-transparent overflow-hidden" dir="ltr">
        {/* Sidebar - Collapsible */}
        <motion.aside
          initial={false}
          animate={{ width: sidebarOpen ? 300 : 72 }}
          className="h-full border-r border-[#E6B23C]/10 bg-[#0D0A07]/95 flex flex-col z-[60] relative"
        >
          {/* Top Icons */}
          <div className="p-4 flex flex-col items-center gap-6">
            {/* Menu Button */}
            <div className="relative group">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="h-10 w-10 flex items-center justify-center rounded-lg hover:bg-[#E6B23C]/10 text-[#A08E70] hover:text-[#E6B23C] transition-all"
              >
                <Menu size={22} />
              </button>
              {!sidebarOpen && (
                <span className="absolute left-full ml-4 px-2 py-1 bg-[#1A1208] border border-[#E6B23C]/20 text-[#E6B23C] text-[10px] uppercase font-bold tracking-widest rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none">
                  {t("chat.sidebar.expand")}
                </span>
              )}
            </div>

            {/* New Chat Button */}
            <div className="relative group">
              <button
                onClick={() => window.location.reload()}
                className="h-10 w-10 flex items-center justify-center rounded-lg hover:bg-[#E6B23C]/10 text-[#A08E70] hover:text-[#E6B23C] transition-all"
              >
                <SquarePen size={22} />
              </button>
              {!sidebarOpen && (
                <span className="absolute left-full ml-4 px-2 py-1 bg-[#1A1208] border border-[#E6B23C]/20 text-[#E6B23C] text-[10px] uppercase font-bold tracking-widest rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none">
                  {t("chat.sidebar.new")}
                </span>
              )}
            </div>
          </div>

          {/* Expanded Content */}
          <AnimatePresence>
            {sidebarOpen && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex-1 flex flex-col px-4 pb-4 overflow-hidden"
              >
                {/* Search Bar */}
                <div className="relative mt-2 mb-6">
                  <Search className={`absolute ${isRTL ? 'right-3' : 'left-3'} top-1/2 -translate-y-1/2 text-[#A08E70]/40`} size={14} />
                  <input
                    type="text"
                    placeholder={t("chat.sidebar.search")}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className={`w-full h-10 ${isRTL ? 'pr-9 pl-4' : 'pl-9 pr-4'} rounded-lg bg-[#1A1208] border border-[#E6B23C]/10 text-xs text-[#F5E6D0] placeholder:text-[#A08E70]/30 focus:outline-none focus:border-[#E6B23C]/30 transition-all`}
                  />
                </div>

                {/* Chat List */}
                <div className="flex-1 overflow-y-auto space-y-1 trending-scrollbar-hide">
                  <div className="px-2 mb-2 text-[10px] font-bold uppercase tracking-[0.2em] text-[#A08E70]/60">{t("chat.sidebar.history")}</div>
                  {chatHistory.map((chat) => (
                    <button
                      key={chat.id}
                      className="w-full flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-[#E6B23C]/5 group transition-all text-left"
                    >
                      <MessageSquare size={14} className="text-[#A08E70] group-hover:text-[#E6B23C] shrink-0" />
                      <div className="flex-1 overflow-hidden">
                        <div className="text-[11px] font-medium text-[#A08E70] group-hover:text-[#F5E6D0] truncate transition-colors">{chat.title}</div>
                        <div className="text-[9px] text-[#A08E70]/40 mt-0.5">{chat.date}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.aside>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-transparent">
          {/* Messages Area - Positioned below the fixed header area */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto trending-scrollbar-hide relative mt-48 md:mt-42">
            <div className="max-w-5xl mx-auto w-full p-4 md:p-8 space-y-8 pb-32" style={{ direction: 'ltr' }}>
              <AnimatePresence>
                {messages.map((msg) =>
                  msg.role === "assistant" ? (
                    <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
                      className="flex flex-col gap-3 max-w-3xl"
                    >
                      <div className="flex flex-col gap-3 w-full">
                        {msg.isSearching && (
                          <AnimatePresence>
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="flex items-center gap-3 px-4 py-2.5 bg-[#E6B23C]/5 border border-[#E6B23C]/10 rounded-xl w-fit text-[#E6B23C] shadow-[0_0_20px_rgba(230,178,60,0.05)] overflow-hidden mb-2"
                            >
                              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 3, ease: "linear" }}>
                                <Scroll size={14} />
                              </motion.div>
                              <span className="text-[10px] font-bold tracking-widest uppercase">{t("chat.searching")}</span>
                            </motion.div>
                          </AnimatePresence>
                        )}

                        <div
                          className="text-[#D4C4A8] text-sm md:text-base leading-relaxed font-normal tracking-wide"
                          style={{ direction: isRTL ? 'rtl' : 'ltr', textAlign: isRTL ? 'right' : 'left' }}
                        >
                          {renderMessageText(msg.text)}
                        </div>

                        <div className={`flex items-center gap-6 mt-1 opacity-40 hover:opacity-100 transition-opacity ${isRTL ? 'flex-row-reverse' : ''}`}>
                          <span className="text-[10px] font-medium tracking-tighter text-[#A08E70]">
                            {new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                          <button
                            onClick={() => handleCopy(msg.text, msg.id)}
                            className="flex items-center gap-2 text-[#A08E70] hover:text-[#E6B23C] transition-colors"
                          >
                            {copiedId === msg.id ? (
                              <>
                                <Check size={12} className="text-[#E6B23C]" />
                                <span className="text-[9px] font-bold uppercase tracking-widest text-[#E6B23C]">{t("chat.copied")}</span>
                              </>
                            ) : (
                              <>
                                <Copy size={12} />
                                <span className="text-[9px] font-bold uppercase tracking-widest">{t("chat.copy")}</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
                      className="flex justify-end max-w-2xl ml-auto group"
                    >
                      <div className="flex flex-col items-end gap-3 text-right">
                        <div className="px-6 py-3 rounded-[24px] bg-[#E6B23C]/10 border border-[#E6B23C]/20 shadow-[0_4px_20px_rgba(230,178,60,0.05)]">
                          <div
                            className="text-[#E6B23C] text-sm md:text-base leading-relaxed font-normal tracking-wide"
                            style={{ direction: isRTL ? 'rtl' : 'ltr', textAlign: isRTL ? 'right' : 'left' }}
                          >
                            {renderMessageText(msg.text)}
                          </div>
                        </div>

                        <div className={`flex items-center gap-6 mt-1 px-2 ${isRTL ? 'flex-row-reverse' : ''}`}>
                          <button
                            onClick={() => handleCopy(msg.text, msg.id)}
                            className="flex items-center gap-2 text-[#A08E70] hover:text-[#E6B23C] transition-all opacity-0 group-hover:opacity-100"
                          >
                            {copiedId === msg.id ? (
                              <>
                                <Check size={12} className="text-[#E6B23C]" />
                                <span className="text-[9px] font-bold uppercase tracking-widest text-[#E6B23C]">{t("chat.copied")}</span>
                              </>
                            ) : (
                              <>
                                <Copy size={12} />
                                <span className="text-[9px] font-bold uppercase tracking-widest">{t("chat.copy")}</span>
                              </>
                            )}
                          </button>
                          <span className="text-[10px] font-medium tracking-tighter text-[#A08E70] opacity-40">
                            {new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </div>
                      </div>
                    </motion.div>
                  )
                )}
                {isTyping && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-3 items-start px-2">
                    <motion.div
                      animate={{
                        scale: [1, 1.25, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{
                        repeat: Infinity,
                        duration: 2,
                        ease: "easeInOut"
                      }}
                      className="h-3 w-3 rounded-full bg-[#E6B23C] shadow-[0_0_10px_rgba(230,178,60,0.5)]"
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Recording overlay */}
            <AnimatePresence>
              {recordingState === "recording" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none"
                  style={{ background: "rgba(13,10,7,0.6)", backdropFilter: "blur(4px)" }}
                >
                  <div className="relative flex flex-col items-center gap-8 pointer-events-auto">
                    {/* The Liquid Blob Container */}
                    <div className="relative h-32 w-32 md:h-[152px] md:w-[152px] flex items-center justify-center">
                      {/* Organic Glow (Behind) */}
                      <motion.div
                        animate={{
                          scale: [1, 1.4, 1],
                          opacity: [0.3, 0.1, 0.3],
                          borderRadius: [
                            "60% 40% 30% 70% / 60% 30% 70% 40%",
                            "30% 60% 70% 40% / 50% 60% 30% 60%",
                            "60% 40% 30% 70% / 60% 30% 70% 40%"
                          ]
                        }}
                        transition={{ repeat: Infinity, duration: 8, ease: "easeInOut" }}
                        className="absolute inset-0 bg-[#E6B23C] blur-3xl pointer-events-none"
                      />

                      {/* Rotating Liquid Shape */}
                      <motion.div
                        animate={{
                          borderRadius: [
                            "60% 40% 30% 70% / 60% 30% 70% 40%",
                            "30% 60% 70% 40% / 50% 60% 30% 60%",
                            "60% 40% 30% 70% / 60% 30% 70% 40%"
                          ],
                          rotate: [0, 360],
                        }}
                        transition={{
                          duration: 15,
                          repeat: Infinity,
                          ease: "linear"
                        }}
                        className="absolute inset-0 bg-gradient-to-br from-[#FFE6A9] via-[#E6B23C] to-[#B48B2D] shadow-[0_0_60px_rgba(230,178,60,0.4)]"
                      />

                      {/* Static Heartbeat Pulse (Icon stays upright) */}
                      <motion.div
                        animate={{ scale: [1, 1.25, 1] }}
                        transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
                        className="relative z-10"
                      >
                        <Mic size={36} className="text-[#1A1208] drop-shadow-lg" />
                      </motion.div>
                    </div>

                    {/* Status & Cancel Button (Outside the rotation) */}
                    <div className="flex flex-col items-center gap-4">
                      <motion.div
                        animate={{ opacity: [1, 0.4, 1] }}
                        transition={{ repeat: Infinity, duration: 2 }}
                      >
                        <span className="text-sm font-bold tracking-[0.5em] uppercase text-[#E6B23C] drop-shadow-md">{t("chat.listening")}</span>
                      </motion.div>

                      <button
                        onClick={cancelRecording}
                        className="mt-4 px-10 py-3 rounded-full bg-[#0D0A07]/40 backdrop-blur-md border border-[#E6B23C]/20 text-[#E6B23C] text-[11px] font-bold uppercase tracking-[0.3em] hover:bg-[#E6B23C] hover:text-[#0D0A07] transition-all hover:scale-105 active:scale-95 shadow-xl"
                      >
                        {t("chat.cancel")}
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Input Bar - Floating & Minimal - Centered relative to screen */}
          <div className="w-full shrink-0 z-10">
            <div className="p-4 md:p-8 md:pb-12 bg-transparent max-w-5xl mx-auto">
              <div className="flex gap-3 md:gap-4 items-center max-w-4xl mx-auto relative">
                {recordingState === "processing" ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex-1 h-14 px-8 rounded-full bg-[#1A1208]/50 backdrop-blur-xl border border-[#E6B23C]/20 flex items-center gap-3"
                  >
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                      className="h-4 w-4 rounded-full border-2 border-[#E6B23C] border-t-transparent"
                    />
                    <span className="text-[11px] font-bold tracking-[0.3em] uppercase text-[#E6B23C]">{t("chat.transcribing")}</span>
                  </motion.div>
                ) : (
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => {
                      setInput(e.target.value);
                      e.target.style.height = "auto";
                      e.target.style.height = `${e.target.scrollHeight}px`;
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    disabled={recordingState !== "idle"}
                    placeholder={t("chat.placeholder", { name: entityName })}
                    rows={1}
                    className="flex-1 min-h-[56px] max-h-48 py-4 px-8 rounded-[28px] bg-[#1A1208]/50 backdrop-blur-xl border border-[#E6B23C]/10 text-base placeholder:text-[#A08E70]/40 focus:outline-none focus:border-[#E6B23C]/30 focus:bg-[#1A1208]/80 focus:shadow-[0_0_30px_rgba(230,178,60,0.05)] transition-all disabled:opacity-50 resize-none overflow-y-auto trending-scrollbar-hide"
                    style={{ color: "#E6B23C", caretColor: "#E6B23C", direction: isRTL ? 'rtl' : 'ltr' }}
                  />
                )}

                {/* Smart send/mic button */}
                <AnimatePresence mode="wait">
                  {input.trim() ? (
                    <motion.div key="send" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: "spring", stiffness: 400, damping: 20 }}>
                      <Button
                        onClick={() => sendMessage()}
                        disabled={isTyping}
                        className="h-12 w-12 shrink-0 rounded-full bg-gradient-to-br from-[#E6B23C] to-[#D4A030] hover:from-[#FFD369] hover:to-[#E6B23C] text-[#0D0A07] transition-all hover:scale-105 hover:shadow-[0_0_20px_rgba(230,178,60,0.3)] disabled:opacity-30 disabled:hover:scale-100"
                      >
                        <Send size={18} />
                      </Button>
                    </motion.div>
                  ) : (
                    <motion.div key="mic" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: "spring", stiffness: 400, damping: 20 }}>
                      <button
                        onClick={recordingState === "idle" ? startRecording : cancelRecording}
                        className="group relative h-14 w-14 shrink-0 rounded-full flex items-center justify-center leading-none transition-all hover:scale-105"
                        style={
                          recordingState === "recording"
                            ? { background: "linear-gradient(135deg, #C53030, #9B2C2C)", boxShadow: "0 0 20px rgba(197,48,48,0.5)", color: "#fff" }
                            : { background: "#0D0A07", border: "1px solid rgba(230,178,60,0.3)", color: "#E6B23C" }
                        }
                      >
                        {recordingState === "recording" ? <MicOff size={18} /> : (
                          <svg width="24" height="22" viewBox="0 0 24 20" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" style={{ display: "block", margin: "auto" }}>
                            <path d="M2 9v2" />
                            <path d="M6 5v10" />
                            <path d="M10 2v16" />
                            <path d="M14 5v10" />
                            <path d="M18 9v2" />
                            <path d="M22 7v6" />
                          </svg>
                        )}

                        {/* Premium Tooltip */}
                        <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-[#0D0A07] border border-[#E6B23C]/30 text-[#E6B23C] text-[10px] uppercase font-bold tracking-wider rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap shadow-[0_0_10px_rgba(230,178,60,0.15)]">
                          {language === "AR" ? "استخدم الصوت" : language === "FR" ? "Utiliser la voix" : "Use voice"}
                        </span>
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </div>
      </div>
    </PageShell>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="h-full flex-1" style={{ background: "#0D0A07" }} />}>
      <ChatContent />
    </Suspense>
  );
}
