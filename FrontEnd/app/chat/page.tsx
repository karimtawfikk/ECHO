"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import Image from "next/image";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { Send, Scroll, Mic, MicOff, X, Volume2, VolumeX } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Suspense } from "react";
import { loadResultFromSession } from "../../lib/services/recognition";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  ts: number;
  audioUrl?: string;
  isSearching?: boolean;
}

type RecordingState = "idle" | "recording" | "processing";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";
const CHAT_API = `${API_BASE}/api/v1/chat/chat`;
const STT_API = `${API_BASE}/api/v1/chat/transcribe`;
const TUT_AVATAR = "/tut.png";

// Voice auto-stop config
const SILENCE_THRESHOLD = 0.015;
const SILENCE_DURATION_MS = 1500;
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
  const sp = useSearchParams();
  const entityName = sp.get("entity") ?? "Ancient Spirit";
  const entityType = sp.get("type") || "pharaoh";

  const getEntityImageUrl = (name: string, type: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";
    const isPharaoh = type === "pharaoh" || type === "king";
    if (isPharaoh) {
      if (name === "Akhenaton") return `${baseUrl}/static/images/pharaohs/Akhenaton.JPG`;
      if (name === "Cleopatra VII Philopator") return `${baseUrl}/static/images/pharaohs/Cleopatra%20VII%20Philopator.jpg`;
      if (name === "Hatshepsut") return `${baseUrl}/static/images/pharaohs/Hatshepsut.JPG`;
      if (name === "Ramesses II") return `${baseUrl}/static/images/pharaohs/Ramesses%20II.jpg`;
      if (name === "Tutankhamun") return `${baseUrl}/static/images/pharaohs/Tutankhamun.jpg`;
    } else {
      if (name === "Pyramids of Giza") return `${baseUrl}/static/images/landmarks/Pyramids%20of%20Giza.webp`;
      if (name === "Sphinx") return `${baseUrl}/static/images/landmarks/Sphinx.jpg`;
      if (name === "Temple of Karnak") return `${baseUrl}/static/images/landmarks/Temple%20of%20Karnak.jpg`;
      if (name === "Temple of Luxor") return `${baseUrl}/static/images/landmarks/Temple%20of%20Luxor.jpg`;
      if (name === "The Great Temple of Ramesses II at Abu Simbel") return `${baseUrl}/static/images/landmarks/The%20Great%20Temple%20of%20Ramesses%20II%20at%20Abu%20Simbel.webp`;
    }
    return null;
  };

  const staticUrl = getEntityImageUrl(entityName, entityType);

  // Fallback: if no static image, try the user's uploaded image from sessionStorage
  const [avatarUrl, setAvatarUrl] = useState<string | null>(staticUrl);
  useEffect(() => {
    if (!staticUrl) {
      const payload = loadResultFromSession();
      if (payload?.imageDataUrl) {
        setAvatarUrl(payload.imageDataUrl);
      }
    }
  }, [staticUrl]);
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "assistant", text: `Greetings, I am ${entityName}. You stand before a legacy that spans millennia. Ask me anything about my reign, my world, or the secrets of the ancients.`, ts: Date.now() },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [threadId] = useState(() => `thread_${Math.random().toString(36).slice(2)}`);

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

    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", text: trimmed, ts: Date.now() }]);
    setInput("");
    setIsTyping(true);
    const assistantMsgId = crypto.randomUUID();
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
        let errDetail = "The ancient scrolls are currently unreadable (API error).";
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
          setMessages((m) => [...m, { id: assistantMsgId, role: "assistant", text: "I'm sorry, that lies beyond what my stones remember.", ts: Date.now() }]);
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

      const errorMsg = err.message || "The connection to the ancient realm was disrupted.";
      setMessages((m) => [...m, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: `[System Error] ${errorMsg}`,
        ts: Date.now(),
      }]);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping, threadId]);

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
        const chunks = audioChunksRef.current;
        if (chunks.length === 0) { setRecordingState("idle"); return; }

        const blob = new Blob(chunks, { type: mime || "audio/webm" });
        setRecordingState("processing");

        try {
          const form = new FormData();
          form.append("audio", blob, "recording.webm");
          const r = await fetch(STT_API, { method: "POST", body: form });
          const d = await r.json();
          if (d.text?.trim()) {
            await sendMessage(d.text.trim(), true);
          }
        } catch {
          console.error("[STT] Transcription failed");
        } finally {
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

  return (
    <>
      <audio ref={audioRef} />
      <div className="flex-1 flex flex-col h-full w-full max-w-4xl mx-auto shadow-2xl glass-surface overflow-hidden md:border-x md:border-[#E6B23C]/10">
        {/* Chat Header */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 md:gap-4 p-3 md:p-4 bg-[#1A1208]/80 backdrop-blur-md border-b border-[#E6B23C]/10 shrink-0 z-10"
        >
          <Link href="/result" className="flex items-center justify-center h-10 w-10 text-xl rounded-full hover:bg-[#E6B23C]/10 text-[#A08E70] hover:text-[#E6B23C] transition-colors">
            ←
          </Link>
          <div className="relative">
            <div className="h-12 w-12 md:h-14 md:w-14 rounded-full bg-gradient-to-br from-[#E6B23C] to-[#D4A030] p-[2px] shadow-[0_0_20px_rgba(230,178,60,0.3)]">
              <div className="h-full w-full rounded-full bg-[#1A1208] overflow-hidden flex items-center justify-center">
                {avatarUrl ? (
                  <img src={avatarUrl} alt={entityName} className="w-full h-full object-cover object-center" onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
                ) : (
                  <span className="text-[#E6B23C] text-3xl leading-none">☥</span>
                )}
              </div>
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 h-3 w-3 md:h-4 md:w-4 rounded-full bg-[#22C55E] border-2 border-[#0D0A07] shadow-[0_0_8px_rgba(42,123,111,0.5)]" />
          </div>
          <div className="flex-1">
            <h1 className="font-heading text-lg md:text-xl font-bold text-[#F5E6D0]">{entityName}</h1>
            <div className="text-[9px] md:text-[10px] font-bold tracking-[0.2em] text-[#22C55E] uppercase">Online</div>
          </div>
        </motion.div>

        {/* Messages Area */}
        <div className="flex-1 flex flex-col overflow-hidden relative" style={{ minHeight: "0" }}>
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-3 trending-scrollbar-hide pb-20">
            <AnimatePresence>
              {messages.map((msg) =>
                msg.role === "assistant" ? (
                  <motion.div key={msg.id} initial={{ opacity: 0, x: -20, scale: 0.95 }} animate={{ opacity: 1, x: 0, scale: 1 }} transition={{ type: "spring", damping: 20 }}
                    className="flex gap-3 max-w-[90%] md:max-w-[85%]"
                  >
                    <div className="h-8 w-8 rounded-full bg-gradient-to-br from-[#E6B23C]/20 to-[#E6B23C]/5 flex-shrink-0 flex items-center justify-center mt-1">
                      <span className="text-[#E6B23C] text-sm leading-none">☥</span>
                    </div>
                    <div className="bubble-assistant px-5 py-3.5 shadow-lg break-words whitespace-pre-wrap w-full">
                      {msg.isSearching && (
                        <AnimatePresence>
                          <motion.div
                            initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                            animate={{ opacity: 1, height: 'auto', marginBottom: 12 }}
                            exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                            className="flex items-center gap-3 px-4 py-2.5 bg-[#E6B23C]/10 border border-[#E6B23C]/20 rounded-xl w-fit text-[#E6B23C] shadow-[0_0_15px_rgba(230,178,60,0.1)] overflow-hidden"
                          >
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 3, ease: "linear" }}>
                              <Scroll size={14} />
                            </motion.div>
                            <span className="text-[10px] font-bold tracking-widest uppercase">Consulting modern scrolls...</span>
                          </motion.div>
                        </AnimatePresence>
                      )}
                      {msg.text && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                          <p className="text-sm leading-relaxed">{renderMessageText(msg.text)}</p>
                        </motion.div>
                      )}

                      <div className="flex items-center gap-2 mt-1.5">
                        <span className="text-[9px] text-[#A08E70]/60">{new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key={msg.id} initial={{ opacity: 0, x: 20, scale: 0.95 }} animate={{ opacity: 1, x: 0, scale: 1 }} transition={{ type: "spring", damping: 20 }}
                    className="flex justify-end max-w-[90%] md:max-w-[85%] ml-auto"
                  >
                    <div className="bubble-user px-5 py-3.5 shadow-lg break-words whitespace-pre-wrap">
                      <p className="text-sm leading-relaxed font-medium">{renderMessageText(msg.text)}</p>
                      <div className="text-[9px] text-[#0D0A07]/50 mt-1.5 text-right">{new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</div>
                    </div>
                  </motion.div>
                )
              )}
              {isTyping && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3 items-center">
                  <div className="h-8 w-8 rounded-full bg-gradient-to-br from-[#E6B23C]/20 to-[#E6B23C]/5 flex-shrink-0 flex items-center justify-center">
                    <span className="text-[#E6B23C] text-sm leading-none">☥</span>
                  </div>
                  <div className="bubble-assistant px-5 py-3.5 flex gap-1.5">
                    {[0, 1, 2].map((i) => (
                      <motion.div key={i} animate={{ y: [-3, 3, -3], opacity: [0.4, 1, 0.4] }} transition={{ repeat: Infinity, duration: 1, delay: i * 0.15 }}
                        className="h-2.5 w-2.5 rounded-full bg-[#E6B23C]"
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Recording overlay */}
          <AnimatePresence>
            {recordingState === "recording" && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute bottom-20 left-4 right-4 z-20"
              >
                <div className="flex items-center gap-3 px-4 py-2.5 rounded-2xl" style={{ background: "rgba(180,30,30,0.12)", border: "1px solid rgba(220,50,50,0.3)" }}>
                  <motion.div
                    animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                    className="h-3 w-3 rounded-full shrink-0"
                    style={{ background: "#E53E3E", boxShadow: "0 0 10px rgba(229,62,62,0.6)" }}
                  />
                  <span className="flex-1 text-xs font-semibold tracking-widest uppercase" style={{ color: "#FC8181" }}>Listening…</span>
                  <button
                    onClick={cancelRecording}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-semibold transition-all hover:scale-105"
                    style={{ background: "rgba(229,62,62,0.15)", color: "#FC8181", border: "1px solid rgba(229,62,62,0.3)" }}
                  >
                    <X size={12} />
                    Cancel
                  </button>
                </div>
              </motion.div>
            )}
            {recordingState === "processing" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute bottom-20 left-4 right-4 z-20"
              >
                <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl" style={{ background: "rgba(230,178,60,0.08)", border: "1px solid rgba(230,178,60,0.2)" }}>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                    className="h-3.5 w-3.5 rounded-full border-2 border-t-transparent"
                    style={{ borderColor: "#E6B23C", borderTopColor: "transparent" }}
                  />
                  <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: "#E6B23C" }}>Transcribing…</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Input Bar */}
          <div className="p-3 md:p-4 bg-[#1A1208]/90 backdrop-blur-lg border-t border-[#E6B23C]/[0.1] shrink-0 z-10">
            <div className="flex gap-2 md:gap-3 items-center">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={recordingState !== "idle"}
                placeholder={`Message ${entityName}...`}
                className="flex-1 h-12 px-5 rounded-full bg-[#0D0A07] border border-[#E6B23C]/20 text-sm placeholder:text-[#A08E70]/50 focus:outline-none focus:border-[#E6B23C]/40 focus:shadow-[0_0_15px_rgba(230,178,60,0.1)] transition-all disabled:opacity-50"
                style={{ color: "#E6B23C", caretColor: "#E6B23C" }}
              />

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
                        Use voice
                      </span>
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default function ChatPage() {
  return (
    <PageShell fullScreen>
      <Suspense fallback={<div className="h-full flex-1" style={{ background: "#0D0A07" }} />}>
        <ChatContent />
      </Suspense>
    </PageShell>
  );
}
