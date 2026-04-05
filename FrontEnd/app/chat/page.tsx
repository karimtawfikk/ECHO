"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import Image from "next/image";
import PageShell from "../../components/feature/PageShell";
import { Button } from "../../components/ui/button";
import { Send, Scroll, Mic, MicOff, X, Volume2, VolumeX } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Suspense } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  ts: number;
  audioUrl?: string;
}

type RecordingState = "idle" | "recording" | "processing";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";
const CHAT_API = `${API_BASE}/api/v1/pharaoh/chat`;
const STT_API  = `${API_BASE}/api/v1/pharaoh/transcribe`;
const TUT_AVATAR = "/tut.png";

// Voice auto-stop config
const SILENCE_THRESHOLD = 0.015;
const SILENCE_DURATION_MS = 1500;
const MIN_DURATION_MS = 1000;

function ChatContent() {
  const sp = useSearchParams();
  const entityName = sp.get("entity") || "Ramesses II";
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "assistant", text: `Greetings, traveler. I am ${entityName}, Son of Ra. You stand before a legacy that spans millennia. Ask me anything about my reign, my world, or the secrets of the ancients.`, ts: Date.now() },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [muteAudio, setMuteAudio] = useState(false);
  const [threadId] = useState(() => `thread_${Math.random().toString(36).slice(2)}`);

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
  const sendMessage = useCallback(async (text?: string) => {
    const trimmed = (text ?? input).trim();
    if (!trimmed || isTyping) return;

    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", text: trimmed, ts: Date.now() }]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch(CHAT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          thread_id: threadId,
          voice_mode: !muteAudio,
          entity: entityName,
        }),
      });

      if (!res.ok) {
        let errDetail = "The ancient scrolls are currently unreadable (API error).";
        try {
          const errData = await res.json();
          errDetail = errData.detail || errDetail;
        } catch {
          // Fallback to text if JSON fails
          const text = await res.text();
          if (text) errDetail = text;
        }
        throw new Error(errDetail);
      }

      const data = await res.json();
      const replyText = data.response ?? "The sands of time veil my words. Ask again, traveler.";
      const audioUrl = data.audio_url ? `${API_BASE}${data.audio_url}` : undefined;

      setMessages((m) => [...m, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: replyText,
        ts: Date.now(),
        audioUrl,
      }]);

      // Auto-play TTS
      if (audioUrl && !muteAudio && audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play().catch(() => {});
      }
    } catch (err: any) {
      console.error("[Chat] Error:", err);
      const errorMessage = err.message || "The connection to the ancient realm was disrupted. Please try again.";
      
      setMessages((m) => [...m, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: errorMessage,
        ts: Date.now(),
      }]);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping, threadId, muteAudio]);

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
            await sendMessage(d.text.trim());
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
                <Image src={TUT_AVATAR} alt={entityName} width={56} height={56} className="object-cover" onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
              </div>
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 h-3 w-3 md:h-4 md:w-4 rounded-full bg-[#2A7B6F] border-2 border-[#0D0A07] shadow-[0_0_8px_rgba(42,123,111,0.5)]" />
          </div>
          <div className="flex-1">
            <h1 className="font-heading text-lg md:text-xl font-bold text-[#F5E6D0]">{entityName}</h1>
            <div className="text-[9px] md:text-[10px] font-bold tracking-[0.2em] text-[#2A7B6F] uppercase">Online • Living Dialogue</div>
          </div>
          {/* Mute toggle */}
          <button
            onClick={() => setMuteAudio(m => !m)}
            className="h-9 w-9 md:h-10 md:w-10 rounded-full flex items-center justify-center transition-all"
            style={{
              background: muteAudio ? "rgba(230,178,60,0.06)" : "rgba(230,178,60,0.12)",
              border: "1px solid rgba(230,178,60,0.2)",
              color: muteAudio ? "#A08E70" : "#E6B23C",
            }}
            title={muteAudio ? "Unmute voice" : "Mute voice"}
          >
            {muteAudio ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>
        </motion.div>

        {/* Messages Area */}
        <div className="flex-1 flex flex-col overflow-hidden relative" style={{ minHeight: "0" }}>
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-3 trending-scrollbar-hide pb-20">
            <AnimatePresence>
              {messages.map((msg) =>
                msg.role === "assistant" ? (
                  <motion.div key={msg.id} initial={{ opacity: 0, x: -20, scale: 0.95 }} animate={{ opacity: 1, x: 0, scale: 1 }} transition={{ type: "spring", damping: 20 }}
                    className="flex gap-3 max-w-[80%]"
                  >
                    <div className="h-8 w-8 rounded-full bg-gradient-to-br from-[#E6B23C]/20 to-[#E6B23C]/5 flex-shrink-0 flex items-center justify-center mt-1">
                      <Scroll size={14} className="text-[#E6B23C]" />
                    </div>
                    <div className="bubble-assistant px-5 py-3.5 shadow-lg break-words break-all">
                      <p className="text-sm leading-relaxed">{msg.text}</p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className="text-[9px] text-[#A08E70]/60">{new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                        {msg.audioUrl && !muteAudio && (
                          <button
                            onClick={() => { if (audioRef.current) { audioRef.current.src = msg.audioUrl!; audioRef.current.play().catch(() => {}); } }}
                            className="opacity-50 hover:opacity-100 transition-opacity"
                            title="Replay voice"
                          >
                            <Volume2 size={10} color="#E6B23C" />
                          </button>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key={msg.id} initial={{ opacity: 0, x: 20, scale: 0.95 }} animate={{ opacity: 1, x: 0, scale: 1 }} transition={{ type: "spring", damping: 20 }}
                    className="flex justify-end max-w-[80%] ml-auto"
                  >
                    <div className="bubble-user px-5 py-3.5 shadow-lg break-words break-all">
                      <p className="text-sm leading-relaxed font-medium">{msg.text}</p>
                      <div className="text-[9px] text-[#0D0A07]/50 mt-1.5 text-right">{new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</div>
                    </div>
                  </motion.div>
                )
              )}
              {isTyping && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3 items-center">
                  <div className="h-8 w-8 rounded-full bg-gradient-to-br from-[#E6B23C]/20 to-[#E6B23C]/5 flex-shrink-0 flex items-center justify-center">
                    <Scroll size={14} className="text-[#E6B23C]" />
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
                  <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: "#E6B23C" }}>Transcribing…</span>
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
                disabled={recordingState !== "idle" || isTyping}
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
                      className="h-12 w-12 shrink-0 rounded-full flex items-center justify-center transition-all hover:scale-105"
                      style={
                        recordingState === "recording"
                          ? { background: "linear-gradient(135deg, #C53030, #9B2C2C)", boxShadow: "0 0 20px rgba(197,48,48,0.5)", color: "#fff" }
                          : { background: "rgba(230,178,60,0.12)", border: "1px solid rgba(230,178,60,0.3)", color: "#E6B23C" }
                      }
                    >
                      {recordingState === "recording" ? <MicOff size={18} /> : <Mic size={18} />}
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
