"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import PageShell from "../../components/layout/PageShell";
import { Button } from "../../components/ui/button";
import { Send, Scroll, Mic, MicOff, Check, Copy, PanelLeft, SquarePen, Search, MessageSquare, ArrowLeft, MoreHorizontal, Pin, Pencil, Trash2, SlidersHorizontal, Plus, X, ChevronDown, Volume2, VolumeX } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Suspense } from "react";
import { createPortal } from "react-dom";
import { useLanguage } from "../../context/LanguageContext";
import { loadResultFromSession } from "../../lib/services/recognition";
import { createClient } from "../../lib/supabase/client";
import { cleanEntityName } from "../../lib/utils";


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
  const router = useRouter();
  const sp = useSearchParams();
  const entityName = sp.get("entity") ?? "Ancient Spirit";
  const cleanDisplayName = cleanEntityName(entityName);
  const entityType = sp.get("type") || "pharaoh";
  const convIdFromUrl = sp.get("conv");

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

  const [dbEntities, setDbEntities] = useState<{ pharaohs: any[]; landmarks: any[] } | null>(null);

  useEffect(() => {
    async function loadEntities() {
      try {
        const dbRes = await fetch(`${API_BASE}/api/v1/entities/all`);
        if (dbRes.ok) {
          const dbData = await dbRes.json();
          setDbEntities(dbData);
        }
      } catch (dbErr) {
        console.error("Error loading dbEntities in chat:", dbErr);
      }
    }
    loadEntities();
  }, []);

  const getEntityImage = (name: string, type: string) => {
    const isPharaoh = type.toLowerCase().includes("pharaoh") || type.toLowerCase().includes("king");
    const cleanName = (n: string) => cleanEntityName(n);
    const targetClean = cleanName(name).toLowerCase();

    // Try dynamic entities first to find image from DB
    if (dbEntities) {
      const list = isPharaoh ? dbEntities.pharaohs : dbEntities.landmarks;
      let found = list.find((e: any) => e.name.toLowerCase() === name.toLowerCase());
      if (!found) {
        found = list.find((e: any) => cleanName(e.name).toLowerCase() === targetClean);
      }
      if (!found) {
        found = list.find((e: any) =>
          e.name.toLowerCase().includes(targetClean) ||
          targetClean.includes(cleanName(e.name).toLowerCase())
        );
      }
      // Scan for composite sub-entities if not found at root level
      if (!found) {
        for (const parent of list) {
          if (parent.composite_entities_data) {
            const subMatch = parent.composite_entities_data.find(
              (sub: any) =>
                sub.name.toLowerCase() === name.toLowerCase() ||
                cleanName(sub.name).toLowerCase() === targetClean
            );
            if (subMatch) {
              found = subMatch;
              break;
            }
          }
        }
      }
      if (found && found.image) {
        if (found.image.startsWith('/') || found.image.startsWith('http')) return found.image;
        if (found.image.startsWith("data/")) return `${API_BASE}/api/v1/assets/r2/${encodeURI(found.image)}`;
        return isPharaoh
          ? "/images/pharaohs/Tutankhamun.jpg"
          : "/images/landmarks/Pyramids of Giza.webp";
      }
    }
    return isPharaoh
      ? "/images/pharaohs/Tutankhamun.jpg"
      : "/images/landmarks/Pyramids of Giza.webp";
  };

  const isPharaoh = entityType === "pharaoh" || entityType === "king";
  const staticUrl = getAssumedImageUrl(entityName, isPharaoh);

  // Fallback: if no static image, try the user's uploaded image from sessionStorage
  // Load entity metadata (period/location) from session or dynamic lookup
  const [statusText, setStatusText] = useState("");
  useEffect(() => {
    const payload = loadResultFromSession();
    const cleanName = (n: string) => cleanEntityName(n);

    const isDeity = (e: any) => {
      if (!e) return false;
      const type = (e.type || "").toLowerCase();
      const dynasty = (e.dynasty || "").toLowerCase();
      return (
        type.includes("god") ||
        type.includes("goddess")
      );
    };

    // Check direct match in session
    const directMatch = payload?.result?.entity && payload?.result?.entity?.name === entityName;
    // Check composite match in session
    const compositeSubMatch = !directMatch && payload?.result?.entity?.composite_entities_data?.find(
      (sub: any) => sub.name === entityName || cleanName(sub.name) === cleanName(entityName)
    );

    if (directMatch && payload?.result?.entity) {
      const e = payload.result.entity;
      if (isDeity(e)) {
        setStatusText("Gods & Deities");
      } else if (entityType === "landmark") {
        setStatusText(e.location || "Ancient Landmark");
      } else {
        setStatusText(e.period || e.dynasty || "Ancient Pharaoh");
      }
    } else if (compositeSubMatch) {
      const sub = compositeSubMatch as any;
      const parent = payload?.result?.entity;
      if (isDeity(sub)) {
        setStatusText("Gods & Deities");
      } else if (entityType === "landmark") {
        setStatusText(sub.location || parent?.location || "Ancient Landmark");
      } else {
        // Inherit parent's period or dynasty if the sub-entity doesn't have its own
        setStatusText(sub.period || parent?.period || sub.dynasty || parent?.dynasty || "Ancient Pharaoh");
      }
    } else {
      // Dynamic lookup from DB or mocks
      const isPharaoh = entityType === "pharaoh" || entityType === "king";
      let found: any = null;

      if (dbEntities) {
        const list = isPharaoh ? dbEntities.pharaohs : dbEntities.landmarks;
        const targetClean = cleanName(entityName).toLowerCase();

        // 1. Direct search
        found = list.find((e: any) =>
          e.name.toLowerCase() === entityName.toLowerCase() ||
          cleanName(e.name).toLowerCase() === targetClean
        );

        // 2. Composite sub-entity search
        if (!found) {
          for (const parent of list) {
            if (parent.composite_entities_data) {
              const subMatch = parent.composite_entities_data.find(
                (sub: any) =>
                  sub.name.toLowerCase() === entityName.toLowerCase() ||
                  cleanName(sub.name).toLowerCase() === targetClean
              );
              if (subMatch) {
                found = {
                  ...subMatch,
                  // Inherit parent's period, dynasty, and location if the sub-entity doesn't have its own
                  period: subMatch.period || parent.period,
                  dynasty: subMatch.dynasty || parent.dynasty,
                  location: subMatch.location || parent.location
                };
                break;
              }
            }
          }
        }
      }

      if (found) {
        if (isDeity(found)) {
          setStatusText("Gods & Deities");
        } else if (entityType === "landmark") {
          setStatusText((found as any).location || "Ancient Landmark");
        } else {
          setStatusText((found as any).period || (found as any).dynasty || "Ancient Pharaoh");
        }
      } else {
        setStatusText(entityType === "landmark" ? "Ancient Landmark" : "Ancient Pharaoh");
      }
    }
  }, [entityType, entityName, dbEntities]);

  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);

  useEffect(() => {
    const payload = loadResultFromSession();
    // Prioritize captured image if it matches the current entity
    if (payload?.imageDataUrl && payload?.result?.entity?.name === entityName) {
      setAvatarUrl(payload.imageDataUrl);
    } else if (staticUrl) {
      // Fallback to professional archive image
      setAvatarUrl(staticUrl);
    } else {
      // Look up image dynamically from mocks / R2 assets
      setAvatarUrl(getEntityImage(entityName, entityType));
    }
  }, [staticUrl, entityName, entityType, dbEntities]);

  const [messages, setMessages] = useState<Message[]>([]);
  const [rewriterMessages, setRewriterMessages] = useState<Message[]>([]);
  const [isAudioMuted, setIsAudioMuted] = useState(false);
  const [playingMsgId, setPlayingMsgId] = useState<string | null>(null);

  // Lock body scrolling on mobile to prevent page scroll behind chat
  useEffect(() => {
    const isMobile = window.innerWidth < 768;
    if (isMobile) {
      document.documentElement.style.overflow = 'hidden';
      document.body.style.overflow = 'hidden';
      document.documentElement.style.overscrollBehavior = 'none';
      document.body.style.overscrollBehavior = 'none';
    }
    return () => {
      document.documentElement.style.overflow = '';
      document.body.style.overflow = '';
      document.documentElement.style.overscrollBehavior = '';
      document.body.style.overscrollBehavior = '';
    };
  }, []);

  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [threadId] = useState(() => `thread_${Math.random().toString(36).slice(2)}`);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [userProfile, setUserProfile] = useState<any>(null);
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [supabaseConvId, setSupabaseConvId] = useState<string | null>(null);
  const supabase = createClient();
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [showAllChats, setShowAllChats] = useState(false);
  const [allChatsSearch, setAllChatsSearch] = useState("");
  const [editTitle, setEditTitle] = useState("");
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [expandedEntity, setExpandedEntity] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  const [filterType, setFilterType] = useState<string | null>(null);
  const [filterMonth, setFilterMonth] = useState<number | null>(null);
  const [showMainFilters, setShowMainFilters] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);

  const [sortBy, setSortBy] = useState<'name' | 'recent'>('name');

  // Group and Sort Logic
  const groupedChats: [string, any[]][] = (Object.entries(
    chatHistory
      .filter(chat => {
        // Search Filter
        const matchesSearch = (chat.title?.toLowerCase().includes(allChatsSearch.toLowerCase())) ||
          (chat.entity_name?.toLowerCase().includes(allChatsSearch.toLowerCase()));
        if (!matchesSearch) return false;

        // Type Filter
        if (filterType && chat.entity_type !== filterType) return false;

        // Date Filter (Months)
        if (filterMonth !== null) {
          const chatDate = new Date(chat.created_at);
          const filterDate = new Date();
          filterDate.setMonth(filterDate.getMonth() - filterMonth);

          if (!isNaN(chatDate.getTime())) {
            if (chatDate.getTime() < filterDate.getTime()) return false;
          }
        }

        return true;
      })
      .reduce((acc, chat) => {
        const key = chat.entity_name || "Unknown";
        if (!acc[key]) acc[key] = [];
        acc[key].push(chat);
        return acc;
      }, {} as Record<string, any[]>)
  ) as [string, any[]][]).sort((a: any, b: any) => {
    if (sortBy === 'name') {
      return a[0].localeCompare(b[0]);
    } else {
      // Sort by the newest chat in each group
      const latestA = Math.max(...a[1].map((c: any) => new Date(c.created_at).getTime()));
      const latestB = Math.max(...b[1].map((c: any) => new Date(c.created_at).getTime()));
      return latestB - latestA;
    }
  });

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Close menu when clicking outside or scrolling
  useEffect(() => {
    const handleClose = (e: any) => {
      if (openMenuId && !e.target.closest('.chat-menu-container') && !e.target.closest('.portal-menu')) {
        setOpenMenuId(null);
      }
    };
    window.addEventListener('mousedown', handleClose);
    window.addEventListener('scroll', () => setOpenMenuId(null), true);
    return () => {
      window.removeEventListener('mousedown', handleClose);
      window.removeEventListener('scroll', () => setOpenMenuId(null), true);
    };
  }, [openMenuId]);

  // 1. Initial Data Load
  useEffect(() => {
    const fetchData = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const { data: profile } = await supabase
          .from('profiles')
          .select('*')
          .eq('id', user.id)
          .single();
        if (profile) setUserProfile(profile);

        const { data: history } = await supabase
          .from('conversations')
          .select('*')
          .eq('user_id', user.id)
          .order('is_pinned', { ascending: false })
          .order('created_at', { ascending: false });

        if (history) setChatHistory(history);

        // 1b & 1c. Load previous messages if conv ID exists
        if (convIdFromUrl) {
          setIsHistoryLoading(true);
          setSupabaseConvId(convIdFromUrl);
          try {
            const [msgsRes, rewriterRes] = await Promise.all([
              supabase
                .from('chat_messages')
                .select('*')
                .eq('conversation_id', convIdFromUrl)
                .order('created_at', { ascending: true }),
              supabase
                .from('chat_messages_rewriter')
                .select('*')
                .eq('conversation_id', convIdFromUrl)
                .order('created_at', { ascending: true })
            ]);

            if (msgsRes.data && msgsRes.data.length > 0) {
              const welcomePattern = t("chat.welcome", { name: cleanDisplayName }).split("{name}")[0];
              const filtered = msgsRes.data
                .filter((m: any, idx: number) => {
                  // Skip if it's the very first message and looks like a greeting
                  if (idx === 0 && m.role === "assistant" && m.content.includes("Greetings")) return false;
                  return true;
                })
                .map((m: any) => ({
                  id: m.id,
                  role: m.role,
                  text: m.content,
                  ts: new Date(m.created_at).getTime()
                }));
              setMessages(filtered);
            }

            if (rewriterRes.data && rewriterRes.data.length > 0) {
              setRewriterMessages(rewriterRes.data.map((m: any) => ({
                id: m.id,
                role: m.role,
                text: m.content,
                ts: new Date(m.created_at).getTime()
              })));
            }
          } finally {
            setIsHistoryLoading(false);
          }
        }
      }
    };
    fetchData();
  }, [supabase, convIdFromUrl]);

  // [LOGIC] This function handles the full deletion of a chat session.
  // It deletes the main record from 'conversations' and all related messages
  // in 'chat_messages' and 'chat_messages_rewriter' (Foreign Key cascading).
  const handleDeleteChat = async (id: string) => {
    if (!id) return;
    setDeleteConfirmId(null); // Close modal before starting process

    try {
      // Explicitly delete from related tables first to ensure no foreign key violations
      // and to satisfy the requirement of deleting all depending rows.
      await Promise.all([
        supabase.from('chat_messages').delete().eq('conversation_id', id),
        supabase.from('chat_messages_rewriter').delete().eq('conversation_id', id)
      ]);

      const { error } = await supabase
        .from('conversations')
        .delete()
        .eq('id', id);

      if (error) throw error;

      // Update local history list
      setChatHistory(prev => prev.filter(c => c.id !== id));

      // If we are currently viewing this chat, redirect to a fresh chat session for this entity
      if (supabaseConvId === id) {
        window.location.href = `/chat?entity=${entityName}&type=${entityType}`;
      }
    } catch (err) {
      console.error("Error deleting chat:", err);
      alert("Failed to delete chat. Please try again.");
    }
  };

  const handleTogglePin = async (chat: any) => {
    const newStatus = !chat.is_pinned;
    try {
      const { error } = await supabase
        .from('conversations')
        .update({ is_pinned: newStatus })
        .eq('id', chat.id);

      if (error) throw error;

      // Update local state and sort
      setChatHistory(prev => {
        const updated = prev.map(c => c.id === chat.id ? { ...c, is_pinned: newStatus } : c);
        return [...updated].sort((a, b) => {
          if (a.is_pinned !== b.is_pinned) return a.is_pinned ? -1 : 1;
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        });
      });
    } catch (err: any) {
      console.error("Error toggling pin:", err);
    }
  };

  const handleRename = async (id: string, title: string) => {
    if (!title.trim()) {
      setRenamingId(null);
      return;
    }

    try {
      const { error } = await supabase
        .from('conversations')
        .update({ title: title.trim() })
        .eq('id', id);

      if (error) throw error;

      setChatHistory(prev => prev.map(c => c.id === id ? { ...c, title: title.trim() } : c));
    } catch (err) {
      console.error("Error renaming chat:", err);
    } finally {
      setRenamingId(null);
    }
  };


  // 3. Persistent Storage Helper
  const saveToSupabase = async (userText: string, aiText: string, rewrittenQuery?: string) => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    let convId = supabaseConvId;

    // Create conversation if it doesn't exist
    if (!convId) {
      const { data: newConv } = await supabase
        .from('conversations')
        .insert({
          user_id: user.id,
          entity_name: entityName,
          entity_type: entityType === 'pharaoh' ? 'pharaoh' : null,
          entity_location: entityType === 'landmark' ? statusText : null,
          title: userText.slice(0, 40) + (userText.length > 40 ? '...' : '')
        })
        .select()
        .single();

      if (newConv) {
        convId = newConv.id;
        setSupabaseConvId(convId);
        setChatHistory(prev => [newConv, ...prev]);
      }
    }

    if (convId) {
      // Save User Message to main chat
      await supabase.from('chat_messages').insert({
        conversation_id: convId,
        role: 'user',
        content: userText
      });

      // Save Assistant Message to main chat
      await supabase.from('chat_messages').insert({
        conversation_id: convId,
        role: 'assistant',
        content: aiText
      });

      // Update conversation title to show the latest response as preview
      const latestTitle = aiText.slice(0, 100) + (aiText.length > 100 ? '...' : '');
      await supabase
        .from('conversations')
        .update({ title: latestTitle })
        .eq('id', convId);

      // Sync local history state
      setChatHistory(prev => prev.map(c => c.id === convId ? { ...c, title: latestTitle } : c));

      // Save to Rewriter history if we have a rewritten query
      if (rewrittenQuery) {
        await supabase.from('chat_messages_rewriter').insert({
          conversation_id: convId,
          role: 'user',
          content: userText
        });
        await supabase.from('chat_messages_rewriter').insert({
          conversation_id: convId,
          role: 'assistant',
          content: rewrittenQuery
        });

        // Update local rewriter state
        setRewriterMessages(prev => [
          ...prev,
          { id: generateId(), role: 'user', text: userText, ts: Date.now() },
          { id: generateId(), role: 'assistant', text: rewrittenQuery, ts: Date.now() }
        ]);
      }
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // 4. Initialize session with backend (includes history for short-term memory)
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const initChat = async () => {
      if (!userProfile || isInitialized || isHistoryLoading) return;
      console.log("[FRONTEND] Initializing chat session:", threadId, "history count:", messages.length, "rewriter history:", rewriterMessages.length);

      try {
        const response = await fetch(`${API_BASE}/api/v1/chat/init`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            thread_id: threadId,
            user_id: userProfile.id,
            entity: entityName,
            entity_type: entityType,
            context: userProfile.user_metadata ? JSON.stringify(userProfile.user_metadata) : "{}",
            history: messages.slice(-10).map(m => ({
              role: m.role,
              content: m.text
            })),
            rewriter_history: rewriterMessages.slice(-10).map(m => ({
              role: m.role,
              content: m.text
            }))
          }),
        });
        if (response.ok) {
          console.log("[FRONTEND] Init successful");
          setIsInitialized(true);
        } else {
          console.error("[FRONTEND] Init failed:", response.status, await response.text());
        }
      } catch (err) {
        console.error("Init error:", err);
      }
    };

    if (userProfile) {
      if (!convIdFromUrl) {
        initChat();
      } else if (!isHistoryLoading) {
        initChat();
      }
    }
  }, [threadId, entityName, entityType, userProfile, messages.length, rewriterMessages.length, isInitialized, convIdFromUrl, isHistoryLoading]);

  // Reset initialization when chat identity changes
  useEffect(() => {
    setIsInitialized(false);
  }, [threadId, entityName]);

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
          user_id: userProfile?.id,
          voice_mode: useVoice,
          entity: entityName,
          entity_type: entityType,
          context: userProfile?.user_metadata ? JSON.stringify(userProfile.user_metadata) : null,
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
      let lastSearchQuery = ""; // Capture rewritten query
      let firstChunk = true;
      let buffer = "";

      // Typewriter drain interval - separates network speed from visual speed
      const typewriterId = setInterval(() => {
        if (displayedText.length < fullText.length) {
          const nextChunk = fullText.slice(displayedText.length, displayedText.length + 8);
          displayedText += nextChunk;
          setMessages((m) => m.map(msg => msg.id === assistantMsgId ? { ...msg, text: displayedText, isSearching: false } : msg));
        } else if (isStreamComplete) {
          clearInterval(typewriterId);
          // ── ZERO LATENCY SAVE ──
          // Once typewriter finishes, save to Supabase
          saveToSupabase(trimmed, fullText, lastSearchQuery);
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

              // Capture search query if present in event (memory)
              if (data.search) lastSearchQuery = data.search;
              if (data.search_query) lastSearchQuery = data.search_query;

              // ── Agentic Search Indicator ──
              if (data.tool === "tavily_search" || data.event === "on_tool_start" || data.tool_calls || data.name === "tavily_search_results_json" || data.name === "search_tool") {
                if (firstChunk) {
                  setIsTyping(false);
                  setMessages((m) => [...m, { id: assistantMsgId, role: "assistant", text: "", ts: Date.now(), isSearching: true }]);
                  firstChunk = false;
                } else {
                  setMessages((m) => m.map(msg => msg.id === assistantMsgId ? { ...msg, isSearching: true } : msg));
                }
                continue;
              }

              // Only create the assistant message bubble if we actually have text to show
              // (If it's searching, the bubble was already created in the indicator block above)
              if (firstChunk && data.text !== undefined) {
                setIsTyping(false);
                setMessages((m) => [...m, { id: assistantMsgId, role: "assistant", text: "", ts: Date.now() }]);
                firstChunk = false;
              }

              if (data.text !== undefined) {
                fullText += data.text; // Just append to fullText; setInterval handles display
              }

              // Also capture search query if it comes in a different event type
              if (data.search_query) {
                lastSearchQuery = data.search_query;
              }

              if (data.audio_url) {
                const url = data.audio_url.startsWith("data:") ? data.audio_url : `${API_BASE}${data.audio_url}`;
                setMessages((m) => m.map(msg => msg.id === assistantMsgId ? { ...msg, audioUrl: url } : msg));
                if (useVoice && audioRef.current) {
                  audioRef.current.src = url;
                  setPlayingMsgId(assistantMsgId);
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
    <div className={`w-full pt-[84px] md:pt-[110px] relative ${sidebarOpen ? 'hidden md:block' : ''}`}>
      <div className="w-full max-w-5xl mx-auto relative flex flex-col items-center pb-4 px-3 md:px-4">

        <div className="flex flex-col items-center text-center gap-1 md:gap-2 pointer-events-auto">
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
            className="h-12 w-12 md:h-16 md:w-16 rounded-full bg-gradient-to-br from-[#E6B23C] to-[#D4A030] p-[2px]"
          >
            <div className="h-full w-full rounded-full bg-[#0D0A07] overflow-hidden flex items-center justify-center">
              {avatarUrl ? (
                <img
                  key={avatarUrl}
                  src={avatarUrl}
                  alt={cleanDisplayName}
                  className={`w-full h-full object-cover ${isPharaoh ? "object-top" : "object-center"}`}
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />
              ) : (
                <span className="text-[#E6B23C] text-4xl leading-none">☥</span>
              )}
            </div>
          </motion.div>
          <div className="space-y-0.5 md:space-y-0.5 space-y-0">
            <h1 className="font-heading text-xl md:text-3xl font-bold text-[#F5E6D0] tracking-wide">{cleanDisplayName}</h1>
            <div className="text-[8px] md:text-[10px] font-bold tracking-[0.4em] text-[#E6B23C] uppercase opacity-70">{statusText}</div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <PageShell fullScreen headerExtension={showAllChats ? null : chatHeader}>
      <audio
        ref={audioRef}
        muted={isAudioMuted}
        onEnded={() => setPlayingMsgId(null)}
        onPause={() => setPlayingMsgId(null)}
      />
      <div className="flex h-full w-full bg-transparent overflow-hidden" dir="ltr">
        {/* Sidebar - Collapsible */}
        <motion.aside
          initial={false}
          animate={{ width: sidebarOpen ? (typeof window !== 'undefined' && window.innerWidth < 768 ? '100vw' : 300) : 56 }}
          className={`h-full flex flex-row z-[60] absolute md:relative left-0 top-0 ${sidebarOpen ? 'bg-[#0D0A07] border-r border-[#E6B23C]/10' : 'bg-transparent border-none md:bg-[#0D0A07] md:border-r md:border-[#E6B23C]/10'}`}
        >
          {/* Narrow Left Column - Always visible */}
          <div className={`w-[56px] h-full flex flex-col items-center py-4 gap-4 shrink-0 ${sidebarOpen ? 'border-r border-[#E6B23C]/5' : 'border-none md:border-r md:border-[#E6B23C]/5'}`}>
            {/* Return Button */}
            <div className="relative group">
              <button
                onClick={() => router.back()}
                className="h-8 w-8 flex items-center justify-center rounded-lg hover:bg-[#E6B23C]/10 active:bg-[#E6B23C]/20 active:ring-2 active:ring-[#E6B23C]/60 active:scale-95 text-[#A08E70] hover:text-[#E6B23C] transition-all border-none outline-none"
              >
                <ArrowLeft size={16} />
              </button>
              <span className={`absolute ${isRTL ? 'right-full mr-4' : 'left-full ml-4'} top-1/2 -translate-y-1/2 px-2 py-1 bg-[#1A1208] border border-[#E6B23C]/20 text-[#E6B23C] text-[10px] capitalize font-bold tracking-widest rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none`}>
                Return
              </span>
            </div>

            {/* Menu Toggle Button */}
            <div className="relative group">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="h-8 w-8 flex items-center justify-center rounded-lg hover:bg-[#E6B23C]/10 active:bg-[#E6B23C]/20 active:ring-2 active:ring-[#E6B23C]/60 active:scale-95 text-[#A08E70] hover:text-[#E6B23C] transition-all border-none outline-none"
              >
                <PanelLeft size={16} />
              </button>
              <span className={`absolute ${isRTL ? 'right-full mr-4' : 'left-full ml-4'} top-1/2 -translate-y-1/2 px-2 py-1 bg-[#1A1208] border border-[#E6B23C]/20 text-[#E6B23C] text-[10px] capitalize font-bold tracking-widest rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none`}>
                {sidebarOpen
                  ? (t("chat.sidebar.collapse").charAt(0).toUpperCase() + t("chat.sidebar.collapse").slice(1).toLowerCase())
                  : (t("chat.sidebar.expand").charAt(0).toUpperCase() + t("chat.sidebar.expand").slice(1).toLowerCase())
                }
              </span>
            </div>


          </div>

          {/* Expanded Content */}
          <AnimatePresence>
            {sidebarOpen && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex-1 md:w-[244px] md:flex-none h-full flex flex-col px-3 pb-4 overflow-hidden shrink-0"
              >
                {/* All Chats Shortcut */}
                {chatHistory.length > 0 && (
                  <button
                    onClick={() => { setShowAllChats(true); setSidebarOpen(false); }}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all text-left group mb-4 mt-6 ${showAllChats
                      ? 'bg-[#E6B23C]/15 border border-[#E6B23C]/20 shadow-[0_4px_20px_rgba(0,0,0,0.3)]'
                      : 'hover:bg-white/[0.03] border border-transparent'}`}
                  >
                    <div className="flex-1 overflow-hidden">
                      <div className="text-[11px] text-[#E6B23C] font-bold uppercase tracking-[0.15em] mb-1">
                        ECHO
                      </div>
                      <div className={`text-[13px] md:text-[14px] font-medium transition-colors ${showAllChats ? 'text-[#F5E6D0]' : 'text-[#A08E70]'} group-hover:text-[#F5E6D0]`}>
                        All Chats
                      </div>
                    </div>
                  </button>
                )}

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
                  <div className="px-2 mb-2 text-[10px] font-bold uppercase tracking-[0.2em] text-[#A08E70]/60">Recent Chats</div>
                  {chatHistory
                    .filter(chat =>
                      chat.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      chat.entity_name?.toLowerCase().includes(searchQuery.toLowerCase())
                    )
                    .map((chat) => (
                      <div key={chat.id} className="relative group/item chat-menu-container">
                        <button
                          onClick={() => window.location.href = `/chat?entity=${chat.entity_name}&type=${chat.entity_type || 'landmark'}&conv=${chat.id}`}
                          className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl transition-all text-left ${supabaseConvId === chat.id && !showAllChats
                            ? 'bg-[#E6B23C]/15 border border-[#E6B23C]/20 shadow-[0_4px_20px_rgba(0,0,0,0.3)]'
                            : 'hover:bg-white/[0.03] border border-transparent'
                            }`}
                        >
                          <div className="flex-1 overflow-hidden pr-6">
                            <div className="text-[11px] text-[#E6B23C] font-bold uppercase tracking-[0.15em] mb-1 flex items-center justify-between">
                              <span>{chat.entity_name && cleanEntityName(chat.entity_name)}</span>
                              {chat.is_pinned && (
                                <Pin size={12} className="text-[#E6B23C] shrink-0" />
                              )}
                            </div>
                            <div className={`text-[13px] md:text-[14px] font-medium transition-colors ${(supabaseConvId === chat.id && !showAllChats) ? 'text-[#F5E6D0]' : 'text-[#A08E70]'} group-hover:text-[#F5E6D0] flex items-center justify-between gap-2`}>
                              {renamingId === chat.id ? (
                                <input
                                  autoFocus
                                  className="bg-transparent border-none p-0 w-full outline-none text-[#F5E6D0] text-[13px] shadow-none focus:ring-0"
                                  value={editTitle}
                                  onChange={(e) => setEditTitle(e.target.value)}
                                  onBlur={() => handleRename(chat.id, editTitle)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') handleRename(chat.id, editTitle);
                                    if (e.key === 'Escape') setRenamingId(null);
                                  }}
                                  onFocus={(e) => e.target.select()}
                                  onClick={(e) => e.stopPropagation()}
                                />
                              ) : (
                                <span className="truncate">{chat.title || "New Chat"}</span>
                              )}
                            </div>
                          </div>
                        </button>

                        {/* Three dots button - visible on hover */}
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 z-10">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              const rect = e.currentTarget.getBoundingClientRect();
                              // Position it at the top-right of the button
                              setMenuPos({ x: rect.right + 12, y: rect.top + 8 });
                              setOpenMenuId(openMenuId === chat.id ? null : chat.id);
                            }}
                            className={`p-1.5 rounded-full transition-all relative ${openMenuId === chat.id
                              ? 'bg-[#E6B23C] text-[#0D0A07] opacity-100'
                              : 'opacity-0 group-hover/item:opacity-100 text-[#A08E70] hover:bg-[#E6B23C]/20 hover:text-[#E6B23C]'
                              }`}
                          >
                            <MoreHorizontal size={16} />

                            {/* Dropdown Menu - using Portal to avoid clipping */}
                            {openMenuId === chat.id && typeof document !== 'undefined' && createPortal(
                              <div className="portal-menu" style={{ position: 'fixed', zIndex: 9999 }}>
                                <AnimatePresence mode="wait">
                                  <motion.div
                                    initial={{ opacity: 0, scale: 0.9, x: -10, y: 10 }}
                                    animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
                                    exit={{ opacity: 0, scale: 0.9, x: -10, y: 10 }}
                                    style={{
                                      position: 'fixed',
                                      left: menuPos.x,
                                      bottom: window.innerHeight - menuPos.y,
                                    }}
                                    className="w-40 bg-[#0D0A07]/95 border border-[#E6B23C]/30 rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.8)] py-2 px-1.5 backdrop-blur-xl overflow-hidden"
                                  >
                                    <button
                                      onClick={(e) => { e.stopPropagation(); setOpenMenuId(null); handleTogglePin(chat); }}
                                      className="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-medium text-[#A08E70] hover:text-[#F5E6D0] transition-all border border-transparent hover:border-[#E6B23C]/30 hover:bg-[#E6B23C]/5 rounded-lg group"
                                    >
                                      <Pin size={14} className={`group-hover:scale-110 transition-transform ${chat.is_pinned ? 'text-[#E6B23C]' : ''}`} />
                                      <span>{chat.is_pinned ? 'Unpin chat' : 'Pin chat'}</span>
                                    </button>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setOpenMenuId(null);
                                        setEditTitle(chat.title || "New Chat");
                                        setRenamingId(chat.id);
                                      }}
                                      className="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-medium text-[#A08E70] hover:text-[#F5E6D0] transition-all border border-transparent hover:border-[#E6B23C]/30 hover:bg-[#E6B23C]/5 rounded-lg group"
                                    >
                                      <Pencil size={14} className="group-hover:scale-110 transition-transform" />
                                      <span>Rename</span>
                                    </button>
                                    <div className="h-[1px] bg-[#E6B23C]/10 my-1 mx-2" />
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setOpenMenuId(null);
                                        setDeleteConfirmId(chat.id);
                                      }}
                                      className="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-medium text-[#EF4444] hover:text-[#FF5555] transition-all border border-transparent hover:border-[#EF4444]/40 hover:bg-[#EF4444]/10 rounded-lg group"
                                    >
                                      <Trash2 size={14} className="group-hover:scale-110 transition-transform" />
                                      <span>Delete</span>
                                    </button>
                                  </motion.div>
                                </AnimatePresence>
                              </div>,
                              document.body
                            )}
                          </button>
                        </div>
                      </div>
                    ))}

                  {/* Bottom padding */}
                  <div className="h-4" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.aside>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-transparent">
          {/* New Chat Button (Top Right) */}
          <div className="absolute top-4 right-4 md:right-6 z-[55]">
            <button
              onClick={() => window.location.href = `/chat?entity=${entityName}&type=${entityType}`}
              className="h-8 w-8 flex items-center justify-center rounded-lg hover:bg-[#E6B23C]/10 active:bg-[#E6B23C]/20 active:ring-2 active:ring-[#E6B23C]/60 active:scale-95 text-[#A08E70] hover:text-[#E6B23C] transition-all border-none outline-none group"
            >
              <SquarePen size={16} />
              <span className={`absolute ${isRTL ? 'left-full ml-4' : 'right-full mr-4'} top-1/2 -translate-y-1/2 px-2 py-1 bg-[#1A1208] border border-[#E6B23C]/20 text-[#E6B23C] text-[10px] capitalize font-bold tracking-widest rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none`}>
                New Chat
              </span>
            </button>
          </div>
          {showAllChats ? (
            <div className="flex-1 flex flex-col overflow-hidden bg-[#0D0A07]/30 backdrop-blur-sm">
              <div className={`max-w-5xl mx-auto w-full pt-32 px-8 md:px-12 ${showMainFilters && !filterType && filterMonth === null && sortBy === 'name' ? 'mb-14' : 'mb-2'}`}>

                {/* Search & Filter Header - Centered */}
                <div className="flex flex-col gap-6 max-w-2xl mx-auto">
                  <div className="flex items-center gap-4">
                    <div className="relative flex-1">
                      <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-[#E6B23C]/40" size={20} />
                      <input
                        type="text"
                        placeholder="Search chat history..."
                        value={allChatsSearch}
                        onChange={(e) => setAllChatsSearch(e.target.value)}
                        className="w-full bg-[#0D0A07] border border-[#E6B23C]/10 rounded-2xl py-4 pl-14 pr-6 text-[16px] text-[#F5E6D0] focus:outline-none focus:border-[#E6B23C]/40 transition-all shadow-inner"
                      />
                    </div>
                    <div className="relative">
                      <button
                        onClick={() => setShowMainFilters(!showMainFilters)}
                        className={`h-[58px] w-[58px] shrink-0 rounded-2xl transition-all flex items-center justify-center ${showMainFilters ? 'bg-[#E6B23C] text-[#0D0A07]' : 'bg-transparent text-[#A08E70] hover:text-[#E6B23C]'}`}
                      >
                        <SlidersHorizontal size={22} className={showMainFilters ? "" : "opacity-80"} />
                      </button>
                    </div>
                  </div>

                  {/* Filter Pills Bar (As requested in image) */}
                  <div className="flex flex-wrap items-center gap-3">

                    {/* Active Filter Pills */}
                    {filterType && (
                      <div className="relative group">
                        <div className="flex items-center bg-[#0D0A07] border border-[#E6B23C]/20 rounded-full h-9 overflow-hidden">
                          <button
                            onClick={() => setFilterType(null)}
                            className="h-full px-3 flex items-center justify-center hover:bg-red-500/10 text-[#A08E70] hover:text-red-500 transition-colors border-r border-[#E6B23C]/10"
                          >
                            <X size={14} />
                          </button>
                          <div className="px-3 flex items-center gap-2">
                            <span className="text-[11px] text-[#A08E70] font-medium">Type |</span>
                            <button
                              onClick={() => setActiveDropdown(activeDropdown === 'type' ? null : 'type')}
                              className="flex items-center gap-1 text-[11px] text-[#E6B23C] font-bold hover:text-[#FFD369] transition-colors"
                            >
                              <span className="capitalize">{filterType === 'pharaoh' ? 'Pharaohs' : 'Landmarks'}</span>
                              <ChevronDown size={12} className={`transition-transform ${activeDropdown === 'type' ? 'rotate-180' : ''}`} />
                            </button>
                          </div>
                        </div>

                        <AnimatePresence>
                          {activeDropdown === 'type' && (
                            <motion.div
                              initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }}
                              className="absolute left-10 top-full mt-2 w-32 bg-[#1A1208] border border-[#E6B23C]/20 rounded-xl py-1 shadow-2xl z-[60]"
                            >
                              <button onClick={() => { setFilterType('pharaoh'); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Pharaohs</button>
                              <button onClick={() => { setFilterType('landmark'); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Landmarks</button>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    {filterMonth !== null && (
                      <div className="relative group">
                        <div className="flex items-center bg-[#0D0A07] border border-[#E6B23C]/20 rounded-full h-9 overflow-hidden">
                          <button
                            onClick={() => setFilterMonth(null)}
                            className="h-full px-3 flex items-center justify-center hover:bg-red-500/10 text-[#A08E70] hover:text-red-500 transition-colors border-r border-[#E6B23C]/10"
                          >
                            <X size={14} />
                          </button>
                          <div className="px-3 flex items-center gap-2">
                            <span className="text-[11px] text-[#A08E70] font-medium">Date |</span>
                            <button
                              onClick={() => setActiveDropdown(activeDropdown === 'date' ? null : 'date')}
                              className="flex items-center gap-1 text-[11px] text-[#E6B23C] font-bold hover:text-[#FFD369] transition-colors"
                            >
                              <span>{filterMonth === 1 ? 'Last Month' : `Last ${filterMonth} Months`}</span>
                              <ChevronDown size={12} className={`transition-transform ${activeDropdown === 'date' ? 'rotate-180' : ''}`} />
                            </button>
                          </div>
                        </div>

                        <AnimatePresence>
                          {activeDropdown === 'date' && (
                            <motion.div
                              initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }}
                              className="absolute left-10 top-full mt-2 w-40 bg-[#1A1208] border border-[#E6B23C]/20 rounded-xl py-1 shadow-2xl z-[60]"
                            >
                              {[1, 3, 6, 12].map(m => (
                                <button key={m} onClick={() => { setFilterMonth(m); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">
                                  {m === 1 ? 'Last Month' : `Last ${m} Months`}
                                </button>
                              ))}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    {sortBy !== 'name' && (
                      <div className="relative group">
                        <div className="flex items-center bg-[#0D0A07] border border-[#E6B23C]/20 rounded-full h-9 overflow-hidden">
                          <button
                            onClick={() => setSortBy('name')}
                            className="h-full px-3 flex items-center justify-center hover:bg-red-500/10 text-[#A08E70] hover:text-red-500 transition-colors border-r border-[#E6B23C]/10"
                          >
                            <X size={14} />
                          </button>
                          <div className="px-3 flex items-center gap-2">
                            <span className="text-[11px] text-[#A08E70] font-medium">Sort |</span>
                            <button
                              onClick={() => setActiveDropdown(activeDropdown === 'sort' ? null : 'sort')}
                              className="flex items-center gap-1 text-[11px] text-[#E6B23C] font-bold hover:text-[#FFD369] transition-colors"
                            >
                              <span>Most Recent</span>
                              <ChevronDown size={12} className={`transition-transform ${activeDropdown === 'sort' ? 'rotate-180' : ''}`} />
                            </button>
                          </div>
                        </div>

                        <AnimatePresence>
                          {activeDropdown === 'sort' && (
                            <motion.div
                              initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }}
                              className="absolute left-10 top-full mt-2 w-32 bg-[#1A1208] border border-[#E6B23C]/20 rounded-xl py-1 shadow-2xl z-[60]"
                            >
                              <button onClick={() => { setSortBy('name'); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Alphabetical</button>
                              <button onClick={() => { setSortBy('recent'); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Most Recent</button>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    {(filterType || filterMonth !== null || sortBy !== 'name') && (
                      <button
                        onClick={() => { setFilterType(null); setFilterMonth(null); setSortBy('name'); setActiveDropdown(null); }}
                        className="text-[11px] font-bold text-[#A08E70] hover:text-red-500 flex items-center gap-1.5 ml-2 mr-4 transition-colors group whitespace-nowrap"
                      >
                        <X size={14} className="group-hover:scale-110 transition-transform" />
                        Clear All
                      </button>
                    )}

                    {/* Inactive filters appearing after Clear All only when toggled */}
                    {showMainFilters && (
                      <>
                        {!filterType && (
                          <div className="relative">
                            <button
                              onClick={() => setActiveDropdown(activeDropdown === 'pill-type' ? null : 'pill-type')}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-[#E6B23C]/10 bg-[#E6B23C]/5 text-[#A08E70] hover:text-[#E6B23C] hover:border-[#E6B23C]/30 transition-all text-[11px] font-medium"
                            >
                              Entity Type
                              <ChevronDown size={12} className={`transition-transform ${activeDropdown === 'pill-type' ? 'rotate-180' : ''}`} />
                            </button>
                            <AnimatePresence>
                              {activeDropdown === 'pill-type' && (
                                <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }} className="absolute left-0 top-full mt-2 w-32 bg-[#1A1208] border border-[#E6B23C]/20 rounded-xl py-1 shadow-2xl z-[70]">
                                  <button onClick={() => { setFilterType('pharaoh'); setActiveDropdown(null); }} className="w-full px-3 py-2 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Pharaohs</button>
                                  <button onClick={() => { setFilterType('landmark'); setActiveDropdown(null); }} className="w-full px-3 py-2 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Landmarks</button>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )}

                        {filterMonth === null && (
                          <div className="relative">
                            <button
                              onClick={() => setActiveDropdown(activeDropdown === 'pill-date' ? null : 'pill-date')}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-[#E6B23C]/10 bg-[#E6B23C]/5 text-[#A08E70] hover:text-[#E6B23C] hover:border-[#E6B23C]/30 transition-all text-[11px] font-medium"
                            >
                              Time Period
                              <ChevronDown size={12} className={`transition-transform ${activeDropdown === 'pill-date' ? 'rotate-180' : ''}`} />
                            </button>
                            <AnimatePresence>
                              {activeDropdown === 'pill-date' && (
                                <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }} className="absolute left-0 top-full mt-2 w-40 bg-[#1A1208] border border-[#E6B23C]/20 rounded-xl py-1 shadow-2xl z-[70]">
                                  {[1, 3, 6, 12].map(m => (
                                    <button key={m} onClick={() => { setFilterMonth(m); setActiveDropdown(null); }} className="w-full px-3 py-2 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">{m === 1 ? 'Last Month' : `Last ${m} Months`}</button>
                                  ))}
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )}

                        {sortBy === 'name' && (
                          <div className="relative">
                            <button
                              onClick={() => setActiveDropdown(activeDropdown === 'pill-sort' ? null : 'pill-sort')}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-[#E6B23C]/10 bg-[#E6B23C]/5 text-[#A08E70] hover:text-[#E6B23C] hover:border-[#E6B23C]/30 transition-all text-[11px] font-medium"
                            >
                              Sort Order
                              <ChevronDown size={12} className={`transition-transform ${activeDropdown === 'pill-sort' ? 'rotate-180' : ''}`} />
                            </button>
                            <AnimatePresence>
                              {activeDropdown === 'pill-sort' && (
                                <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 5 }} className="absolute left-0 top-full mt-2 w-36 bg-[#1A1208] border border-[#E6B23C]/20 rounded-xl py-1 shadow-2xl z-[70]">
                                  <button onClick={() => { setSortBy('recent'); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Most Recent</button>
                                  <button onClick={() => { setSortBy('name'); setActiveDropdown(null); }} className="w-full px-3 py-1.5 text-left text-[11px] text-[#A08E70] hover:bg-[#E6B23C]/10 hover:text-[#E6B23C]">Alphabetical</button>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Grouped List - Now full width for scrolling but inner content centered */}
              <div className={`flex-1 overflow-x-hidden ${groupedChats.length > 0 ? 'overflow-y-auto' : 'overflow-hidden'} trending-scrollbar-hide`}>
                <div className="max-w-5xl mx-auto w-full px-8 md:px-12 pb-24">
                  {groupedChats.length > 0 ? (
                    <div className="bg-[#1A1208]/30 border border-[#E6B23C]/10 rounded-3xl p-6 md:p-10 backdrop-blur-md shadow-[0_10px_40px_rgba(0,0,0,0.4)]">
                      {groupedChats.map(([entity, chats]: [string, any[]]) => (
                        <div key={entity} className="mb-4">
                          <button
                            onClick={() => setExpandedEntity(expandedEntity === entity ? null : entity)}
                            className="w-full text-left group/header py-4"
                          >
                            <h3 className="text-[13px] font-bold uppercase tracking-[0.4em] text-[#E6B23C]/50 flex items-center gap-4 group-hover/header:text-[#E6B23C] transition-all">
                              <span className="min-w-fit">{cleanEntityName(entity)}</span>
                              <div className="flex-1 h-[1px] bg-[#E6B23C]/10 group-hover/header:bg-[#E6B23C]/30" />
                              <span className="text-[11px] font-mono opacity-40 group-hover/header:opacity-100">{chats.length} {chats.length === 1 ? 'RECORD' : 'RECORDS'}</span>
                            </h3>
                          </button>

                          <AnimatePresence>
                            {expandedEntity === entity && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="overflow-hidden"
                              >
                                <div className="space-y-4 pt-2 pb-8">
                                  {[...chats]
                                    .sort((a: any, b: any) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                                    .map((chat: any) => (
                                      <div
                                        key={chat.id}
                                        onClick={() => {
                                          setShowAllChats(false);
                                          window.location.href = `/chat?entity=${chat.entity_name}&type=${chat.entity_type || 'landmark'}&conv=${chat.id}`;
                                        }}
                                        className="group cursor-pointer border-b border-[#E6B23C]/5 p-6 -mx-6 rounded-2xl hover:bg-[#E6B23C]/5 hover:border-[#E6B23C]/20 transition-all active:scale-[0.98]"
                                      >
                                        <div className="flex items-start justify-between mb-1">
                                          <h4 className="text-lg font-medium text-[#A08E70] transition-colors line-clamp-1">
                                            {chat.title || "New Chat"}
                                          </h4>
                                          <span className="text-[10px] text-[#A08E70] font-mono group-hover:text-[#E6B23C]/60 transition-colors">
                                            {new Date(chat.created_at).toLocaleDateString()}
                                          </span>
                                        </div>
                                        <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#E6B23C] group-hover:text-[#FFD369] transition-colors">
                                          {new Date(chat.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </div>
                                      </div>
                                    ))}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-24 text-center">
                      <div className="w-20 h-20 bg-[#E6B23C]/5 rounded-full flex items-center justify-center mb-6 border border-[#E6B23C]/10">
                        <Search size={32} className="text-[#E6B23C]/40" />
                      </div>
                      <h3 className="text-xl font-bold text-[#E6B23C] mb-3 uppercase tracking-[0.2em]">No Records Found</h3>
                      <p className="text-[#A08E70] text-sm max-w-sm leading-relaxed mx-auto">
                        We couldn't find any conversations matching your current filters or search terms.
                        Try adjusting your criteria or clearing all filters.
                      </p>
                      <button
                        onClick={() => { setFilterType(null); setFilterMonth(null); setSortBy('name'); setAllChatsSearch(''); setActiveDropdown(null); }}
                        className="mt-10 px-8 py-3 rounded-xl border border-[#E6B23C]/20 text-[#E6B23C] text-xs font-bold uppercase tracking-widest hover:bg-[#E6B23C] hover:text-[#0D0A07] transition-all"
                      >
                        Clear All Filters
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Messages Area - Positioned below the fixed header area */}
              <div ref={scrollRef} className="flex-1 overflow-y-auto trending-scrollbar-hide relative mt-36 md:mt-42">
                <div className="max-w-5xl mx-auto w-full pl-12 pr-4 pt-4 pb-32 md:p-8 space-y-8" style={{ direction: 'ltr' }}>
                  <AnimatePresence>
                    {/* Static Welcome Message */}
                    <motion.div key="welcome-message-static" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
                      className="flex flex-col gap-3 max-w-3xl mb-12"
                    >
                      <div className="flex flex-col gap-3 w-full">
                        <div
                          className="text-[#D4C4A8] text-[13px] md:text-base leading-relaxed font-normal tracking-wide"
                          style={{ direction: isRTL ? 'rtl' : 'ltr', textAlign: isRTL ? 'right' : 'left' }}
                        >
                          {renderMessageText(t("chat.welcome", { name: cleanDisplayName }))}
                        </div>

                        <div className={`flex items-center gap-6 mt-1 opacity-40 hover:opacity-100 transition-opacity ${isRTL ? 'flex-row-reverse' : ''}`}>
                          <button
                            onClick={() => handleCopy(t("chat.welcome", { name: cleanDisplayName }), "welcome-msg")}
                            className="flex items-center gap-2 text-[#A08E70] hover:text-[#E6B23C] transition-colors"
                          >
                            {copiedId === "welcome-msg" ? (
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
                              className="text-[#D4C4A8] text-[13px] md:text-base leading-relaxed font-normal tracking-wide"
                              style={{ direction: isRTL ? 'rtl' : 'ltr', textAlign: isRTL ? 'right' : 'left' }}
                            >
                              {renderMessageText(msg.text)}
                            </div>

                            <div className={`flex items-center gap-6 mt-1 transition-opacity ${isRTL ? 'flex-row-reverse' : ''} ${playingMsgId === msg.id ? 'opacity-100' : 'opacity-40 hover:opacity-100'}`}>
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
                              {playingMsgId === msg.id && (
                                <button
                                  onClick={() => setIsAudioMuted(!isAudioMuted)}
                                  className={`flex items-center gap-2 transition-colors ${isAudioMuted ? 'text-red-500' : 'text-[#E6B23C]'}`}
                                  title={isAudioMuted ? "Unmute" : "Mute"}
                                >
                                  {isAudioMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
                                  <span className={`text-[9px] font-bold uppercase tracking-widest ${isAudioMuted ? 'text-red-500' : 'text-[#E6B23C]'}`}>
                                    {isAudioMuted ? "Unmute" : "Mute"}
                                  </span>
                                </button>
                              )}
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
                                className="text-[#E6B23C] text-[13px] md:text-base leading-relaxed font-normal tracking-wide"
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
                        <div className="relative h-24 w-24 md:h-[152px] md:w-[152px] flex items-center justify-center">
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
                            <Mic size={28} className="text-[#1A1208] drop-shadow-lg md:w-9 md:h-9" />
                          </motion.div>
                        </div>

                        {/* Status & Cancel Button (Outside the rotation) */}
                        <div className="flex flex-col items-center gap-4">
                          <motion.div
                            animate={{ opacity: [1, 0.4, 1] }}
                            transition={{ repeat: Infinity, duration: 2 }}
                          >
                            <span className="text-xs md:text-sm font-bold tracking-[0.5em] uppercase text-[#E6B23C] drop-shadow-md">{t("chat.listening")}</span>
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
                <div className="pl-12 pr-4 pt-4 pb-4 md:p-8 md:pb-12 bg-transparent max-w-5xl mx-auto">
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
                      <div className="flex-1 min-h-[56px] max-h-48 py-4 px-8 rounded-[28px] bg-[#1A1208]/50 backdrop-blur-xl border border-[#E6B23C]/10 focus-within:border-[#E6B23C]/30 focus-within:bg-[#1A1208]/80 focus-within:shadow-[0_0_30px_rgba(230,178,60,0.05)] transition-all flex items-center relative z-[60]">
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
                          placeholder={t("chat.placeholder", { name: cleanDisplayName })}
                          rows={1}
                          className="w-full bg-transparent text-base placeholder:text-[#E6B23C]/50 focus:outline-none disabled:opacity-50 resize-none overflow-y-auto trending-scrollbar-hide"
                          style={{ color: "#E6B23C", caretColor: "#E6B23C", direction: isRTL ? 'rtl' : 'ltr', transform: 'translateZ(0)' }}
                        />
                      </div>
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
            </>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {isMounted && createPortal(
        <AnimatePresence>
          {deleteConfirmId && (
            <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setDeleteConfirmId(null)}
                className="absolute inset-0 bg-black/40 backdrop-blur-[1px]"
              />
              <motion.div
                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9, y: 20 }}
                onClick={(e) => e.stopPropagation()}
                className="relative w-full max-w-sm bg-[#0D0A07] border border-[#E6B23C]/20 rounded-3xl shadow-[0_30px_70px_rgba(0,0,0,1)] overflow-hidden"
              >
                <div className="p-8">
                  <h3 className="text-xl font-bold text-[#E6B23C] mb-3">Delete chat</h3>
                  <p className="text-sm text-[#A08E70] leading-relaxed mb-8">
                    Are you sure you want to delete this chat? This action cannot be undone.
                  </p>
                  <div className="flex items-center justify-end gap-3">
                    <button
                      onClick={() => setDeleteConfirmId(null)}
                      className="px-5 py-2.5 rounded-xl text-sm font-medium text-[#A08E70] hover:text-[#F5E6D0] hover:bg-white/5 transition-all"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => handleDeleteChat(deleteConfirmId)}
                      className="px-6 py-2.5 rounded-xl text-sm font-medium bg-[#EF4444] text-white hover:bg-[#FF5555] transition-all shadow-[0_4px_15px_rgba(239,68,68,0.4)]"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>,
        document.body
      )}


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
