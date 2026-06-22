"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import PageShell from "../../components/layout/PageShell";
import { motion, AnimatePresence, useScroll, useTransform, useSpring } from "framer-motion";
import { Search, Crown, MapPin, ChevronDown, ChevronRight, MessageSquare, Video, Scroll, Sparkles, X, History } from "lucide-react";

import { saveResultToSession } from "../../lib/services/recognition";
import { useLanguage } from "../../context/LanguageContext";
import { cleanEntityName } from "../../lib/utils";
import type { RecognitionEntity, RecognitionResult } from "../../lib/types";
import { Suspense } from "react";

// ── Period ordering for pharaohs ──────────────────────────────────────────
const PERIOD_ORDER = [
  "Old Kingdom",
  "Middle Kingdom",
  "New Kingdom",
  "Late Period",
  "Argead Period",
  "Ptolemic Period",
];

// ── Governorate pin coordinates on the SVG map ────────────────────────────
const GOVERNORATE_PINS: Record<string, { x: number; y: number; label: string }> = {
  "Alexandria Governorate, Egypt": { x: 110, y: 55, label: "Alexandria" },
  "Giza Governorate, Egypt": { x: 135, y: 85, label: "Giza" },
  "Giza, Giza Governorate, Egypt": { x: 135, y: 85, label: "Giza" },
  "Faiyum Governorate, Egypt": { x: 130, y: 105, label: "Faiyum" },
  "Beni Suef Governorate, Egypt": { x: 140, y: 125, label: "Beni Suef" },
  "Minya Governorate, Egypt": { x: 145, y: 175, label: "Minya" },
  "Sohag Governorate, Egypt": { x: 170, y: 255, label: "Sohag" },
  "Qena Governorate, Egypt": { x: 185, y: 285, label: "Qena" },
  "New Valley Governorate, Egypt": { x: 70, y: 280, label: "New Valley" },
  "Luxor Governorate, Egypt": { x: 185, y: 315, label: "Luxor" },
  "Aswan Governorate, Egypt": { x: 190, y: 385, label: "Aswan" },
};

// Normalize location keys (trim leading spaces)
function normalizeLocation(loc: string): string {
  return loc.trim();
}

// ── Dynasty String to Number mapping for chronological sorting ─────────────
const DYNASTY_NUMBERS: Record<string, number> = {
  "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
  "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
  "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
  "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
  "nineteenth": 19, "twentieth": 20, "twenty first": 21, "twenty second": 22,
  "twenty third": 23, "twenty fourth": 24, "twenty fifth": 25,
  "twenty sixth": 26, "twenty seventh": 27, "twenty eighth": 28,
  "twenty ninth": 29, "thirtieth": 30, "thirty first": 31
};

function getDynastyNumber(dynasty: string): number {
  const normalized = dynasty.toLowerCase().replace(" dynasty", "").trim();
  return DYNASTY_NUMBERS[normalized] || 999; // Put 'Other' / 'Gods' at the end
}

// ── Entity Card ───────────────────────────────────────────────────────────
function EntityCard({ entity, type, onNavigate }: { entity: RecognitionEntity; type: "pharaoh" | "landmark"; onNavigate: () => void }) {
  const { t } = useLanguage();
  const cleanName = cleanEntityName(entity.name);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onNavigate}
      className="group cursor-pointer rounded-xl border border-[#E6B23C]/8 bg-[#1A1208]/50 hover:border-[#E6B23C]/20 hover:bg-[#1A1208]/80 transition-all p-4 backdrop-blur-sm"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-heading text-base font-bold text-[#E6B23C] truncate transition-colors">
            {cleanName}
          </h3>
          {entity.type && type !== "pharaoh" && (
            <span className="text-[11px] text-[#A08E70]/70 uppercase tracking-wider">
              {t("result.badge.landmark")}
            </span>
          )}
          {entity.description && (
            <p className="text-xs text-[#A08E70]/60 mt-1.5 line-clamp-2 leading-relaxed">
              {entity.description}
            </p>
          )}
        </div>
        <ChevronRight size={14} className="text-[#E6B23C]/30 group-hover:text-[#E6B23C] transition-colors mt-1 shrink-0" />
      </div>
    </motion.div>
  );
}

// ── Collapsible Dynasty Group ──────────────────────────────────────────────
function DynastyGroup({ dynasty, entities, type, onNavigate }: {
  dynasty: string;
  entities: RecognitionEntity[];
  type: "pharaoh" | "landmark";
  onNavigate: (entity: RecognitionEntity) => void;
}) {
  const { t, isRTL } = useLanguage();
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-[#E6B23C]/8 rounded-2xl overflow-hidden bg-[#0D0A07]/50 backdrop-blur-sm">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-[#E6B23C]/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Scroll size={15} className="text-[#B8860B] shrink-0" />
          <span className="text-sm font-bold text-[#EADBB8] tracking-wide">{dynasty}</span>
          <span className={`text-[10px] text-[#A08E70]/50 font-bold tracking-widest uppercase ${isRTL ? 'mr-1' : 'ml-1'}`}>
            {entities.length} {entities.length === 1 ? t("common.entity") : t("common.entities")}
          </span>
        </div>
        <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={16} className="text-[#E6B23C]/40" />
        </motion.div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 grid gap-2 sm:grid-cols-2">
              {entities.map((e) => (
                <EntityCard key={e.id} entity={e} type={type} onNavigate={() => onNavigate(e)} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Egypt SVG Map ──────────────────────────────────────────────────────────
function EgyptMap({
  pins,
  selectedCity,
  onSelectCity,
}: {
  pins: { label: string; x: number; y: number; count: number }[];
  selectedCity: string | null;
  onSelectCity: (label: string | null) => void;
}) {
  return (
    <div className="relative w-full max-w-2xl mx-auto overflow-hidden rounded-2xl md:rounded-3xl bg-transparent">
      <div className="absolute inset-0 bg-gradient-to-t from-transparent via-transparent to-transparent z-10 pointer-events-none" />

      <div className="relative aspect-square w-full">
        <img
          src="/images/maps/egypt_map.png"
          alt="Map of Egypt"
          className="absolute inset-0 w-full h-full object-cover md:object-contain opacity-80"
          style={{ filter: "brightness(0.85) saturate(0.75) sepia(0.2) contrast(1.35)" }}
        />

        <svg
          viewBox="0 0 300 300"
          className="absolute inset-0 w-full h-full z-20"

        >
          {pins.map((pin) => {
            const isActive = selectedCity === pin.label;
            const shortName = pin.label.split(',')[0];

            // ── PIN CALIBRATION — Research-based Geographical Coordinates ───────
            let x = pin.x;
            let y = pin.y;

            if (shortName === "Alexandria") { x = 141; y = 15; }
            if (shortName === "Giza") { x = 172; y = 45; }
            if (shortName === "Faiyum") { x = 166; y = 71; }
            if (shortName === "Beni Suef") { x = 173; y = 78; }
            if (shortName === "Minya") { x = 165; y = 110; }
            if (shortName === "Sohag") { x = 199; y = 165; }
            if (shortName === "Qena") { x = 217; y = 165; }
            if (shortName === "Luxor") { x = 212; y = 182; }
            if (shortName === "Aswan") { x = 222; y = 216; }
            if (shortName === "New Valley") { x = 180; y = 186; }

            return (
              <g
                key={pin.label}
                onClick={() => onSelectCity(isActive ? null : pin.label)}
                className="cursor-pointer group"
              >
                {/* Pulsing Base (Implier) */}
                {!isActive && (
                  <circle cx={x} cy={y} r="6" fill="#1A1005" fillOpacity="0.2">
                    <animate attributeName="r" values="4;8;4" dur="3s" repeatCount="indefinite" />
                    <animate attributeName="fill-opacity" values="0.4;0.1;0.4" dur="3s" repeatCount="indefinite" />
                  </circle>
                )}

                {/* Selection Pulse Ring */}
                {isActive && (
                  <circle cx={x} cy={y} r="14" fill="none" stroke="#E6B23C" strokeWidth="1" strokeOpacity="0.4">
                    <animate attributeName="r" values="8;18;8" dur="2s" repeatCount="indefinite" />
                    <animate attributeName="stroke-opacity" values="0.4;0;0.4" dur="2s" repeatCount="indefinite" />
                  </circle>
                )}

                {/* Golden Map Pin */}
                <g transform={`translate(${x}, ${y - 1}) scale(${isActive ? 1.3 : 1})`}>
                  {/* Pin Shape */}
                  <path
                    d="M0,0 C-1,-1 -4,-4 -4,-7 A4,4 0 1,1 4,-7 C4,-4 1,-1 0,0 Z"
                    fill={isActive ? "#E6B23C" : "#D4A017"}
                    stroke="#0D0A07"
                    strokeWidth="0.5"
                  />
                  {/* Inner Circle (Cutout) */}
                  <circle
                    cx="0"
                    cy="-7"
                    r="1.5"
                    fill="#0D0A07"
                  />
                </g>

                {/* City Label */}
                <text
                  x={x}
                  y={y - -5}
                  fill={isActive ? "#E6B23C" : "#4A3728"}
                  fontSize="8"
                  fontWeight="900"
                  textAnchor="middle"
                  className="select-none pointer-events-none transition-all duration-300 group-hover:fill-[#D4A017] drop-shadow-[0_1px_2px_rgba(255,255,255,0.2)]"
                  style={{ fontFamily: 'var(--font-cormorant), serif' }}
                >
                  {shortName}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}



// ── Cache variables to persist dynamic data across route transitions ───────
let cachedPharaohs: RecognitionEntity[] | null = null;
let cachedLandmarks: RecognitionEntity[] | null = null;

// ── Main Explore Content ──────────────────────────────────────────────────
function ExploreContent() {
  const { t, isRTL } = useLanguage();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [activeTab, setActiveTab] = useState<"pharaohs" | "landmarks">("pharaohs");
  const [search, setSearch] = useState("");
  const [pharaohs, setPharaohs] = useState<RecognitionEntity[]>(() => {
    return cachedPharaohs || [];
  });
  const [landmarks, setLandmarks] = useState<RecognitionEntity[]>(() => {
    return cachedLandmarks || [];
  });
  const [isLoading, setIsLoading] = useState(() => !cachedPharaohs);

  useEffect(() => {
    if (cachedPharaohs && cachedLandmarks) {
      return; // Already cached once, skip fetching
    }
    let active = true;
    async function loadData() {
      try {
        setIsLoading(true);
        const { fetchAllEntities } = await import("../../lib/services/entities");
        const data = await fetchAllEntities("");
        if (active) {
          if (data.pharaohs && data.pharaohs.length > 0) {
            setPharaohs(data.pharaohs);
            cachedPharaohs = data.pharaohs;
          }
          if (data.landmarks && data.landmarks.length > 0) {
            setLandmarks(data.landmarks);
            cachedLandmarks = data.landmarks;
          }
        }
      } catch (err) {
        console.error("Failed to fetch entities from DB:", err);
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    }
    loadData();
    return () => {
      active = false;
    };
  }, []);

  const [selectedCity, setSelectedCity] = useState<string | null>(null);
  const landmarksListRef = useRef<HTMLDivElement>(null);

  // Reset scroll position when selected city changes
  useEffect(() => {
    if (selectedCity && landmarksListRef.current) {
      landmarksListRef.current.scrollTop = 0;
    }
  }, [selectedCity]);

  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start 20%", "end end"]
  });

  const scaleY = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001
  });

  // ── Render Helpers ──────────────────────────────────────────────────
  const ScrollLine = () => (
    <div className="absolute left-5 md:left-1/2 top-0 bottom-0 w-0.5 -translate-x-1/2 pointer-events-none z-0">
      {/* Background track */}
      <div className="absolute inset-0 bg-white/[0.05]" />

      {/* Active shining line */}
      <motion.div
        style={{ scaleY, originY: 0 }}
        className="absolute inset-0 bg-gradient-to-b from-[#E6B23C] via-[#E6B23C] to-white/50 shadow-[0_0_15px_rgba(230,178,60,0.5)]"
      />

      {/* Scrolling dot */}
      <motion.div
        style={{ top: useTransform(scaleY, [0, 1], ["0%", "100%"]) }}
        className="absolute left-1/2 -translate-x-1/2 w-2.5 h-2.5 bg-[#E6B23C] rounded-full shadow-[0_0_15px_rgba(230,178,60,0.8)] z-10"
      />
    </div>
  );

  // Sync tab with URL on mount and param changes
  useEffect(() => {
    const tabParam = searchParams.get('tab') as "pharaohs" | "landmarks";
    if (tabParam && (tabParam === "pharaohs" || tabParam === "landmarks")) {
      setActiveTab(tabParam);
    }
  }, [searchParams]);

  const handleTabChange = (tab: "pharaohs" | "landmarks") => {
    setActiveTab(tab);
    setSearch("");
    setSelectedCity(null);
    router.replace(`/explore?tab=${tab}`, { scroll: false });
  };

  // Navigate to result page
  function handleEntityClick(entity: RecognitionEntity, type: "pharaoh" | "landmark") {
    const result: RecognitionResult = {
      source: "explore",
      type: type,
      name: entity.name,
      category: type,
      confidence: 1.0,
      binary_confidence: 1.0,
      entity: entity,
      debug_info: null,
    };
    saveResultToSession({ result, imageDataUrl: null });
    router.push(`/result?entity=${encodeURIComponent(entity.name)}&type=${type}`);
  }

  // ── Pharaohs: group by period → dynasty ──────────────────────────
  const filteredPharaohs = useMemo(() => {
    if (!search) return pharaohs;
    const q = search.toLowerCase();
    return pharaohs.filter((p) => {
      const cleanName = cleanEntityName(p.name);
      const parts = cleanName.toLowerCase().split(/[\s-]/);
      return parts.some(part => part.startsWith(q)) || cleanName.toLowerCase().startsWith(q);
    });
  }, [pharaohs, search]);

  const pharaohsByPeriod = useMemo(() => {
    const groups: Record<string, Record<string, RecognitionEntity[]>> = {};

    for (const p of pharaohs) {
      const period = p.period || "Other";
      const dynasty = p.dynasty || (p.type?.includes("God") || p.type?.includes("Goddess") ? "Gods & Deities" : "Other");

      if (!groups[period]) groups[period] = {};
      if (!groups[period][dynasty]) groups[period][dynasty] = [];
      groups[period][dynasty].push(p);
    }

    // Sort periods by PERIOD_ORDER
    const sorted: { period: string; dynasties: { dynasty: string; entities: RecognitionEntity[] }[] }[] = [];

    // Known periods first
    for (const period of PERIOD_ORDER) {
      if (groups[period]) {
        const dynasties = Object.entries(groups[period])
          .map(([dynasty, entities]) => ({ dynasty, entities: entities.sort((a, b) => a.name.localeCompare(b.name)) }))
          .sort((a, b) => getDynastyNumber(a.dynasty) - getDynastyNumber(b.dynasty));
        sorted.push({ period, dynasties });
        delete groups[period];
      }
    }

    // Remaining periods (nulls mapped to "Other")
    for (const [period, dynastyMap] of Object.entries(groups)) {
      const dynasties = Object.entries(dynastyMap)
        .map(([dynasty, entities]) => ({ dynasty, entities: entities.sort((a, b) => a.name.localeCompare(b.name)) }))
        .sort((a, b) => getDynastyNumber(a.dynasty) - getDynastyNumber(b.dynasty));
      sorted.push({ period, dynasties });
    }

    return sorted;
  }, [filteredPharaohs]);

  // ── Landmarks: group by normalized location ──────────────────────
  const filteredLandmarks = useMemo(() => {
    if (!search && !selectedCity) return landmarks;
    let result = landmarks;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter((l) => {
        const cleanName = cleanEntityName(l.name);
        const parts = cleanName.toLowerCase().split(/[\s-]/);
        return parts.some(part => part.startsWith(q)) || cleanName.toLowerCase().startsWith(q);
      });
    }
    if (selectedCity) {
      result = result.filter((l) => {
        const pin = GOVERNORATE_PINS[normalizeLocation(l.location || "")];
        return pin && pin.label === selectedCity;
      });
    }
    return result;
  }, [landmarks, search, selectedCity]);

  const landmarksByCity = useMemo(() => {
    const groups: Record<string, RecognitionEntity[]> = {};
    const cityFiltered = selectedCity
      ? landmarks.filter((l) => {
        const pin = GOVERNORATE_PINS[normalizeLocation(l.location || "")];
        return pin && pin.label === selectedCity;
      })
      : landmarks;

    for (const l of cityFiltered) {
      const loc = normalizeLocation(l.location || "Unknown");
      const pin = GOVERNORATE_PINS[loc];
      const label = pin?.label || loc.replace(/ Governorate, Egypt$/, "");
      if (!groups[label]) groups[label] = [];
      groups[label].push(l);
    }
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
  }, [filteredLandmarks]);

  // Map pins data
  const mapPins = useMemo(() => {
    const cityCount: Record<string, { x: number; y: number; count: number }> = {};
    for (const l of landmarks) {
      const loc = normalizeLocation(l.location || "");
      const pin = GOVERNORATE_PINS[loc];
      if (pin) {
        if (!cityCount[pin.label]) cityCount[pin.label] = { x: pin.x, y: pin.y, count: 0 };
        cityCount[pin.label].count++;
      }
    }
    return Object.entries(cityCount).map(([label, data]) => ({ label, ...data }));
  }, [landmarks]);

  return (
    <PageShell>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >

        <h1
          className="font-heading text-4xl md:text-5xl font-bold tracking-wide uppercase text-[#F5E6D0] mb-3"
          style={{ fontFamily: "var(--font-cormorant), serif" }}
        >
          {t("explore.title").split(" ").map((word, i, arr) => {
            const isArchive = word.toLowerCase().includes("archive") || word.includes("الأرشيف");
            return (
              <span key={i} className={isArchive ? "text-[#E6B23C]" : ""}>
                {word}{i < arr.length - 1 ? " " : ""}
              </span>
            );
          })}
        </h1>
        <p className="text-[#A08E70] text-base max-w-lg mx-auto" style={{ fontFamily: "var(--font-cormorant), serif" }}>
          {t("explore.subtitle")}
        </p>
      </motion.div>

      {/* Tabs */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="flex justify-center gap-2 mb-8"
      >
        {(["pharaohs", "landmarks"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => handleTabChange(tab)}
            className={`px-6 py-3 rounded-xl text-sm font-bold tracking-wider uppercase transition-all ${activeTab === tab
              ? "bg-[#E6B23C] text-[#0D0A07] shadow-[0_4px_20px_rgba(230,178,60,0.3)]"
              : "bg-[#E6B23C]/[0.06] border border-[#E6B23C]/10 text-[#A08E70] hover:text-[#E6B23C] hover:border-[#E6B23C]/20"
              }`}
          >
            {tab === "pharaohs" ? (
              <span className="flex items-center gap-2"><Crown size={14} /> {t("explore.tab.pharaohs")}</span>
            ) : (
              <span className="flex items-center gap-2"><MapPin size={14} /> {t("explore.tab.landmarks")}</span>
            )}
          </button>
        ))}
      </motion.div>

      {/* Search */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="max-w-lg mx-auto mb-10 relative"
      >
        <Search size={16} className={`absolute ${isRTL ? 'right-4' : 'left-4'} top-1/2 -translate-y-1/2 text-[#A08E70]/40`} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={activeTab === "pharaohs" ? t("explore.search.pharaohs") : t("explore.search.landmarks")}
          className={`w-full h-12 ${isRTL ? 'pr-11 pl-10' : 'pl-11 pr-10'} rounded-xl bg-[#0D0A07] border border-[#E6B23C]/15 text-sm text-[#F5E6D0] placeholder:text-[#A08E70]/40 focus:outline-none focus:border-[#E6B23C]/30 focus:shadow-[0_0_15px_rgba(230,178,60,0.08)] transition-all`}
          style={{ caretColor: "#E6B23C" }}
        />
        {search && (
          <button onClick={() => setSearch("")} className={`absolute ${isRTL ? 'left-4' : 'right-4'} top-1/2 -translate-y-1/2 text-[#A08E70]/40 hover:text-[#E6B23C] transition-colors`}>
            <X size={14} />
          </button>
        )}
      </motion.div>

      {/* Search Results Overlay */}
      <div className="relative w-full z-50">
        <AnimatePresence>
          {search && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="absolute left-0 right-0 top-0 flex justify-center px-4"
            >
              <div className="w-full max-w-5xl">
                {(activeTab === "pharaohs" ? filteredPharaohs : filteredLandmarks).length === 0 ? (
                  <div className="max-w-md mx-auto text-center py-16 bg-[#0D0A07]/60 backdrop-blur-md rounded-3xl border border-[#E6B23C]/10 shadow-[0_20px_60px_rgba(0,0,0,0.5)]">
                    <Search size={40} className="mx-auto mb-4 opacity-30 text-[#E6B23C]" />
                    <p className="text-sm text-[#A08E70]">{t("explore.search.no_results", { search })}</p>
                  </div>
                ) : (
                  <div className="flex flex-wrap justify-center gap-4 max-h-[70vh] overflow-y-auto pb-10 pt-2 px-2">
                    {(activeTab === "pharaohs" ? filteredPharaohs : filteredLandmarks).map((entity) => {
                      const entityType = activeTab === "pharaohs" ? "pharaoh" : "landmark";
                      return (
                        <div key={entity.id} className="w-full sm:w-[320px]">
                          <EntityCard
                            entity={entity}
                            type={entityType}
                            onNavigate={() => handleEntityClick(entity, entityType)}
                          />
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <div ref={containerRef} className={`relative transition-all duration-700 ${search ? 'blur-3xl pointer-events-none' : ''}`}>
        {/* Loading */}
        {isLoading && (
          <div className="flex justify-center py-20">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
            >
              <Scroll size={32} className="text-[#E6B23C]/40" />
            </motion.div>
          </div>
        )}

        {/* ── PHARAOHS TAB ─────────────────────────────────────────────── */}
        {!isLoading && activeTab === "pharaohs" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-6 max-w-4xl mx-auto"
          >
            {pharaohsByPeriod.length === 0 && (
              <div className="text-center py-16 text-[#A08E70]/50">
                <Scroll size={40} className="mx-auto mb-4 opacity-30" />
                <p className="text-sm">{t("explore.search.no_pharaohs")}</p>
              </div>
            )}

            <div className="relative pt-10 pb-10">
              <ScrollLine />
              {pharaohsByPeriod.map(({ period, dynasties }, idx) => {
                const isLeft = idx % 2 === 0;
                return (
                  <motion.div
                    key={period}
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-10% 0px -10% 0px" }}
                    className={`relative flex md:justify-between items-center w-full mb-16 ${isLeft ? 'md:flex-row-reverse' : ''}`}
                  >
                    {/* Spacer for desktop */}
                    <div className="hidden md:block w-5/12" />

                    {/* Timeline dot */}
                    <div className="absolute left-5 md:left-1/2 w-4 h-4 rounded-full bg-[#1A1208] border-2 border-[#E6B23C] -translate-x-1/2 z-10 shadow-[0_0_15px_rgba(230,178,60,0.5)]" />

                    {/* Content card */}
                    <motion.div
                      initial={{ filter: "brightness(0.5) opacity(0.6)" }}
                      whileInView={{ filter: "brightness(1) opacity(1)" }}
                      viewport={{ margin: "-35% 0px -35% 0px" }}
                      transition={{ duration: 0.5 }}
                      className="w-full pl-12 md:pl-0 md:w-5/12"
                    >
                      <div className={`flex flex-col ${isLeft ? 'md:items-end md:text-right' : 'md:items-start md:text-left'} mb-4`}>
                        <span className="text-sm font-bold tracking-[0.3em] text-[#E6B23C] uppercase drop-shadow-[0_0_8px_rgba(230,178,60,0.4)]">
                          {period}
                        </span>
                      </div>

                      <div className="space-y-3">
                        {dynasties.map(({ dynasty, entities }) => (
                          <DynastyGroup
                            key={dynasty}
                            dynasty={dynasty}
                            entities={entities}
                            type="pharaoh"
                            onNavigate={(e) => handleEntityClick(e, "pharaoh")}
                          />
                        ))}
                      </div>
                    </motion.div>
                  </motion.div>
                );
              })}
            </div>

            <div className="text-center pt-8 pb-4 text-[10px] text-[#A08E70]/30 uppercase tracking-widest">
              {filteredPharaohs.length} {t("common.entity")} / {pharaohs.length} {t("common.entities")}
            </div>
          </motion.div>
        )}

        {/* ── LANDMARKS TAB ────────────────────────────────────────────── */}
        {!isLoading && activeTab === "landmarks" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-6xl mx-auto px-4"
          >


            <AnimatePresence>
              {!selectedCity && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex justify-center mb-8"
                >
                  <div className="inline-flex items-center gap-3 px-6 py-2.5 rounded-full bg-[#E6B23C]/5 border border-[#E6B23C]/10 backdrop-blur-sm">
                    <span className="text-[11px] font-bold tracking-[0.2em] text-[#A08E70] uppercase">
                      {t("explore.map.instruction")}
                    </span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="relative flex flex-col md:flex-row items-start justify-center gap-8">
              {/* Centered Large Map */}
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{
                  opacity: 1,
                  scale: 1,
                  x: selectedCity ? (window.innerWidth > 768 ? -220 : 0) : 0
                }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="relative z-0 w-full"
              >
                <EgyptMap
                  pins={mapPins}
                  selectedCity={selectedCity}
                  onSelectCity={setSelectedCity}
                />
              </motion.div>

              {/* Landmarks Overlay (Sliding Panel) */}
              <AnimatePresence>
                {selectedCity && (
                  <motion.div
                    initial={{ x: "100%", opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: "100%", opacity: 0 }}
                    transition={{ type: "spring", damping: 25, stiffness: 200 }}
                    className="absolute top-0 right-0 h-full w-full md:w-[450px] bg-[#0D0A07]/95 backdrop-blur-xl border-l border-[#E6B23C]/20 z-[100] shadow-[-20px_0_50px_rgba(0,0,0,0.5)] flex flex-col rounded-r-3xl overflow-hidden"
                  >
                    {/* Header */}
                    <div className="p-8 border-b border-[#E6B23C]/10 flex items-center justify-between bg-gradient-to-r from-[#1A1208] to-[#0D0A07]">
                      <div>
                        <div className="flex items-center gap-2 text-[#E6B23C] mb-1">
                          <MapPin size={16} />
                          <span className="text-[10px] font-bold tracking-[0.3em] uppercase">{t("explore.map.region")}</span>
                        </div>
                        <h2 className="text-2xl font-bold text-[#F5E6D0] uppercase tracking-wider font-heading">
                          {selectedCity.split(",")[0]}
                        </h2>
                      </div>
                      <button
                        onClick={() => setSelectedCity(null)}
                        className="p-3 rounded-full bg-[#E6B23C]/10 text-[#E6B23C] hover:bg-[#E6B23C]/20 transition-all border border-[#E6B23C]/20"
                      >
                        <X size={20} />
                      </button>
                    </div>

                    {/* Landmarks List */}
                    <div
                      ref={landmarksListRef}
                      className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar"
                    >
                      {landmarksByCity
                        .filter(([city]) => city === selectedCity)
                        .flatMap(([_, entities]) => entities)
                        .map((e) => (
                          <EntityCard
                            key={e.id}
                            entity={e}
                            type="landmark"
                            onNavigate={() => handleEntityClick(e, "landmark")}
                          />
                        ))
                      }
                    </div>

                    {/* Footer */}
                    <div className="p-6 border-t border-[#E6B23C]/10 bg-[#0D0A07] text-center">
                      <button
                        onClick={() => setSelectedCity(null)}
                        className="text-xs text-[#A08E70] hover:text-[#E6B23C] transition-colors underline underline-offset-4 uppercase tracking-widest font-bold"
                      >
                        {t("explore.map.close")}
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </div>
    </PageShell>
  );
}

export default function ExplorePage() {
  return (
    <Suspense fallback={<div className="min-h-screen" style={{ background: "#0D0A07" }} />}>
      <ExploreContent />
    </Suspense>
  );
}
