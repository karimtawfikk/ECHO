"""
ANCIENT EGYPT VIDEO SCRIPT GENERATOR (GROUNDED + ROBUST) — MULTI-MODEL (LANDMARKS)
========================================================================
Stage Models:
- Facts   → qwen2.5:7b
- Script  → llama3.1:latest
- Rewrite → qwen2.5:7b
"""

import re
import gc
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =============================================================================
# PATHS
# =============================================================================
DOCS_DIR = Path("docs")
OUT_DIR = Path("qwen2.5-llama3.1_Landmark_Scripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODELS
# =============================================================================
MODEL_FACTS   = "qwen2.5:7b"
MODEL_SCRIPT  = "llama3.1:8b"
MODEL_REWRITE = "qwen2.5:7b"

OLLAMA_URL = "http://localhost:11434/api/chat"

# =============================================================================
# GENERATION SETTINGS
# =============================================================================
SAFE_CHUNK_TOKENS = 1200
CHUNK_OVERLAP_TOKENS = 200

FACTS_MAX_NEW = 600
SCRIPT_MAX_NEW = 520
REWRITE_MAX_NEW = 520

# default targets (normal docs)
TARGET_MIN_WORDS = 140
TARGET_MAX_WORDS = 220
TARGET_IDEAL = 185

FACTS_TEMP = 0.0
FACTS_TOP_P = 0.9

SCRIPT_TEMP = 0.75
SCRIPT_TOP_P = 0.92

REWRITE_TEMP = 0.15
REWRITE_TOP_P = 0.95

REPEAT_PENALTY = 1.18

CLAIMS_DIR = OUT_DIR / "_claims_json"
CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_DIR = OUT_DIR / "_selected_json"
SELECTED_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LANDMARK TYPE RULES (UPDATED FOR YOUR FULL LIST)
# Order matters: specific first, then common classes.
# =============================================================================
LANDMARK_TYPE_RULES = [
    # specific / special names
    ("osireion",   [r"\bosireion\b"]),
    ("ramesseum",  [r"\bramesseum\b"]),                      # treat like temple complex
    ("sphinx",     [r"\bsphinx\b"]),
    ("statue",     [r"\bcolossi\b", r"\bcolossus\b", r"\bstatue\b"]),
    ("fortress",   [r"\bqasr\b", r"\bfortress\b", r"\bcastle\b"]),

    # common landmark classes
    ("kiosk",      [r"\bkiosk\b"]),
    ("stela",      [r"\bstela\b", r"\bstele\b"]),
    ("pyramid",    [r"\bpyramid\b", r"\bpyramids\b"]),
    ("tomb",       [r"\btomb\b", r"\bmausoleum\b", r"\bnecropolis\b"]),
    ("temple",     [r"\btemple\b", r"\bmammisi\b", r"\bsanctuary\b", r"\bchapel\b", r"\bspeos\b"]),

    # scenic / general
    ("mountain",   [r"\bel qurn\b", r"\bqurn\b", r"\bmount\b", r"\bpeak\b", r"\bhill\b"]),
    ("complex",    [r"\bcomplex\b", r"\bsite\b"]),
]

# =============================================================================
# ADAPTIVE CONTROLS FOR SMALL DOCS
# =============================================================================
def facts_range_for_doc(doc_words: int) -> Tuple[int, int]:
    # small docs: ask for fewer facts to avoid repetition/junk
    if doc_words <= 450:
        return 4, 7
    if doc_words <= 800:
        return 6, 10
    return 8, 12

def targets_for_doc(doc_words: int) -> Tuple[int, int, int]:
    # small docs: allow slightly shorter scripts to reduce forced repetition
    if doc_words <= 450:
        return 120, 190, 160
    return TARGET_MIN_WORDS, TARGET_MAX_WORDS, TARGET_IDEAL

# =============================================================================
# NAME UTILITIES
# =============================================================================
_ROMAN_ONLY = re.compile(r"^(?=[IVXLCDM]+$)[IVXLCDM]+$", re.IGNORECASE)

def clear_mem():
    gc.collect()

def normalize_name_key(stem: str) -> str:
    s = stem.lower().replace(".txt", "")
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_for_display(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"([A-Za-z0-9])\(", r"\1 (", s)
    s = re.sub(r"\)([A-Za-z0-9])", r") \1", s)
    return s

def smart_title(s: str) -> str:
    parts = re.split(r"(\s+)", s.strip())
    out = []
    for p in parts:
        if p.isspace() or p == "":
            out.append(p)
            continue

        token = p
        core = re.sub(r"^[^\w]+|[^\w]+$", "", token)

        if core and _ROMAN_ONLY.match(core):
            out.append(token.replace(core, core.upper()))
            continue

        if any(ch.isdigit() for ch in core):
            out.append(token)
            continue

        out.append(token[0].upper() + token[1:].lower() if len(token) > 1 else token.upper())

    return "".join(out)

def extract_display_name(filename_stem: str) -> str:
    raw = normalize_for_display(filename_stem)
    return smart_title(raw)

# =============================================================================
# LANDMARK TYPE
# =============================================================================
def get_entity_type(filename_stem: str) -> str:
    key = normalize_name_key(filename_stem)
    for t, patterns in LANDMARK_TYPE_RULES:
        for pat in patterns:
            if re.search(pat, key, flags=re.IGNORECASE):
                return t
    return "landmark"

# =============================================================================
# OLLAMA CALL
# =============================================================================
def ollama_chat(model, system, user, max_new, temperature, top_p):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "num_predict": max_new,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": REPEAT_PENALTY,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}).get("content") or "").strip()

# =============================================================================
# CHUNKING
# =============================================================================
def approx_token_count(text: str) -> int:
    return int(len(text.split()) / 0.75) + 1

def chunk_text_approx_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return [text]

    chunks, cur, cur_tok = [], [], 0

    for p in paras:
        t = approx_token_count(p)
        if cur and cur_tok + t > max_tokens:
            chunk = "\n\n".join(cur).strip()
            chunks.append(chunk)

            words = chunk.split()
            ov_words = int(overlap_tokens * 0.75)
            tail = " ".join(words[-ov_words:]) if ov_words > 0 and len(words) > ov_words else chunk
            cur = [tail, p]
            cur_tok = approx_token_count(tail) + t
        else:
            cur.append(p)
            cur_tok += t

    if cur:
        chunks.append("\n\n".join(cur).strip())

    out, seen = [], set()
    for c in chunks:
        k = re.sub(r"\s+", " ", c.strip())
        if k and k not in seen:
            out.append(c)
            seen.add(k)
    return out

# =============================================================================
# CLEANING / FINALIZATION
# =============================================================================
def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

def clean_output(text: str) -> str:
    text = text.replace("</s>", "").replace("<s>", "").strip()
    text = re.sub(r"^\(\d+\s*lines?\)\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^\s*["“”\'\-\•\*]+\s*', "", text)
    return text.strip()

def force_complete_sentence(text: str) -> str:
    text = clean_output(text)
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    m = re.search(r"^(.+[.!?])\s", text + " ")
    if m:
        return m.group(1).strip()
    return text

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def pick_best_containing_sentence(full_text: str, fragment: str) -> Optional[str]:
    frag = re.sub(r"\s+", " ", fragment).strip()
    if len(frag) < 25:
        return None
    words = frag.split()
    anchor = " ".join(words[:10])
    anchor_re = re.escape(anchor)
    for s in split_sentences(full_text):
        if re.search(anchor_re, s, flags=re.IGNORECASE):
            return s
    return None

def ensure_sentence_boundary(full_text: str, s: str, max_words: int) -> str:
    s0 = re.sub(r"\s+", " ", (s or "")).strip()
    if not s0:
        return s0
    repaired = pick_best_containing_sentence(full_text, s0)
    if repaired:
        s0 = repaired
    words = s0.split()
    if len(words) <= max_words:
        return s0
    cut = " ".join(words[:max_words]).strip()
    m = re.search(r"^(.+?[.!?;:])\s", cut + " ")
    return m.group(1).strip() if m else cut

def dedupe_claims(claims: List[Dict], full_text: Optional[str] = None) -> List[Dict]:
    seen = set()
    out = []
    for c in claims:
        claim = (c.get("claim") or "").strip()
        ev = (c.get("evidence") or "").strip()

        if full_text:
            claim = ensure_sentence_boundary(full_text, claim, max_words=34)
            ev = ensure_sentence_boundary(full_text, ev, max_words=95)

        if len(claim) < 12 or len(ev.split()) < 8:
            continue

        key = re.sub(r"[^a-z0-9 ]+", "", claim.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"claim": claim, "evidence": ev})
    return out

# =============================================================================
# FACT EXTRACTION JSON (LANDMARKS)
# =============================================================================
def facts_prompt(name: str, landmark_type: str, chunk_text: str, n_min: int, n_max: int) -> str:
    place_line = f"the landmark {name}"
    if landmark_type != "landmark":
        place_line = f"the {landmark_type} {name}"

    return f"""
Extract facts ONLY from the TEXT about {place_line}.

Return STRICT JSON ONLY:
[
  {{"claim":"...","evidence":"..."}} ,
  ...
]

Rules:
- evidence must be an exact quote copied from the text (10-80 words).
- claim must be directly supported by the evidence.
- Return {n_min} to {n_max} items.
- Do NOT guess. Do NOT add outside knowledge.

TEXT:
\"\"\"{chunk_text}\"\"\"
""".strip()

def try_parse_json_list(s: str) -> Optional[List[Dict]]:
    m = re.search(r"\[\s*\{.*?\}\s*\]", s, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        data = json.loads(blob)
        if isinstance(data, list):
            out = []
            for it in data:
                if isinstance(it, dict) and "claim" in it and "evidence" in it:
                    out.append({"claim": str(it["claim"]).strip(),
                                "evidence": str(it["evidence"]).strip()})
            return out
    except Exception:
        return None
    return None

# =============================================================================
# EXTRACTIVE FALLBACK (GUARANTEED GROUNDED)
# =============================================================================
def score_sentence(landmark_type: str, s: str) -> int:
    t = s.lower()
    score = 0

    if re.search(r"\b\d{3,4}\b", t) or "bc" in t or "bce" in t or "ad" in t or "ce" in t:
        score += 4
    if any(k in t for k in ["dynasty", "reign", "ptolemaic", "roman", "new kingdom", "middle kingdom", "old kingdom", "amarna"]):
        score += 4

    if any(k in t for k in ["located", "situated", "west bank", "east bank", "nile", "aswan", "luxor", "dahshur", "nubia", "lake nasser", "alexandria"]):
        score += 3

    if any(k in t for k in ["pylon", "hypostyle", "sanctuary", "naos", "courtyard", "colonnade", "chapel",
                            "relief", "inscription", "columns", "architrave", "causeway", "chamber"]):
        score += 3

    if any(k in t for k in ["dedicated", "worship", "cult", "god", "goddess", "amun", "isis", "hathor", "thoth", "osiris", "horus", "serapis"]):
        score += 3

    if any(k in t for k in ["relocated", "moved", "unesco", "campaign", "aswan high dam", "saved", "flooding", "lake nasser"]):
        score += 4

    # type-specific boosts
    if landmark_type == "tomb" and any(k in t for k in ["mummy", "sarcophagus", "burial", "necropolis", "mausoleum", "poem"]):
        score += 4
    if landmark_type == "pyramid" and any(k in t for k in ["mudbrick", "casing", "substructure", "entrance", "chamber", "sarcophagus"]):
        score += 4
    if landmark_type == "stela" and any(k in t for k in ["inscribed", "stela", "stele", "text", "decree"]):
        score += 4
    if landmark_type in ("monument", "statue", "sphinx") and any(k in t for k in ["column", "pillar", "granite", "diocletian", "serapeum", "guardian", "lion", "colossi"]):
        score += 4
    if landmark_type == "ramesseum" and any(k in t for k in ["mortuary", "temple", "ramesses", "rameses", "thebes", "west bank"]):
        score += 3
    if landmark_type == "fortress" and any(k in t for k in ["fortified", "garrison", "walls", "gateway"]):
        score += 3
    if landmark_type == "mountain" and any(k in t for k in ["peak", "summit", "climb", "trail", "view", "ridge"]):
        score += 3

    return score

def extractive_claims(full_text: str, landmark_type: str, k: int = 16) -> List[Dict]:
    sents = split_sentences(full_text)
    ranked = sorted(sents, key=lambda s: score_sentence(landmark_type, s), reverse=True)

    out = []
    for s in ranked:
        evidence = ensure_sentence_boundary(full_text, s, max_words=110)
        claim = ensure_sentence_boundary(full_text, s, max_words=40)
        out.append({"claim": claim, "evidence": evidence})
        if len(out) >= k:
            break

    return dedupe_claims(out, full_text=full_text)

# =============================================================================
# SELECT TOP FACTS
# =============================================================================
def claim_score_for_video(landmark_type: str, claim: str) -> int:
    c = claim.lower()
    score = 0

    if re.search(r"\b\d{3,4}\b", c) or "bce" in c or "bc" in c or "ad" in c or "ce" in c:
        score += 3
    if any(x in c for x in ["dynasty", "ptolemaic", "roman", "new kingdom", "middle kingdom", "old kingdom", "amarna"]):
        score += 4

    if any(x in c for x in ["located", "situated", "west bank", "east bank", "nile", "aswan", "luxor", "nubia",
                             "dahshur", "lake nasser", "alexandria", "philae"]):
        score += 4

    if any(x in c for x in ["built", "constructed", "expanded", "added", "restored", "converted"]):
        score += 3

    if any(x in c for x in ["dedicated", "worship", "cult", "temple", "god", "goddess", "amun", "isis", "hathor", "thoth", "osiris", "horus", "serapis"]):
        score += 3

    if any(x in c for x in ["pylon", "hypostyle", "sanctuary", "naos", "courtyard", "colonnade",
                             "columns", "reliefs", "inscriptions", "causeway", "chamber"]):
        score += 3

    if any(x in c for x in ["relocated", "moved", "unesco", "aswan high dam", "lake nasser", "campaign"]):
        score += 5

    # type-specific boosts
    if landmark_type == "tomb" and any(x in c for x in ["mummy", "sarcophagus", "burial", "poem"]):
        score += 4
    if landmark_type == "pyramid" and any(x in c for x in ["mudbrick", "casing", "substructure", "chambers", "entrance"]):
        score += 4
    if landmark_type == "stela" and any(x in c for x in ["inscribed", "decree", "text", "stela", "stele"]):
        score += 4
    if landmark_type in ("monument", "statue", "sphinx") and any(x in c for x in ["column", "pillar", "granite", "diocletian", "serapeum", "guardian", "lion", "colossi"]):
        score += 4
    if landmark_type == "ramesseum" and any(x in c for x in ["mortuary", "temple", "ramesses", "rameses"]):
        score += 3
    if landmark_type == "fortress" and any(x in c for x in ["fortified", "garrison", "walls", "gateway"]):
        score += 3
    if landmark_type == "mountain" and any(x in c for x in ["peak", "summit", "climb", "trail", "ridge"]):
        score += 3

    return score

def select_top_claims(landmark_type: str, claims: List[Dict], full_text: str, k: int = 9) -> List[Dict]:
    claims = dedupe_claims(claims, full_text=full_text)
    ranked = sorted(claims, key=lambda x: claim_score_for_video(landmark_type, x["claim"]), reverse=True)
    return ranked[:k]

# =============================================================================
# SCRIPT PROMPTS (ADAPTIVE TARGETS + UPDATED TONES)
# =============================================================================
SYSTEM_SCRIPT = (
    "You write short, engaging museum-style video scripts in very simple English. "
    "You must not invent facts. You must obey the allowed facts list. "
    "No lists. No headings. No bullet points. No numbering."
)

def script_prompt(name: str, landmark_type: str, selected: List[Dict], tmin: int, tmax: int) -> str:
    tone = {
        "temple":   "Calm, awe-filled, and vivid, but simple English.",
        "ramesseum":"Calm, awe-filled, and vivid, but simple English.",
        "tomb":     "Personal and mysterious, but simple English.",
        "pyramid":  "Grand and curious, but simple English.",
        "kiosk":    "Elegant and intriguing, but simple English.",
        "stela":    "Story-like and informative, but simple English.",
        "monument": "Bold and surprising, but simple English.",
        "statue":   "Grand and human, but simple English.",
        "sphinx":   "Mysterious and watchful, but simple English.",
        "mountain": "Atmospheric and scenic, but simple English.",
        "osireion": "Mystical and ancient, but simple English.",
        "fortress": "Bold and historic, but simple English.",
        "complex":  "Guided-tour style, but simple English.",
        "landmark": "Guided-tour style, but simple English.",
    }.get(landmark_type, "Guided-tour style, but simple English.")

    facts_block = "\n".join([f"- FACT: {c['claim']}\n  QUOTE: \"{c['evidence']}\"" for c in selected])

    return f"""
Write an engaging ~1 to ~1.5 minute video script about the {landmark_type} {name}.

HARD RULES:
- Use ONLY the facts/quotes below. Do NOT add new facts, dates, places, names, or achievements.
- Each fact may be used at most once.
- Do NOT mention “the document” or “the text says”.
- Output ONLY the script text (no title, no bullets, no numbering, no parentheses, no quotation marks).
- Smooth transitions, no repetition.
- Very simple, conversational English.
- Word count: {tmin}-{tmax}.

STRUCTURE (follow exactly):
1) Hook (1 sentence)
2) Where it is + what it is (1 sentence)
3) Main story (6-9 sentences, smooth transitions)
4) Why it matters today (1 sentence)

TONE: {tone}

ALLOWED FACTS/QUOTES:
{facts_block}

SCRIPT:
""".strip()

def rewrite_prompt(name: str, landmark_type: str, script: str, selected: List[Dict], tmin: int, tmax: int, tideal: int) -> str:
    facts_block = "\n".join([f"- FACT: {c['claim']}\n  QUOTE: \"{c['evidence']}\"" for c in selected])
    return f"""
Rewrite the script about the {landmark_type} {name} to {tmin}-{tmax} words (ideal {tideal}).
Make it smoother and more engaging, in very simple English.

HARD RULES:
- Do NOT add new facts. Use ONLY the allowed facts/quotes.
- No title, no bullets, no numbering, no parentheses, no quotation marks.
- Remove repetition. Keep clear flow and transitions.
- End with a strong final sentence.

ALLOWED FACTS/QUOTES:
{facts_block}

CURRENT SCRIPT:
{script}

REWRITE:
""".strip()

# =============================================================================
# PIPELINE
# =============================================================================
def extract_grounded_claims(name: str, landmark_type: str, full_text: str, doc_id: str) -> List[Dict]:
    chunks = chunk_text_approx_tokens(full_text, SAFE_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
    all_claims: List[Dict] = []

    doc_words = len(full_text.split())
    n_min, n_max = facts_range_for_doc(doc_words)

    system = "You extract grounded facts as STRICT JSON only. Never explain. Never add text. Only valid JSON list."

    for ch in chunks:
        user = facts_prompt(name, landmark_type, ch, n_min=n_min, n_max=n_max)
        raw = ollama_chat(MODEL_FACTS, system, user, FACTS_MAX_NEW, FACTS_TEMP, FACTS_TOP_P)
        parsed = try_parse_json_list(raw)
        if parsed:
            all_claims.extend(parsed)

    all_claims = dedupe_claims(all_claims, full_text=full_text)

    # PASS 2 (adaptive)
    if len(all_claims) < max(6, n_min) and len(chunks) > 1:
        n2_min = min(10, n_min + 2)
        n2_max = min(14, n_max + 2)
        for ch in chunks[: max(2, len(chunks)//2)]:
            user = facts_prompt(name, landmark_type, ch, n_min=n2_min, n_max=n2_max)
            raw = ollama_chat(MODEL_FACTS, system, user, FACTS_MAX_NEW, FACTS_TEMP, FACTS_TOP_P)
            parsed = try_parse_json_list(raw)
            if parsed:
                all_claims.extend(parsed)
        all_claims = dedupe_claims(all_claims, full_text=full_text)

    # Extractive fallback
    if len(all_claims) < 6:
        all_claims = extractive_claims(full_text, landmark_type, k=16)

    (CLAIMS_DIR / f"{doc_id}.json").write_text(
        json.dumps(all_claims, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return all_claims

def build_script(name: str, landmark_type: str, claims: List[Dict], full_text: str) -> Tuple[str, List[Dict]]:
    doc_words = len(full_text.split())
    tmin, tmax, tideal = targets_for_doc(doc_words)

    k = 10 if landmark_type in ("temple", "pyramid", "complex", "ramesseum") else 9
    selected = select_top_claims(landmark_type, claims, full_text=full_text, k=k)

    (SELECTED_DIR / f"{normalize_name_key(name).replace(' ', '_')}.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if len(selected) < 4:
        selected = dedupe_claims(claims, full_text=full_text)[:8]

    if len(selected) < 4:
        return "Not enough grounded facts in the document to generate a script safely.", selected

    draft = ollama_chat(
        MODEL_SCRIPT,
        SYSTEM_SCRIPT,
        script_prompt(name, landmark_type, selected, tmin=tmin, tmax=tmax),
        SCRIPT_MAX_NEW,
        SCRIPT_TEMP,
        SCRIPT_TOP_P
    )
    draft = force_complete_sentence(draft)

    final = ollama_chat(
        MODEL_REWRITE,
        SYSTEM_SCRIPT,
        rewrite_prompt(name, landmark_type, draft, selected, tmin=tmin, tmax=tmax, tideal=tideal),
        REWRITE_MAX_NEW,
        REWRITE_TEMP,
        REWRITE_TOP_P
    )
    final = force_complete_sentence(final)

    wc_f, wc_d = count_words(final), count_words(draft)
    in_f = tmin <= wc_f <= tmax
    in_d = tmin <= wc_d <= tmax
    if not in_f and in_d:
        final = draft

    return final, selected

# =============================================================================
# SAVE OUTPUT (UPDATED ICONS)
# =============================================================================
def save_output(out_file: Path, name: str, landmark_type: str,
                script: str, used: List[Dict], doc_words: int, facts_total: int):
    wc = count_words(script)
    tmin, tmax, tideal = targets_for_doc(doc_words)
    ok = tmin <= wc <= tmax

    icon = {
        "temple": "🏛️",
        "ramesseum": "🏛️",
        "tomb": "⚱️",
        "pyramid": "🔺",
        "kiosk": "🏛️",
        "stela": "🪨",
        "monument": "🗿",
        "statue": "🗿",
        "sphinx": "🦁",
        "mountain": "⛰️",
        "osireion": "🌀",
        "fortress": "🛡️",
        "complex": "🧭",
        "landmark": "📍",
    }.get(landmark_type, "📍")

    facts_lines = []
    for idx, c in enumerate(used, 1):
        facts_lines.append(f"{idx}. {c['claim']}\n   Evidence: \"{c['evidence']}\"")
    facts_block = "\n".join(facts_lines) if facts_lines else "[]"

    text = f"""===========================================================================
{icon} ANCIENT EGYPT VIDEO SCRIPT: {name}
===========================================================================
Type: {landmark_type.upper()}

{script}

===========================================================================
GROUNDED FACTS USED (CLAIM + EVIDENCE)
===========================================================================
{facts_block}

===========================================================================
METADATA
===========================================================================
Word Count: {wc} (target {tmin}-{tmax}, ideal {tideal}) => {'✅ OK' if ok else '⚠️ CHECK'}
Source Doc Size: {doc_words} words
facts_used={len(used)} | facts_total={facts_total}
Models: extractor={MODEL_FACTS} | narration={MODEL_SCRIPT} | rewrite={MODEL_REWRITE}
===========================================================================
"""
    out_file.write_text(text, encoding="utf-8")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("============================================================")
    print("ANCIENT EGYPT VIDEO SCRIPT GENERATOR — MULTI MODEL (LANDMARKS)")
    print("============================================================")
    print("Facts Model  :", MODEL_FACTS)
    print("Script Model :", MODEL_SCRIPT)
    print("Rewrite Model:", MODEL_REWRITE)
    print("Docs :", DOCS_DIR.resolve())
    print("Out  :", OUT_DIR.resolve())
    print(f"Default target words: {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} (ideal {TARGET_IDEAL})")
    print("Small docs (<=450w) target words: 120-190 (ideal 160)")
    print("============================================================\n")

    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {DOCS_DIR.resolve()}")

    out_dir = OUT_DIR / "qwen2.5-llama3.1_landmarks_grounded"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(files, 1):
        stem = fp.stem
        name = extract_display_name(stem)
        landmark_type = get_entity_type(stem)

        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        doc_words = len(text.split())

        tmin, tmax, tideal = targets_for_doc(doc_words)
        icon = {"temple": "🏛️","ramesseum": "🏛️","tomb": "⚱️","pyramid": "🔺","kiosk": "🏛️","stela": "🪨","monument": "🗿",
        "statue": "🗿","sphinx": "🦁","mountain": "⛰️","osireion": "🌀","fortress": "🛡️","complex": "🧭","landmark": "📍",
        }.get(landmark_type, "📍")
        print(f"[{i}/{len(files)}] {icon} {name} | {landmark_type} | {doc_words}w | target {tmin}-{tmax}")

        try:
            claims = extract_grounded_claims(
                name=name,
                landmark_type=landmark_type,
                full_text=text,
                doc_id=normalize_name_key(stem).replace(" ", "_")
            )

            script, used = build_script(name, landmark_type, claims, full_text=text)

            out_file = out_dir / fp.name
            save_output(out_file, name, landmark_type, script, used, doc_words, facts_total=len(claims))

            wc = count_words(script)
            status = "✅OK" if tmin <= wc <= tmax else "⚠️CHECK"
            print(f"  -> saved | {wc}w {status} | facts_used={len(used)} | facts_total={len(claims)}\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            continue

        clear_mem()

    print("============================================================")
    print(f"🎉 DONE. Outputs saved to: {out_dir.resolve()}")
    print("============================================================")

if __name__ == "__main__":
    main()