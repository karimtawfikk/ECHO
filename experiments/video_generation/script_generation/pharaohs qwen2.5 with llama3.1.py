import re
import gc
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional

DOCS_DIR = Path("docs")
OUT_DIR = Path("qwen2.5-llama3.1_Scripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FACTS   = "qwen2.5:7b"
MODEL_SCRIPT  = "llama3.1:latest"
MODEL_REWRITE = "qwen2.5:7b"

OLLAMA_URL = "http://localhost:11434/api/chat"


SAFE_CHUNK_TOKENS = 1200
CHUNK_OVERLAP_TOKENS = 200

FACTS_MAX_NEW = 600
SCRIPT_MAX_NEW = 520
REWRITE_MAX_NEW = 520

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


ENTITY_MAP = {
    'amun': 'god',
    'amun ra': 'god',
    'anubis god': 'god',
    'mut': 'goddess',
    'khonsu': 'god',
    'isis': 'goddess',
    'hathor': 'goddess',
    'hauron': 'god',
    'horus': 'god',
    'osiris': 'god',
    'ptah': 'god',
    'rahorakhty': 'god',
    'sekhmet': 'goddess',
    'seth': 'god',
    'anath': 'goddess',
    'serapis': 'god',
    'bat': 'goddess',
    'taweret': 'goddess',

    'amenirdis': 'family',
    'arsinoe ii': 'family',
    'arsinoe iii': 'family',
    'isis (mother of thutmose iii)': 'family',
    'khamerernebty ii': 'family',
    'meresankh iii': 'family',
    'mutnofret': 'family',
    'nofret': 'family',
    'thuya': 'family',
    'tiye': 'family',
    'ankhsenamun': 'family',
    'yuya': 'family',

    'khasekhemwy': 'pharaoh',
    'djoser': 'pharaoh',
    'sneferu': 'pharaoh',
    'khufu': 'pharaoh',
    'khafre': 'pharaoh',
    'menkaura': 'pharaoh',
    'shepseskaf': 'pharaoh',
    'userkaf': 'pharaoh',
    'raneferef': 'pharaoh',
    'nyuserra': 'pharaoh',
    'teti': 'pharaoh',
    'pepy i': 'pharaoh',
    'mentuhotep ii': 'pharaoh',
    'amenemhet i': 'pharaoh',
    'sesostris i (senusret i)': 'pharaoh',
    'sesostris iii (senusret iii)': 'pharaoh',
    'sesostris iv (senusret iv)': 'pharaoh',
    'amenemhet iii': 'pharaoh',
    'hor awibre': 'pharaoh',
    'sobekhotep iv': 'pharaoh',
    'sobekhotep v': 'pharaoh',
    'sobekemsaf i': 'pharaoh',
    'ahmose i': 'pharaoh',
    'hatshepsut': 'pharaoh',
    'thutmose iii': 'pharaoh',
    'amenhotep ii': 'pharaoh',
    'thutmose iv': 'pharaoh',
    'amenhotep iii': 'pharaoh',
    'amenhotep iv(akhenaten)': 'pharaoh',
    'nefertiti': 'pharaoh',
    'smenkhkare': 'pharaoh',
    'tutankhamun': 'pharaoh',
    'horemheb': 'pharaoh',
    'seti i': 'pharaoh',
    'rameses ii': 'pharaoh',
    'rameses iii': 'pharaoh',
    'merenptah': 'pharaoh',
    'sety ii': 'pharaoh',
    'psusennes i': 'pharaoh',
    'osorkon ii': 'pharaoh',
    'shabaka': 'pharaoh',
    'ahmose ii': 'pharaoh',
    'amasis': 'pharaoh',
    'hakor(achoris)': 'pharaoh',
    'achoris': 'pharaoh',
    'nectanebo i': 'pharaoh',
    'nectanebo ii': 'pharaoh',
    'alexander the great': 'pharaoh',
    'cleopatra vii': 'pharaoh',
    'ptolemy i': 'pharaoh',
    'ptolemy ii': 'pharaoh',
    'ptolemy iii': 'pharaoh',
}


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

def get_entity_type(filename_stem: str) -> str:
    key = normalize_name_key(filename_stem)
    if "goddess" in key: 
        return "goddess"
    if re.search(r"\bgod\b", key): 
        return "god"
    if "mother" in key or "wife" in key or "daughter" in key: 
        return "family"
    if key in ENTITY_MAP: 
        return ENTITY_MAP[key]
    for k, t in ENTITY_MAP.items():
        if k in key or key in k:
            return t
    return "pharaoh"

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

def approx_token_count(text: str) -> int:
    return int(len(text.split()) / 0.75) + 1

def chunk_text_approx_tokens(text: str, max_tokens: int,  overlap_tokens: int) -> List[str]:
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

    out = []
    seen = set()
    for c in chunks:
        key = re.sub(r"\s+", " ", c.strip())
        if key and key not in seen:
            out.append(c)
            seen.add(key)

    return out

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

def facts_prompt(name: str, entity_type: str, chunk_text: str, n_min: int, n_max: int) -> str:
    role_line = {
        "pharaoh": f"Pharaoh {name}",
        "god": f"the god {name}",
        "goddess": f"the goddess {name}",
        "family": f"{name}",
    }.get(entity_type, name)

    return f"""
Extract facts ONLY from the TEXT about {role_line}.

Return STRICT JSON ONLY:
[
  {{"claim":"...","evidence":"..."}},
  ...
]

Rules:
- evidence must be an exact quote copied from the text (12-60 words).
- claim must be directly supported by the evidence.
- Return {n_min} to {n_max} items.
- Do NOT guess. Do NOT add outside knowledge.

TEXT:
\"\"\"{chunk_text}\"\"\"
""".strip()

def try_parse_json_list(s: str) -> Optional[List[Dict]]:
    m = re.search(r"\[\s*\{.?\}\s\]", s, flags=re.DOTALL)
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

def score_sentence(entity_type: str, s: str) -> int:
    t = s.lower()
    score = 0
    if re.search(r"\b\d{3,4}\b", t) or "bc" in t or "bce" in t:
        score += 4
    if any(k in t for k in ["dynasty", "reign", "ruled", "co-regent", "coruler", "co-ruler"]):
        score += 4

    if entity_type == "pharaoh":
            keys = ["hyksos", "avaris", "campaign", "battle", "siege", "expulsion",
                    "new kingdom", "memphis", "delta", "nubia", "canaan", "palestine",
                    "thebes", "karnak", "temple", "pyramid", "tomb", "succession", "co-regent"]
    elif entity_type in ("god", "goddess"):
            keys = ["worship", "cult", "temple", "domain", "symbol", "iconography", "myth",
                    "afterlife", "mummification", "sun", "moon", "sphinx", "spell", "protect"]
    else:
            keys = ["daughter", "wife", "mother", "regent", "influence", "title", "tomb", "burial"]

    for k in keys:
        if k in t:
            score += 2
    return score

def extractive_claims(full_text: str, entity_type: str, k: int = 16) -> List[Dict]:
    sents = split_sentences(full_text)
    ranked = sorted(sents, key=lambda s: score_sentence(entity_type, s), reverse=True)

    out = []
    for s in ranked:
        evidence = ensure_sentence_boundary(full_text, s, max_words=110)
        claim = ensure_sentence_boundary(full_text, s, max_words=40)
        out.append({"claim": claim, "evidence": evidence})
        if len(out) >= k:
            break

    return dedupe_claims(out, full_text=full_text)


def claim_score_for_video(entity_type: str, claim: str) -> int:
    c = claim.lower()
    score = 0

    if re.search(r"\b\d{3,4}\b", c) or "bce" in c or "bc" in c:
        score += 3
    if any(x in c for x in ["dynasty", "reign", "ruled", "co-regent", "co-ruler", "succession"]):
        score += 4

    if entity_type == "pharaoh":
        if any(x in c for x in ["hyksos", "avaris", "siege", "battle", "campaign", "expulsion", "conquest"]):
            score += 4
        if any(x in c for x in ["temple", "pyramid", "tomb", "karnak", "thebes", "memphis", "delta"]):
            score += 3
        if any(x in c for x in ["trade", "economy", "flood", "fayoum", "sinai", "quarries", "mines"]):
            score += 2

    elif entity_type in ("god", "goddess"):
        if any(x in c for x in ["worship", "cult", "temple", "ritual", "spell", "protector", "healing"]):
            score += 4
        if any(x in c for x in ["symbol", "iconography", "depicted", "associated", "sphinx", "moon", "sun"]):
            score += 3

    else:  
        if any(x in c for x in ["mother", "wife", "daughter", "regent", "queen", "priestess"]):
            score += 4
        if any(x in c for x in ["tomb", "burial", "title", "influence"]):
            score += 2

    return score
    
def select_top_claims(entity_type: str, claims: List[Dict], full_text: str, k: int = 9) -> List[Dict]:
    claims = dedupe_claims(claims, full_text=full_text)
    ranked = sorted(claims, key=lambda x: claim_score_for_video(entity_type, x["claim"]), reverse=True)
    return ranked[:k]


SYSTEM_SCRIPT = (
    "You write short, engaging museum-style video scripts in very simple English. "
    "You must not invent facts. You must obey the allowed facts list. "
    "No lists. No headings. No bullet points. No numbering"
)

def script_prompt(name: str, entity_type: str, selected: List[Dict]) -> str:
    tone = {
        "pharaoh": "Epic and powerful, but simple English.",
        "god": "Mystical and fascinating, but simple English.",
        "goddess": "Mystical and fascinating, but simple English.",
        "family": "Personal and compelling, but simple English.",
    }.get(entity_type, "Simple and engaging.")

    facts_block = "\n".join(
        [f"- FACT: {c['claim']}\n  QUOTE: \"{c['evidence']}\"" for c in selected]
    )

    return f"""
Write an engaging ~1 to ~1.5 minute video script about {name}.

HARD RULES:
- Use ONLY the facts/quotes below. Do NOT add new facts, dates, places, names, or achievements.
- Each fact may be used at most once.
- Do NOT mention “the document” or “the text says”.
- Output ONLY the script text (no title, no bullets, no numbering, no parentheses, no quotation marks).
- Smooth transitions, no repetition.
- Simple, conversational English.
- Word count: {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}.

STRUCTURE (follow exactly):
1) Hook (1 sentence)
2) Context (1 sentence)
3) Main story (6-9 sentences, smooth transitions)
5) Legacy ending (1 sentence)

TONE: {tone}

ALLOWED FACTS/QUOTES:
{facts_block}

SCRIPT:
""".strip()

def rewrite_prompt(name: str, script: str, selected: List[Dict]) -> str:
    facts_block = "\n".join(
        [f"- FACT: {c['claim']}\n  QUOTE: \"{c['evidence']}\"" for c in selected]
    )
    return f"""
Rewrite the script about {name} to {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} words (ideal {TARGET_IDEAL}).
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

def extract_grounded_claims(name: str, entity_type: str, full_text: str, doc_id: str) -> List[Dict]:

    chunks = chunk_text_approx_tokens(full_text, SAFE_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
    all_claims: List[Dict] = []
    raw_dump: List[str] = []

    system = "You extract grounded facts as STRICT JSON only. Never explain. Never add text. Only valid JSON list."

    for idx, ch in enumerate(chunks):
        user = facts_prompt(name, entity_type, ch, n_min=6, n_max=10)
        raw = ollama_chat(MODEL_FACTS,system, user, FACTS_MAX_NEW, FACTS_TEMP, FACTS_TOP_P)
        raw_dump.append(f"\n--- PASS1 CHUNK {idx} ---\n{raw}\n")
        parsed = try_parse_json_list(raw)
        if parsed:
            all_claims.extend(parsed)

    all_claims = dedupe_claims(all_claims, full_text=full_text)
    # PASS 2: if still low facts, ask for more (helps big docs that only gave 4-5)
    if len(all_claims) < 8 and len(chunks) > 1:
        for idx, ch in enumerate(chunks[: max(2, len(chunks)//2)]):
            user = facts_prompt(name, entity_type, ch, n_min=8, n_max=12)
            raw = ollama_chat(MODEL_FACTS,system, user, max_new=FACTS_MAX_NEW, temperature=FACTS_TEMP, top_p=FACTS_TOP_P)
            raw_dump.append(f"\n--- PASS2 CHUNK {idx} ---\n{raw}\n")
            parsed = try_parse_json_list(raw)
            if parsed:
                all_claims.extend(parsed)

        all_claims = dedupe_claims(all_claims, full_text=full_text)

    if len(all_claims) < 6:
        all_claims = extractive_claims(full_text, entity_type, k=16)

    (CLAIMS_DIR / f"{doc_id}.json").write_text(json.dumps(all_claims, ensure_ascii=False, indent=2),encoding="utf-8")
    return all_claims


def build_script(name: str, entity_type: str, claims: List[Dict], full_text: str) -> Tuple[str, List[Dict]]:
    k = 10 if entity_type in ("pharaoh",) else 9
    selected = select_top_claims(entity_type, claims, full_text=full_text, k=k)

    (SELECTED_DIR / f"{normalize_name_key(name).replace(' ', '_')}.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),encoding="utf-8"
    )

    if len(selected) < 4:
        selected = dedupe_claims(claims)[:8]

    if len(selected) < 4:
        return "Not enough grounded facts in the document to generate a script safely.", selected


    draft = ollama_chat(
        MODEL_SCRIPT,
        SYSTEM_SCRIPT,
        script_prompt(name, entity_type, selected),
        max_new=SCRIPT_MAX_NEW,
        temperature=SCRIPT_TEMP,
        top_p=SCRIPT_TOP_P
    )
    draft = force_complete_sentence(draft)

    final = ollama_chat(
        MODEL_REWRITE,
        SYSTEM_SCRIPT,
        rewrite_prompt(name, draft, selected),
        max_new=REWRITE_MAX_NEW,
        temperature=REWRITE_TEMP,
        top_p=REWRITE_TOP_P
    )
    final = force_complete_sentence(final)

    wc_f, wc_d = count_words(final), count_words(draft)
    in_f = TARGET_MIN_WORDS <= wc_f <= TARGET_MAX_WORDS
    in_d = TARGET_MIN_WORDS <= wc_d <= TARGET_MAX_WORDS
    
    if not in_f and in_d:
        final = draft

    return final, selected


def save_output(out_file: Path, name: str, entity_type: str,
                script: str, used: List[Dict], doc_words: int, facts_total: int):
    wc = count_words(script)
    ok = TARGET_MIN_WORDS <= wc <= TARGET_MAX_WORDS
    icon = {"pharaoh": "👑", "god": "✨", "goddess": "✨", "family": "👥"}.get(entity_type, "📜")

    facts_lines = []
    for idx, c in enumerate(used, 1):
        facts_lines.append(f"{idx}. {c['claim']}\n   Evidence: \"{c['evidence']}\"")
    facts_block = "\n".join(facts_lines) if facts_lines else "[]"

    text = f"""{icon} VIDEO SCRIPT: {name}
Type: {entity_type.upper()}

{script}

GROUNDED FACTS USED (CLAIM + EVIDENCE)
{facts_block}

METADATA
Word Count: {wc} (target {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}) => {'✅ OK' if ok else '⚠️ CHECK'}
Source Doc Size: {doc_words} words
facts_used={len(used)} | facts_total={facts_total}
Models: extractor={MODEL_FACTS} | narration={MODEL_SCRIPT} | rewrite={MODEL_REWRITE}
"""
    out_file.write_text(text, encoding="utf-8")
    
    
def main():
    print("VIDEO SCRIPT GENERATOR — MULTI MODEL")
    print("Facts Model  :", MODEL_FACTS)
    print("Script Model :", MODEL_SCRIPT)
    print("Rewrite Model:", MODEL_REWRITE)
    print("Docs :", DOCS_DIR.resolve())
    print("Out  :", OUT_DIR.resolve())
    print(f"Target words: {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} (ideal {TARGET_IDEAL})")

    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {DOCS_DIR.resolve()}")

    out_dir = OUT_DIR / "qwen2.5-llama3.1_grounded"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(files, 1):

        stem = fp.stem
        name = extract_display_name(stem)
        entity_type = get_entity_type(stem)

        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        doc_words = len(text.split())

        print(f"[{i}/{len(files)}] {name} | {entity_type} | {doc_words}w")

        try:
            claims = extract_grounded_claims(
                name=name,
                entity_type=entity_type,
                full_text=text,
                doc_id=normalize_name_key(stem).replace(" ", "_")
            )

            script, used = build_script(name, entity_type, claims, full_text=text)

            out_file = out_dir / fp.name
            save_output(out_file, name, entity_type, script, used, doc_words, facts_total=len(claims))

            wc = count_words(script)
            status = "OK" if TARGET_MIN_WORDS <= wc <= TARGET_MAX_WORDS else "CHECK"
            print(f"  -> saved | {wc}w {status} | facts_used={len(used)} | facts_total={len(claims)}\n")

        except Exception as e:
            print(f" Error: {e}\n")
            continue

        clear_mem()

    print(f"DONE. Outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()