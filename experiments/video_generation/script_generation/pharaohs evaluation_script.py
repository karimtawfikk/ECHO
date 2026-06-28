import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util
import textstat


DOCS_DIR = Path("docs")
BASE_RUNS_DIR = Path("scripts//Scripts")

EVAL_DIR = BASE_RUNS_DIR / "_eval_all"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DETAILS_CSV = EVAL_DIR / "all_runs_details.csv"
MASTER_PROBLEMS_CSV = EVAL_DIR / "problem_cases_all.csv"
MASTER_LEADERBOARD_STAGE1_CSV = EVAL_DIR / "leaderboard_by_stage1.csv"
MASTER_LEADERBOARD_STAGE2_CSV = EVAL_DIR / "leaderboard_by_stage2.csv"
MASTER_LEADERBOARD_FINAL_CSV  = EVAL_DIR / "leaderboard_by_final.csv"

EMBED_MODEL = "all-MiniLM-L6-v2"

TARGET_MIN_WORDS = 140
TARGET_MAX_WORDS = 220

EXCLUDE_RUN_DIR_NAMES = {"_debug_raw", "_eval", "_eval_all", "__pycache__"}


SCRIPT_HEADER_RE = re.compile(r"^={10,}\n.*?VIDEO SCRIPT:.*?\n={10,}\nType:.*?\n\n", re.DOTALL)
FACTS_BLOCK_RE = re.compile(
    r"=+\nGROUNDED FACTS USED \(CLAIM \+ EVIDENCE\)\n=+\n(.*?)\n\n=+\nMETADATA\n=+",
    re.DOTALL
)
METADATA_RE = re.compile(
    r"Word Count:\s*(\d+).*\nSource Doc Size:\s*(\d+)\s*words.*\nfacts_used=(\d+)\s*\|\s*facts_total=(\d+)",
    re.DOTALL
)
FACT_ITEM_RE = re.compile(
    r"^\s*\d+\.\s*(.*?)\n\s*Evidence:\s*\"(.*?)\"\s*$",
    re.DOTALL | re.MULTILINE
)

YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
DYNASTY_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\s+dynasty\b", re.IGNORECASE)
CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

STOP_CAP = {
    "The", "A", "An", "And", "Or", "But", "In", "On", "At", "To", "From", "Of", "For",
    "With", "As", "By", "Into", "Over", "After", "Before", "During", "Within", "Without",
    "This", "That", "These", "Those", "It", "Its", "He", "She", "They", "We", "You", "I",
    "One", "Today", "Later", "Finally",
}

DIGIT_RE = re.compile(r"\b\d+\b")
ORDINAL_WORDS = ["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh","twelfth"]
NUMBER_WORDS = ["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve"]
ORDINAL_SUFFIX_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)

LEGACY_KEYWORDS = ["legacy","lasting","remembered","history","influence","endures","left behind","still","today","mark","shaped"]


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def split_sentences(text: str) -> List[str]:
    text = norm_ws(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def repetition_3gram(script: str) -> float:
    words = re.findall(r"\b\w+\b", (script or "").lower())
    if len(words) < 20:
        return 0.0
    grams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    return float(1.0 - safe_div(len(set(grams)), len(grams))) if grams else 0.0

def readability_metrics(script: str) -> Dict:
    s = script or ""
    try:
        fre = float(textstat.flesch_reading_ease(s))
    except Exception:
        fre = 0.0
    sents = split_sentences(s)
    avg_sent_words = float(np.mean([count_words(x) for x in sents])) if sents else 0.0
    return {"flesch_reading_ease": fre, "avg_sentence_words": avg_sent_words, "n_sentences": len(sents)}

def parse_generated_output(txt_path: Path) -> Dict:
    raw = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")

    m = METADATA_RE.search(raw)
    wc_rep = int(m.group(1)) if m else None
    doc_words_rep = int(m.group(2)) if m else None
    facts_used_rep = int(m.group(3)) if m else None
    facts_total_rep = int(m.group(4)) if m else None

    raw = raw.lstrip("\ufeff\n\r\t ")
    script = SCRIPT_HEADER_RE.sub("", raw, count=1)
    cut = re.search(r"\n=+\nGROUNDED FACTS USED", script)
    if cut:
        script = script[:cut.start()]
    script = script.strip()

    used_facts = []
    fb = FACTS_BLOCK_RE.search(raw)
    if fb:
        facts_block = fb.group(1).strip()
        for m2 in FACT_ITEM_RE.finditer(facts_block):
            used_facts.append({
                "claim": norm_ws(m2.group(1)),
                "evidence": norm_ws(m2.group(2)),
            })

    return {
        "file": txt_path.name,
        "script": script,
        "used_facts": used_facts,
        "word_count_reported": wc_rep,
        "doc_words_reported": doc_words_rep,
        "facts_used_reported": facts_used_rep,
        "facts_total_reported": facts_total_rep,
    }

def evidence_in_source_rate(facts: List[Dict], source_text: str) -> Tuple[float, int, int]:
    src = norm_ws(source_text)
    if not facts:
        return 0.0, 0, 0
    ok = 0
    for f in facts:
        ev = norm_ws(f.get("evidence", ""))
        if ev and ev in src:
            ok += 1
    return safe_div(ok, len(facts)), ok, len(facts)

def evidence_length_stats(facts: List[Dict]) -> Dict:
    lens = [count_words(f.get("evidence", "")) for f in facts if f.get("evidence")]
    if not lens:
        return {"evidence_words_mean": 0.0, "evidence_words_min": 0, "evidence_words_max": 0}
    return {
        "evidence_words_mean": float(np.mean(lens)),
        "evidence_words_min": int(np.min(lens)),
        "evidence_words_max": int(np.max(lens)),
    }

def evidence_len_ok_rate(used_facts: List[Dict], min_w=12, max_w=60) -> float:
    if not used_facts:
        return 0.0
    lens = [count_words(f.get("evidence","")) for f in used_facts if f.get("evidence")]
    if not lens:
        return 0.0
    ok = sum(1 for L in lens if min_w <= L <= max_w)
    return ok / len(lens)

def duplicate_claim_rate(facts: List[Dict]) -> float:
    claims = [norm_ws(f.get("claim", "")).lower() for f in facts if f.get("claim")]
    if not claims:
        return 0.0
    return float(1.0 - safe_div(len(set(claims)), len(claims)))


def semantic_redundancy_claims(used_facts: List[Dict], model: SentenceTransformer) -> float:
    claims = [f.get("claim","").strip() for f in used_facts if f.get("claim")]
    if len(claims) < 2:
        return 0.0

    emb = model.encode(claims, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb, emb).cpu().numpy()

    vals = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            vals.append(sim[i, j])
    return float(np.mean(vals)) if vals else 0.0

def claim_evidence_semantic_mean(used_facts: List[Dict], model: SentenceTransformer) -> float:
    pairs = [(f.get("claim",""), f.get("evidence","")) for f in used_facts if f.get("claim") and f.get("evidence")]
    if not pairs:
        return 0.0

    claims = [c for c, _ in pairs]
    evids  = [e for _, e in pairs]

    emb_c = model.encode(claims, convert_to_tensor=True, normalize_embeddings=True)
    emb_e = model.encode(evids,  convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(emb_c, emb_e).diagonal().cpu().numpy()
    return float(np.mean(sims))

def extract_cap_phrases(text: str) -> List[str]:
    out = []
    for m in CAP_PHRASE_RE.finditer(text or ""):
        ph = m.group(1).strip()
        if ph in STOP_CAP:
            continue
        out.append(ph)
    seen, uniq = set(), []
    for x in out:
        k = x.lower()
        if k not in seen:
            uniq.append(x)
            seen.add(k)
    return uniq

def hallucination_flags(script: str, allowed_text: str, filename_stem: str) -> Dict:
    a_norm = norm_ws(allowed_text).lower()

    years = sorted(set(YEAR_RE.findall(script or "")))
    new_years = [y for y in years if re.search(rf"\b{re.escape(y)}\b", a_norm) is None]

    dyns = [m.group(0) for m in DYNASTY_RE.finditer(script or "")]
    new_dyns = [d for d in dyns if re.search(rf"\b{re.escape(d.lower())}\b", a_norm) is None]

    caps = extract_cap_phrases(script or "")
    name_parts = set([p.lower() for p in re.split(r"\s+", norm_ws(filename_stem)) if p])

    new_caps = []
    for c in caps:
        cl = c.lower()
        if cl in a_norm:
            continue
        words = [w.lower() for w in c.split()]
        if words and all(w in name_parts for w in words):
            continue
        new_caps.append(c)

    risk = float(len(new_years) * 2 + len(new_dyns) * 2 + len(new_caps))
    return {
        "hallucination_risk": risk,
        "new_years": new_years,
        "new_dynasties": new_dyns,
        "new_cap_phrases": new_caps,
    }

def number_risk_flags(script: str, allowed_text: str) -> Dict:
    s = (script or "").lower()
    a = norm_ws(allowed_text).lower()

    script_nums = set(DIGIT_RE.findall(s))
    script_ord_suffix_tokens = set(ORDINAL_SUFFIX_RE.findall(s))
    script_ord_words = set([w for w in ORDINAL_WORDS if re.search(rf"\b{re.escape(w)}\b", s)])
    script_num_words = set([w for w in NUMBER_WORDS if re.search(rf"\b{re.escape(w)}\b", s)])

    tokens = []
    tokens.extend(sorted(script_nums))
    tokens.extend(sorted(script_ord_suffix_tokens))
    tokens.extend(sorted(script_ord_words))
    tokens.extend(sorted(script_num_words))

    new_tokens = []
    for t in tokens:
        if t.isdigit():
            if re.search(rf"\b{re.escape(t)}\b", a) is None:
                new_tokens.append(t)
        else:
            if re.search(rf"\b{re.escape(t)}\b", a) is None:
                new_tokens.append(t)

    seen, new_unique = set(), []
    for t in new_tokens:
        if t not in seen:
            new_unique.append(t)
            seen.add(t)

    risk = 0.0
    for t in new_unique:
        if t.isdigit() and int(t) >= 100:
            risk += 2.0
        else:
            risk += 1.0

    return {"new_numbers_risk": float(risk), "new_numbers": new_unique}

def structure_metrics(script: str) -> Dict:
    sents = split_sentences(script or "")
    n = len(sents)

    hook_words = count_words(sents[0]) if n >= 1 else 0
    hook_ok = int(n >= 1 and hook_words <= 25)

    context_ok = int(n >= 2)

    main_story = max(0, n - 3)
    main_story_ok = int(6 <= main_story <= 9)

    ending = (sents[-1].lower() if n >= 1 else "")
    ending_has_legacy = int(any(k in ending for k in LEGACY_KEYWORDS))

    total_sent_ok = int(9 <= n <= 12)

    structure_ok = int(total_sent_ok and main_story_ok and ending_has_legacy)
    structure_score = (hook_ok + context_ok + main_story_ok + ending_has_legacy + total_sent_ok) / 5.0

    return {
        "total_sentences": n,
        "hook_words": hook_words,
        "hook_ok": hook_ok,
        "context_ok": context_ok,
        "main_story_sentences": main_story,
        "main_story_ok": main_story_ok,
        "ending_has_legacy_keyword": ending_has_legacy,
        "total_sent_ok_9_12": total_sent_ok,
        "structure_ok": structure_ok,
        "structure_score": float(structure_score),
    }

def sentence_fact_alignment(script: str, facts: List[Dict], model: SentenceTransformer) -> Dict:
    sents = split_sentences(script or "")
    claims = [f["claim"] for f in facts if f.get("claim")]
    if not sents or not claims:
        return {"alignment_mean_max": 0.0, "coverage_ge_045": 0.0, "coverage_ge_055": 0.0}

    emb_s = model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    emb_c = model.encode(claims, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(emb_s, emb_c)
    max_per_sent = sims.max(dim=1).values.cpu().numpy()

    return {
        "alignment_mean_max": float(np.mean(max_per_sent)),
        "coverage_ge_045": float(np.mean(max_per_sent >= 0.45)),
        "coverage_ge_055": float(np.mean(max_per_sent >= 0.55)),
    }

def facts_used_coverage_rate(script: str, used_facts: List[Dict], model: SentenceTransformer, threshold: float = 0.55) -> float:
    if not used_facts:
        return 0.0
    sents = split_sentences(script or "")
    if not sents:
        return 0.0

    claims = [f["claim"] for f in used_facts if f.get("claim")]
    if not claims:
        return 0.0

    emb_s = model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
    emb_c = model.encode(claims, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(emb_c, emb_s)
    max_per_claim = sims.max(dim=1).values.cpu().numpy()
    return float(np.mean(max_per_claim >= threshold))


def stage1_score(row: Dict) -> float:
    evid = float(row["evidence_in_source_rate"])
    dup = float(row["duplicate_claim_rate_used"])
    k = float(row["used_facts_count"])

    ev_len_ok = float(row.get("evidence_len_ok_rate", 0.0))
    sem_red = float(row.get("claim_semantic_redundancy_mean", 0.0))

    ce = float(row.get("claim_evidence_semantic_mean", 0.0))
    ce = max(0.0, min(1.0, ce))

    k_norm = max(0.0, min(1.0, k / 10.0))

    dup_norm = max(0.0, min(1.0, dup / 0.4))
    dup_good = 1.0 - dup_norm

    sem_nonred = 1.0 - max(0.0, min(1.0, (sem_red - 0.55) / 0.30))  # penalize if mean sim > ~0.55

    return float(55.0 * evid + 10.0 * ev_len_ok + 10.0 * dup_good + 10.0 * k_norm +  10.0 * sem_nonred + 5* ce)

def stage2_score(row: Dict) -> float:
    wc = float(row["word_count_calc"])
    len_ok = 1.0 if (TARGET_MIN_WORDS <= wc <= TARGET_MAX_WORDS) else 0.0

    halluc = float(row["hallucination_risk"])
    numrisk = float(row["new_numbers_risk"])

    align = float(row["alignment_mean_max"])
    cov = float(row["facts_used_coverage_055"])
    struct = float(row["structure_score"])

    halluc_norm = math.exp(-0.35 * halluc)
    num_norm = math.exp(-0.45 * numrisk)

    align = max(0.0, min(1.0, align))
    cov = max(0.0, min(1.0, cov))
    struct = max(0.0, min(1.0, struct))

    return float(
        30.0 * halluc_norm +
        15.0 * num_norm +
        25.0 * align +
        15.0 * cov +
        10.0 * len_ok +
        5.0  * struct
    )

def final_score(stage1: float, stage2: float) -> float:
    return float(0.55 * stage1 + 0.45 * stage2)


def find_run_dirs(base: Path) -> List[Path]:
    run_dirs = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name in EXCLUDE_RUN_DIR_NAMES:
            continue
        if any(p.glob("*.txt")):
            run_dirs.append(p)
    return run_dirs


def main():
    print("Evaluate All Runs (Stage1 + Stage2 + Final)")
    print("Docs base :", DOCS_DIR.resolve())
    print("Runs base :", BASE_RUNS_DIR.resolve())
    print("Eval out  :", EVAL_DIR.resolve())
    print(f"Target words: {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}")

    if not DOCS_DIR.exists():
        raise SystemExit(f"DOCS_DIR not found: {DOCS_DIR.resolve()}")
    if not BASE_RUNS_DIR.exists():
        raise SystemExit(f"BASE_RUNS_DIR not found: {BASE_RUNS_DIR.resolve()}")

    run_dirs = find_run_dirs(BASE_RUNS_DIR)
    if not run_dirs:
        raise SystemExit(f"No run folders found under: {BASE_RUNS_DIR.resolve()}")

    print("Found run folders:")
    for rd in run_dirs:
        print(" -", rd.name)
    print()

    print("Loading embedding model:", EMBED_MODEL)
    emb_model = SentenceTransformer(EMBED_MODEL)

    all_rows = []
    leaderboard_rows = []
    all_problems = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        run_files = sorted(run_dir.glob("*.txt"))
        if not run_files:
            continue

        run_eval_dir = EVAL_DIR / run_name
        run_eval_dir.mkdir(parents=True, exist_ok=True)

        run_details = []
        run_problems = []

        for out_fp in run_files:
            parsed = parse_generated_output(out_fp)

            src_path = DOCS_DIR / out_fp.name
            if not src_path.exists():
                prob = {"run": run_name, "file": out_fp.name, "problem": "missing_source_doc"}
                run_problems.append(prob); all_problems.append(prob)
                continue

            src_text = src_path.read_text(encoding="utf-8", errors="ignore")

            script = parsed["script"]
            used_facts = parsed["used_facts"]

            if len(used_facts) == 0:
                prob = {"run": run_name, "file": out_fp.name, "problem": "no_used_facts_parsed"}
                run_problems.append(prob); all_problems.append(prob)
                continue

            allowed_text = " ".join([f"{u.get('claim','')} {u.get('evidence','')}" for u in used_facts])

  
            ev_rate, ev_ok, ev_total = evidence_in_source_rate(used_facts, src_text)
            ev_stats = evidence_length_stats(used_facts)
            dup_rate = duplicate_claim_rate(used_facts)
            ev_len_ok = evidence_len_ok_rate(used_facts)
            sem_red = semantic_redundancy_claims(used_facts, emb_model)
            ce_sem = claim_evidence_semantic_mean(used_facts, emb_model)


            wc_calc = count_words(script)
            rep = repetition_3gram(script)
            read = readability_metrics(script)
            align = sentence_fact_alignment(script, used_facts, emb_model)
            hall = hallucination_flags(script, allowed_text, filename_stem=out_fp.stem)
            facts_cov = facts_used_coverage_rate(script, used_facts, emb_model, threshold=0.55)
            num_flags = number_risk_flags(script, allowed_text)
            struct = structure_metrics(script)

            row = {
                "run": run_name,
                "file": out_fp.name,
                "output_path": str(out_fp),

                "word_count_reported": parsed["word_count_reported"],
                "doc_words_reported": parsed["doc_words_reported"],
                "facts_used_reported": parsed["facts_used_reported"],
                "facts_total_reported": parsed["facts_total_reported"],

         
                "used_facts_count": len(used_facts),
                "evidence_in_source_rate": ev_rate,
                "evidence_in_source_ok": ev_ok,
                "evidence_in_source_total": ev_total,
                "duplicate_claim_rate_used": dup_rate,
                "evidence_len_ok_rate": ev_len_ok,
                "claim_semantic_redundancy_mean": sem_red,
                "claim_evidence_semantic_mean": ce_sem,
                **ev_stats,


                "word_count_calc": wc_calc,
                "length_ok": int(TARGET_MIN_WORDS <= wc_calc <= TARGET_MAX_WORDS),
                "repetition_3gram": rep,
                **read,
                **align,
                "facts_used_coverage_055": facts_cov,

                "hallucination_risk": hall["hallucination_risk"],
                "new_years": ";".join(hall["new_years"]) if hall["new_years"] else "",
                "new_dynasties": ";".join(hall["new_dynasties"]) if hall["new_dynasties"] else "",
                "new_cap_phrases": ";".join(hall["new_cap_phrases"][:25]) if hall["new_cap_phrases"] else "",

                "new_numbers_risk": num_flags["new_numbers_risk"],
                "new_numbers": ";".join(num_flags["new_numbers"]) if num_flags["new_numbers"] else "",

                **struct,
            }

            row["stage1_score"] = stage1_score(row)
            row["stage2_score"] = stage2_score(row)
            row["final_score"] = final_score(row["stage1_score"], row["stage2_score"])

            run_details.append(row)

            if ev_rate < 0.8:
                prob = {"run": run_name, "file": out_fp.name, "problem": f"low_evidence_in_source_rate={ev_rate:.2f}"}
                run_problems.append(prob); all_problems.append(prob)
            if row["hallucination_risk"] > 0:
                prob = {"run": run_name, "file": out_fp.name, "problem": f"hallucination_risk={row['hallucination_risk']:.0f}"}
                run_problems.append(prob); all_problems.append(prob)
            if row["new_numbers_risk"] > 0:
                prob = {"run": run_name, "file": out_fp.name, "problem": f"new_numbers_risk={row['new_numbers_risk']:.0f}"}
                run_problems.append(prob); all_problems.append(prob)
            if row["length_ok"] == 0:
                prob = {"run": run_name, "file": out_fp.name, "problem": f"length_out_of_range wc={wc_calc}"}
                run_problems.append(prob); all_problems.append(prob)
            if facts_cov < 0.6 and len(used_facts) >= 6:
                prob = {"run": run_name, "file": out_fp.name, "problem": f"low_facts_coverage={facts_cov:.2f}"}
                run_problems.append(prob); all_problems.append(prob)
            if struct["structure_score"] < 0.6:
                prob = {"run": run_name, "file": out_fp.name, "problem": f"weak_structure_score={struct['structure_score']:.2f}"}
                run_problems.append(prob); all_problems.append(prob)

        run_df = pd.DataFrame(run_details)
        run_df.to_csv(run_eval_dir / "details.csv", index=False, encoding="utf-8")
        if run_problems:
            pd.DataFrame(run_problems).to_csv(run_eval_dir / "problem_cases.csv", index=False, encoding="utf-8")

        if not run_df.empty:
            summary = {
                "run": run_name,
                "n_files": int(len(run_df)),

                "mean_final_score": float(run_df["final_score"].mean()),
                "median_final_score": float(run_df["final_score"].median()),

                "mean_stage1_score": float(run_df["stage1_score"].mean()),
                "mean_stage2_score": float(run_df["stage2_score"].mean()),

                "length_ok_rate": float(run_df["length_ok"].mean()),
                "mean_evidence_len_ok_rate": float(run_df["evidence_len_ok_rate"].mean()),
                "mean_claim_semantic_redundancy_mean": float(run_df["claim_semantic_redundancy_mean"].mean()),
                "mean_claim_evidence_semantic_mean": float(run_df["claim_evidence_semantic_mean"].mean()),

                "mean_evidence_in_source_rate": float(run_df["evidence_in_source_rate"].mean()),
                "mean_hallucination_risk": float(run_df["hallucination_risk"].mean()),
                "mean_new_numbers_risk": float(run_df["new_numbers_risk"].mean()),
                "mean_alignment": float(run_df["alignment_mean_max"].mean()),
                "mean_facts_used_coverage_055": float(run_df["facts_used_coverage_055"].mean()),
                "mean_structure_score": float(run_df["structure_score"].mean()),
                "structure_ok_rate": float(run_df["structure_ok"].mean()),
            }
            leaderboard_rows.append(summary)

        all_rows.extend(run_details)
        print(f"Run evaluated: {run_name} | files: {len(run_details)}")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(MASTER_DETAILS_CSV, index=False, encoding="utf-8")

    lb_df = pd.DataFrame(leaderboard_rows)

    if not lb_df.empty:
        shared_cols = [
            "run", "n_files",

            "mean_stage1_score", "mean_stage2_score", "mean_final_score", "median_final_score",
   
            "mean_evidence_in_source_rate",
            "mean_evidence_len_ok_rate",
            "mean_claim_semantic_redundancy_mean",
            "mean_claim_evidence_semantic_mean",
      
            "mean_hallucination_risk", "mean_new_numbers_risk",
            "length_ok_rate",
            "mean_alignment", "mean_facts_used_coverage_055",
            "mean_structure_score", "structure_ok_rate",
        ]

 
        shared_cols = [c for c in shared_cols if c in lb_df.columns]
        base_lb = lb_df[shared_cols].copy()

        # Rank by Stage 1
        lb_stage1 = base_lb.sort_values("mean_stage1_score", ascending=False)
        lb_stage1.to_csv(MASTER_LEADERBOARD_STAGE1_CSV, index=False, encoding="utf-8")

        # Rank by Stage 2
        lb_stage2 = base_lb.sort_values("mean_stage2_score", ascending=False)
        lb_stage2.to_csv(MASTER_LEADERBOARD_STAGE2_CSV, index=False, encoding="utf-8")

        # Rank by Final
        lb_final = base_lb.sort_values("mean_final_score", ascending=False)
        lb_final.to_csv(MASTER_LEADERBOARD_FINAL_CSV, index=False, encoding="utf-8")

    else:
        pd.DataFrame().to_csv(MASTER_LEADERBOARD_STAGE1_CSV, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(MASTER_LEADERBOARD_STAGE2_CSV, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(MASTER_LEADERBOARD_FINAL_CSV, index=False, encoding="utf-8")


    if all_problems:
        pd.DataFrame(all_problems).to_csv(MASTER_PROBLEMS_CSV, index=False, encoding="utf-8")

    print("DONE")
    print("Master details         :", MASTER_DETAILS_CSV.resolve())
    print("Leaderboard (Stage 1)  :", MASTER_LEADERBOARD_STAGE1_CSV.resolve())
    print("Leaderboard (Stage 2)  :", MASTER_LEADERBOARD_STAGE2_CSV.resolve())
    print("Leaderboard (Final)    :", MASTER_LEADERBOARD_FINAL_CSV.resolve())
    if all_problems:
        print("All problem cases      :", MASTER_PROBLEMS_CSV.resolve())

    if not lb_df.empty:
        print("\nTop runs by Stage 1:")
        print(lb_stage1.head(10).to_string(index=False))

        print("\nTop runs by Stage 2:")
        print(lb_stage2.head(10).to_string(index=False))

        print("\nTop runs by Final:")
        print(lb_final.head(10).to_string(index=False))

if __name__ == "__main__":
    main()