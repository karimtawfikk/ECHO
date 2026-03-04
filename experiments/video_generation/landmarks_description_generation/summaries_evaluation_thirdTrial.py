import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
import textstat
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# ======================
# CONFIG
# ======================
DOCS_DIR = Path("docs")
OUTPUTS_DIR = Path("outputs")

TOP_K_KEYWORDS = 30
NGRAM_N = 3

# compression "sweet spot" (tune if you want)
COMP_TARGET_LOW = 0.20
COMP_TARGET_HIGH = 0.40


# NLI / factual consistency
NLI_MODEL_ID = "facebook/bart-large-mnli"
NLI_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NLI_SENTENCES = 6  # evaluate up to first N summary sentences (None = all)
MIN_SENT_WORDS = 4     # ignore very short sentences

# Retrieval-based premise building
RETRIEVAL_TOP_K = 8              # how many doc sentences to retrieve per summary sentence
RETRIEVAL_MAX_DOC_SENTS = 350    # cap doc sentences for speed (None = no cap)
RETRIEVAL_PREMISE_MAX_WORDS = 300  # keep premise short for NLI

# MNLI label mapping for BART:
# 0 = contradiction, 1 = neutral, 2 = entailment
CONTRADICTION_ID = 0
NEUTRAL_ID = 1
ENTAILMENT_ID = 2


# Weights for final score (tweak)
# readability is computed but not used in final score by default
WEIGHTS = {
    "factual_consistency": 0.28,   # entailment rate
    "contradiction_rate_inv": 0.10, # 1 - contradiction rate (higher better)
    "semantic_sim": 0.24,
    "entity_recall": 0.16,
    "entity_precision": 0.10,
    "keyword_recall": 0.08,
    "compression_score": 0.02,
    "redundancy_score": 0.02,
}


# ======================
# LOAD MODELS
# ======================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

print("Loading NLI model:", NLI_MODEL_ID)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(NLI_DEVICE)
nli_model.eval()
if NLI_DEVICE == "cuda":
    nli_model = nli_model.half()   # optional: lower VRAM + faster

# ======================
# HELPERS
# ======================
def wc(text: str) -> int:
    return len(text.split())

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_entities(text: str) -> set[str]:
    doc = nlp(text)
    return set(normalize_text(ent.text) for ent in doc.ents if ent.text.strip())

def semantic_similarity(orig: str, summ: str) -> float:
    emb = embedder.encode([orig, summ])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def compression_ratio(orig: str, summ: str) -> float:
    return wc(summ) / max(wc(orig), 1)

def compression_sweetspot_score(r: float, low=COMP_TARGET_LOW, high=COMP_TARGET_HIGH) -> float:
    """
    Score in [0,1], best if r in [low, high].
    Penalize being too short or too long.
    """
    if low <= r <= high:
        return 1.0
    if r < low:
        # linearly drop to 0 at r=0
        return max(0.0, r / low)
    # r > high: drop as it gets longer
    # 0 at r = 2*high (adjust if you want)
    return max(0.0, 1.0 - (r - high) / high)

def keyword_set_tfidf(text: str, top_k: int = TOP_K_KEYWORDS) -> set[str]:
    """
    Extract top TF-IDF terms from a single document by fitting on its sentences.
    """
    # split into pseudo-docs (sentences/lines)
    parts = [p.strip() for p in re.split(r"[.\n]+", text) if p.strip()]
    if len(parts) < 2:
        parts = [text]

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=3000,
    )
    X = vec.fit_transform(parts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())

    if vocab.size == 0:
        return set()

    top_idx = np.argsort(scores)[::-1][:top_k]
    return set(normalize_text(w) for w in vocab[top_idx])

def keyword_recall(orig_keywords: set[str], summ_text: str) -> float:
    if not orig_keywords:
        return 1.0
    s = normalize_text(summ_text)
    hit = sum(1 for kw in orig_keywords if kw in s)
    return hit / len(orig_keywords)

def entity_scores(orig_ents: set[str], summ_ents: set[str]) -> tuple[float, float]:
    """
    Returns (recall, precision)
    recall: how many original entities appear in summary
    precision: how many summary entities appear in original (hallucination proxy)
    """
    if not orig_ents:
        return (1.0, 1.0)

    inter = orig_ents & summ_ents
    recall = len(inter) / len(orig_ents)

    if not summ_ents:
        precision = 0.0
    else:
        precision = len(inter) / len(summ_ents)

    return recall, precision

def redundancy_ngram_score(text: str, n: int = NGRAM_N) -> float:
    """
    Score in [0,1], higher is better (less repetition).
    Compute repeated n-gram ratio.
    """
    tokens = normalize_text(text).split()
    if len(tokens) < n * 2:
        return 1.0
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    # repetition ratio = 1 - unique/total ; convert to 'goodness' by 1 - repetition
    repetition = 1.0 - (unique / max(total, 1))
    return float(1.0 - repetition)

def readability_score(text: str) -> float:
    """
    Convert Flesch Reading Ease to [0,1] roughly.
    Higher = easier to read.
    """
    try:
        flesch = textstat.flesch_reading_ease(text)
        # typical range ~[-50, 120]; map to [0,1]
        return float(np.clip((flesch + 50) / 170, 0, 1))
    except Exception:
        return 0.5
    
def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter, filters out very short sentences.
    """
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    sents = [s for s in sents if wc(s) >= MIN_SENT_WORDS]
    if MAX_NLI_SENTENCES is not None:
        sents = sents[:MAX_NLI_SENTENCES]
    return sents

def take_first_n_sents(sents: list[str], n: int | None) -> list[str]:
    if n is None:
        return sents
    return sents[:n]

def truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def build_doc_sentence_bank(orig: str) -> list[str]:
    doc_sents = split_sentences(orig)
    if RETRIEVAL_MAX_DOC_SENTS is not None:
        doc_sents = doc_sents[:RETRIEVAL_MAX_DOC_SENTS]
    return doc_sents

def retrieval_premise_for_hypothesis(doc_sents: list[str], hyp_sent: str) -> str:
    """
    Retrieve top-K most similar doc sentences to the hypothesis sentence using embeddings,
    then join them into a short premise for NLI.
    """
    if not doc_sents:
        return ""

    embs = embedder.encode([hyp_sent] + doc_sents)
    h = embs[0:1]
    D = embs[1:]
    sims = cosine_similarity(h, D)[0]

    topk = min(RETRIEVAL_TOP_K, len(doc_sents))
    idx = np.argsort(sims)[::-1][:topk]

    premise = " ".join([doc_sents[i] for i in idx])
    premise = truncate_to_words(premise, RETRIEVAL_PREMISE_MAX_WORDS)
    return premise

def nli_rates_retrieval(orig: str, summ: str) -> tuple[float, float, float]:
    """
    Retrieval-based NLI:
      For each summary sentence (hypothesis), build a premise from top-K relevant doc sentences,
      then run MNLI.

    Returns (entailment_rate, contradiction_rate, neutral_rate).
    """
    hyp_sents = take_first_n_sents(split_sentences(summ), MAX_NLI_SENTENCES)
    if not hyp_sents:
        return 0.0, 0.0, 0.0

    doc_sents = build_doc_sentence_bank(orig)
    premises = [retrieval_premise_for_hypothesis(doc_sents, h) for h in hyp_sents]

    # fallback if retrieval yields empty premise
    premises = [p if p.strip() else truncate_to_words(orig, RETRIEVAL_PREMISE_MAX_WORDS) for p in premises]

    inputs = nli_tokenizer(
        premises,
        hyp_sents,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )
    inputs = {k: v.to(NLI_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = nli_model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    ent = float((preds == ENTAILMENT_ID).mean())
    con = float((preds == CONTRADICTION_ID).mean())
    neu = float((preds == NEUTRAL_ID).mean())
    return ent, con, neu


def weighted_score(row: dict) -> float:
    return sum(row[k] * w for k, w in WEIGHTS.items())


# ======================
# MAIN EVAL
# ======================
def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing docs dir: {DOCS_DIR.resolve()}")
    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Missing outputs dir: {OUTPUTS_DIR.resolve()}")

    model_dirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    docs = sorted([p for p in DOCS_DIR.glob("*.txt")])

    print("Docs:", DOCS_DIR.resolve())
    print("Outputs:", OUTPUTS_DIR.resolve())
    print(f"Found {len(docs)} docs and {len(model_dirs)} model folders")

    if not docs or not model_dirs:
        return

    rows = []
    per_doc_rows = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nEvaluating: {model_name}")

        metrics = {
            "compression": [],
            "compression_score": [],
            "semantic_sim": [],
            "entity_recall": [],
            "entity_precision": [],
            "keyword_recall": [],
            "redundancy_score": [],
            "readability": [],
            "factual_consistency": [],       # entailment rate
            "contradiction_rate": [],
            "neutral_rate": [],
            "contradiction_rate_inv": [],    # 1 - contradiction_rate
        }

        for doc_path in tqdm(docs):
            summ_path = model_dir / doc_path.name
            if not summ_path.exists():
                continue

            orig = doc_path.read_text(encoding="utf-8", errors="ignore").strip()
            summ = summ_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not orig or not summ:
                continue

            comp = compression_ratio(orig, summ)
            comp_s = compression_sweetspot_score(comp)

            sim = semantic_similarity(orig, summ)

            orig_ents = get_entities(orig)
            summ_ents = get_entities(summ)
            ent_r, ent_p = entity_scores(orig_ents, summ_ents)

            orig_kw = keyword_set_tfidf(orig, TOP_K_KEYWORDS)
            kw_r = keyword_recall(orig_kw, summ)

            red = redundancy_ngram_score(summ, NGRAM_N)
            read = readability_score(summ)

            ent_rate, con_rate, neu_rate = nli_rates_retrieval(orig, summ)
            con_inv = 1.0 - con_rate

            metrics["compression"].append(comp)
            metrics["compression_score"].append(comp_s)
            metrics["semantic_sim"].append(sim)
            metrics["entity_recall"].append(ent_r)
            metrics["entity_precision"].append(ent_p)
            metrics["keyword_recall"].append(kw_r)
            metrics["redundancy_score"].append(red)
            metrics["readability"].append(read)

            metrics["factual_consistency"].append(ent_rate)
            metrics["contradiction_rate"].append(con_rate)
            metrics["neutral_rate"].append(neu_rate)
            metrics["contradiction_rate_inv"].append(con_inv)

            per_doc_rows.append({
                "model": model_name,
                "doc": doc_path.name,
                "compression": comp,
                "compression_score": comp_s,
                "semantic_sim": sim,
                "entity_recall": ent_r,
                "entity_precision": ent_p,
                "keyword_recall": kw_r,
                "redundancy_score": red,
                "readability": read,
                "factual_consistency": ent_rate,
                "contradiction_rate": con_rate,
                "neutral_rate": neu_rate,
                "contradiction_rate_inv": con_inv,
            })

        if not metrics["semantic_sim"]:
            continue

        agg = {
            "model": model_name,
            **{k: float(np.mean(v)) for k, v in metrics.items()},
        }
        agg["final_score"] = weighted_score(agg)
        rows.append(agg)

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    print("\n===== MODEL RANKING =====")
    print(df[["model", "final_score",
              "factual_consistency", "contradiction_rate", "neutral_rate",
              "semantic_sim", "entity_recall", "entity_precision",
              "keyword_recall", "compression", "redundancy_score", "readability"]])

    df.to_csv("model_ranking_thirdTrial.csv", index=False)

    df_doc = pd.DataFrame(per_doc_rows)
    df_doc.to_csv("per_doc_metrics_thirdTrial.csv", index=False)

    print("\nSaved: model_ranking_thirdTrial.csv, per_doc_metrics_thirdTrial.csv")


if __name__ == "__main__":
    main()
