# ECHO Chatbot Module — Comprehensive Documentation

> **Project:** ECHO – AI-Powered Ancient Egypt Explorer  
> **Module:** Conversational Historical Chatbot  
> **Type:** Graduation Project Documentation  
> **Final Version:** Phase 6 (Production)

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Objectives](#2-objectives)
3. [Data Pipeline](#3-data-pipeline)
4. [Development Phases](#4-development-phases)
5. [System Architecture — Final (Phase 6)](#5-system-architecture--final-phase-6)
6. [Embedding Model Selection & Evaluation](#6-embedding-model-selection--evaluation)
7. [Retrieval System](#7-retrieval-system)
8. [Reranking](#8-reranking)
9. [Query Rewriting](#9-query-rewriting)
10. [Generation & Persona Prompting](#10-generation--persona-prompting)
11. [Tool-Augmented Generation (Web Search Fallback)](#11-tool-augmented-generation-web-search-fallback)
12. [Conversational Memory](#12-conversational-memory)
13. [Speech Pipeline (STT & TTS)](#13-speech-pipeline-stt--tts)
14. [Multilingual Support](#14-multilingual-support)
15. [API Design](#15-api-design)
16. [Database Schema](#16-database-schema)
17. [Dockerization & Deployment](#17-dockerization--deployment)
18. [Evaluation & Results](#18-evaluation--results)
19. [Frameworks & Libraries](#19-frameworks--libraries)
20. [Challenges & Lessons Learned](#20-challenges--lessons-learned)
21. [Future Work](#21-future-work)

---

## 1. Module Overview

The ECHO Chatbot Module is a persona-based Retrieval-Augmented Generation (RAG) system that enables users to converse with ancient Egyptian pharaohs and landmarks as if speaking to the historical entity itself. The chatbot combines dense vector retrieval, cross-encoder reranking, structured prompt engineering, tool-augmented generation, conversational memory, and a full voice pipeline to deliver an educational, immersive, and historically accurate conversational experience.

The user selects an entity type (`pharaoh` or `landmark`) and a specific entity name (e.g., "Ramesses II" or "Temple of Karnak"). The chatbot then answers in the first-person perspective of that entity, grounding all responses strictly in retrieved historical context stored in a PostgreSQL vector database.

---

## 2. Objectives

1. **Historical Accuracy** — Generate factually correct answers grounded in verified historical documents; minimize hallucination.
2. **Immersive Persona** — The chatbot speaks as the selected pharaoh or landmark in first person, making the experience educational and engaging.
3. **Retrieval Quality** — Use dense vector retrieval combined with cross-encoder reranking to surface the most relevant context chunks.
4. **Multilingual Interaction** — Respond in the same language the user uses (English, Arabic, French, German, Italian, Spanish, Portuguese).
5. **Voice Interaction** — Support speech-to-text (STT) input and text-to-speech (TTS) output with language-aware and gender-aware voice selection.
6. **Modularity** — Build the system as a LangGraph state machine where each stage (rewrite → retrieve → rerank → generate → tools → TTS) is an independent, testable node.
7. **Production Readiness** — Package the chatbot as a Dockerized FastAPI service with GPU support, model preloading, and connection pooling.

---

## 3. Data Pipeline

### 3.1 Raw Document Collection

Historical documents were collected and curated for two entity categories:

| Category   | Document Count | Storage Path                        |
|------------|---------------|--------------------------------------|
| Pharaohs   | 80            | `data/chatbot/raw/pharaohs_docs/`    |
| Landmarks  | 52            | `data/chatbot/raw/landmarks_docs/`   |
| **Total**  | **132**       |                                      |

Each document is a plain-text `.txt` file containing detailed historical information about a single entity. Pharaoh documents cover entities ranging from Old Kingdom rulers (Khufu, Djoser, Sneferu) through Ptolemaic rulers (Cleopatra VII, Ptolemy I Soter) and also include Egyptian gods (Amun, Osiris, Isis, Horus, Anubis). Landmark documents cover pyramids, temples, tombs, and monuments across Egypt.

Document sizes vary significantly—some files are under 500 bytes (e.g., `Ra-Horakhty (God).txt` at 220 bytes), while others exceed 20 KB (e.g., `Shepsekaf.txt` at 25,734 bytes; `Raneferef.txt` at 21,774 bytes). This variance was a key factor in embedding model selection.

### 3.2 Text Chunking

Documents were split into text chunks suitable for embedding and retrieval. The chunking process was handled during the embedding creation phase using the notebook at `experiments/chatbot/create_documents_embeddings/create_embeddings.ipynb`.

### 3.3 Embedding Generation

Each text chunk was embedded using the **Qwen/Qwen3-Embedding-0.6B** model. Embeddings were generated at the model's native 1024 dimensions, then truncated to **768 dimensions** using Matryoshka Representation Learning (MRL). After truncation, embeddings were L2-normalized.

The embedding pipeline:
1. Load text chunk
2. Encode with `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")`
3. Normalize embeddings (`normalize_embeddings=True`)
4. Truncate to 768 dimensions
5. Re-normalize after truncation
6. Store in PostgreSQL via pgvector

### 3.4 Storage in PostgreSQL (pgvector)

Embeddings are stored in PostgreSQL using the `pgvector` extension. Two text tables exist:

- `pharaohs_texts` — stores pharaoh text chunks with 768-dimensional embeddings
- `landmarks_texts` — stores landmark text chunks with 768-dimensional embeddings

Each table uses an **HNSW index** for fast approximate nearest-neighbor search:

```sql
-- Index configuration (both tables)
CREATE INDEX hnsw_idx_pharaoh_text_embedding
ON pharaohs_texts USING hnsw (text_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 3.5 Evaluation Dataset Generation

A synthetic evaluation dataset was generated for benchmarking:

- `synthetic_dataset.csv` — 132 evaluation entries (one per entity)
- `shrunk_dataset_132.csv` — condensed version used for final evaluation runs
- The dataset was split into two parts (`eval_part_1.csv`, `eval_part_2.csv`) for parallel evaluation

The test case creation process is documented in `experiments/chatbot/echo_chatbot/evaluation_scripts/test_cases/create_test_dataset.ipynb`.

---

## 4. Development Phases

The chatbot was developed iteratively through **6 progressive phases**, each adding a new capability. All phase scripts are preserved in `experiments/chatbot/echo_chatbot/chatbot_phases/`.

### Phase 1 — Basic RAG (`phase1.py`)

The foundational phase implementing a minimal Retrieve → Generate pipeline.

- **Graph:** `retriever → generator → END`
- **Embedding:** Local `Qwen/Qwen3-Embedding-0.6B` on CUDA
- **LLM:** `Qwen/Qwen3-32B` via Groq (single model for everything)
- **Retrieval:** Hardcoded to "Ramesses II" only; entity-specific vector search via pgvector
- **History:** Full conversation history passed to generator (unfiltered)
- **Limitations:** No query rewriting, no reranking, single entity only, no persona separation between rewriter and generator

### Phase 2 — Query Rewriting + Dual LLM (`phase2.py`)

Added a dedicated query rewriter node and separated the LLMs for rewriting vs. generation.

- **Graph:** `rewriter → retriever → generator → END`
- **Embedding:** Switched to **Cloudflare Workers AI** (`@cf/qwen/qwen3-embedding-0.6b`) — the embedding model was hosted remotely via Cloudflare's Workers AI API
- **Query Rewriter LLM:** `Qwen/Qwen3-32B` via Groq (temperature=0.2, reasoning hidden)
- **Generator LLM:** `openai/gpt-oss-120b` via Groq (temperature=0.8, reasoning medium)
- **Key Change:** Introduced separate conversation history windows — the rewriter sees `HumanMessage + search_query` pairs, while the generator sees `HumanMessage + generator_response` pairs. This prevents the generator from being confused by internal search queries.
- **Token Streaming:** Generator output is streamed token-by-token to the console

### Phase 3 — Reranking (`phase3.py`)

Added a cross-encoder reranking step between retrieval and generation.

- **Graph:** `rewriter → retriever → reranker → generator → END`
- **Reranker:** `jina-reranker-v3` via Jina API (top_n=3)
- **Embedding:** Still Cloudflare Workers AI
- **Key Change:** Retrieved top-10 chunks are compressed to top-3 by the Jina cross-encoder reranker, significantly improving the relevance of context passed to the generator

### Phase 4 — Tool-Augmented Generation (Agentic RAG) (`phase 4.py`)

The generator became an agent with access to web search as a fallback tool.

- **Graph:** `rewriter → retriever → reranker → generator ⇄ tools → END`
- **Tool:** `TavilySearch` (max_results=5, search_depth="advanced")
- **Embedding:** Still Cloudflare Workers AI
- **Key Change:** The generator LLM is bound to tools via `.bind_tools(tools)`. If retrieved context is insufficient, the model can invoke Tavily web search exactly once. A circuit breaker prevents double-searching: if the model has already searched (detected by the presence of `ToolMessage` in current turn), it is instructed not to search again.
- **Out-of-Scope Handling:** If the rewriter marks a query as `OUT_OF_SCOPE`, the generator returns a graceful refusal in persona voice

### Phase 5 — Text-to-Speech with Inworld AI (`phase5.py`)

Added voice output using Inworld AI's TTS API.

- **Graph:** `rewriter → retriever → reranker → generator ⇄ tools → tts_gate → tts → END`
- **TTS Provider:** Inworld AI (`inworld-tts-1.5-mini` model with a custom "ancient_egyptian_pharaoh" voice)
- **TTS Gate:** Uses LangGraph's `interrupt()` to ask the user whether to generate speech before the TTS node runs
- **Embedding:** Still Cloudflare Workers AI
- **Note:** This phase was later superseded by Phase 6 which switched to Edge TTS for better multilingual support and lower latency

### Phase 6 — Final Production System (`phase6.py` → `src/chatbot_api/runtime.py`)

The fully-featured final system incorporating all improvements and switching critical infrastructure.

- **Graph:** `rewriter → retriever → reranker → generator ⇄ tools → tts_router → tts → END`

**Critical infrastructure changes in Phase 6:**

1. **Embedding Model Migration:** Switched from **Cloudflare Workers AI** (remote API) to **local Qwen/Qwen3-Embedding-0.6B** via `sentence-transformers`. The Cloudflare Workers AI embedding endpoint introduced significant latency (~200-400ms per call). Running the model locally on GPU reduced embedding latency dramatically and eliminated external API dependency for the core retrieval path.

2. **TTS Migration:** Switched from Inworld AI TTS to **Edge TTS** (`edge-tts` library). Edge TTS provides automatic language detection and gender-aware voice selection across 7 languages.

3. **STT Integration:** Added speech-to-text via **Groq Whisper** (`whisper-large-v3`), supporting voice input from microphone.

4. **Background Model Loading:** The embedding model is loaded in a background thread at startup to avoid blocking the main application.

5. **Dynamic Entity Resolution:** Supports any pharaoh or landmark in the database — no longer hardcoded to a single entity.

6. **User Memory System:** The rewriter detects and stores personal information (name, age, language preference, interests) as key-value pairs in session memory.

7. **Entity-Specific Prompts:** Separate prompt templates for pharaohs vs. landmarks, with different persona voices and contextual framing.

8. **Production Refactoring:** The Phase 6 experiment script was refactored into a clean class-based architecture in `src/chatbot_api/runtime.py` (`EchoChatbotRuntime`) for production deployment.

---

## 5. System Architecture — Final (Phase 6)

### 5.1 High-Level Pipeline

```
User Input (text or voice)
    │
    ▼
┌─────────────────┐
│  STT (Whisper)   │  ← voice input only
└────────┬────────┘
         ▼
┌─────────────────┐
│  Query Rewriter  │  Qwen3-32B via Groq
│  + Memory Detect │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Retriever       │  Qwen3-Embedding-0.6B (local GPU) → pgvector Top-10
└────────┬────────┘
         ▼
┌─────────────────┐
│  Reranker        │  Jina Reranker v3 → Top-3
└────────┬────────┘
         ▼
┌─────────────────┐
│  Generator       │  GPT-OSS-120B via Groq (persona prompt)
│  + Tool Decision │
└────────┬────────┘
        ╱ ╲
       ╱   ╲
  No Tool   Tool Call
      │         │
      │    ┌────┴────┐
      │    │ Tavily   │  Web Search (max 4 results)
      │    │ Search   │
      │    └────┬────┘
      │         │
      │    ┌────┴────┐
      │    │Generator │  Second pass (no more tools)
      │    │(retry)   │
      │    └────┬────┘
      ▼         ▼
┌─────────────────┐
│  TTS (Edge TTS)  │  ← voice mode only
└────────┬────────┘
         ▼
    Response (text + optional audio)
```

### 5.2 LangGraph State Machine

The chatbot is implemented as a LangGraph `StateGraph` with the following state definition:

```python
class AgentState(TypedDict):
    session_id: str
    query: str              # Original user query
    search_query: str       # Rewritten search query
    messages: Annotated[list, add_messages]  # Full message history
    context: list[str]      # Retrieved text chunks
    response: str           # Final generated response
```

**Graph Nodes:**
- `rewriter` — Rewrites queries, detects memory, marks out-of-scope
- `retriever` — Embeds query and performs pgvector similarity search
- `reranker` — Cross-encoder reranking with Jina
- `generator` — Persona-based response generation with optional tool calls
- `tools` — LangGraph `ToolNode` wrapping Tavily search

**Graph Edges:**
```
rewriter → retriever → reranker → generator
generator → tools (conditional: if tool call)
generator → END (conditional: if no tool call)
tools → generator (loop back for second generation pass)
```

### 5.3 Session Management

The production runtime (`EchoChatbotRuntime`) maintains an in-memory session store:

```python
self.sessions: dict[str, dict[str, object]] = {}
```

Each session tracks:
- `entity_type` — "pharaoh" or "landmark"
- `entity_name` — e.g., "Ramesses II"
- `entity_id` — Database primary key (resolved once at session start)
- `gender` — For pharaohs (used for TTS voice selection)
- `user_memory` — Key-value pairs extracted from conversation
- `messages` — Full conversation history

Sessions are initialized on first request and reset when the entity changes.

---

## 6. Embedding Model Selection & Evaluation

### 6.1 Initial Approach: Cloudflare Workers AI

In Phases 2–5, embeddings were generated using **Cloudflare Workers AI**, hosting the `@cf/qwen/qwen3-embedding-0.6b` model remotely:

```python
from langchain_cloudflare import CloudflareWorkersAIEmbeddings

embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)
```

**Problems with this approach:**
- **Latency:** Each embedding API call added ~200-400ms of network round-trip, significantly impacting the retrieval step
- **Reliability:** Dependent on Cloudflare API availability
- **Cost:** API calls incur usage costs at scale

### 6.2 Migration to Local Inference

In Phase 6, the embedding model was switched to **local GPU inference** using `sentence-transformers`:

```python
from sentence_transformers import SentenceTransformer

qwen_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cuda",
    tokenizer_kwargs={"padding_side": "left"},
    token=HF_TOKEN
)
```

This eliminated network latency entirely and made embedding generation a sub-100ms local operation.

### 6.3 Model Comparison: Qwen vs. mxbai

Two embedding models were formally evaluated using a custom benchmark of 340+ query–document pairs across both pharaohs and landmarks:

| Metric     | Qwen3-Embedding-0.6B | mxbai-embed-large |
|------------|----------------------|-------------------|
| Recall@1   | 0.8127               | 0.8210            |
| Recall@5   | 0.9156               | 0.9213            |
| MRR        | 0.8535               | 0.8598            |
| NDCG@5     | 0.8691               | 0.8752            |

**mxbai was marginally better by ~1-2% across all metrics**, but **Qwen was chosen** for the following reasons:

1. **Superior Context Handling:** Qwen supports a context window of **32,768 tokens** vs. mxbai's **512 tokens** (standard BERT window). Since some landmark documents exceed 512 tokens, mxbai would truncate critical historical details.

2. **Matryoshka Representation Learning (MRL):** Qwen was explicitly trained with MRL, allowing dimensional truncation from 1024 → 768 → 512 while retaining ~95% retrieval performance. This provides storage efficiency without retraining.

3. **MTEB Leaderboard Position:** Qwen3-Embedding consistently ranks in the **Top 10** on the MTEB multilingual leaderboard, significantly higher than mxbai (#61).

4. **Generative LLM Backbone:** Built on Qwen2.5, it inherits deeper reasoning and multilingual capabilities than traditional BERT-based models.

### 6.4 MRL Dimensional Truncation Evaluation

Three embedding dimensions were evaluated:

| Dimension | Recall@1 | Recall@5 | MRR    | NDCG@5 | Avg Search Time |
|-----------|----------|----------|--------|--------|-----------------|
| 1024      | 0.8127   | 0.9156   | 0.8535 | 0.8691 | 5.84ms          |
| **768**   | **0.8035** | **0.9184** | **0.8475** | **0.8652** | **4.72ms** |
| 512       | 0.7766   | 0.9002   | 0.8254 | 0.8442 | 3.77ms          |

**768 dimensions was selected** as the optimal trade-off:
- Only ~1% quality loss vs. full 1024 dimensions
- **20% faster search time** than 1024 dimensions
- Significantly better than 512 across all metrics

---

## 7. Retrieval System

### 7.1 Vector Search Query

Retrieval is performed using a parameterized SQL query with pgvector's cosine distance operator (`<=>`):

```sql
SELECT text_chunk
FROM {texts_table}
WHERE {entity_id_col} = :entity_id
ORDER BY text_embedding <=> :embedding
LIMIT 10;
```

Key design decisions:
- **Entity-scoped retrieval:** Search is constrained to chunks belonging to the selected entity only. This eliminates cross-entity noise (e.g., retrieving Khufu chunks when talking to Ramesses II).
- **Top-10 retrieval:** 10 chunks are retrieved by the bi-encoder, then compressed to 3 by the reranker.
- **HNSW indexing:** Approximate nearest-neighbor search with `m=16, ef_construction=64` for fast sub-10ms queries.

### 7.2 Embedding at Query Time

The user's rewritten query is embedded at runtime:

```python
def get_embedding(self, text_value: str) -> list[float]:
    embeddings = self.qwen_model.encode(
        text_value,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    sliced = embeddings[:768]  # MRL truncation
    norm = np.linalg.norm(sliced)
    return (sliced / norm if norm > 0 else sliced).tolist()
```

---

## 8. Reranking

### 8.1 Jina Reranker v3

After bi-encoder retrieval returns 10 candidates, a cross-encoder reranker compresses them to the top 3:

```python
reranker = JinaRerank(
    model="jina-reranker-v3",
    top_n=3,
    jina_api_key=JINA_API_KEY,
)
```

### 8.2 Impact of Reranking (Quantified)

A dedicated evaluation measured the reranker's effect:

| Metric          | Without Reranker | With Reranker | Improvement |
|-----------------|-----------------|---------------|-------------|
| Recall@1        | 0.7045          | 0.8712        | **+23.6%**  |
| MRR             | 0.8106          | 0.9268        | **+14.3%**  |
| NDCG@3          | 0.8438          | 0.9419        | **+11.6%**  |
| Recall@3        | 0.9394          | 0.9848        | **+4.8%**   |

The reranker's most significant impact is on **Recall@1 (+23.6%)**, meaning the most relevant chunk is now ranked first in 87.12% of cases (vs. 70.45% without reranking). This is critical because the generator relies most heavily on the top-ranked context.

The cross-attention mechanism of the Jina cross-encoder is superior to simple cosine similarity for this domain—it successfully "rescues" relevant historical context from lower-ranked positions and prioritizes it for the generator.

---

## 9. Query Rewriting

### 9.1 Purpose

User queries are typically conversational and first-person (e.g., "Tell me about your father"). Vector retrieval works better with explicit, factual, third-person queries (e.g., "Who was the biological father of Ramesses II?"). The query rewriter bridges this gap.

### 9.2 Rewriter LLM

- **Model:** `Qwen/Qwen3-32B` via Groq
- **Temperature:** 0.2 (low for deterministic rewrites)
- **Max Tokens:** 1024
- **Reasoning:** Hidden (no chain-of-thought exposed to user)

### 9.3 Rewriting Capabilities

The rewriter prompt handles multiple tasks:

1. **Scope Classification:**
   - **(A) Conversational/Meta** — queries about the user themselves or conversation memory → handled normally
   - **(B) Domain** — queries about the entity or ancient Egyptian history → rewritten
   - **(C) Out-of-Domain** — modern/unrelated topics → marked with `OUT_OF_SCOPE`

2. **Subject Resolution:** Pronouns ("he", "she", "it", "that") are resolved using the last 10 conversation turns.

3. **Language Translation:** Non-English queries are translated to English for retrieval (the generator still responds in the user's language).

4. **Memory Detection:** If the user states personal information ("My name is Jacob", "I'm 18"), the rewriter extracts it as `[MEMORY]: name=Jacob, age=18`.

### 9.4 Separate History Windows

The rewriter and generator see **different conversation history windows:**

- **Rewriter:** Sees `HumanMessage` + `search_query` pairs (last 10 turns) — optimized for reference resolution
- **Generator:** Sees `HumanMessage` + `generator_response` pairs (last 5 turns / 10 messages) — optimized for conversational coherence

---

## 10. Generation & Persona Prompting

### 10.1 Generator LLM

- **Model:** `openai/gpt-oss-120b` via Groq
- **Temperature:** 0.8 (creative enough for persona voice)
- **Top-p:** 0.95
- **Max Tokens:** 4,096
- **Reasoning:** Medium effort, hidden format

### 10.2 Persona Prompt Structure

Two separate persona templates exist (pharaoh and landmark). The pharaoh template:

```yaml
# THE SOVEREIGN IDENTITY
You are {pharaoh_name}, speaking from your legacy in ancient Egypt.

# MULTILINGUAL MANDATE
- Respond in the SAME LANGUAGE as the User's Query.

# GUIDELINES
- First Person: Speak in 1st person perspective
- Do not be a "helpful assistant." Be {pharaoh_name} sharing memories.
- Answer Scope: Only derive your answer from the provided context ONLY.

# YOUR MEMORIES (Context)
{context}

# DIALOGUE HISTORY
{chat_history}

# USER INFORMATION
{user_info}

# SEARCH TOOL
- If your MEMORIES do not contain the answer, use the search tool.
- ONLY ONE search attempt per user question.
```

### 10.3 Out-of-Scope Handling

When the rewriter marks a query as `OUT_OF_SCOPE`, the generator returns a graceful in-character refusal:

- **Pharaoh:** "I'm sorry but you speak of a time that is not mine. My eyes see only the borders of my own reign."
- **Landmark:** "I'm sorry, that lies beyond what my stones remember."

### 10.4 Streaming

The production API streams responses token-by-token via Server-Sent Events (SSE):

```python
@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chatbot_service.stream_chat(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

## 11. Tool-Augmented Generation (Web Search Fallback)

### 11.1 Tavily Search Integration

The generator LLM is bound to a Tavily web search tool:

```python
search_tool = TavilySearch(max_results=4, search_depth="advanced")
generator_llm = ChatGroq(...).bind_tools([search_tool])
```

### 11.2 Circuit Breaker Pattern

To prevent infinite search loops:

1. The generator decides whether to call the search tool based on context sufficiency
2. If it calls the tool, the search results are merged with the retrieved database context
3. The generator is invoked a **second time** with an explicit instruction: "You have already consulted the modern scrolls. Do not call the search tool again."
4. If the answer is still not found: "The gods have veiled that specific moment from my sight for now."

---

## 12. Conversational Memory

### 12.1 Memory Extraction

The query rewriter detects explicit user statements and extracts them as key-value pairs:

```
User: My name is Jacob and I'm 18, please speak Arabic
→ [MEMORY]: name=Jacob, age=18, language=Arabic
```

**Critical rule:** Memory is only extracted from declarative statements, never from questions. "What's my name?" does not trigger memory extraction.

### 12.2 Memory Storage

Memory is stored per-session as a list of `key=value` strings. Duplicate keys are overwritten (e.g., if the user corrects their name). Memory is included in the generator prompt under `# USER INFORMATION`.

---

## 13. Speech Pipeline (STT & TTS)

### 13.1 Speech-to-Text (STT)

- **Model:** Groq Whisper (`whisper-large-v3`)
- **Input:** Audio file uploaded via `/voice/transcribe` endpoint
- **Temperature:** 0 (deterministic transcription)

```python
transcription = self.groq_client.audio.transcriptions.create(
    file=(filename, audio_bytes),
    model="whisper-large-v3",
    temperature=0,
)
```

### 13.2 Text-to-Speech (TTS)

- **Engine:** Edge TTS (`edge-tts` library)
- **Rate:** +9%
- **Pitch:** -10Hz

Voice selection is automatic based on detected language and entity gender:

| Language | Male Voice               | Female Voice              |
|----------|--------------------------|---------------------------|
| English  | en-CA-LiamNeural         | en-US-JennyNeural         |
| Arabic   | ar-EG-ShakirNeural       | ar-BH-LailaNeural         |
| French   | fr-FR-HenriNeural        | fr-FR-DeniseNeural        |
| Spanish  | es-ES-AlvaroNeural       | es-ES-ElviraNeural        |
| German   | de-DE-ConradNeural       | de-DE-KatjaNeural         |
| Italian  | it-IT-DiegoNeural        | it-IT-ElsaNeural          |
| Portuguese | pt-BR-AntonioNeural    | pt-BR-FranciscaNeural     |

**Default voice** (fallback): `en-US-ChristopherNeural`

For pharaohs, the gender is retrieved from the database (`pharaohs.gender` column). For landmarks, male voice is always used.

---

## 14. Multilingual Support

The chatbot supports multilingual interaction through multiple mechanisms:

1. **Query Rewriting:** Non-English queries are translated to English for retrieval (English retrieval corpus)
2. **Response Generation:** The persona prompt mandates responding in the same language as the user's query
3. **Language Detection:** `langdetect` is used to identify the response language for TTS voice selection
4. **TTS Voices:** 7 languages supported with gender-appropriate neural voices

---

## 15. API Design

### 15.1 FastAPI Application

The chatbot is served as a FastAPI application (`src/chatbot_api/app.py`):

```python
app = FastAPI(title="ECHO Chatbot API", version="0.1.0")
```

### 15.2 Endpoints

| Method | Path               | Description                                      | Request Body         |
|--------|--------------------|-------------------------------------------------|----------------------|
| GET    | `/health`          | Health check                                     | —                    |
| POST   | `/chat`            | Send message, receive streamed response (SSE)   | `ChatRequest`        |
| POST   | `/voice/transcribe`| Upload audio file, receive transcription         | `audio: UploadFile`  |
| POST   | `/voice/speak`     | Convert text to speech audio                     | `SpeechRequest`      |

### 15.3 Request/Response Schemas

```python
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    entity_type: str = Field(..., pattern="^(pharaoh|landmark)$")
    entity_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1)
    entity_type: str | None = Field(default=None, pattern="^(pharaoh|landmark)$")
    entity_name: str | None = None
```

### 15.4 Thread Safety

The `ChatbotService` wraps all runtime calls with a `threading.Lock` to ensure thread-safe access to the shared embedding model and session state:

```python
class ChatbotService:
    def __init__(self):
        self._lock = Lock()

    def stream_chat(self, request: ChatRequest):
        def event_stream():
            with self._lock:
                yield from chatbot_runtime.stream_chat(...)
        return event_stream()
```

### 15.5 Model Preloading

On startup, the embedding model is loaded and warmed up with a realistic query to eliminate cold-start latency:

```python
@app.on_event("startup")
def preload_models():
    chatbot_runtime.warmup_embedding()
```

Warmup text: `"Tell me about Ramesses II, his reign, military campaigns, monuments, and legacy in ancient Egypt."`

---

## 16. Database Schema

### 16.1 Entity Tables

```
pharaohs
├── id (PK)
├── name (String, indexed)
├── dynasty (String, nullable)
├── type (String, nullable)
├── description (String, nullable)
├── period (String, nullable)
├── composite_entity (String, nullable)
└── gender (String, nullable)

landmarks
├── id (PK)
├── name (String, indexed)
├── description (String, nullable)
└── location (String, nullable)
```

### 16.2 Text Embedding Tables

```
pharaohs_texts
├── id (PK)
├── pharaoh_id (FK → pharaohs.id)
├── text_chunk (Text)
└── text_embedding (Vector(768), HNSW indexed)

landmarks_texts
├── id (PK)
├── landmark_id (FK → landmarks.id)
├── text_chunk (Text)
└── text_embedding (Vector(768), HNSW indexed)
```

### 16.3 Database Connection

PostgreSQL with connection pooling via SQLAlchemy:

```python
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

---

## 17. Dockerization & Deployment

### 17.1 Dockerfile (`Dockerfile.chatbot`)

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG HF_TOKEN

WORKDIR /app

COPY requirements.chatbot.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.chatbot.txt

# Pre-download and cache the embedding model during build
RUN test -n "$HF_TOKEN"
RUN HF_TOKEN="$HF_TOKEN" python -c "from os import getenv; \
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cpu', \
    token=getenv('HF_TOKEN'), tokenizer_kwargs={'padding_side': 'left'})"

COPY src/chatbot_api src/chatbot_api
COPY src/models src/models
COPY src/db src/db
COPY experiments/chatbot/echo_chatbot/resources experiments/chatbot/echo_chatbot/resources

EXPOSE 8000

CMD ["uvicorn", "src.chatbot_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key design decisions:**
- The Qwen embedding model is **downloaded during the Docker build** (not at runtime), ensuring fast container startup
- `HF_TOKEN` is required as a build argument to authenticate with Hugging Face
- Only the necessary source code directories are copied (not the entire repo)
- The `resources/` directory (prompts.yaml, queries.sql) is included for runtime access

### 17.2 Docker Compose (`docker-compose.yml`)

```yaml
services:
  chatbot:
    build:
      context: .
      dockerfile: Dockerfile.chatbot
      args:
        HF_TOKEN: "${HF_TOKEN}"
    container_name: "${CHATBOT_CONTAINER_NAME:-echo-chatbot-api}"
    environment:
      APP_ENV: "production"
      DATABASE_URL: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@host.docker.internal:5432/${POSTGRES_DB}"
      NVIDIA_DRIVER_CAPABILITIES: "compute,video,utility"
      HF_TOKEN: "${HF_TOKEN}"
      GROQ_API_KEY1: "${GROQ_API_KEY1}"
      GROQ_API_KEY2: "${GROQ_API_KEY2}"
      JINA_API_KEY: "${JINA_API_KEY}"
      TAVILY_API_KEY: "${TAVILY_API_KEY}"
      HF_HOME: /root/.cache/huggingface
    volumes:
      - hf_cache_chatbot:/root/.cache/huggingface
    ports:
      - "8000:8000"
    gpus: all
```

**Key features:**
- **GPU passthrough:** `gpus: all` enables NVIDIA GPU access for the embedding model
- **HuggingFace cache volume:** `hf_cache_chatbot` persists downloaded models across container restarts
- **Host networking for DB:** Uses `host.docker.internal` to connect to PostgreSQL running on the host
- **Two Groq API keys:** Separate keys for the rewriter and generator to avoid rate limiting

### 17.3 Environment Variables

| Variable           | Purpose                                    |
|--------------------|--------------------------------------------|
| `DATABASE_URL`     | PostgreSQL connection string               |
| `HF_TOKEN`         | Hugging Face authentication token          |
| `GROQ_API_KEY1`    | Groq API key for query rewriter            |
| `GROQ_API_KEY2`    | Groq API key for generator                 |
| `JINA_API_KEY`     | Jina reranker API key                      |
| `TAVILY_API_KEY`   | Tavily web search API key                  |
| `R2_ACCOUNT_ID`    | Cloudflare R2 account (legacy/data sync)   |
| `CF_AI_API`        | Cloudflare AI API key (legacy embedding)   |

---

## 18. Evaluation & Results

### 18.1 Evaluation Framework

The chatbot was evaluated using the **RAGAS framework** (Retrieval-Augmented Generation Assessment), measuring:

- **Faithfulness** — Is the answer supported by the retrieved context?
- **Answer Relevancy** — Is the answer relevant to the user's question?
- **Context Precision** — Are the retrieved contexts relevant?
- **Context Recall** — Does the retrieved context cover the ground truth?
- **Answer Accuracy** — How correct is the final answer?

### 18.2 Final ECHO Agent Results (Triple-Trial)

Three independent evaluation runs were conducted to account for LLM non-determinism:

| Metric              | Run 1   | Run 2   | Run 3   | **Mean ± Std**         |
|---------------------|---------|---------|---------|------------------------|
| Faithfulness        | 0.961   | 0.977   | 0.963   | **0.9670 ± 0.0087**   |
| Answer Relevancy    | 0.775   | 0.768   | 0.767   | **0.7700 ± 0.0044**   |
| Context Precision   | 0.935   | 0.935   | 0.937   | **0.9357 ± 0.0011**   |
| Context Recall      | 0.977   | 0.999   | 0.992   | **0.9893 ± 0.0112**   |
| **Answer Accuracy** |         |         |         | **0.9745**             |
| Hallucination Rate  |         |         |         | **0.0330 ± 0.0087**   |

**Key findings:**

- **Context Precision (σ = 0.0011)** — The lowest variance across all trials. The Jina reranker provides near-deterministic ranking logic, confirming that retrieval quality is a consistent structural feature, not luck.
- **Faithfulness (μ = 0.9670)** — Elite-level grounding. The GPT-120B generator strictly adheres to retrieved context. The low variance proves the persona does not cause hallucination drift.
- **Answer Relevancy (μ = 0.7700)** — The lowest mean but highest "tightness" (σ = 0.0044). This is not an error but a consistent stylistic effect: the RAGAS evaluator penalizes persona metaphors and ancient greetings as "non-relevant" to the raw query.
- **Context Recall (μ = 0.9893)** — Near saturation (1.0), indicating the retrieval pipeline captures ground-truth information in 98.9% of cases.

### 18.3 Generator Model Comparison (LLM-Only, No RAG)

Three LLMs were tested without the RAG pipeline to isolate model quality:

| Model           | Answer Relevancy | Answer Accuracy | Confabulation |
|-----------------|------------------|-----------------|---------------|
| GPT-OSS-120B   | 0.719            | 0.254           | 0.746         |
| Qwen3-32B      | 0.750            | 0.242           | 0.758         |
| Llama 3 70B    | 0.736            | 0.356           | 0.644         |

**vs. Final ECHO Agent with RAG:**
| System          | Answer Accuracy | Confabulation Reduction |
|-----------------|-----------------|--------------------------|
| ECHO Agent      | **0.9745**      | **96.6% grounded**       |

The RAG pipeline improves answer accuracy by **~3.8×** over the best standalone LLM, proving that retrieval quality and grounding matter far more than model size alone.

### 18.4 Baseline RAG vs. Final System

Two baseline RAG evaluations (without query rewriting, reranking, or tool augmentation):

| Metric              | Baseline Run 1 | Baseline Run 2 | **Final Agent** |
|---------------------|----------------|----------------|-----------------|
| Faithfulness        | 0.973          | 0.942          | **0.967**       |
| Answer Relevancy    | 0.758          | 0.738          | **0.770**       |
| Context Precision   | 0.862          | 0.899          | **0.936**       |
| Context Recall      | 0.985          | 1.000          | **0.989**       |
| Answer Accuracy     | 0.863          | 0.894          | **0.975**       |

The final system improves over baseline primarily through **better context precision (+8.5%)** and **answer accuracy (+10%)**, driven by query rewriting and reranking.

### 18.5 Efficiency Results

Measured across all 132 evaluation queries (100% success rate):

**Time to First Token (TTFT):**

| Percentile | Latency |
|------------|---------|
| P50        | 1.14s   |
| P95        | 1.37s   |
| P99        | 1.75s   |
| Mean       | 1.17s   |
| Min        | 0.96s   |
| Max        | 3.19s   |

**End-to-End Latency:**

| Percentile | Latency |
|------------|---------|
| P50        | 1.86s   |
| P95        | 2.65s   |
| P99        | 4.09s   |
| Mean       | 1.98s   |
| Max        | 7.45s   |

**Component Latency Breakdown:**

| Component  | Avg Time | % of Total |
|------------|----------|------------|
| Rewriter   | 0.411s   | 21.0%      |
| Retriever  | 0.358s   | 18.3%      |
| Reranker   | 0.251s   | 12.8%      |
| Generator  | 0.938s   | 47.9%      |

**Token Usage:**

| LLM              | Total Tokens | Avg/Query | % of Total |
|-------------------|-------------|-----------|------------|
| Generator (120B) | 150,664     | 1,141     | 62.9%      |
| Rewriter (32B)   | 88,868      | 673       | 37.1%      |
| **Combined**     | **239,532** |           |            |

- Generator throughput: **281.1 tokens/sec**
- Success rate: **132/132 (100%)**

---

## 19. Frameworks & Libraries

### 19.1 Core Framework Stack

| Component         | Technology                  | Purpose                              |
|-------------------|-----------------------------|--------------------------------------|
| Web Framework     | FastAPI + Uvicorn           | REST API server                      |
| State Machine     | LangGraph                  | Agentic workflow orchestration       |
| LLM Interface     | LangChain-Groq             | Groq-hosted LLM integration         |
| Prompt Engine     | LangChain-Core             | Prompt templates, output parsers     |
| Web Search        | LangChain-Tavily           | Tavily search tool integration       |
| Reranker          | LangChain-Community (Jina) | Jina reranker v3 integration         |
| Embeddings        | sentence-transformers      | Local Qwen embedding model           |
| DB ORM            | SQLAlchemy                  | Database models, queries, sessions   |
| Vector Extension  | pgvector                    | PostgreSQL vector similarity search  |
| Migrations        | Alembic                     | Database schema migrations           |
| Validation        | Pydantic                    | Request/response schema validation   |

### 19.2 AI/ML Models

| Role              | Model                       | Provider      | Parameters |
|-------------------|-----------------------------|---------------|------------|
| Embedding         | Qwen/Qwen3-Embedding-0.6B  | Local (GPU)   | 0.6B       |
| Query Rewriter    | Qwen/Qwen3-32B             | Groq Cloud    | 32B        |
| Generator         | openai/gpt-oss-120b        | Groq Cloud    | 120B       |
| STT               | Whisper Large v3            | Groq Cloud    | 1.5B       |
| Reranker          | jina-reranker-v3            | Jina Cloud    | —          |
| TTS               | Edge TTS (Neural voices)    | Microsoft Edge| —          |

### 19.3 Infrastructure

| Component         | Technology                  |
|-------------------|-----------------------------|
| Database          | PostgreSQL + pgvector       |
| Containerization  | Docker + Docker Compose     |
| GPU Runtime       | NVIDIA CUDA                 |
| Config            | python-dotenv, PyYAML       |
| Evaluation        | RAGAS framework             |

---

## 20. Challenges & Lessons Learned

### Challenge 1: Cloudflare Workers AI → Local Embedding

**Problem:** In Phases 2–5, embeddings were generated via Cloudflare Workers AI. Each API call added 200–400ms of network latency to every retrieval operation. This made the system noticeably slower and introduced an external dependency.

**Solution:** In Phase 6, the embedding model was migrated to local GPU inference using `sentence-transformers`. This eliminated network latency entirely, reduced embedding time to sub-100ms, and removed the Cloudflare API dependency from the critical path.

**Lesson:** For real-time interactive systems, local inference on available GPU hardware is vastly superior to cloud API calls for lightweight models like embedding encoders.

### Challenge 2: Persona vs. Evaluation Metrics

**Problem:** The persona-based speaking style (pharaoh metaphors, ancient greetings, first-person historical framing) consistently lowered the Answer Relevancy score in RAGAS evaluation, even when the answers were factually correct and grounded.

**Solution:** Accepted the trade-off. The lower relevancy score (0.77) is not an error but a consistent stylistic choice confirmed by the lowest standard deviation (σ = 0.0044) across all metrics. The persona is essential to the project's educational and experiential goals.

**Lesson:** Standard evaluation metrics may not fully capture the value of domain-specific design choices. Contextual interpretation of metrics is necessary.

### Challenge 3: Retrieval Ranking Noise

**Problem:** Dense bi-encoder retrieval achieved good recall (Recall@3 = 0.94) but poor ranking precision (Recall@1 = 0.70). Relevant chunks were present in the top-10 but not ranked first, leading to suboptimal context for the generator.

**Solution:** Added Jina Reranker v3 as a cross-encoder reranking step, improving Recall@1 by 23.6%.

**Lesson:** Bi-encoder retrieval provides good recall but is insufficient for precision-critical applications. Cross-encoder reranking is essential for RAG systems.

### Challenge 4: Long Historical Documents

**Problem:** Some historical documents exceed 20KB (e.g., Shepsekaf at 25,734 bytes). Short-context embedding models like mxbai (512 tokens) would truncate critical information.

**Solution:** Selected Qwen3-Embedding-0.6B with its 32K token context window, ensuring complete document understanding without truncation.

### Challenge 5: Out-of-Domain Queries

**Problem:** Users ask unrelated modern questions that the chatbot should not attempt to answer in persona.

**Solution:** The query rewriter classifies queries into Domain, Conversational, and Out-of-Domain categories. Out-of-domain queries are marked with `OUT_OF_SCOPE` and receive a graceful in-character refusal.

### Challenge 6: Dual LLM API Rate Limiting

**Problem:** Using a single Groq API key for both the rewriter and generator could trigger rate limits under load.

**Solution:** Two separate Groq API keys (`GROQ_API_KEY1` for rewriter, `GROQ_API_KEY2` for generator) distribute the load across independent rate limit pools.

### Challenge 7: Docker Image Size & Startup Time

**Problem:** The Qwen embedding model needs to be available at container startup. Downloading it at runtime would add significant startup latency.

**Solution:** The model is downloaded during the Docker build phase (`RUN ... SentenceTransformer(...)` in the Dockerfile). A persistent volume (`hf_cache_chatbot`) ensures the model persists across container restarts.

### Challenge 8: Thread Safety

**Problem:** The embedding model and session state are shared resources accessed by concurrent API requests.

**Solution:** A `threading.Lock` in `ChatbotService` ensures all runtime operations are serialized, preventing race conditions on the GPU-bound embedding model.

---

## 21. Future Work

1. **Arabic-first RAG corpus** — Add a native Arabic document collection for direct Arabic retrieval
2. **Streaming TTS** — Stream TTS audio in real-time as text is generated
3. **Long-term memory** — Persist user memory across sessions with database-backed storage
4. **Fine-tuned embedding model** — Fine-tune Qwen on Egyptian historical domain for better retrieval
5. **Hybrid retrieval** — Combine dense vector search with BM25 keyword search
6. **GPU-optimized reranking** — Replace the Jina API call with a local cross-encoder model
7. **Evaluation automation** — CI/CD pipeline for automated RAGAS evaluation on every code change
8. **Multi-turn tool use** — Allow multiple search attempts for complex multi-part questions

---

## Appendix A: File Structure

```
ECHO/
├── src/
│   ├── chatbot_api/
│   │   ├── __init__.py          # Re-exports app
│   │   ├── main.py              # Entry point
│   │   ├── app.py               # FastAPI app, routes, startup
│   │   ├── runtime.py           # EchoChatbotRuntime class (695 lines)
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── service.py           # Thread-safe service layer
│   ├── models/
│   │   ├── pharaohs.py          # Pharaoh SQLAlchemy model
│   │   ├── pharaohs_text.py     # PharaohText model (vector column)
│   │   ├── landmarks.py         # Landmark SQLAlchemy model
│   │   └── landmarks_text.py    # LandmarkText model (vector column)
│   └── db/
│       └── session.py           # SQLAlchemy engine, session, pool config
├── experiments/
│   └── chatbot/
│       ├── echo_chatbot/
│       │   ├── chatbot_phases/
│       │   │   ├── phase1.py    # Basic RAG
│       │   │   ├── phase2.py    # + Query rewriting + Cloudflare embeddings
│       │   │   ├── phase3.py    # + Jina reranking
│       │   │   ├── phase 4.py   # + Tavily tool-augmented generation
│       │   │   ├── phase5.py    # + Inworld TTS
│       │   │   └── phase6.py    # + Edge TTS + local embeddings + memory
│       │   ├── evaluation_scripts/
│       │   │   ├── agents_llm_evaluation/    # RAGAS evaluation scripts
│       │   │   ├── evaluation_graphs/        # LangGraph evaluation graphs
│       │   │   ├── reranker_evaluation/      # Reranker impact evaluation
│       │   │   ├── responses/                # Response generation scripts
│       │   │   └── test_cases/               # Synthetic dataset creation
│       │   └── resources/
│       │       ├── prompts.yaml              # All prompt templates
│       │       ├── queries.sql               # Vector search SQL template
│       │       └── evaluation_prompt_baseline.yaml
│       └── create_documents_embeddings/
│           ├── create_embeddings.ipynb        # Embedding generation notebook
│           ├── evaluate_models.ipynb          # Qwen vs mxbai evaluation
│           └── evaluate_qwen_mrl.py           # MRL dimension evaluation
├── data/
│   └── chatbot/
│       ├── raw/
│       │   ├── pharaohs_docs/   # 80 pharaoh text files
│       │   └── landmarks_docs/  # 52 landmark text files
│       ├── embeddings/          # ChromaDB embedding stores (6 variants)
│       └── outputs/
│           ├── echo_agent_evaluation/
│           │   ├── evaluation_data/     # Evaluation datasets (132 cases)
│           │   ├── final_results/       # RAG Triad results + reranker effect
│           │   ├── ragas_evaluation_results/
│           │   └── responses/           # Generated responses per model
│           ├── efficiency_evaluation_results/
│           │   ├── results.txt          # TTFT, E2E, component latency
│           │   └── langsmith_analysis/  # Visualizations + CSV metrics
│           └── embedding_model_evaluation/
│               ├── Qwen vs Mxbai.txt
│               ├── Qwen MRL evaluation.txt
│               └── Justification.txt
├── Dockerfile.chatbot
├── docker-compose.yml
├── requirements.chatbot.txt
└── .env.example
```

---

## Appendix B: Running the Chatbot

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.chatbot.txt

# 2. Set environment variables (copy and fill .env.example)
cp .env.example .env

# 3. Run database migrations
alembic upgrade head

# 4. Start the server
uvicorn src.chatbot_api.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Build and start
docker compose up --build

# The chatbot API will be available at http://localhost:8000
# Health check: GET http://localhost:8000/health
```

### Example API Call

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-1",
    "entity_type": "pharaoh",
    "entity_name": "Ramesses II",
    "message": "Tell me about the Battle of Kadesh"
  }'
```
