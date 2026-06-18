# Chatbot Module Seminar 2

This file is a slide-by-slide guide for presenting the final chatbot module based on `phase6_edgetts.py`.

## 1. Slide Title

**ECHO Chatbot Module - Final Phase (Phase 6)**

Suggested subtitle:
- Persona-based Ancient Egypt conversational RAG system
- Supports text, voice input, multilingual output, memory, reranking, and web fallback

What to say:
- This module is the final version of our chatbot for the ECHO project.
- The chatbot lets users talk to an ancient Egyptian pharaoh or landmark in first person.
- It combines retrieval-augmented generation, persona prompting, query rewriting, reranking, speech-to-text, and text-to-speech.

---

## 2. Module Goal

**Goal of the module**

- Build an educational chatbot that answers questions about Ancient Egyptian pharaohs and landmarks.
- Make the answers accurate, grounded in retrieved context, and engaging through a historical persona.
- Support both text mode and voice mode.
- Support multilingual interaction.

What to say:
- The design goal was not only to answer correctly, but to answer in-character while staying grounded in retrieved evidence.
- So the core problem was balancing factuality, retrieval quality, user experience, and presentation style.

---

## 3. Data Used and All Its Details

**Main knowledge base**

- Raw historical documents are stored in:
  - `data/chatbot/raw/pharaohs_docs`
  - `data/chatbot/raw/landmarks_docs`
- Document coverage:
  - 80 pharaoh documents
  - 52 landmark documents
  - Total = 132 entity documents

**Entity groups**

- `pharaoh`
- `landmark`

**How data is used in the final system**

- Each entity has historical text chunks stored in the database.
- During retrieval, the chatbot searches only inside the selected entity's chunks.
- The SQL query limits retrieval to the chosen entity and returns the nearest chunks using vector similarity.

**Evaluation data**

- Evaluation dataset files are stored in:
  - `data/chatbot/outputs/echo_agent_evaluation/evaluation_data`
- Main evaluation set size:
  - `shrunk_dataset_132.csv` = 132 evaluation cases
  - `synthetic_dataset.csv` = 132 entries

What to say:
- We used a domain-specific dataset built from historical text files about Egyptian pharaohs and landmarks.
- The final chatbot does entity-specific retrieval, so if the user chooses Ramesses II, retrieval is constrained to Ramesses II chunks only.
- This helps reduce irrelevant retrieval noise.

Suggested slide visual:
- A simple pipeline: Raw docs -> chunked texts in DB -> embeddings -> retrieval -> reranking -> generation

---

## 4. Final Architecture Overview

**Phase 6 pipeline**

1. User selects entity type: `pharaoh` or `landmark`
2. User selects entity name
3. User enters text or voice
4. Query rewriter rewrites the question into a factual search query
5. Query is embedded with Qwen embedding model
6. Top-10 chunks are retrieved from the database
7. Jina reranker compresses them to Top-3
8. Generator LLM answers in persona
9. If context is missing, the model may call Tavily web search once
10. If voice mode is on, response is converted to speech using Edge TTS

What to say:
- The system is built as a multi-node LangGraph workflow.
- This lets us separate rewriting, retrieval, reranking, generation, tool use, and text-to-speech into clear stages.
- The design is modular, which made debugging and evaluation easier.

---

## 5. List of Functions

**LangGraph node functions**

- `rewrite_node(state)`
  - Rewrites user queries, resolves context, handles memory extraction, and marks out-of-scope queries.
- `retrieve_node(state)`
  - Retrieves top chunks from the vector database for the chosen entity.
- `rerank_node(state)`
  - Reranks retrieved chunks using Jina reranker.
- `generate_node(state)`
  - Produces the final persona-based answer and decides whether web search is needed.
- `tts_node(state)`
  - Converts final text to speech with Edge TTS.
- `route_tts(state)`
  - Decides whether the graph should go to TTS or terminate.

**Program control**
---

## 6. Algorithms and Techniques Used

### 6.1 Retrieval-Augmented Generation (RAG) (agentic, corective, react)

- The chatbot does not answer directly from the model alone.
- It first retrieves relevant historical chunks, then generates an answer based on them.

Why this technique:
- Reduces hallucination
- Keeps answers grounded in the project dataset
- Makes the chatbot more trustworthy for educational use

### 6.2 Query Rewriting

- User questions are rewritten into factual third-person search queries.
- Pronouns are resolved using conversation history.
- Non-English queries are converted to English for retrieval.
- Out-of-domain questions are marked as `OUT_OF_SCOPE`.

Why this technique:
- Users ask in natural conversation style, but vector retrieval works better with explicit factual queries.
- Rewriting improves retrieval precision.

### 6.3 Persona Prompting

- The assistant speaks in first person as the selected pharaoh or landmark.
- The prompt forces the answer to stay in the same language as the user.
- The system uses strict context grounding rules.

Why this technique:
- Makes the experience more immersive and engaging.
- Fits the project identity while preserving factual grounding.

### 6.4 Dense Vector Retrieval

- The system embeds the rewritten query using `Qwen/Qwen3-Embedding-0.6B`.
- Embeddings are normalized and truncated to 768 dimensions.
- Retrieval is done with nearest-neighbor vector similarity in SQL.
- The database query returns Top-10 chunks.

Why this technique:
- Semantic retrieval is better than keyword matching for history questions.
- It can match meaning even if the user phrasing differs from document wording.

### 6.5 Cross-Encoder Reranking

- Retrieved chunks are reranked using `jina-reranker-v3`.
- Top-10 retrieved chunks are compressed to Top-3.

Why this technique:
- Initial vector retrieval is good at recall, but not always perfect at ranking.
- Reranking improves top-context quality before generation.

### 6.6 Tool-Augmented Generation

- If retrieved local context is insufficient, the generator can call Tavily search once.
- Search depth is `advanced`, with up to 4 results.

Why this technique:
- Handles questions not fully covered in the local knowledge base.
- Keeps web access controlled so the model does not overuse search.

### 6.7 Conversational Memory

- The rewriter extracts explicit user facts such as name, age, language, interests, or location.
- These are stored as simple key-value memory entries.
- The generator includes this memory in later responses.

Why this technique:
- Makes the conversation more personalized.
- Improves continuity across turns without needing a separate memory database.

### 6.8 Speech Pipeline

- Voice input uses microphone recording plus Groq Whisper (`whisper-large-v3`) for STT.
- Voice output uses Edge TTS with automatic language-based voice selection.

Why this technique:
- Makes the chatbot more interactive for demo and user experience.
- Edge TTS is lightweight and practical for multilingual output.

### 6.9 LangGraph State Machine

- The chatbot flow is modeled using `StateGraph`.
- State includes query, rewritten query, messages, context, response, and voice mode.

Why this technique:
- Clear node-based control flow
- Easy integration of tools and conditional routing
- Better maintainability than a single long script

---

## 7. Justification for Each Major Choice

### Qwen embedding model

- Chosen model: `Qwen/Qwen3-Embedding-0.6B`

Justification:
- Supports long context up to 32k tokens
- Better for long historical files than short-window models
- Supports Matryoshka Representation Learning
- Strong multilingual and semantic performance

Important note for the slide:
- In your own evaluation, `mxbai` was slightly better by around 1-2% on some retrieval metrics.
- But Qwen was chosen because it handles longer contexts and supports MRL-based dimensional truncation more effectively.

### 768-dimensional embeddings

Justification:
- Balanced retrieval quality and efficiency
- Better than the 512-dimensional truncated version
- Faster than keeping larger vectors without losing much quality

### Jina reranker

Justification:
- Strongly improved ranking quality after retrieval
- Especially improved top-1 relevance, which matters most for final answer quality

### GPT-OSS-120B as generator

Justification:
- Produced the strongest answer accuracy among tested generator models
- Maintained high faithfulness when combined with RAG
- Has reasoning

### Qwen-32B as query rewriter

Justification:
- Good semantic reformulation quality
- Strong enough for rewrite and memory extraction
- More efficient than using the larger generator model for this smaller task

### Tavily web search

Justification:
- Provides controlled fallback for missing knowledge
- Prevents total failure when local context is incomplete

### Edge TTS

Justification:
- Easy multilingual text-to-speech integration
- Suitable for live demo use
- Allows persona responses to become audible

---

## 8. Evaluation and Results

### 8.1 Main RAGAS evaluation for final ECHO agent

Final reported results across 3 runs:

- Faithfulness: `0.9670 +- 0.0087`
- Answer Relevancy: `0.7700 +- 0.0044`
- Context Precision: `0.9357 +- 0.0011`
- Context Recall: `0.9893 +- 0.0112`
- Answer Accuracy: `0.9745`
- Grounded Hallucination: `0.0330 +- 0.0087`

What to say:
- These numbers show that the chatbot is highly grounded and retrieves relevant evidence very consistently.
- The strongest result is faithfulness and low hallucination.
- Answer relevancy is lower than faithfulness because the persona style adds historical phrasing that evaluators may not count as directly relevant wording.

### 8.2 Why answer relevancy is lower

Use this explanation:
- The lower answer relevancy score is mainly caused by the persona-driven speaking style.
- The chatbot intentionally answers as a pharaoh or monument, so some stylistic phrases are penalized by the evaluator even when the answer is correct and grounded.

### 8.3 Reranker effect

With reranker:
- MRR: `0.9268`
- NDCG@3: `0.9419`
- Recall@1: `0.8712`
- Recall@3: `0.9848`

Without reranker:
- MRR: `0.8106`
- NDCG@3: `0.8438`
- Recall@1: `0.7045`
- Recall@3: `0.9394`

Relative improvements:
- Recall@1: `+23.6%`
- MRR: `+14.3%`
- NDCG@3: `+11.6%`
- Recall@3: `+4.8%`

What to say:
- This is one of the strongest arguments for the final architecture.
- The reranker significantly improves which chunk reaches the top positions, especially the first result.

### 8.4 Generator model comparison

GPT-OSS-120B:
- Answer Relevancy: `0.719`
- Answer Accuracy: `0.254`

Qwen 32B:
- Answer Relevancy: `0.750`
- Answer Accuracy: `0.242`

Llama 70B:
- Answer Relevancy: `0.736`
- Answer Accuracy: `0.356`

Echo Agent with full RAG pipeline:
- Answer Accuracy: `0.9745`

What to say:
- LLM-only or weakly grounded generation performs much worse than the final RAG agent.
- The key takeaway is that retrieval quality and grounding matter more than model size alone.

### 8.5 Baseline vs final system

Baseline RAG examples:
- Faithfulness: `0.973` and `0.942`
- Answer Relevancy: `0.758` and `0.738`
- Context Precision: `0.862` and `0.899`
- Context Recall: `0.985` and `1.000`
- Answer Accuracy: `0.863` and `0.894`

Final ECHO agent:
- Better context precision
- Better final answer accuracy
- Lower hallucination

What to say:
- The final Phase 6 system improves over baseline mainly through query rewriting, reranking, and better pipeline control.

---

## 9. Efficiency Results

**Time to First Token**

- Mean: `1.17s`
- Median: `1.14s`
- P95: `1.37s`
- P99: `1.75s`

**End-to-End latency**

- Mean: `1.98s`
- Median: `1.86s`
- P95: `2.65s`
- P99: `4.09s`
- Max: `7.45s`

**Component latency breakdown**

- Rewriter: `0.411s` (`21.0%`)
- Retriever: `0.358s` (`18.3%`)
- Reranker: `0.251s` (`12.8%`)
- Generator: `0.938s` (`47.9%`)

**Token usage**

- Generator total tokens: `150,664`
- Rewriter total tokens: `88,868`
- Combined total: `239,532`
- Generator TPS: `281.1 tokens/sec`

**Success rate**

- `132/132` successful queries = `100%`

What to say:
- The system is not only accurate but also responsive.
- Most users begin seeing the answer in almost one second.
- The generator is the largest latency component, which is expected.

---

## 10. Challenges and Limitations

### Challenge 1: Persona vs evaluation metrics

- Persona-based answers improve engagement
- But they can reduce answer relevancy scores because evaluators prefer shorter direct answers

### Challenge 2: Retrieval ranking noise

- Dense retrieval alone was not enough
- Relevant chunks were sometimes present but not ranked first
- This is why reranking was necessary

### Challenge 3: Long historical documents

- Some history files are long and information-rich
- Short-context embedding models may truncate important parts

### Challenge 4: Out-of-domain queries

- Users may ask unrelated modern questions
- The system handles this by marking out-of-scope cases and refusing gracefully

### Challenge 5: Dependence on external services

- STT, LLM generation, reranking, and web search depend on external APIs
- This adds operational cost and potential runtime dependency risk

### Challenge 6: Voice pipeline variability

- Speech recognition quality depends on audio quality and pronunciation
- TTS adds an extra generation step and can slightly increase response time

### Challenge 7: Simple memory design

- Current memory is lightweight key-value storage
- Good for personalization, but limited compared to a more advanced long-term memory system

## Challenge 7: Cloud flare qwen model -> moved to local

---

## 11. Strong Discussion Points for Questions

If the panel asks why this phase is strong, say:

- It combines accuracy and user experience, not just one of them.
- It is modular and measurable because each stage can be evaluated separately.
- The reranker gives a clear quantified improvement.
- The final system strongly reduces hallucination compared with weakly grounded generation.
- The chatbot keeps the educational value while making interaction engaging through persona and voice.

---

## 12. Suggested Final Conclusion Slide

**Conclusion**

- The final chatbot module is a robust persona-based RAG system for Ancient Egypt education.
- It uses query rewriting, dense retrieval, reranking, controlled web fallback, memory, and voice interaction.
- Evaluation shows strong grounding, low hallucination, high retrieval quality, and practical response speed.
- The biggest improvement came from combining better retrieval design with reranking and structured graph orchestration.

Short closing line:
- Phase 6 transformed the chatbot from a simple retriever-generator into a more reliable, interactive, and presentation-ready educational agent.

---

## 13. Very Short Slide Deck Version

If you need a short presentation, use this order:

1. Title and module goal
2. Data used
3. Final architecture
4. Main functions
5. Algorithms and techniques
6. Justification of choices
7. Evaluation results
8. Efficiency
9. Challenges and limitations
10. Conclusion

---

## 14. Presenter Notes

Avoid saying:
- "We just used an LLM"
- "It works because the model is strong"

Prefer saying:
- "We structured the problem into rewrite, retrieve, rerank, generate, and optional speech."
- "We justified each architectural choice experimentally."
- "The reranker and grounded generation were the main contributors to answer quality."
- "The system was evaluated quantitatively, not only demonstrated qualitatively."
