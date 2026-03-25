# Chatbot Module Seminar 2 - Short Deck

This is a shorter 10-slide version of the chatbot seminar content, designed for direct presentation use.

## Slide 1 - Title

**ECHO Chatbot Module - Phase 6 Final System**

Subtitle:
- Persona-based Ancient Egypt conversational RAG chatbot

Say:
- This is the final version of the chatbot module in the ECHO project.
- It allows users to talk to an ancient Egyptian pharaoh or landmark through text or voice.
- The system focuses on both historical accuracy and immersive interaction.

---

## Slide 2 - Module Objective

**Objective**

- Build an educational chatbot for Ancient Egyptian pharaohs and landmarks
- Keep answers grounded in retrieved evidence
- Make responses engaging through first-person persona
- Support multilingual interaction and voice mode

Say:
- Our goal was not just question answering, but grounded persona-based conversation.
- So we aimed to balance factuality, retrieval quality, and user experience.

---

## Slide 3 - Data Used

**Dataset**

- `80` pharaoh documents
- `52` landmark documents
- `132` total entity documents
- Evaluation dataset: `132` test cases

**How the data is used**

- User first selects an entity type and entity name
- Retrieval is restricted to that entity only
- The system searches historical text chunks stored in the database

Say:
- This entity-specific retrieval design reduces irrelevant context and improves precision.
- It also makes the chatbot feel more consistent because each conversation stays tied to one selected historical identity.

---

## Slide 4 - Final Architecture

**Phase 6 Pipeline**

1. User enters text or voice
2. Query rewriter reformulates the question
3. Query is embedded using Qwen embeddings
4. Top-10 chunks are retrieved
5. Jina reranker compresses them to Top-3
6. Generator LLM answers in persona
7. Tavily search is used once if local context is insufficient
8. Edge TTS generates speech in voice mode

Say:
- The chatbot is implemented as a LangGraph workflow, so each stage is modular and easy to evaluate.
- The most important improvement in Phase 6 is that the answer is generated only after retrieval and reranking.

---

## Slide 5 - Main Functions and Techniques

**Core functions**

- `rewrite_node()` for query reformulation and memory extraction
- `retrieve_node()` for vector retrieval
- `rerank_node()` for reranking retrieved chunks
- `generate_node()` for persona-based grounded answer generation
- `tts_node()` for speech output

**Main techniques**

- Retrieval-Augmented Generation
- Query rewriting
- Persona prompting
- Dense vector retrieval
- Cross-encoder reranking
- Controlled web fallback
- Speech-to-text and text-to-speech

Say:
- These four core nodes form the intelligence pipeline of the chatbot.
- Around them, we added voice handling, memory, and optional web fallback for a more complete user experience.

---

## Slide 6 - Justification of Design Choices

**Why these choices?**

- `Qwen/Qwen3-Embedding-0.6B`
  - Chosen for long context support, multilingual ability, and MRL support
- `768-dim embeddings`
  - Better balance between quality and speed than smaller truncated vectors
- `Jina reranker`
  - Improves top-ranked retrieval quality significantly
- `GPT-OSS-120B`
  - Strong final grounded generation performance
- `Qwen-32B rewriter`
  - Good for reformulation without using the larger model unnecessarily
- `Edge TTS`
  - Practical multilingual speech output for demos

Say:
- We did not choose components randomly.
- Each component was selected either because it improved retrieval quality, reduced hallucination, or improved usability.

---

## Slide 7 - Main Evaluation Results

**Final ECHO Agent Results**

- Faithfulness: `0.9670 +- 0.0087`
- Answer Relevancy: `0.7700 +- 0.0044`
- Context Precision: `0.9357 +- 0.0011`
- Context Recall: `0.9893 +- 0.0112`
- Answer Accuracy: `0.9745`
- Grounded Hallucination: `0.0330 +- 0.0087`

Say:
- These results show that the system is highly grounded and very reliable.
- The strongest signals are high faithfulness, very high context recall, and low hallucination.
- Answer relevancy is a bit lower mainly because the persona style adds extra phrasing that evaluation metrics may penalize.

---

## Slide 8 - Reranker Impact

**Effect of Jina Reranker**

- Recall@1: `0.7045 -> 0.8712` (`+23.6%`)
- MRR: `0.8106 -> 0.9268` (`+14.3%`)
- NDCG@3: `0.8438 -> 0.9419` (`+11.6%`)
- Recall@3: `0.9394 -> 0.9848` (`+4.8%`)

Say:
- This is one of the clearest improvements in the entire module.
- The reranker dramatically improved which chunk appears first, which directly improves final answer quality.
- It solved the ranking-noise problem of pure dense retrieval.

---

## Slide 9 - Efficiency and Practicality

**Efficiency**

- TTFT mean: `1.17s`
- TTFT P95: `1.37s`
- End-to-end mean latency: `1.98s`
- End-to-end P95: `2.65s`
- Success rate: `132/132 = 100%`

**Latency breakdown**

- Rewriter: `21.0%`
- Retriever: `18.3%`
- Reranker: `12.8%`
- Generator: `47.9%`

Say:
- The system is not only accurate, but also responsive enough for a live conversational setting.
- Most users see the response begin in nearly one second.
- The generator is the main latency source, which is expected in LLM-based systems.

---

## Slide 10 - Challenges, Limitations, and Conclusion

**Challenges and limitations**

- Persona style can reduce some evaluation metrics
- Dense retrieval alone was not enough, so reranking was required
- Some historical documents are long and complex
- External APIs introduce dependency and cost
- Voice quality depends on microphone and speech conditions
- Memory is lightweight, not a full long-term memory system

**Conclusion**

- Phase 6 produced a grounded, modular, and interactive chatbot
- The main gains came from query rewriting, reranking, and graph-based orchestration
- The final result is both educationally useful and presentation-ready

Say:
- Overall, Phase 6 moved the chatbot from a basic RAG system into a more reliable and polished conversational agent.
- The most important contribution is that the architecture is justified by real evaluation results, not only by qualitative demos.
