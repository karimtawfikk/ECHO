from pathlib import Path
import sys
import warnings

import asyncio


import pandas as pd
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

from datasets import Dataset
import os
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.prompt import PydanticPrompt
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementInput, 
    NLIStatementOutput, 
    StatementFaithfulnessAnswer
)
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,AnswerAccuracy
from ragas.metrics._answer_relevance import ( 
    ResponseRelevanceInput, 
    ResponseRelevanceOutput
)
from ragas.run_config import RunConfig

from ragas.metrics._context_recall import (
    ContextRecallClassificationPrompt, 
    QCA, 
    ContextRecallClassifications, 
    ContextRecallClassification
)
from langchain_cloudflare import CloudflareWorkersAIEmbeddings

CF_WORKERSAI_ACCOUNTID = "b9ae04d03a4c782dcf03546ca9c8240e"
CF_AI_API              = "9JqTj_5tV-Te6HN0MDVUQQr_lt-Ay2cj9Q7x2egR"

"""ragas_emb = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)"""

ragas_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# ============================================================================
# Custom Ragas Prompts
# ============================================================================
class CustomNLIPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = """You are a factual auditor for a historical RAG system for ancient egypt. The system's 'answer' is written in a 1st-person narrative persona (e.g., a King, a Queen, or an Ancient Monument).

Your goal is to verify factual faithfulness based ONLY on historical claims.

STRICT EVALUATION RULES:
1. **Fact vs. Flavor**: Separate 'historical facts' (names, dates, locations, military events, architectural builds) from 'narrative flavor' (first-person pronouns, emotional expressions, poetic descriptions, and metaphorical personification).
2. **Metaphorical Equivalence**: Treat poetic descriptions of historical entities as equivalent to their literal names (e.g., descriptions of invaders, symbols of a nation, or poetic names for locations).
3. **Ignore Honorifics**: Titles, praise, and self-referential introductory phrases (e.g., "I, the eternal stone," "My reign was glorious") should not be checked for faithfulness.
4. **Context Check**: A claim is 'Faithful' (1) if the underlying historical event or data point is supported by the context, regardless of the creative language used to describe it. It is 'Unfaithful' (0) only if it introduces a historical fact that contradicts or is entirely absent from the context.
5. Break down the answer into simple, atomic facts. Ignore the 'Thee' and 'Thou' and extract only the historical claims.
"""
    input_model = NLIStatementInput
    output_model = NLIStatementOutput

    examples = [
        # Example 1: Basic persona stripping
        (
            NLIStatementInput(
                context="The Great Pyramid was built for Pharaoh Khufu around 2560 BCE.",
                statements=["I, the eternal stone of Khufu, was placed here in 2560 BCE."]
            ),
            NLIStatementOutput(
                reason="The 1st person persona is flavor; the date and association match.", 
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="I, the eternal stone of Khufu, was placed here in 2560 BCE.",
                        reason="The 1st person persona is flavor; the date and association with Khufu match the context.",
                        verdict=1
                    )
                ]
            )
        ),
        
        # Example 2: Paraphrasing and synonyms
        (
            NLIStatementInput(
                context="Senwosret III built defensive walls along the Nubian frontier.",
                statements=[
                    "I raised stone barriers to shield the Black Land from southern invasion.",
                    "My engineers constructed towering fortifications at the border."
                ]
            ),
            NLIStatementOutput(
                reason="Both statements describe the same defensive structures using different wording.",
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="I raised stone barriers to shield the Black Land from southern invasion.",
                        reason="'Stone barriers' = 'defensive walls'. 'Black Land' = Egypt. 'Southern invasion' aligns with Nubian frontier. Core fact is present.",
                        verdict=1
                    ),
                    StatementFaithfulnessAnswer(
                        statement="My engineers constructed towering fortifications at the border.",
                        reason="'Fortifications' = 'defensive walls'. 'Border' = 'frontier'. 'Towering' is descriptive flavor. Core fact matches.",
                        verdict=1
                    )
                ]
            )
        ),
        
        # Example 3: Multiple facts in one statement
        (
            NLIStatementInput(
                context="Pepi I built Ka-chapels at Memphis and constructed pyramids at Abydos and Dendera.",
                statements=[
                    "I caused sacred houses for my Ka to rise in the capital and raised eternal monuments at the city of Osiris and the dwelling of Hathor."
                ]
            ),
            NLIStatementOutput(
                reason="Statement contains multiple facts, all supported by context with different wording.",
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="I caused sacred houses for my Ka to rise in the capital and raised eternal monuments at the city of Osiris and the dwelling of Hathor.",
                        reason="'Sacred houses for Ka' = Ka-chapels. 'Capital' = Memphis. 'Eternal monuments' = pyramids. 'City of Osiris' = Abydos. 'Dwelling of Hathor' = Dendera. All facts present.",
                        verdict=1
                    )
                ]
            )
        ),
        
        # Example 4: Emotional/poetic language (should be ignored)
        (
            NLIStatementInput(
                context="The mummy was identified in 1976 at the Egyptian Museum in Cairo.",
                statements=[
                    "My body, preserved through centuries, was recognized by scholars in 1976.",
                    "Visitors to Cairo's museum can now gaze upon my eternal form with wonder and reverence."
                ]
            ),
            NLIStatementOutput(
                reason="First statement has facts supported by context. Second is pure descriptive flavor.",
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="My body, preserved through centuries, was recognized by scholars in 1976.",
                        reason="'Recognized' = 'identified'. Date 1976 matches. 'Preserved through centuries' is flavor but doesn't contradict. Core fact is present.",
                        verdict=1
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Visitors to Cairo's museum can now gaze upon my eternal form with wonder and reverence.",
                        reason="Context mentions Egyptian Museum in Cairo. 'Gaze upon', 'wonder', 'reverence' are emotional flavor but the location fact (Cairo museum) is correct.",
                        verdict=1
                    )
                ]
            )
        ),
        
        # Example 5: Unfaithful hallucination
        (
            NLIStatementInput(
                context="Hatshepsut ruled as pharaoh for approximately 22 years during the 18th Dynasty.",
                statements=[
                    "I ruled for 30 glorious years and expanded Egypt's borders to the distant lands of Punt and Nubia."
                ]
            ),
            NLIStatementOutput(
                reason="Statement contains factual errors not supported by context.",
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="I ruled for 30 glorious years and expanded Egypt's borders to the distant lands of Punt and Nubia.",
                        reason="Context states 22 years, not 30 years. No mention of border expansion to Punt/Nubia. This introduces incorrect facts.",
                        verdict=0
                    )
                ]
            )
        ),
        
        # Example 6: Partial truth (some facts correct, some not)
        (
            NLIStatementInput(
                context="The tomb KV46 was discovered in the Valley of the Kings.",
                statements=[
                    "My tomb KV46 was found in the Valley of the Kings by Howard Carter in 1922."
                ]
            ),
            NLIStatementOutput(
                reason="Location and tomb number correct, but discoverer and date are not mentioned in context.",
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="My tomb KV46 was found in the Valley of the Kings by Howard Carter in 1922.",
                        reason="KV46 and Valley of the Kings are correct. However, 'Howard Carter in 1922' is not mentioned in context. Adding unverified facts = unfaithful.",
                        verdict=0
                    )
                ]
            )
        )
    ]

class CustomRelevancePrompt(PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]):
    instruction = """You are a question generator for a historical RAG evaluation system. 
    
Your task: Given an answer written in 1st-person Ancient Egyptian persona, generate a question that would produce this EXACT answer.

CRITICAL RULES:
1. **Match the specificity**: If the answer mentions specific details (dates, measurements, locations, relationships), the question MUST ask about those EXACT details.
2. **Use 2nd person address**: Always use "you/your" when addressing the entity.
3. **Focus on the PRIMARY FACT**: Identify what the answer is mainly about and ask directly about that.
4. **Ignore narrative structure**: Don't ask "how did X happen?" - ask "what/when/where/who is X?"
5. **Include key nouns from the answer**: If answer mentions "KV55 mummy" or "funerary chapel", include those terms in the question.
6. **Match the question type to the answer type**:
   - Answer gives a number/measurement → Ask "What are the dimensions/measurements?"
   - Answer gives a location → Ask "Where is/was...?"
   - Answer gives a date → Ask "When was...?"
   - Answer gives a relationship → Ask "What is your relationship to...?"
   - Answer gives a discovery → Ask "When/where was... discovered?"

EXAMPLES OF CORRECT SPECIFICITY:

Answer: "The genetic tests on the KV‑55 mummy show 99.99% probability that the individual was the father of Tutankhamun, and osteological assessments place his age at death between 19 and 22 years."
✓ GOOD: "What does the analysis of KV55 tell you about your relationship to Tutankhamun and your age at death?"
✗ BAD: "Who was buried in KV55?"

Answer: "The statue was uncovered in 1981 during excavations beside the Mosque of Sheikh Naqshadi in Akhmim, Sohag Governorate."
✓ GOOD: "When and where was your statue discovered?"
✗ BAD: "Where are you located?"

Answer: "I measure roughly 15 × 20 metres and rise to 15.85 metres; my fourteen columns are in a four‑by‑five pattern."
✓ GOOD: "What are your dimensions and how are your columns arranged?"
✗ BAD: "How big are you?"

Generate ONE question that matches the answer's level of detail and specificity.
"""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    
    examples = [
        (
            ResponseRelevanceInput(
                response="I became ruler of the underworld after Set murdered and dismembered me, Isis gathered and reassembled my body, revived me, and I then withdrew to the realm of the dead, assuming the throne as king and judge of souls."
            ),
            ResponseRelevanceOutput(
                question="How did you become the ruler of the underworld?",
                noncommittal=0
            )
        ),
        (
            ResponseRelevanceInput(
                response="The genetic tests on the KV‑55 mummy show a 99.99999981 % probability that the individual was the father of Tutankhamun, and osteological assessments place his age at death between 19 and 22 years (some estimates extend to 20–25 years)."
            ),
            ResponseRelevanceOutput(
                question="What does the analysis of KV55 tell you about your relationship to Tutankhamun and your age at death?",
                noncommittal=0
            )
        ),
        (
            ResponseRelevanceInput(
                response="The mound lies in the same relative spot within the enclosure that later housed Djoser's Step Pyramid, and it is considered a forerunner of the step pyramids, representing an early evolutionary stage of Egyptian royal mortuary architecture."
            ),
            ResponseRelevanceOutput(
                question="You are located near which structure, and what significance does your mound hold in the evolution of Egyptian pyramids?",
                noncommittal=0
            )
        ),
        (
            ResponseRelevanceInput(
                response="The statue of Meret Amun was uncovered in 1981 during excavations beside the Mosque of Sheikh Naqshadi in the town of Akhmim, within Sohag Governorate, Egypt."
            ),
            ResponseRelevanceOutput(
                question="When and where was your statue discovered?",
                noncommittal=0
            )
        ),
        (
            ResponseRelevanceInput(
                response="I measure roughly 15 × 20 metres and rise to a height of 15.85 metres; my fourteen massive sandstone columns are laid out in a four‑by‑five pattern."
            ),
            ResponseRelevanceOutput(
                question="What are your dimensions and how are your columns arranged?",
                noncommittal=0
            )
        ),
        (
            ResponseRelevanceInput(
                response="The walls of my funerary chapel at Medinet Habu depict the ritual driving of the four calves and the consecration of the meret‑chests, both performed for the benefit of my adoptive mother, Amenirdis I."
            ),
            ResponseRelevanceOutput(
                question="What rituals are depicted on the walls of your funerary chapel at Medinet Habu that were performed for your adoptive mother?",
                noncommittal=0
            )
        ),
        (
            ResponseRelevanceInput(
                response="I was interred in tomb KV 46 in the Valley of the Kings, and my mummy now rests in the Egyptian Museum in Cairo, where it was positively identified in 1976 and linked to the Elder Lady of KV 35 through genetic testing."
            ),
            ResponseRelevanceOutput(
                question="Where were you buried and where is your mummy currently located?",
                noncommittal=0
            )
        )
    ]


class CustomRecallExtractionPrompt(ContextRecallClassificationPrompt):
    instruction = """
Given a context and a poetic reference answer (labeled 'answer'), analyze each sentence 
in the answer. Classify if the historical/factual claim in that sentence is supported 
by the context.

GUIDELINES:
1. The 'expected_output' is written in a poetic, first-person Pharaoh persona. 
2. STRIP AWAY all persona elements (e.g., 'I recall', 'My spirit') and extract ONLY the underlying historical data points.
3. Ignore all first-person roleplay, metaphors, flavor text, and poetic descriptions.
4. Extract only dates, names, physical locations, archaeological finds, and specific historical events.
5. **CRITICAL**: If the core fact is present but expressed differently, mark as 1 (attributed).
   - Example: Context says "temple built", answer says "I raised a shrine" → SAME FACT → 1
   - Example: Context says "defensive walls", answer says "stone barriers" → SAME FACT → 1
6. Use only 'Yes' (1) or 'No' (0) as a binary classification.
"""
    input_model = QCA
    output_model = ContextRecallClassifications

    examples = [
        (
            QCA(
                question="When was Neith confirmed as Teti's wife?",
                context="In January 2021, Zahi Hawass discovered the funerary temple of Queen Neith near Teti's pyramid.",
                answer="I, Teti, saw the scholars find the temple of my beloved Neith in January 2021."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="I, Teti, saw the scholars find the temple of my beloved Neith in January 2021.",
                        reason="The context confirms the discovery of Neith's temple in January 2021. The first-person 'I' and 'beloved' are persona flavor and should be ignored.",
                        attributed=1
                    )
                ]
            )
        ),
        (
            QCA(
                question="What defensive structures did Senwosret build?",
                context="Senusret III raised defensive walls and built fortifications along the Nubian frontier.",
                answer="I, Kha-kha-per-re, commanded my engineers to erect towering stone barriers that would shield the Black Land from southern invasion."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="I commanded my engineers to erect towering stone barriers that would shield the Black Land from southern invasion.",
                        reason="Context mentions 'defensive walls' and 'fortifications'. The answer's 'stone barriers' and 'shield' are synonymous descriptions of the same defensive structures. The poetic language ('towering', 'Black Land') is flavor.",
                        attributed=1
                    )
                ]
            )
        ),
        (
            QCA(
                question="What building projects did Pepi I undertake?",
                context="Pepi I constructed Ka-chapels at Memphis and built pyramids at Abydos and Dendera.",
                answer="I caused sacred houses for my Ka to rise in the capital, and raised eternal monuments at the city of Osiris and the dwelling of Hathor."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="I caused sacred houses for my Ka to rise in the capital.",
                        reason="Context confirms Ka-chapels at Memphis (the capital). 'Sacred houses for my Ka' = Ka-chapels. Same fact.",
                        attributed=1
                    ),
                    ContextRecallClassification(
                        statement="I raised eternal monuments at the city of Osiris and the dwelling of Hathor.",
                        reason="Context confirms pyramids at Abydos (city of Osiris) and Dendera (dwelling of Hathor). 'Eternal monuments' = pyramids. Same facts.",
                        attributed=1
                    )
                ]
            )
        ),
        (
            QCA(
                question="What was found in the burial shafts?",
                context="52 burial shafts were found containing New Kingdom sarcophagi.",
                answer="My spirit watched as they lifted fifty-two boxes from the sands of time."
            ),
            ContextRecallClassifications(
                classifications=[
                    ContextRecallClassification(
                        statement="My spirit watched as they lifted fifty-two boxes from the sands of time.",
                        reason="The context confirms 52 burial shafts (boxes). 'My spirit watched' and 'sands of time' are metaphors.",
                        attributed=1
                    )
                ]
            )
        )
    ]


# ============================================================================
# Load Agent Responses from CSV
# ============================================================================

def load_agent_responses(csv_path: str) -> List[Dict[str, Any]]:
    """Load pre-collected agent responses from CSV"""
    print(f"\nLoading agent responses from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Convert contexts string back to list
    df['contexts'] = df['contexts'].apply(lambda x: x.split('|||') if isinstance(x, str) else [])
    
    # Convert to list of dicts
    results = df.to_dict('records')
    
    print(f"  ✓ Loaded {len(results)} agent responses")
    print(f"  • Successful: {sum(1 for r in results if r.get('success', False))}")
    print(f"  • Unique entities: {df['entity_name'].nunique()}")
    
    return results


# ============================================================================
# Groq Key Manager
# ============================================================================

"""class GroqKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_index = 0

    def get_current_key(self):
        return self.keys[self.current_index]

    def rotate_key(self):
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"🔄 Swapping to API Key {self.current_index + 1}...")
        return self.get_current_key()"""

import os

class GroqKeyManager:
    def __init__(self, keys):
        self.keys = keys
        # Start at the last index (e.g., index 8 for 9 keys)
        self.current_index = len(self.keys) - 1 if keys else 0

    def get_current_key(self):
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def rotate_key(self):
        if not self.keys:
            return None
            
        # Subtract 1 to move backward. 
        # Python's % operator handles negative numbers: -1 % 9 = 8
        self.current_index = (self.current_index - 1) % len(self.keys)
        
        print(f"🔄 Swapping to API Key {self.current_index + 1}...")
        return self.get_current_key()

# --- Setup ---
# This fetches Keys 1 through 9 from your environment


def extract_first_paragraph(text):
    if not text:
        return text
    
    text = text.replace('**', '').strip()
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) > 1:
        return paragraphs[0]
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Return first paragraph
    return paragraphs[0] if paragraphs else text


def compute_ragas_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute Ragas metrics with manual Strictness 3 for Answer Relevancy"""
    print("\n" + "="*80)
    print("Computing Ragas Metrics (Enhanced Relevancy Mode)")
    print("="*80 + "\n")

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    successful_results = [r for r in results if r.get("success") and r.get("answer")]
    
    keys = [os.getenv(f"GROQ_API_KEY{i}") for i in range(1, 10)]
    # Filter out any None values if some keys aren't set
    valid_keys = [k for k in keys if k]

    manager = GroqKeyManager(valid_keys)


    
    all_individual_results = []
    i = 0
    retry_count = {} 

    print(f"🚀 Evaluating {len(successful_results)} responses")
    print(f"🔑 Using {len(manager.keys)} API keys with rotation every 5 samples (due to high call volume)\n")

    while i < len(successful_results):
        # Rotate more frequently because we are doing 4 calls per sample now
        if i > 0 and i % 5 == 0:
            manager.rotate_key()
        
        item = successful_results[i]
        entity_name = item.get("entity_name", "Unknown")
        entity_type = item.get("entity_type", "Unknown")
        
        print(f"\n[Sample {i+1}/{len(successful_results)}] Entity: {entity_name} ({entity_type})")
        
        if i not in retry_count:
            retry_count[i] = 0
        # Prepare wrappers with the current key
        evaluator_llm = ChatGroq(
            model="moonshotai/kimi-k2-instruct-0905", 
            api_key=manager.get_current_key(),
            temperature=0,
            max_tokens=3000
        )

        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        ragas_emb_wrapped = LangchainEmbeddingsWrapper(ragas_emb)

        # Pre-process strings
        clean_answer = extract_first_paragraph(item["answer"])
        clean_ground_truth = extract_first_paragraph(item.get("ground_truth", ""))

        single_dataset = Dataset.from_dict({
            "user_input": [item["question"]],
            "response": [clean_answer],
            "retrieved_contexts": [item["contexts"]],
            "reference": [clean_ground_truth]
        })

        try:
            # --- PHASE 1: CORE METRICS ---
            print(f"  → Phase 1: Calculating Faithfulness & Context Metrics...")
            f_metric = Faithfulness(llm=ragas_llm)
            f_metric.set_prompts(n_l_i_statement_prompt=CustomNLIPrompt())
            
            recall_metric = ContextRecall(llm=ragas_llm)
            recall_metric.context_recall_prompt = CustomRecallExtractionPrompt()

            r_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb_wrapped,strictness=1)
            r_metric.question_generation_prompt = CustomRelevancePrompt()


            core_res = evaluate(
                dataset=single_dataset,
                metrics=[f_metric, recall_metric, r_metric,ContextPrecision(llm=ragas_llm), AnswerAccuracy(llm=ragas_llm) ],
                llm=ragas_llm,
                embeddings=ragas_emb_wrapped,
                run_config=RunConfig(timeout=180, max_workers=1)
            )
            core_df = core_res.to_pandas()
            '''relevancy_scores = []
            print(f"  → Phase 2: Running Relevancy (3 trials):", end=" ", flush=True)
            for trial in range(3):
                rel_res = evaluate(
                    dataset=single_dataset,
                    metrics=[r_metric],
                    llm=ragas_llm,
                    embeddings=ragas_emb_wrapped,
                    run_config=RunConfig(timeout=120, max_workers=1)
                )
                score = rel_res.to_pandas()['answer_relevancy'].iloc[0]
                relevancy_scores.append(score)
                print(f"[{score:.3f}]", end=" ", flush=True)
            
            best_relevancy = max(relevancy_scores)
            print(f" | SCORES: ", relevancy_scores)
            print(f" | Selected Max: {best_relevancy:.3f}")'''

            # Combine and Log
            final_row = core_df.iloc[0].to_dict()
            #final_row['answer_relevancy'] = best_relevancy
            
            
            
            print(f"  → Results: F:{final_row['faithfulness']:.2f} | R:{final_row['answer_relevancy']:.2f} | C-Rec:{final_row['context_recall']:.2f} | C-Pre:{final_row['context_precision']:.2f} | Acc:{final_row['nv_accuracy']:.2f}")
            
           


            all_individual_results.append(pd.DataFrame([final_row]))
            retry_count[i] = 0
            i += 1
            
            print(f"✅ Success [{i}/{len(successful_results)}]")

        except Exception as e:
            err_msg = str(e)
            if any(x in err_msg.lower() for x in ["rate limit", "rate_limit", "429", "tokens per day"]):
                print(f"🚨 Rate Limit! Rotating key and sleeping...")
                manager.rotate_key()
                time.sleep(20)
            elif "StringIO" in err_msg and retry_count[i] < 1:
                retry_count[i] += 1
                print(f"⚠️ Parser glitch, retrying sample...")
                time.sleep(5)
            else:
                print(f"❌ Skipping sample {i+1} due to error: {err_msg[:100]}")
                i += 1

    if not all_individual_results:
        return {"error": "No successful evaluations"}

    final_df = pd.concat(all_individual_results, ignore_index=True)
    summary = final_df.mean(numeric_only=True).to_dict()
    
    print("\n✅ Ragas evaluation complete!")
    print(f"📊 Evaluated {len(all_individual_results)}/{len(successful_results)} samples successfully")
    
    return {k: float(v) for k, v in summary.items()}



# ============================================================================
# Compute Custom Metrics
# ============================================================================

def compute_custom_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute custom performance metrics"""
    print("\nComputing custom metrics...")
    
    successful_results = [r for r in results if r.get("success", False)]
    
    metrics = {
        "total_queries": len(results),
        "successful_queries": len(successful_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "avg_response_time": np.mean([r.get("response_time", 0) for r in successful_results]) if successful_results else 0,
        "min_response_time": np.min([r.get("response_time", 0) for r in successful_results]) if successful_results else 0,
        "max_response_time": np.max([r.get("response_time", 0) for r in successful_results]) if successful_results else 0,
        "median_response_time": np.median([r.get("response_time", 0) for r in successful_results]) if successful_results else 0,
        "avg_context_chunks": np.mean([r.get("context_count", 0) for r in successful_results]) if successful_results else 0,
        "avg_answer_length": np.mean([r.get("answer_length", 0) for r in successful_results]) if successful_results else 0,
    }
    
    entity_types = {}
    for r in results:
        etype = r.get("entity_type", "unknown")
        if etype not in entity_types:
            entity_types[etype] = {"total": 0, "successful": 0}
        entity_types[etype]["total"] += 1
        if r.get("success", False):
            entity_types[etype]["successful"] += 1
    
    metrics["entity_type_performance"] = {
        etype: stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        for etype, stats in entity_types.items()
    }
    
    retrieved = sum(1 for r in successful_results if r.get("context_count", 0) > 0)
    metrics["retrieval_success_rate"] = retrieved / len(successful_results) if successful_results else 0
    
    print("  ✓ Custom metrics computed successfully!")
    return metrics


# ============================================================================
# Generate Visualizations
# ============================================================================

def generate_visualizations(results: List[Dict], ragas_scores: Dict, custom_metrics: Dict, output_dir: Path):
    """Generate visualization charts"""
    print("\nGenerating visualizations...")
    
    output_dir.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ancient Egypt RAG Chatbot - Evaluation Results', fontsize=16, fontweight='bold')
    
    if ragas_scores:
        metrics_df = pd.DataFrame({
            'Metric': ['Faithfulness', 'Answer\nRelevancy', 'Context\nPrecision', 'Context\nRecall', 'Answer\nAccuracy'],
            'Score': [
                ragas_scores.get('faithfulness', 0),
                ragas_scores.get('answer_relevancy', 0),
                ragas_scores.get('context_precision', 0),
                ragas_scores.get('context_recall', 0),
                ragas_scores.get('nv_accuracy', 0)
            ]
        })
        sns.barplot(data=metrics_df, x='Metric', y='Score', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('RAG Triad Metrics', fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
        axes[0, 0].legend()
    
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        times = [r.get('response_time', 0) for r in successful_results]
        axes[0, 1].hist(times, bins=20, edgecolor='black', color='skyblue')
        axes[0, 1].set_title('Response Time Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=np.mean(times), color='r', linestyle='--', label=f'Mean: {np.mean(times):.2f}s')
        axes[0, 1].legend()
    
    entity_perf = custom_metrics.get('entity_type_performance', {})
    if entity_perf:
        etypes = list(entity_perf.keys())
        scores = list(entity_perf.values())
        axes[0, 2].bar(etypes, scores, color='coral', edgecolor='black')
        axes[0, 2].set_title('Success Rate by Entity Type', fontweight='bold')
        axes[0, 2].set_ylabel('Success Rate')
        axes[0, 2].set_ylim(0, 1)
    
    if successful_results:
        context_lengths = [r.get('context_count', 0) for r in successful_results]
        max_contexts = max(context_lengths) if context_lengths else 10
        axes[1, 0].hist(context_lengths, bins=range(0, max_contexts+2), edgecolor='black', color='lightgreen')
        axes[1, 0].set_title('Retrieved Context Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Chunks Retrieved')
        axes[1, 0].set_ylabel('Frequency')
    
    success_count = custom_metrics.get('successful_queries', 0)
    failure_count = custom_metrics.get('total_queries', 0) - success_count
    axes[1, 1].pie([success_count, failure_count], labels=['Success', 'Failure'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[1, 1].set_title('Query Success Rate', fontweight='bold')
    
    axes[1, 2].axis('off')
    summary_data = [
        ['Total Queries', f"{custom_metrics['total_queries']}"],
        ['Success Rate', f"{custom_metrics['success_rate']*100:.1f}%"],
        ['Avg Response Time', f"{custom_metrics['avg_response_time']:.2f}s"],
        ['Retrieval Success', f"{custom_metrics['retrieval_success_rate']*100:.1f}%"],
    ]
    if ragas_scores:
        summary_data.extend([
            ['Faithfulness', f"{ragas_scores.get('faithfulness', 0):.3f}"],
            ['Answer Relevancy', f"{ragas_scores.get('answer_relevancy', 0):.3f}"],
        ])
    
    table = axes[1, 2].table(cellText=summary_data, cellLoc='left', loc='center',
                             colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('Performance Summary', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'evaluation_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_path}")
    
    plt.close()


# ============================================================================
# Save JSON Report
# ============================================================================

def save_json_report(results: List[Dict], ragas_scores: Dict, custom_metrics: Dict, output_dir: Path):
    """Save detailed results as JSON"""
    print("\nSaving JSON report...")
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(results),
            "evaluation_model": "kimi"
        },
        "ragas_metrics": ragas_scores,
        "custom_metrics": custom_metrics,
        "detailed_results": results
    }
    
    output_path = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved JSON report to {output_path}")
    
    return output_path


# ============================================================================
# Generate Markdown Report
# ============================================================================

def generate_markdown_report(results: List[Dict], ragas_scores: Dict, custom_metrics: Dict, output_dir: Path):
    """Generate comprehensive markdown report"""
    print("\nGenerating Markdown report...")
    
    md = f"""# Ancient Egypt RAG Chatbot - Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Model:** Groq gpt-oss-20b  
**Test Cases:** {len(results)}

---

## Executive Summary

This evaluation assesses the Ancient Egypt RAG chatbot across {len(results)} test cases covering {len(set(r.get('entity_name', '') for r in results))} unique entities (pharaohs and landmarks).

### Key Findings

"""
    
    if ragas_scores:
        avg_score = np.mean(list(ragas_scores.values()))
        md += f"- **Overall RAG Performance:** {avg_score:.2f}/1.00\n"
    
    md += f"- **System Success Rate:** {custom_metrics['success_rate']*100:.1f}%\n"
    md += f"- **Average Response Time:** {custom_metrics['avg_response_time']:.2f} seconds\n"
    md += f"- **Retrieval Success Rate:** {custom_metrics['retrieval_success_rate']*100:.1f}%\n"
    
    md += "\n---\n\n## RAG Triad Metrics\n\n"
    
    if ragas_scores:
        md += "| Metric | Score | Target | Status | Interpretation |\n"
        md += "|--------|-------|--------|--------|----------------|\n"
        
        metrics_info = {
            "faithfulness": ("0.85", "Did the Pharaoh hallucinate? (Answer vs Context)"),
            "answer_relevancy": ("0.80", "Did it answer the question? (Answer vs Question)"),
            "context_precision": ("0.75", "Did retrieval find the best chunks? (Context vs Answer)"),
            "context_recall": ("0.75", "Does context contain the answer? (Answer coverage)"),
            "answer_accuracy": ("0.80", "How accurate is the answer vs ground truth? (Answer vs Reference)")
        }
        
        for metric_name, score in ragas_scores.items():
            target, description = metrics_info.get(metric_name, ("0.75", "N/A"))
            status = "✓ Pass" if score >= float(target) else "✗ Needs Improvement"
            md += f"| {metric_name.replace('_', ' ').title()} | {score:.3f} | {target} | {status} | {description} |\n"
    
    md += "\n---\n\n## Performance Breakdown\n\n"
    md += "### By Entity Type\n\n"
    
    for etype, perf in custom_metrics.get('entity_type_performance', {}).items():
        count = sum(1 for r in results if r.get('entity_type') == etype)
        md += f"| {etype.title()} | {perf*100:.1f}% | {count} |\n"
    
    md += "\n### Response Time Statistics\n\n"
    md += f"| Average | {custom_metrics['avg_response_time']:.2f}s |\n"
    md += f"| Median | {custom_metrics['median_response_time']:.2f}s |\n"
    md += f"| Min | {custom_metrics['min_response_time']:.2f}s |\n"
    md += f"| Max | {custom_metrics['max_response_time']:.2f}s |\n"
    
    md += "\n\n---\n\n*End of Report*"
    
    output_path = output_dir / 'evaluation_report.md'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"  ✓ Saved Markdown report to {output_path}")
    
    return output_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" Ragas Evaluation on Pre-Collected Agent Responses")
    print("="*80 + "\n")
    
    # Input: Pre-collected agent responses CSV
    responses_csv = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\agent_responses\agent_responses_pt1.csv"
    
    # Output directory
    output_dir = Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\ragas_evaluation_results\run3\pt2")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load pre-collected responses
    results = load_agent_responses(responses_csv)
    
    # Step 2: Compute Ragas metrics
    ragas_scores = compute_ragas_metrics(results)
    
    # Step 3: Compute custom metrics
    custom_metrics = compute_custom_metrics(results)
    
    # Step 4: Generate visualizations
    generate_visualizations(results, ragas_scores, custom_metrics, output_dir)
    
    # Step 5: Save JSON report
    json_path = save_json_report(results, ragas_scores, custom_metrics, output_dir)
    
    # Step 6: Generate Markdown report
    md_path = generate_markdown_report(results, ragas_scores, custom_metrics, output_dir)
    
    print("\n" + "="*80)
    print(" EVALUATION COMPLETE!")
    print("="*80 + "\n")
    
    print("📊 Results Summary:")
    print(f"  • Total test cases: {custom_metrics['total_queries']}")
    print(f"  • Success rate: {custom_metrics['success_rate']*100:.1f}%")
    print(f"  • Avg response time: {custom_metrics['avg_response_time']:.2f}s")
    
    if ragas_scores:
        print(f"\n📈 RAG Triad Metrics:")
        print(f"  • Faithfulness: {ragas_scores.get('faithfulness', 0):.3f}")
        print(f"  • Answer Relevancy: {ragas_scores.get('answer_relevancy', 0):.3f}")
        print(f"  • Context Precision: {ragas_scores.get('context_precision', 0):.3f}")
        print(f"  • Context Recall: {ragas_scores.get('context_recall', 0):.3f}")
        print(f"  • Answer Accuracy: {ragas_scores.get('nv_accuracy', 0):.3f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  • Visualization: {output_dir}/evaluation_visualization.png")
    print(f"  • JSON Report: {json_path}")
    print(f"  • Markdown Report: {md_path}")
    
    print("\n✅ All evaluation files saved to:", output_dir.absolute())
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()