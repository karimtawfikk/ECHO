from pathlib import Path
import sys
import warnings
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset
import os
from dotenv import load_dotenv
import time

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy
from ragas.run_config import RunConfig

# Embeddings
ragas_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# ============================================================================
# Load Agent Responses from CSV
# ============================================================================

def load_agent_responses(csv_path: str) -> List[Dict[str, Any]]:
    """Load pre-collected agent responses from CSV"""
    print(f"\nLoading agent responses from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    df['contexts'] = df['contexts'].apply(lambda x: x.split('|||') if isinstance(x, str) else [])
    results = df.to_dict('records')
    
    print(f"  ✓ Loaded {len(results)} agent responses")
    print(f"  • Successful: {sum(1 for r in results if r.get('success', False))}")
    
    return results


# ============================================================================
# Groq Key Manager
# ============================================================================

class GroqKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_index = len(self.keys) - 1 if keys else 0

    def get_current_key(self):
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def rotate_key(self):
        if not self.keys:
            return None
        self.current_index = (self.current_index - 1) % len(self.keys)
        print(f"🔄 Swapping to API Key {self.current_index + 1}...")
        return self.get_current_key()


def extract_first_paragraph(text):
    if not text:
        return text
    
    text = text.replace('**', '').strip()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) > 1:
        return paragraphs[0]
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs[0] if paragraphs else text


def compute_answer_accuracy(results: List[Dict]) -> float:
    """Compute ONLY Answer Accuracy metric"""
    print("\n" + "="*80)
    print("Computing Answer Accuracy Only")
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
    valid_keys = [k for k in keys if k]
    manager = GroqKeyManager(valid_keys)

    all_scores = []
    i = 0
    retry_count = {}

    print(f"🚀 Evaluating {len(successful_results)} responses for Answer Accuracy")
    print(f"🔑 Using {len(manager.keys)} API keys with rotation every 10 samples\n")

    while i < len(successful_results):
        
        
        if i > 0 and i % 10 == 0:
            manager.rotate_key()
        
        item = successful_results[i]
        entity_name = item.get("entity_name", "Unknown")
        entity_type = item.get("entity_type", "Unknown")
        
        print(f"\n[Sample {i+1}/{len(successful_results)}] Entity: {entity_name} ({entity_type})")
        
        if i not in retry_count:
            retry_count[i] = 0
        
        evaluator_llm = ChatGroq(
            model="moonshotai/kimi-k2-instruct-0905", 
            api_key=manager.get_current_key(),
            temperature=0,
            max_tokens=3000
        )

        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        ragas_emb_wrapped = LangchainEmbeddingsWrapper(ragas_emb)

        clean_answer = extract_first_paragraph(item["answer"])
        clean_ground_truth = extract_first_paragraph(item.get("ground_truth", ""))

        single_dataset = Dataset.from_dict({
            "user_input": [item["question"]],
            "response": [clean_answer],
            "retrieved_contexts": [item["contexts"]],
            "reference": [clean_ground_truth]
        })

        try:
            res = evaluate(
                dataset=single_dataset,
                metrics=[AnswerAccuracy(llm=ragas_llm)],
                llm=ragas_llm,
                embeddings=ragas_emb_wrapped,
                run_config=RunConfig(timeout=180, max_workers=1)
            )
            
            df = res.to_pandas()
            accuracy = df['nv_accuracy'].iloc[0]
            if pd.isna(accuracy):  # 503 error occurred
                retry_wait = min(60 * (2 ** retry_count[i]), 300)  # Exponential backoff, max 5 min
                print(f"⚠️ 503 Error. Waiting {retry_wait}s before retry #{retry_count[i]+1}...")
                time.sleep(retry_wait)
                retry_count[i] += 1
                
                if retry_count[i] >= 5:  # Give up after 5 retries
                    print(f"❌ Failed after 5 retries. Skipping sample.")
                    all_scores.append(np.nan)  # Keep as NaN
                    i += 1
                # else: don't increment i, retry same sample
                continue



            all_scores.append(accuracy)
            
            print(f"Accuracy: {accuracy:.3f} ✅")
            
            i += 1
            retry_count[i] = 0

        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "over capacity" in err_msg.lower():
                print(f"⏳ Kimi overloaded. Waiting 60s then retrying...")
                time.sleep(60)
                # Don't increment i - retry same sample
                continue
            elif any(x in err_msg.lower() for x in ["rate limit", "rate_limit", "429"]):
                print(f"🚨 Rate Limit! Rotating...")
                manager.rotate_key()
                time.sleep(20)
            elif "StringIO" in err_msg and retry_count[i] < 1:
                retry_count[i] += 1
                print(f"⚠️ Retry...")
                time.sleep(5)
            else:
                print(f"❌ Error: {err_msg[:80]}")
                i += 1

    if not all_scores:
        return 0.0

    avg_accuracy = np.mean(all_scores)
    
    print("\n" + "="*80)
    print(f"📊 FINAL ANSWER ACCURACY: {avg_accuracy:.3f}")
    print(f"   Evaluated: {len(all_scores)}/{len(successful_results)} samples")
    print("="*80 + "\n")
    
    return avg_accuracy


# ============================================================================
# Main Execution
# ============================================================================

def main():
    responses_csv = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\agent_responses\agent_responses_pt1.csv"
    
    results = load_agent_responses(responses_csv)
    final_accuracy = compute_answer_accuracy(results)
    
    print(f"\n✅ Final Answer Accuracy: {final_accuracy:.3f}\n")


if __name__ == "__main__":
    main()