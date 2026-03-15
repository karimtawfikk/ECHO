"""
Ragas Evaluation Script - Evaluates Pre-Collected Agent Responses
Loads agent responses from CSV and computes Ragas metrics without re-running the agent
"""

from pathlib import Path
import sys
import warnings
root_path = Path("c:/Uni/4th Year/GP/ECHO/experiments/chatbot/echo_chatbot/chatbot_phases")
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

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
warnings.filterwarnings("ignore", category=RuntimeError, message=".*Event loop is closed.*")

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
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
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
ragas_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
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
        )
    ]


class CustomRelevancePrompt(PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]):
    instruction = """Generate a question for the given answer. The answer is provided by a historical RAG chatbot acting as an Ancient Egyptian persona.
You are a historical query analyst. Your task is to generate a question that would result in the provided 'answer'. 
The answer is written in a 1st-person narrative persona (e.g., a Pharaoh or Ancient Monument).

STRICT ANALYSIS RULES:
1. **Direct Address**: The question MUST be written in the 2nd person, addressing the entity directly (e.g., Use "You," "Your," "Mighty [Name]").
2. **Fact vs. Flavor**: Ignore 1st-person pronouns ("I", "My"), emotional expressions, and poetic storytelling. Focus only on the 'historical facts' (achievements, names, dates, military events).
3. If the answer contains narrative 'flavor' but answers a specific historical event, the generated question should reflect that event.
4. **Reverse Engineering**: Generate a question that targets the core factual information. If the answer describes military resistance, the question should be about military resistance, even if the answer is written as a poem.
5. **Tone Matching**: The question should sound like it belongs in your dataset—formal, respectful, and inquiring about specific reign achievements or historical events.
6. **Precision**: The generated question must be a direct map to the factual substance of the answer provided.
"""
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    
    examples = [
        (
            ResponseRelevanceInput(response="I, Hakoris, secured the future by naming my son Nephrites as my successor to ensure stability."),
            ResponseRelevanceOutput(question="Mighty Hakoris, how did you ensure political stability and what was the significance of naming your son as heir?", noncommittal=0)
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
3. Ignore all first-person roleplay, metaphors, flavor text, and poetic descriptions (e.g., 'immortal horizons', 'ghostly whispers').
4. Extract only dates, names, physical locations, archaeological finds, and specific historical events.
5. Only extract verifiable facts: names, dates, quantities, and specific locations.
6. Example: If it says "My spirit saw fifty sarcophagi," you extract: "Fifty sarcophagi were present."
7. If the sentence contains a date, name, or location found in the context, mark as 1.
8. Use only 'Yes' (1) or 'No' (0) as a binary classification. Output json with reason.
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

class GroqKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_index = 0

    def get_current_key(self):
        return self.keys[self.current_index]

    def rotate_key(self):
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"🔄 Swapping to API Key {self.current_index + 1}...")
        return self.get_current_key()


# ============================================================================
# Compute Ragas Metrics
# ============================================================================
def compute_ragas_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute Ragas metrics on pre-collected responses"""
    print("\n" + "="*80)
    print("Computing Ragas Metrics")
    print("="*80 + "\n")

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    successful_results = [r for r in results if r.get("success") and r.get("answer")]
    
    keys = [os.getenv(f"GROQ_API_KEY{i}") for i in range(1, 8)]
    manager = GroqKeyManager([k for k in keys if k])
    
    all_individual_results = []
    i = 0
    retry_count = {}  # Track retries per sample

    print(f"🚀 Evaluating {len(successful_results)} responses (Individual Resume Mode)")

    while i < len(successful_results):
        item = successful_results[i]
        
        # Print entity name before evaluation
        entity_name = item.get("entity_name", "Unknown")
        entity_type = item.get("entity_type", "Unknown")
        print(f"\n[Sample {i+1}/{len(successful_results)}] Entity: {entity_name} ({entity_type})")
        
        # Initialize retry counter for this sample
        if i not in retry_count:
            retry_count[i] = 0
        
        evaluator_llm = ChatGroq(
            model="openai/gpt-oss-20b", 
            api_key=manager.get_current_key(),
            temperature=0,
            max_tokens=4096,
            extra_body={"reasoning_effort": "low"} 
        )
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        ragas_emb_wrapped = LangchainEmbeddingsWrapper(ragas_emb)

        f_metric = Faithfulness(llm=ragas_llm)
        f_metric.set_prompts(n_l_i_statement_prompt=CustomNLIPrompt())
        
        r_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb_wrapped, strictness=1)
        r_metric.question_generation_prompt = CustomRelevancePrompt()

        recall_metric = ContextRecall(llm=ragas_llm)
        recall_metric.context_recall_prompt = CustomRecallExtractionPrompt()

        single_dataset = Dataset.from_dict({
            "user_input": [item["question"]],
            "response": [item["answer"]],
            "retrieved_contexts": [item["contexts"]],
            "reference": [item["ground_truth"]]
        })

        try:
            res = evaluate(
                dataset=single_dataset,
                metrics=[
                    f_metric, 
                    r_metric, 
                    ContextPrecision(llm=ragas_llm), 
                    recall_metric
                ],
                llm=ragas_llm,
                embeddings=ragas_emb_wrapped,
                raise_exceptions=True,
                run_config=RunConfig(timeout=180, max_workers=1)
            )
            
            all_individual_results.append(res.to_pandas())
            i += 1
            retry_count[i] = 0  # Reset retry count on success
            print(f"✅ Success [{i}/{len(successful_results)}]")

        except Exception as e:
            err_msg = str(e)
            
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                print(f"🚨 Rate Limit. Swapping to Key {manager.current_index + 2}...")
                manager.rotate_key()
                time.sleep(15)
                
            elif "StringIO" in err_msg:
                retry_count[i] += 1
                
                if retry_count[i] <= 1:
                    # First retry - give it one more chance
                    print(f"⚠️ Parser glitch (attempt {retry_count[i]}/1). Retrying...")
                    time.sleep(5)
                else:
                    # Second failure - skip this sample
                    print(f"❌ Parser glitch persists after 1 retry. Skipping sample {i+1} ({entity_name})...")
                    i += 1
                    
            else:
                print(f"❌ Hard error on sample {i+1} ({entity_name}): {err_msg}")
                i += 1

    final_df = pd.concat(all_individual_results, ignore_index=True)
    summary = final_df.mean(numeric_only=True).to_dict()
    
    print("\n✅ Ragas evaluation complete!")
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
            'Metric': ['Faithfulness', 'Answer\nRelevancy', 'Context\nPrecision', 'Context\nRecall'],
            'Score': [
                ragas_scores.get('faithfulness', 0),
                ragas_scores.get('answer_relevancy', 0),
                ragas_scores.get('context_precision', 0),
                ragas_scores.get('context_recall', 0)
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
            "evaluation_model": "gpt-oss-20b"
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
            "context_recall": ("0.75", "Does context contain the answer? (Answer coverage)")
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
    responses_csv = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\agent_responses\agent_responses_pt1.csv"
    
    # Output directory
    output_dir = Path("data/chatbot/outputs/ragas_evaluation_results")
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
    
    print(f"\n📁 Generated Files:")
    print(f"  • Visualization: {output_dir}/evaluation_visualization.png")
    print(f"  • JSON Report: {json_path}")
    print(f"  • Markdown Report: {md_path}")
    
    print("\n✅ All evaluation files saved to:", output_dir.absolute())
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()