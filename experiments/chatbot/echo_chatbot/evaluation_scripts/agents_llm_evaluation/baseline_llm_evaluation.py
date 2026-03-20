"""
LLM-Only Evaluation Script (No RAG - No Context)
Evaluates Answer Relevancy and Answer Accuracy only
"""

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

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.prompt import PydanticPrompt
from ragas.metrics import AnswerRelevancy, AnswerAccuracy
from ragas.metrics._answer_relevance import ( 
    ResponseRelevanceInput, 
    ResponseRelevanceOutput
)
from ragas.run_config import RunConfig

# ============================================================================
# Embeddings
# ============================================================================

ragas_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# ============================================================================
# Custom Answer Relevancy Prompt
# ============================================================================

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
    ]


# ============================================================================
# Load LLM-Only Responses
# ============================================================================

def load_agent_responses(csv_path: str) -> List[Dict[str, Any]]:
    """Load pre-collected LLM-only responses from CSV"""
    print(f"\nLoading LLM-only responses from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    results = df.to_dict('records')
    
    print(f"  ✓ Loaded {len(results)} responses")
    print(f"  • Successful: {sum(1 for r in results if r.get('success', False))}")
    print(f"  • Unique entities: {df['entity_name'].nunique()}")
    
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


# ============================================================================
# Extract First Paragraph
# ============================================================================

def extract_first_paragraph(text):
    if not text:
        return text
    
    text = text.replace('**', '').strip()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) > 1:
        return paragraphs[0]
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs[0] if paragraphs else text


# ============================================================================
# Compute Ragas Metrics (Answer Relevancy + Answer Accuracy Only)
# ============================================================================

def compute_ragas_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute Answer Relevancy and Answer Accuracy only (no context-based metrics)"""
    print("\n" + "="*80)
    print("Computing LLM-Only Metrics (Answer Relevancy + Answer Accuracy)")
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

    all_individual_results = []
    i = 0
    retry_count = {} 

    print(f"🚀 Evaluating {len(successful_results)} responses")
    print(f"🔑 Using {len(manager.keys)} API keys with rotation every 5 samples\n")

    while i < len(successful_results):
        if i > 0 and i % 5 == 0:
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

        # NO retrieved_contexts field for LLM-only
        single_dataset = Dataset.from_dict({
            "user_input": [item["question"]],
            "response": [clean_answer],
            "reference": [clean_ground_truth]
        })

        try:
            print(f"  → Calculating Answer Relevancy & Answer Accuracy...")
            
            r_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb_wrapped, strictness=1)
            r_metric.question_generation_prompt = CustomRelevancePrompt()

            core_res = evaluate(
                dataset=single_dataset,
                metrics=[r_metric, AnswerAccuracy(llm=ragas_llm)],
                llm=ragas_llm,
                embeddings=ragas_emb_wrapped,
                run_config=RunConfig(timeout=180, max_workers=1)
            )
            core_df = core_res.to_pandas()
            
            final_row = core_df.iloc[0].to_dict()
            
            print(f"  → Results: Relevancy:{final_row['answer_relevancy']:.2f} | Accuracy:{final_row['nv_accuracy']:.2f}")
            
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
    """Compute custom performance metrics (no context-related metrics)"""
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
    
    print("  ✓ Custom metrics computed successfully!")
    return metrics


# ============================================================================
# Generate Visualizations
# ============================================================================

def generate_visualizations(results: List[Dict], ragas_scores: Dict, custom_metrics: Dict, output_dir: Path):
    """Generate visualization charts (no context-related charts)"""
    print("\nGenerating visualizations...")
    
    output_dir.mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ancient Egypt LLM-Only - Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Answer Relevancy + Answer Accuracy
    if ragas_scores:
        metrics_df = pd.DataFrame({
            'Metric': ['Answer\nRelevancy', 'Answer\nAccuracy'],
            'Score': [
                ragas_scores.get('answer_relevancy', 0),
                ragas_scores.get('nv_accuracy', 0)
            ]
        })
        sns.barplot(data=metrics_df, x='Metric', y='Score', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('LLM-Only Metrics', fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
        axes[0, 0].legend()
    
    # Plot 2: Response Time Distribution
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        times = [r.get('response_time', 0) for r in successful_results]
        axes[0, 1].hist(times, bins=20, edgecolor='black', color='skyblue')
        axes[0, 1].set_title('Response Time Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=np.mean(times), color='r', linestyle='--', label=f'Mean: {np.mean(times):.2f}s')
        axes[0, 1].legend()
    
    # Plot 3: Success Rate by Entity Type
    entity_perf = custom_metrics.get('entity_type_performance', {})
    if entity_perf:
        etypes = list(entity_perf.keys())
        scores = list(entity_perf.values())
        axes[1, 0].bar(etypes, scores, color='coral', edgecolor='black')
        axes[1, 0].set_title('Success Rate by Entity Type', fontweight='bold')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: Performance Summary Table
    axes[1, 1].axis('off')
    summary_data = [
        ['Total Queries', f"{custom_metrics['total_queries']}"],
        ['Success Rate', f"{custom_metrics['success_rate']*100:.1f}%"],
        ['Avg Response Time', f"{custom_metrics['avg_response_time']:.2f}s"],
    ]
    if ragas_scores:
        summary_data.extend([
            ['Answer Relevancy', f"{ragas_scores.get('answer_relevancy', 0):.3f}"],
            ['Answer Accuracy', f"{ragas_scores.get('nv_accuracy', 0):.3f}"],
        ])
    
    table = axes[1, 1].table(cellText=summary_data, cellLoc='left', loc='center',
                             colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Performance Summary', fontweight='bold')
    
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
            "evaluation_model": "kimi",
            "evaluation_type": "LLM-Only (No RAG)"
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
    
    md = f"""# Ancient Egypt LLM-Only - Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Model:** Kimi K2 Instruct  
**Test Cases:** {len(results)}  
**Evaluation Type:** LLM-Only (No RAG - Baseline)

---

## Executive Summary

This evaluation assesses the LLM-only baseline (no retrieval) across {len(results)} test cases covering {len(set(r.get('entity_name', '') for r in results))} unique entities (pharaohs and landmarks).

### Key Findings

"""
    
    if ragas_scores:
        avg_score = np.mean(list(ragas_scores.values()))
        md += f"- **Overall Performance:** {avg_score:.2f}/1.00\n"
    
    md += f"- **System Success Rate:** {custom_metrics['success_rate']*100:.1f}%\n"
    md += f"- **Average Response Time:** {custom_metrics['avg_response_time']:.2f} seconds\n"
    
    md += "\n---\n\n## LLM-Only Metrics\n\n"
    
    if ragas_scores:
        md += "| Metric | Score | Target | Status | Interpretation |\n"
        md += "|--------|-------|--------|--------|----------------|\n"
        
        metrics_info = {
            "answer_relevancy": ("0.80", "Did it answer the question? (Answer vs Question)"),
            "nv_accuracy": ("0.80", "How accurate is the answer vs ground truth? (Answer vs Reference)")
        }
        
        for metric_name, score in ragas_scores.items():
            display_name = "Answer Relevancy" if metric_name == "answer_relevancy" else "Answer Accuracy"
            target, description = metrics_info.get(metric_name, ("0.75", "N/A"))
            status = "✓ Pass" if score >= float(target) else "✗ Needs Improvement"
            md += f"| {display_name} | {score:.3f} | {target} | {status} | {description} |\n"
    
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
    print(" LLM-Only Evaluation (No RAG - Baseline)")
    print("="*80 + "\n")
    
    # Input: LLM-only responses CSV
    responses_csv = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\responses\qwen_3_32b_responses\full_agent_response.csv"
    
    # Output directory
    output_dir = Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\ragas_evaluation_results\qwen_3_32b")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load LLM-only responses
    results = load_agent_responses(responses_csv)
    
    # Step 2: Compute Ragas metrics (Answer Relevancy + Answer Accuracy only)
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
        print(f"\n📈 LLM-Only Metrics:")
        print(f"  • Answer Relevancy: {ragas_scores.get('answer_relevancy', 0):.3f}")
        print(f"  • Answer Accuracy: {ragas_scores.get('nv_accuracy', 0):.3f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  • Visualization: {output_dir}/evaluation_visualization.png")
    print(f"  • JSON Report: {json_path}")
    print(f"  • Markdown Report: {md_path}")
    
    print("\n✅ All evaluation files saved to:", output_dir.absolute())
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()