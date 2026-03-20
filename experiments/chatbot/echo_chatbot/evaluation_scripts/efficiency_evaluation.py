"""
Efficiency Evaluation with LangSmith Tracing + Key Rotation
Runs 132 test queries with automatic Groq API key rotation
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import os
from dotenv import load_dotenv

root_path = Path(r"C:\Uni\4th Year\GP\ECHO\experiments\chatbot\echo_chatbot\evaluation_scripts")
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

load_dotenv()


# Import base components
from evaluation_graph import (
    graph, ENTITY_CONFIG, SQL_TEMPLATE, PROMPTS,
    embedding_model, reranker
)
import evaluation_graph as eval_graph

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ============================================================================
# Groq Key Manager (Rotating Keys to Avoid Rate Limits)
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
        print(f"🔄 Rotating to API Key {self.current_index + 1}...")
        return self.get_current_key()


# ============================================================================
# Initialize Key Manager
# ============================================================================

def initialize_key_manager():
    """Load all Groq API keys from environment"""
    keys = [os.getenv(f"GROQ_API_KEY{i}") for i in range(1, 10)]
    valid_keys = [k for k in keys if k]
    
    if not valid_keys:
        raise ValueError("No Groq API keys found in .env file!")
    
    print(f"✅ Loaded {len(valid_keys)} Groq API keys")
    return GroqKeyManager(valid_keys)


# ============================================================================
# Initialize Entity Configuration with Key Rotation
# ============================================================================

def initialize_entity_config(entity_type: str, entity_name: str, key_manager: GroqKeyManager):
    """Set the 3 global variables + rebuild LLMs with current API key"""
    
    eval_graph.ENTITY_TYPE = entity_type
    eval_graph.ENTITY_NAME = entity_name
    
    cfg = ENTITY_CONFIG[entity_type]
    
    # Set SQL query for this entity type
    eval_graph.VECTOR_SQL = SQL_TEMPLATE.format(
        texts_table=cfg["texts_table"],
        entities_table=cfg["entities_table"],
        entity_id_col=cfg["entity_id_col"]
    )
    
    # Rebuild query rewriter LLM with current API key
    current_key = key_manager.get_current_key()
    
    eval_graph.query_rewriter_llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.2,
        max_tokens=1024,
        api_key=current_key,
        extra_body={"reasoning_effort": "default", "reasoning_format": "hidden"}
    )
    
    # Rebuild generator LLM with current API key
    eval_graph.generator_llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        api_key=current_key,
        extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"}
    )
    
    # Set rewrite chain with new LLM
    prompt_key = cfg["prompt_key"]
    eval_graph.rewrite_chain = (
        PromptTemplate.from_template(PROMPTS["rewrite_prompt"][prompt_key]) | 
        eval_graph.query_rewriter_llm | 
        StrOutputParser()
    )
    
    # Set prompt template for this entity
    eval_graph.llm_prompt_template = PromptTemplate.from_template(
        PROMPTS["assistant_persona"][prompt_key]
    )


# ============================================================================
# Run Efficiency Evaluation with Key Rotation
# ============================================================================

def run_efficiency_evaluation(csv_path: str):
    """
    Run all test queries with:
    - LangSmith tracing (automatic latency/token tracking)
    - Key rotation every 5 queries (prevent rate limits)
    """
    print("\n" + "="*80)
    print("🚀 EFFICIENCY EVALUATION WITH LANGSMITH + KEY ROTATION")
    print("="*80 + "\n")
    
    # Verify LangSmith is configured
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        print("❌ ERROR: LANGCHAIN_TRACING_V2 not set in .env")
        print("Add to .env: LANGCHAIN_TRACING_V2=true")
        return
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ ERROR: LANGCHAIN_API_KEY not set in .env")
        print("Add to .env: LANGCHAIN_API_KEY=lsv2_pt_...")
        return
    
    print("✅ LangSmith tracing enabled")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    
    # Initialize key manager
    key_manager = initialize_key_manager()
    print(f"🔑 Key rotation: every 5 queries\n")
    
    # Load test dataset
    print(f"Loading test dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} test cases")
    print(f"  • Unique entities: {df['entity_name'].nunique()}\n")
    
    results = []
    current_entity = None
    
    for idx, row in df.iterrows():
        # Rotate key every 5 queries
        if idx > 0 and idx % 5 == 0:
            key_manager.rotate_key()
        
        entity_key = (row['entity_type'], row['entity_name'])
        
        # Initialize entity config if changed OR when key rotates
        if current_entity != entity_key or (idx > 0 and idx % 5 == 0):
            if current_entity != entity_key:
                print(f"\n[SWITCHING ENTITY] {row['entity_type']}: {row['entity_name']}")
            initialize_entity_config(row['entity_type'], row['entity_name'], key_manager)
            current_entity = entity_key
        
        print(f"[{idx+1}/{len(df)}] {row['entity_name']}: {row['input'][:50]}...")
        
        start_time = time.time()
        
        try:
            # Config with metadata for LangSmith filtering
            config = {
                "configurable": {"thread_id": f"efficiency-eval-{idx}"},
                "metadata": {
                    "evaluation_type": "efficiency",
                    "entity_type": row['entity_type'],
                    "entity_name": row['entity_name'],
                    "query_id": idx,
                    "api_key_index": key_manager.current_index + 1,
                    "run_name": f"{row['entity_name']}: {row['input'][:50]}"
                }
            }
            
            # Run the graph (LangSmith automatically traces everything!)
            response = graph.invoke(
                {
                    "messages": [("user", row["input"])],
                    "query": row["input"],
                    "context": [],
                    "voice_mode": False
                },
                config=config
            )
            
            end_time = time.time()
            
            answer = response.get("response", "")
            
            results.append({
                "query_id": idx,
                "entity_type": row["entity_type"],
                "entity_name": row["entity_name"],
                "query": row["input"],
                "response": answer,
                "total_time": end_time - start_time,
                "api_key_used": key_manager.current_index + 1,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"  ✓ Success ({end_time - start_time:.2f}s) [Key {key_manager.current_index + 1}]\n")
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            
            # Check if rate limit error
            if "rate" in error_msg.lower() or "429" in error_msg:
                print(f"  ⚠️ Rate limit hit! Rotating key and retrying...")
                key_manager.rotate_key()
                initialize_entity_config(row['entity_type'], row['entity_name'], key_manager)
                time.sleep(10)  # Wait before retry
                # Don't increment idx - will retry this query
                continue
            else:
                print(f"  ✗ Error: {error_msg[:100]}\n")
                
                results.append({
                    "query_id": idx,
                    "entity_type": row["entity_type"],
                    "entity_name": row["entity_name"],
                    "query": row["input"],
                    "response": "",
                    "total_time": end_time - start_time,
                    "api_key_used": key_manager.current_index + 1,
                    "success": False,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Small delay between queries
        time.sleep(1.5)
    
    # Save results summary
    results_df = pd.DataFrame(results)
    output_dir = Path("efficiency_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"efficiency_results_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    successful = results_df['success'].sum()
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80 + "\n")
    
    print(f"📊 Results Summary:")
    print(f"  • Total queries: {len(results_df)}")
    print(f"  • Successful: {successful}")
    print(f"  • Failed: {len(results_df) - successful}")
    print(f"  • Success rate: {successful/len(results_df)*100:.1f}%")
    
    if successful > 0:
        success_df = results_df[results_df['success']]
        print(f"  • Avg time: {success_df['total_time'].mean():.2f}s")
        print(f"  • Min time: {success_df['total_time'].min():.2f}s")
        print(f"  • Max time: {success_df['total_time'].max():.2f}s")
        print(f"  • P50: {success_df['total_time'].quantile(0.50):.2f}s")
        print(f"  • P90: {success_df['total_time'].quantile(0.90):.2f}s")
        print(f"  • P99: {success_df['total_time'].quantile(0.99):.2f}s")
    
    print(f"\n💾 Results saved to: {output_path.absolute()}")
    
    print(f"\n🌐 View detailed metrics in LangSmith:")
    print(f"   https://smith.langchain.com/")
    print(f"   Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    
    print("\n📈 LangSmith Dashboard includes:")
    print("   ✓ Latency breakdown (rewriter/retriever/reranker/generator)")
    print("   ✓ Percentiles (P50, P90, P99)")
    print("   ✓ Token counts and TPS")
    print("   ✓ Waterfall charts per query")
    print("   ✓ Filter by entity type/name")
    print("   ✓ Export to CSV")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Path to your test dataset (132 queries)
    csv_path = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\evaluation_data\shrunk_dataset_132.csv"
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"❌ ERROR: Test dataset not found at {csv_path}")
        return
    
    run_efficiency_evaluation(csv_path)


if __name__ == "__main__":
    main()
