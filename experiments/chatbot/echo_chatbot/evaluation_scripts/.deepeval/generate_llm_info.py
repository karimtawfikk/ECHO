"""
Agent Response Collection Script
Runs the Ancient Egypt RAG chatbot on test dataset and saves all responses for later evaluation
"""

from pathlib import Path
import sys

root_path = Path(r"C:\Users\Zeyad\Desktop\4th Year\GP\ECHO\experiments\chatbot\echo_chatbot\chatbot_phases")
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import evaluation_graph
from evaluation_graph import graph, ENTITY_CONFIG, SQL_TEMPLATE, PROMPTS

import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================================
# Helper: Initialize Entity Configuration
# ============================================================================

def initialize_entity_config(entity_type: str, entity_name: str):
    """Set the 3 global variables that change per entity"""
    evaluation_graph.ENTITY_TYPE = entity_type
    evaluation_graph.ENTITY_NAME = entity_name
    
    cfg = ENTITY_CONFIG[entity_type]
    
    # Set SQL query for this entity type
    evaluation_graph.VECTOR_SQL = SQL_TEMPLATE.format(
        texts_table=cfg["texts_table"],
        entities_table=cfg["entities_table"],
        entity_id_col=cfg["entity_id_col"]
    )
    
    # Set rewrite chain for this entity (uses already-initialized query_rewriter_llm)
    prompt_key = cfg["prompt_key"]
    evaluation_graph.rewrite_chain = (
        PromptTemplate.from_template(PROMPTS["rewrite_prompt"][prompt_key]) | 
        evaluation_graph.query_rewriter_llm | 
        StrOutputParser()
    )
    
    # Set prompt template for this entity
    evaluation_graph.llm_prompt_template = PromptTemplate.from_template(
        PROMPTS["assistant_persona"][prompt_key]
    )


# ============================================================================
# Collect Agent Responses
# ============================================================================

def collect_agent_responses(csv_path: str) -> List[Dict[str, Any]]:
    """Run the chatbot on all test cases and collect responses"""
    print(f"\n{'='*80}")
    print("Collecting Agent Responses")
    print(f"{'='*80}\n")
    
    print(f"Loading test dataset from {csv_path}...")
    test_df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(test_df)} test cases")
    print(f"  • Unique entities: {test_df['entity_name'].nunique()}\n")
    
    results = []
    current_entity = None
    
    for idx, row in test_df.iterrows():
        entity_key = (row['entity_type'], row['entity_name'])
        
        if current_entity != entity_key:
            print(f"\n[SWITCHING ENTITY] Initializing {row['entity_type']}: {row['entity_name']}")
            initialize_entity_config(row['entity_type'], row['entity_name'])
            current_entity = entity_key
        
        print(f"[{idx+1}/{len(test_df)}] Testing: {row['entity_name']} - {row['input'][:60]}...")
        
        start_time = time.time()
        
        try:
            config = {"configurable": {"thread_id": f"collect-{idx}"}}
            
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
            contexts = response.get("context", [])
            
            if isinstance(contexts, str):
                contexts = [contexts]
            elif not isinstance(contexts, list):
                contexts = []
            
            results.append({
                "question": row["input"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row["expected_output"],
                "response_time": end_time - start_time,
                "entity_type": row["entity_type"],
                "entity_name": row["entity_name"],
                "success": True,
                "answer_length": len(answer.split()),
                "context_count": len(contexts)
            })
            
            print(f"  ✓ Response time: {end_time - start_time:.2f}s | Contexts: {len(contexts)}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                "question": row["input"],
                "answer": "",
                "contexts": [],
                "ground_truth": row["expected_output"],
                "response_time": 0,
                "entity_type": row["entity_type"],
                "entity_name": row["entity_name"],
                "success": False,
                "error": str(e),
                "answer_length": 0,
                "context_count": 0
            })
        
        time.sleep(1.2)
    
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*80}")
    print(f"Collection complete! {successful}/{len(results)} successful")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# Save to CSV
# ============================================================================

def save_responses_to_csv(results: List[Dict[str, Any]], output_dir: Path):
    """Save collected responses to CSV file"""
    print("\nSaving responses to CSV...")
    
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Convert contexts list to string for CSV storage
    df['contexts'] = df['contexts'].apply(lambda x: '|||'.join(x) if isinstance(x, list) else x)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"agent_responses_{timestamp}.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"  ✓ Saved {len(df)} responses to {output_path}")
    
    print(f"\n📊 Summary Statistics:")
    print(f"  • Total responses: {len(df)}")
    print(f"  • Successful: {df['success'].sum()}")
    print(f"  • Failed: {(~df['success']).sum()}")
    print(f"  • Avg response time: {df[df['success']]['response_time'].mean():.2f}s")
    print(f"  • Avg answer length: {df[df['success']]['answer_length'].mean():.0f} words")
    print(f"  • Avg context count: {df[df['success']]['context_count'].mean():.1f} chunks")
    
    return output_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" Agent Response Collection for Evaluation")
    print("="*80 + "\n")
    
    csv_path = r"C:\Users\Zeyad\Desktop\4th Year\GP\ECHO\data\chatbot\outputs\new_evaluation_data\eval_part_1.csv"
    output_dir = Path("data/chatbot/outputs/new_agent_responses")
    
    results = collect_agent_responses(csv_path)
    
    output_path = save_responses_to_csv(results, output_dir)
    
    print("\n" + "="*80)
    print(" COLLECTION COMPLETE!")
    print("="*80 + "\n")
    
    print(f"✅ Agent responses saved to: {output_path.absolute()}")
    print("\n💡 Next steps:")
    print("  1. Use this CSV file as input for Ragas evaluation")
    print("  2. Run evaluation 3 times on the same responses")
    print("  3. Compare metric stability across runs")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()