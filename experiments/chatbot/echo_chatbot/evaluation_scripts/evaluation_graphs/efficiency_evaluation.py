"""
Efficiency Evaluation with LangSmith Tracing + Key Rotation
Runs 132 test queries - Pure execution, no local stats
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import os
from dotenv import load_dotenv

root_path = Path(r"C:\Uni\4th Year\GP\ECHO\experiments\chatbot\echo_chatbot\evaluation_scripts\evaluation_graphs")
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

load_dotenv()

# Import base components
from echo_agent_evaluation_graph import (
    graph, ENTITY_CONFIG, SQL_TEMPLATE, PROMPTS
)
import echo_agent_evaluation_graph as eval_graph

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine


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
        print(f"🔄 Rotating to API Key {self.current_index + 1}...")
        return self.get_current_key()


# ============================================================================
# Initialize Key Manager
# ============================================================================

def initialize_key_manager():
    """Load all Groq API keys from environment"""
    keys = [os.getenv(f"GROQ_API_KEY{i}") for i in range(1, 11)]
    valid_keys = [k for k in keys if k]
    
    if not valid_keys:
        raise ValueError("No Groq API keys found in .env file!")
    
    print(f"✅ Loaded {len(valid_keys)} Groq API keys")
    return GroqKeyManager(valid_keys)


# ============================================================================
# Initialize Entity Configuration + Cache Entity ID
# ============================================================================

def initialize_entity_config(entity_type: str, entity_name: str, key_manager: GroqKeyManager):
    """Set globals + cache entity_id + rebuild LLMs with current API key"""
    
    eval_graph.ENTITY_TYPE = entity_type
    eval_graph.ENTITY_NAME = entity_name
    
    cfg = ENTITY_CONFIG[entity_type]
    
    # Cache entity ID (one-time lookup per entity)
    table_name = "pharaohs" if entity_type == "pharaoh" else "landmarks"
    
    with Session(engine) as session:
        result = session.execute(
            text(f"SELECT id FROM {table_name} WHERE name = :name"),
            {"name": entity_name}
        ).fetchone()
        
        if result:
            eval_graph.ENTITY_ID = result[0]
        else:
            raise ValueError(f"Entity '{entity_name}' not found in database!")
    
    # Set optimized SQL query
    eval_graph.VECTOR_SQL = SQL_TEMPLATE.format(
        texts_table=cfg["texts_table"],
        entity_id_col=cfg["entity_id_col"]
    )
    
    # Rebuild LLMs with current API key
    current_key = key_manager.get_current_key()
    
    eval_graph.query_rewriter_llm = ChatGroq(
        model_name="qwen/qwen3-32b",
        temperature=0.2,
        max_tokens=1024,
        api_key=current_key,
        extra_body={"reasoning_effort": "none", "reasoning_format": "hidden"}
    )
    
    eval_graph.generator_llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        api_key=current_key,
        extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"}
    )
    
    # Set rewrite chain
    prompt_key = cfg["prompt_key"]
    eval_graph.rewrite_chain = (
        PromptTemplate.from_template(PROMPTS["rewrite_prompt"][prompt_key]) | 
        eval_graph.query_rewriter_llm | 
        StrOutputParser()
    )
    
    # Set prompt template
    eval_graph.llm_prompt_template = PromptTemplate.from_template(
        PROMPTS["assistant_persona"][prompt_key]
    )


# ============================================================================
# Run Efficiency Evaluation
# ============================================================================

def run_efficiency_evaluation(csv_path: str):
    """Run 132 queries with LangSmith tracing"""
    
    print("\n" + "="*80)
    print("🚀 EFFICIENCY EVALUATION - OPTIMIZED SQL + KEY ROTATION")
    print("="*80 + "\n")
    
    # Verify LangSmith
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        print("❌ ERROR: LANGCHAIN_TRACING_V2 not set in .env")
        return
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ ERROR: LANGCHAIN_API_KEY not set in .env")
        return
    
    print("✅ LangSmith tracing enabled")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")
    
    # Initialize key manager
    key_manager = initialize_key_manager()
    print(f"🔑 Key rotation: every 5 queries\n")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} test queries\n")
    
    current_entity = None
    
    for idx, row in df.iterrows():
        # Rotate key every 5 queries
        if idx > 0 and idx % 5 == 0:
            key_manager.rotate_key()
        
        entity_key = (row['entity_type'], row['entity_name'])
        
        # Initialize entity config if changed OR when key rotates
        if current_entity != entity_key or (idx > 0 and idx % 5 == 0):
            if current_entity != entity_key:
                print(f"\n[ENTITY] {row['entity_type']}: {row['entity_name']}")
            initialize_entity_config(row['entity_type'], row['entity_name'], key_manager)
            current_entity = entity_key
        
        print(f"[{idx+1}/{len(df)}] {row['input'][:60]}...")
        
        try:
            # Config with metadata for LangSmith
            config = {
                "configurable": {"thread_id": f"efficiency-eval-optimized-{idx}"},
                "metadata": {
                    "evaluation_type": "efficiency_optimized",
                    "entity_type": row['entity_type'],
                    "entity_name": row['entity_name'],
                    "query_id": idx,
                    "run_name": f"{row['entity_name']}: {row['input'][:50]}"
                }
            }
            
            # Run graph
            graph.invoke(
                {
                    "messages": [("user", row["input"])],
                    "query": row["input"],
                    "context": [],
                    "voice_mode": False
                },
                config=config
            )
            
            print(f"  ✓ Done\n")
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limits
            if "rate" in error_msg.lower() or "429" in error_msg:
                print(f"  ⚠️ Rate limit! Rotating key...")
                key_manager.rotate_key()
                initialize_entity_config(row['entity_type'], row['entity_name'], key_manager)
                time.sleep(10)
                continue
            else:
                print(f"  ✗ Error: {error_msg[:100]}\n")
        
        time.sleep(1.5)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80 + "\n")
    
    print(f"🌐 View metrics in LangSmith:")
    print(f"   https://smith.langchain.com/")
    print(f"   Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print(f"   Filter by: evaluation_type='efficiency_optimized'\n")


# ============================================================================
# Main
# ============================================================================

def main():
    csv_path = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\evaluation_data\shrunk_dataset_132.csv"
    
    if not Path(csv_path).exists():
        print(f"❌ ERROR: Dataset not found at {csv_path}")
        return
    
    run_efficiency_evaluation(csv_path)


if __name__ == "__main__":
    main()