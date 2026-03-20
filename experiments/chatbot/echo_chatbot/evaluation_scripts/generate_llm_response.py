"""
LLM-Only Response Collection Script (No RAG)
Tests LLM's knowledge without retrieval - baseline evaluation
"""

from pathlib import Path
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""


load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# Prompts (No Context, No Chat History)
# ============================================================================

PROMPTS = {
    "pharaoh": """# THE SOVEREIGN IDENTITY
You are {pharaoh_name}, speaking from your legacy in ancient Egypt.

Your goal is to educate the user with accurate and concise historical facts from your knowledge.

**IMPORTANT**: Answer based on your general knowledge and training data about ancient Egypt. Do not make up facts if you're unsure.


# MULTILINGUAL MANDATE
- You must respond in the SAME LANGUAGE as the User's Query.
- If the user asks in Arabic, respond in Arabic. If in English, respond in English and so on for the other languages.

# MANDATORY RESPONSE STRUCTURE
Structure every response in 2 parts:
- **Paragraph 1**: Directly answer the question with concrete facts only (dates, names, events, locations). No elaboration yet.
- **Remaining Paragraphs**: Expand on the WHY and HOW in an immersive, first-person voice.

# GUIDELINES
- **First Person**: Speak in 1st person perspective, you are {pharaoh_name} speaking.
- **Do not be a "helpful assistant."** Be {pharaoh_name} sharing memories.
- Present the information clearly and confidently.
- Don't output **Paragraph 1** or **Remaining Paragraphs**, answer directly.

User Query: {query}""",

    "landmark": """# THE MONUMENTAL IDENTITY
You are {landmark_name}, an ancient monument of Egypt that has stood through centuries of history.
You speak as a timeless structure whose stones remember the past.

Your goal is to educate the user with accurate historical facts about yourself from your knowledge.

**IMPORTANT**: Answer based on your general knowledge and training data about ancient Egypt. Do not make up facts if you're unsure.


# MULTILINGUAL MANDATE
- You must respond in the SAME LANGUAGE as the User's Query.
- If the user asks in Arabic, respond in Arabic. If in English, respond in English and so on for the other languages.

# MANDATORY RESPONSE STRUCTURE
Structure every response in 2 parts:
- **Paragraph 1**: Directly answer the question with concrete facts only (dates, names, events, locations). No elaboration yet.
- **Remaining Paragraphs**: Expand on the WHY and HOW in an immersive, first-person voice.

# GUIDELINES
- **First Person**: Speak in 1st person perspective, you are {landmark_name} speaking.
- **Do not be a "helpful assistant."** Speak as {landmark_name} sharing the memories carved into your stones.
- Present the information clearly and confidently.
- Don't output **Paragraph 1** or **Remaining Paragraphs**, answer directly.

User Query: {query}"""
}


# ============================================================================
# LLM Setup
# ============================================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # gpt-120b (actual model name on Groq)
    api_key=os.getenv("GROQ_API_KEY1"),
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
    streaming=True
)


# ============================================================================
# Collect LLM-Only Responses
# ============================================================================

def collect_llm_only_responses(csv_path: str) -> List[Dict[str, Any]]:
    """Run LLM without RAG on all test cases and collect responses"""
    print(f"\n{'='*80}")
    print("Collecting LLM-Only Responses (No RAG)")
    print(f"{'='*80}\n")
    
    print(f"Loading test dataset from {csv_path}...")
    test_df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(test_df)} test cases")
    print(f"  • Unique entities: {test_df['entity_name'].nunique()}\n")
    
    results = []
    
    for idx, row in test_df.iterrows():
        entity_type = row['entity_type']
        entity_name = row['entity_name']
        query = row['input']
        
        print(f"[{idx+1}/{len(test_df)}] {entity_name} - {query[:60]}...")
        
        # Select prompt template based on entity type
        prompt_template = PROMPTS[entity_type]
        
        # Format prompt with entity name and query
        if entity_type == "pharaoh":
            formatted_prompt = prompt_template.format(
                pharaoh_name=entity_name,
                query=query
            )
        else:  # landmark
            formatted_prompt = prompt_template.format(
                landmark_name=entity_name,
                query=query
            )
        
        start_time = time.time()
        
        try:
            # Stream response token by token
            answer_chunks = []
            print(f"  → Streaming response: ", end="", flush=True)
            
            for chunk in llm.stream(formatted_prompt):
                token = chunk.content
                answer_chunks.append(token)
                print(token, end="", flush=True)
            
            print()  # New line after streaming
            
            answer = "".join(answer_chunks)
            end_time = time.time()
            
            results.append({
                "question": query,
                "answer": answer,
                "ground_truth": row["expected_output"],
                "response_time": end_time - start_time,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "success": True,
                "answer_length": len(answer.split())
            })
            
            print(f"  ✓ Response time: {end_time - start_time:.2f}s | Length: {len(answer.split())} words\n")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
            results.append({
                "question": query,
                "answer": "",
                "ground_truth": row["expected_output"],
                "response_time": 0,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "success": False,
                "error": str(e),
                "answer_length": 0
            })
        
        time.sleep(1.2)  # Rate limiting
    
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
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"llm_only_responses_{timestamp}.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"  ✓ Saved {len(df)} responses to {output_path}")
    
    print(f"\n📊 Summary Statistics:")
    print(f"  • Total responses: {len(df)}")
    print(f"  • Successful: {df['success'].sum()}")
    print(f"  • Failed: {(~df['success']).sum()}")
    print(f"  • Avg response time: {df[df['success']]['response_time'].mean():.2f}s")
    print(f"  • Avg answer length: {df[df['success']]['answer_length'].mean():.0f} words")
    
    return output_path


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" LLM-Only Response Collection (Baseline - No RAG)")
    print("="*80 + "\n")
    
    # Input CSV
    csv_path = r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\evaluation_data\eval_part_2.csv"
    
    # Output directory
    output_dir = Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\outputs\echo_agent_evaluation\llama_3_70b_responses")
    
    # Collect responses
    results = collect_llm_only_responses(csv_path)
    
    # Save to CSV
    output_path = save_responses_to_csv(results, output_dir)
    
    print("\n" + "="*80)
    print(" COLLECTION COMPLETE!")
    print("="*80 + "\n")
    
    print(f"✅ LLM-only responses saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()