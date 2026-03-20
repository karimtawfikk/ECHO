import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))

import warnings
import os
import re
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_cloudflare import CloudflareWorkersAIEmbeddings

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine
import yaml

warnings.filterwarnings("ignore")
load_dotenv()

# ---------------------------------------------------------------------------
# Resources & Config (Kept same as Phase 6)
# ---------------------------------------------------------------------------
def load_resources():
    base_path = Path(__file__).parent.parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_template = f.read()
    with open(base_path / "evaluation_prompt_baseline.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return sql_template, prompts

SQL_TEMPLATE, PROMPTS = load_resources()

GROQ_API_KEY9 = os.getenv("GROQ_API_KEY9")
CF_WORKERSAI_ACCOUNTID = os.getenv("R2_ACCOUNT_ID")
CF_AI_API = os.getenv("CF_AI_API")

GROQ_GENERATOR_MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_DIM = 768

ENTITY_CONFIG = {
    "pharaoh": {
        "texts_table":    "pharaohs_texts",
        "entities_table": "pharaohs",
        "entity_id_col":  "pharaoh_id",
        "prompt_key":     "pharaoh",
        "name_key":       "pharaoh_name",
    },
    "landmark": {
        "texts_table":    "landmarks_texts",
        "entities_table": "landmarks",
        "entity_id_col":  "landmark_id",
        "prompt_key":     "landmark",
        "name_key":       "landmark_name",
    }
}


ENTITY_TYPE = None
ENTITY_NAME = None
VECTOR_SQL  = None
llm_prompt_template  = None

# ---------------------------------------------------------------------------
# State & Models
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str

embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)

generator_llm = ChatGroq(
    model_name=GROQ_GENERATOR_MODEL_NAME,
    temperature=0.7,
    api_key=GROQ_API_KEY9,
    max_tokens=4096,
    top_p=0.95,
    extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"}
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_embedding(text: str):
    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced     = embeddings[:EMBEDDING_DIM]
    norm       = np.linalg.norm(sliced)
    return (sliced / norm if norm > 0 else sliced).tolist()

# ---------------------------------------------------------------------------
# Nodes (Phase 1 Logic)
# ---------------------------------------------------------------------------
def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['query'])

    with Session(engine) as session:
        result = session.execute(
            text(VECTOR_SQL),
            {"entity_name": ENTITY_NAME,
             "embedding":   str(query_embedding)}
        )
        context = [row[0] for row in result]

    return {"context": context}

def generate_node(state: AgentState) -> dict:
    combined_context = "\n\n".join(state.get('context', []))
    

    history_str = "No previous conversation."
    name_key = ENTITY_CONFIG[ENTITY_TYPE]["name_key"]
    
    prompt = llm_prompt_template.format(**{
        name_key:       ENTITY_NAME,
        "context":      combined_context,
        "query":        state['query'],
        "chat_history": history_str,
    })

    print(f"\n{ENTITY_NAME}: ", end="", flush=True)
    full_content = ""

    for chunk in generator_llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_content += chunk.content

    print("\n🏛️ [SOURCE]: Final answer generated using RAG context ONLY.")
    
    return {
        "messages": [AIMessage(content=full_content, name="generator_response")],
        "response": full_content
    }

# ---------------------------------------------------------------------------
# Graph Construction (Linear Phase 1)
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()
graph = workflow.compile()

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    global ENTITY_TYPE, ENTITY_NAME, VECTOR_SQL, llm_prompt_template

    print("--- PHASE 1: NAIVE RAG BASELINE ---")
    ENTITY_TYPE = input("Entity type (pharaoh/landmark): ").strip().lower()
    ENTITY_NAME = input(f"Enter {ENTITY_TYPE} name: ").strip()

    cfg = ENTITY_CONFIG[ENTITY_TYPE]
    VECTOR_SQL = SQL_TEMPLATE.format(
        texts_table=cfg["texts_table"],
        entities_table=cfg["entities_table"],
        entity_id_col=cfg["entity_id_col"]
    )
    prompt_key          = cfg["prompt_key"]
    llm_prompt_template = PromptTemplate.from_template(PROMPTS["assistant_persona"][prompt_key])
    config     = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['q', 'quit']: break

        graph.invoke({
            "messages":[("user", user_input)],
             "query":user_input,
             "context":[],
        },config=config)

if __name__ == "__main__":
    main()