import sys
import os
import yaml
import numpy as np
import warnings
from pathlib import Path
from typing import TypedDict, Annotated, List, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine

warnings.filterwarnings("ignore")
load_dotenv()

# --- CONFIG & RESOURCES ---
def load_resources():
    base_path = Path(__file__).parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()

GROQ_API_KEY1 = os.getenv("GROQ_API_KEY1") 
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2")
JINA_API_KEY = os.getenv("JINA_API_KEY")

GROQ_GENERATOR_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_QUERY_RERWRITER_MODEL_NAME = "qwen/qwen3-32b"
ENTITY_NAME = "Ramesses II"
EMBEDDING_DIM = 768
TOP_K = 3

# --- STATE ---
class AgentState(TypedDict):
    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: List[str]

# --- TOOLS & MODELS ---
embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=os.getenv("R2_ACCOUNT_ID"),
    api_token=os.getenv("CF_AI_API"),
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)

reranker = JinaRerank(
    model="jina-reranker-v3",
    top_n=TOP_K,
    jina_api_key=JINA_API_KEY
)

# Option A Tool Setup
search_tool = TavilySearch(max_results=2)
tools = [search_tool]
tool_node = ToolNode(tools)

# Rewriter LLM
query_rewriter_llm = ChatGroq(
    model_name=GROQ_QUERY_RERWRITER_MODEL_NAME,
    temperature=0.2,
    api_key=GROQ_API_KEY1,
    extra_body={
        "reasoning_effort": "default",
        "reasoning_format": "hidden" 
    } 
)

# Generator LLM (with Tool Binding)
llm = ChatGroq(
    model_name=GROQ_GENERATOR_MODEL_NAME,
    api_key=GROQ_API_KEY2,
    temperature=0.8,
    top_p=0.95,
    extra_body={
        "reasoning_effort": "medium",
        "reasoning_format": "hidden"
    }
)
llm_with_tools = llm.bind_tools(tools)

# --- CHAINS ---
rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
rewrite_chain = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()

# --- NODES ---

def rewrite_node(state: AgentState) -> dict:
    print("--- REWRITING QUERY ---")
    clean_dialouge = [
        msg for msg in state['messages'][:-1] 
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "search_query"
    ]
    history_window = clean_dialouge[-10:] if clean_dialouge else []
    
    dialogue = []
    for msg in history_window:
        role = "User" if isinstance(msg, HumanMessage) else "Search Query"
        dialogue.append(f"{role}: {msg.content}")

    history_str = "\n".join(dialogue) if dialogue else "No history yet."

    search_q = rewrite_chain.invoke({
        "query": state['query'],
        "pharaoh_name": ENTITY_NAME,
        "chat_history": history_str
    }).replace("Search Query:", "").strip()

    return {"messages": [AIMessage(content=search_q, name="search_query")], "search_query": search_q}

def retrieve_node(state: AgentState) -> dict:
    print("--- RETRIEVING FROM DATABASE ---")
    raw_emb = np.array(embedding_model.embed_query(state['search_query']))
    norm = np.linalg.norm(raw_emb[:EMBEDDING_DIM])
    query_embedding = (raw_emb[:EMBEDDING_DIM] / norm).tolist() if norm > 0 else raw_emb[:EMBEDDING_DIM].tolist()
    
    with Session(engine) as session:
        result = session.execute(text(VECTOR_SQL), {
            "pharoah_name": ENTITY_NAME,
            "embedding": str(query_embedding)
        })
        context = [row[0] for row in result]
        
    return {"context": context}

def rerank_node(state: AgentState) -> dict:
    print("--- RERANKING CONTEXT ---")
    
    docs = [Document(page_content=chunk) for chunk in state['context']]
    reranked = reranker.compress_documents(docs, query=state['search_query'])

    return {"context": [doc.page_content for doc in reranked]}

def generate_node(state: AgentState):
    """Option A: The Pharaoh decides if he needs the Search Tool."""
    print(f"--- {ENTITY_NAME} REASONING (AGENTIC) ---")
    
    # 1. Prepare conversation history for the persona
    clean_dialogue = [
        msg for msg in state['messages'] 
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "generator_response"
    ]
    history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else []
    
    dialogue = []
    for msg in history_window:
        role = "User" if isinstance(msg, HumanMessage) else ENTITY_NAME
        dialogue.append(f"{role}: {msg.content}")
    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."

    # 2. Format the Persona Prompt
    system_prompt = PROMPTS['assistant_persona'].format(
        pharaoh_name=ENTITY_NAME,
        context="\n\n".join(state['context']),
        query=state['query'],
        chat_history=history_str
    )
    
    # 3. Invoke LLM with Tools
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    response = llm_with_tools.invoke(messages)
    
    # Return as a generator_response to maintain your history logic
    return {"messages": [AIMessage(content=response.content, tool_calls=response.tool_calls, name="generator_response")]}

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)

workflow.add_node("rewriter", rewrite_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("reranker", rerank_node)
workflow.add_node("gen", generate_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "rewriter")
workflow.add_edge("rewriter", "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_edge("reranker", "pharaoh")

# Agentic Branch: Does Pharaoh want tools or is he done?
workflow.add_conditional_edges(
    "pharaoh",
    tools_condition, 
)

# If tools were used, go back to Pharaoh for final synthesis
workflow.add_edge("tools", "pharaoh")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# --- MAIN LOOP ---
def main():
    print(f"Agentic Pharaoh RAG Ready (Phase 4 - Option A)")
    config = {"configurable": {"thread_id": "pharaoh_v4"}}

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']: break
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)], 
            "query": user_input,
            "context": []
        }
        
        # Using stream to show the steps
        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                # Only print if it's a final text response (not a tool call)
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls and last_msg.name == "generator_response":
                    print(f"\n{ENTITY_NAME}: {last_msg.content}")

if __name__ == "__main__":
    main()