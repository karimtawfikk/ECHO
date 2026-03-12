import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))

import warnings
import os
import re
import io
import asyncio
import threading
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

import edge_tts
from langdetect import detect

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine

import yaml

warnings.filterwarnings("ignore")
load_dotenv()


# ---------------------------------------------------------------------------
# CONFIGURATION - Change these to talk to different entities
# ---------------------------------------------------------------------------

ENTITY_TYPE = "pharaoh"
ENTITY_NAME = "Khufu"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def load_resources():
    base_path = Path(__file__).parent.parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_template = f.read()
    with open(base_path / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return sql_template, prompts

SQL_TEMPLATE, PROMPTS = load_resources()


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

GROQ_API_KEY1          = os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2          = os.getenv("GROQ_API_KEY2")
CF_WORKERSAI_ACCOUNTID = os.getenv("R2_ACCOUNT_ID")
CF_AI_API              = os.getenv("CF_AI_API")
JINA_API_KEY           = os.getenv("JINA_API_KEY")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_GENERATOR_MODEL_NAME      = "openai/gpt-oss-120b"
GROQ_QUERY_REWRITER_MODEL_NAME = "qwen/qwen3-32b"
EDGE_TTS_RATE                  = "+9%"
EDGE_TTS_PITCH                 = "-10Hz"
TOP_K                          = 3
EMBEDDING_DIM                  = 768

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

LANG_TO_VOICE = {
    "en": "en-US-ChristopherNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "ar": "ar-EG-ShakirNeural",
    "de": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
}
DEFAULT_VOICE = "en-US-ChristopherNeural"


# State
class AgentState(TypedDict):
    query:        str
    search_query: str
    messages:     Annotated[list, add_messages]
    context:      List[str]
    response:     str
    voice_mode:   bool


# Build entity-specific configurations
cfg            = ENTITY_CONFIG[ENTITY_TYPE]
VECTOR_SQL     = SQL_TEMPLATE.format(
    texts_table=cfg["texts_table"],
    entities_table=cfg["entities_table"],
    entity_id_col=cfg["entity_id_col"]
)
prompt_key     = cfg["prompt_key"]
name_key       = cfg["name_key"]


# Models & Tools
embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)

reranker = JinaRerank(
    model="jina-reranker-v3",
    top_n=TOP_K,
    jina_api_key=JINA_API_KEY
)

search_tool = TavilySearchResults(max_results=3)
tools       = [search_tool]
tool_node   = ToolNode(tools=tools)

query_rewriter_llm = ChatGroq(
    model_name=GROQ_QUERY_REWRITER_MODEL_NAME,
    temperature=0.2,
    max_tokens=1024,
    api_key=GROQ_API_KEY1,
    extra_body={"reasoning_effort": "default", "reasoning_format": "hidden"}
)

generator_llm = ChatGroq(
    model_name=GROQ_GENERATOR_MODEL_NAME,
    temperature=0.8,
    max_tokens=4096,
    top_p=0.95,
    api_key=GROQ_API_KEY1,
    extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"}
).bind_tools(tools)

rewrite_chain       = PromptTemplate.from_template(PROMPTS["rewrite_prompt"][prompt_key]) | query_rewriter_llm | StrOutputParser()
llm_prompt_template = PromptTemplate.from_template(PROMPTS["assistant_persona"][prompt_key])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str):
    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced     = embeddings[:EMBEDDING_DIM]
    norm       = np.linalg.norm(sliced)
    return (sliced / norm if norm > 0 else sliced).tolist()


def clean_for_tts(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    text = re.sub(r'\n+',           ' ',   text).strip()
    return text


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def rewrite_node(state: AgentState) -> dict:
    clean_dialogue = [
        msg for msg in state['messages'][:-1]
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "search_query"
    ]
    history_window = clean_dialogue[-10:] if clean_dialogue else []

    dialogue = []
    for msg in history_window:
        if isinstance(msg, HumanMessage):
            dialogue.append(f"User: {msg.content}")
        elif getattr(msg, "name", None) == "search_query":
            dialogue.append(f"Search Query: {msg.content}")

    history_str = "\n".join(dialogue) if dialogue else "No history yet."
    
    search_q = rewrite_chain.invoke({
        name_key:       ENTITY_NAME,
        "chat_history": history_str,
        "query":        state["query"]
    }).replace("Search Query:", "").strip()

    if not search_q:
        search_q = state["query"]
    
    print(f"[REWRITER]: {search_q}")
    return {
        "messages":     [AIMessage(content=search_q, name="search_query")],
        "search_query": search_q
    }


def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['search_query'])

    with Session(engine) as session:
        result = session.execute(
            text(VECTOR_SQL),
            {"entity_name": ENTITY_NAME,
             "embedding":   str(query_embedding)}
        )
        context = [row[0] for row in result]

    return {"context": context}


def rerank_node(state: AgentState) -> dict:
    docs     = [Document(page_content=chunk) for chunk in state['context']]
    reranked = reranker.compress_documents(docs, query=state['search_query'])
    return {"context": [doc.page_content for doc in reranked]}


def generate_node(state: AgentState) -> dict:
    if "OUT_OF_SCOPE" in state.get('search_query', ''):
        response_text = "I'm sorry but you speak of a time that is not mine. My eyes see only the borders of my own reign." \
            if ENTITY_TYPE == "pharaoh" else \
            "I'm sorry, that lies beyond what my stones remember."
        print(f"\n{ENTITY_NAME}: {response_text}")
        return {
            "messages": [AIMessage(content=response_text, name="irrelevant_query")],
            "response": response_text
        }

    # FIX: Handle both HumanMessage objects and tuples
    last_human_index = -1
    for i, msg in enumerate(state['messages']):
        # Check if it's a HumanMessage object OR a tuple starting with "user"
        if isinstance(msg, HumanMessage) or (isinstance(msg, tuple) and msg[0] == "user"):
            last_human_index = i
    
    print(f"\n[DEBUG] Total messages: {len(state['messages'])}")
    print(f"[DEBUG] last_human_index: {last_human_index}")

    current_turn_messages  = state['messages'][last_human_index:] if last_human_index != -1 else []
    has_searched           = any(isinstance(msg, ToolMessage) for msg in current_turn_messages)
    current_search_results = [msg.content for msg in current_turn_messages if isinstance(msg, ToolMessage)]
    
    print(f"[DEBUG] current_turn_messages count: {len(current_turn_messages)}")
    print(f"[DEBUG] has_searched: {has_searched}")
    print(f"[DEBUG] Message types in current turn: {[type(m).__name__ for m in current_turn_messages]}")
    
    combined_context = current_search_results + state['context'] if has_searched else state['context']

    clean_dialogue = [
        msg for msg in state['messages']
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "generator_response"
    ]
    history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else []

    dialogue = []
    for msg in history_window:
        if isinstance(msg, HumanMessage):
            dialogue.append(f"User: {msg.content}")
        elif getattr(msg, "name", None) == "generator_response":
            dialogue.append(f"{ENTITY_NAME}: {msg.content}")

    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."
   
    extra_instruction = ""
    if has_searched:
        extra_instruction = (
            "\n\nIMPORTANT: You have already consulted the modern scrolls (Search Tool). "
            "Do not call the search tool again. Answer strictly from the context provided. "
            "If the answer is still missing, say: 'The gods have veiled that specific moment from my sight for now.'"
        )

    prompt = llm_prompt_template.format(**{
        name_key:       ENTITY_NAME,
        "context":      "\n\n".join(combined_context),
        "query":        state['query'],
        "chat_history": history_str,
    }) + extra_instruction

    print(f"\n{ENTITY_NAME}: ", end="", flush=True)
    full_content   = ""
    tool_calls_buf = None

    for chunk in generator_llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_content += chunk.content
        if chunk.tool_calls:
            tool_calls_buf = chunk.tool_calls

    print()

    response = AIMessage(content=full_content, tool_calls=tool_calls_buf or [])

    print(f"[DEBUG] response.tool_calls: {response.tool_calls}")
    print(f"[DEBUG] Will loop? {bool(response.tool_calls and not has_searched)}")

    if response.tool_calls and not has_searched:
        print(f"\n[DECISION]: Consulting modern scrolls via {response.tool_calls[0]['name']}...")
        return {"messages": [response]}

    return {
        "messages": [AIMessage(content=full_content, name="generator_response")],
        "response": full_content
    }


def tts_node(state: AgentState) -> dict:
    if not state.get("response"):
        return {}

    clean_text = clean_for_tts(state["response"])
    output_dir = Path(__file__).parent / "audio"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "response.mp3")

    try:
        lang  = detect(clean_text)
        voice = LANG_TO_VOICE.get(lang, DEFAULT_VOICE)
    except Exception:
        voice = DEFAULT_VOICE
    
    print(f"\n[TTS]: Generating speech via edge-tts ({voice})...")
    async def generate():
        communicate = edge_tts.Communicate(clean_text, voice, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH)
        await communicate.save(output_path)

    try:
        asyncio.run(generate())
        print(f"[TTS]: Saved → {output_path}")
    except Exception as e:
        print(f"[TTS]: Failed — {e}")

    return {}


def route_tts(state: AgentState) -> str:
    return "tts" if state.get("voice_mode") else END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("rewriter",   rewrite_node)
workflow.add_node("retriever",  retrieve_node)
workflow.add_node("reranker",   rerank_node)
workflow.add_node("generator",  generate_node)
workflow.add_node("tools",      tool_node)
workflow.add_node("tts_router", lambda state: {})
workflow.add_node("tts",        tts_node)

workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter",  "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_edge("reranker",  "generator")
workflow.add_conditional_edges(
    "generator",
    tools_condition,
    {
        "tools": "tools",
        END:     "tts_router"
    }
)
workflow.add_edge("tools", "generator")
workflow.add_conditional_edges(
    "tts_router",
    route_tts,
    {
        "tts": "tts",
        END:   END
    }
)
workflow.add_edge("tts", END)

memory = MemorySaver()
graph  = workflow.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n╔══════════════════════════════════╗")
    print("║        ANCIENT EGYPT RAG         ║")
    print("╚══════════════════════════════════╝\n")
    print(f"Now speaking with: {ENTITY_NAME} ({ENTITY_TYPE})")
    print("Commands: 'v' or 'voice' → toggle voice mode | 'q' → quit\n")

    config     = {"configurable": {"thread_id": "1"}}
    voice_mode = False

    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.lower() in ['v', 'voice']:
            voice_mode = not voice_mode
            print(f"[Voice Mode {'ON' if voice_mode else 'OFF'}]")
            continue

        graph.invoke(
            {
                "messages":   [("user", user_input)],
                "query":      user_input,
                "context":    [],
                "voice_mode": voice_mode
            },
            config=config
        )