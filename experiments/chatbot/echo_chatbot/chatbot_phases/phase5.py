import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))

import warnings
import os
import re
import base64
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_tavily import TavilySearch

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine

import yaml
import numpy as np

warnings.filterwarnings("ignore")
load_dotenv()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def load_resources():
    base_path = Path(__file__).parent.parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "prompts.yaml", "r",encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

GROQ_API_KEY1          = os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2          = os.getenv("GROQ_API_KEY2")
CF_WORKERSAI_ACCOUNTID = os.getenv("R2_ACCOUNT_ID")
CF_AI_API              = os.getenv("CF_AI_API")
JINA_API_KEY           = os.getenv("JINA_API_KEY")
INWORLD_API_KEY        = os.getenv("INWORLD_API_KEY")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_GENERATOR_MODEL_NAME      = "openai/gpt-oss-120b"
GROQ_QUERY_REWRITER_MODEL_NAME = "qwen/qwen3-32b"
INWORLD_VOICE_ID               = "default-1ocgrlw5u8sovko4eeeqnw__ancient_egyptian_pharaoh"
INWORLD_MODEL                  = "inworld-tts-1.5-mini"
TOP_K                          = 3
EMBEDDING_DIM                  = 768
ENTITY_NAME                    = "Ramesses II"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:        str
    search_query: str
    messages:     Annotated[list, add_messages]
    context:      List[str]
    response:     str
    tts_enabled:  bool


# ---------------------------------------------------------------------------
# Models & Tools
# ---------------------------------------------------------------------------

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

search_tool = TavilySearch(max_results=4, search_depth="advanced")
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
    api_key=GROQ_API_KEY2,
    extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"}
).bind_tools(tools)


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------

rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt']['pharaoh'])
rewrite_chain           = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()
llm_prompt_template     = PromptTemplate.from_template(PROMPTS['assistant_persona']['pharaoh'])


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
    print("="*50)
    print(history_str)
    print("="*50)
    
    search_q = rewrite_chain.invoke({
        "query":        state['query'],
        "pharaoh_name": ENTITY_NAME,
        "chat_history": history_str
    }).replace("Search Query:", "").strip()
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
            {"pharoah_name": ENTITY_NAME,
             "embedding":    str(query_embedding)}
        )
        context = [row[0] for row in result]

    return {"context": context}


def rerank_node(state: AgentState) -> dict:
    docs     = [Document(page_content=chunk) for chunk in state['context']]
    reranked = reranker.compress_documents(docs, query=state['search_query'])
    return {"context": [doc.page_content for doc in reranked]}


def generate_node(state: AgentState) -> dict:

    if "OUT_OF_SCOPE" in state.get('search_query', ''):
        response_text = "I'm sorry but you speak of a time that is not mine. My eyes see only the borders of my own reign."
        print(f"\n{ENTITY_NAME}: {response_text}")
        return {
            "messages": [AIMessage(content=response_text, name="irrelevant_query")],
            "response": response_text
        }

    last_human_index = -1
    for i, msg in enumerate(state['messages']):
        if isinstance(msg, HumanMessage):
            last_human_index = i

    current_turn_messages  = state['messages'][last_human_index:] if last_human_index != -1 else []
    has_searched           = any(isinstance(msg, ToolMessage) for msg in current_turn_messages)
    current_search_results = [msg.content for msg in current_turn_messages if isinstance(msg, ToolMessage)]
    combined_context       = current_search_results + state['context'] if has_searched else state['context']

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

    prompt = llm_prompt_template.format(
        pharaoh_name=ENTITY_NAME,
        context="\n\n".join(combined_context),
        query=state['query'],
        chat_history=history_str,
    ) + extra_instruction

    response = generator_llm.invoke(prompt)

    if response.tool_calls and not has_searched:
        print(f"\n[PHARAOH DECISION]: Consulting modern scrolls via {response.tool_calls[0]['name']}...")
        return {"messages": [response]}

    print(f"\n{ENTITY_NAME}: {response.content}")
    return {
        "messages": [AIMessage(content=response.content, name="generator_response")],
        "response": response.content
    }


def tts_gate_node(state: AgentState) -> dict:
    user_choice = interrupt("Generate speech for this response? (y/n): ")
    tts_enabled = user_choice.strip().lower() == "y"
    return {"tts_enabled": tts_enabled}


def tts_node(state: AgentState) -> dict:
    if not state.get("tts_enabled") or not state.get("response"):
        return {}

    clean_text = clean_for_tts(state["response"])
    output_dir = Path(__file__).parent / "audio"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "response.mp3")

    print(f"\n[TTS]: Generating speech via Inworld ({INWORLD_VOICE_ID})...")

    try:
        response = requests.post(
            "https://api.inworld.ai/tts/v1/voice",
            headers={
                "Authorization": f"Basic {INWORLD_API_KEY}",
                "Content-Type":  "application/json"
            },
            json={
                "text":    clean_text,
                "voiceId": INWORLD_VOICE_ID,
                "modelId": INWORLD_MODEL,
                "speakingRate": 2.5,
                "temperature":1.3
            }
        )
        response.raise_for_status()

        audio_b64 = response.json()["audioContent"]
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        print(f"[TTS]: Saved → {output_path}")

    except Exception as e:
        print(f"[TTS]: Failed — {e}")

    return {}

def route_tts(state: AgentState) -> str:
    return "tts" if state.get("tts_enabled") else END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("rewriter",  rewrite_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("reranker",  rerank_node)
workflow.add_node("generator", generate_node)
workflow.add_node("tools",     tool_node)
workflow.add_node("tts_gate",  tts_gate_node)
workflow.add_node("tts",       tts_node)

workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter",  "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_edge("reranker",  "generator")
workflow.add_conditional_edges(
    "generator",
    tools_condition,
    {
        "tools": "tools",
        END:     "tts_gate"
    }
)
workflow.add_edge("tools",    "generator")
workflow.add_edge("tts_gate", "tts_gate")
workflow.add_conditional_edges(
    "tts_gate",
    route_tts,
    {
        "tts": "tts",
        END:   END
    }
)
workflow.add_edge("tts", END)

memory = MemorySaver()
graph  = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Pharaoh-RAG Phase 5 — Inworld TTS ({ENTITY_NAME}) Ready:")

    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        initial_state = {
            "messages":    [("user", user_input)],
            "query":       user_input,
            "context":     [],
            "tts_enabled": False
        }

        result = graph.invoke(initial_state, config=config)

        if result.get("__interrupt__"):
            user_choice = input("\nGenerate speech for this response? (y/n): ").strip()
            graph.invoke(Command(resume=user_choice), config=config)


if __name__ == "__main__":
    main()
