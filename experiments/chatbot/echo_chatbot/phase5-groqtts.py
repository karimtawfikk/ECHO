import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import warnings
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import re
import time

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
from groq import Groq

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine

import yaml
import numpy as np
import json

warnings.filterwarnings("ignore")
load_dotenv()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def load_resources():
    base_path = Path(__file__).parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()


# ---------------------------------------------------------------------------
# Environment
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
GROQ_TTS_MODEL                 = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE                 = "troy"
TOP_K                          = 3
EMBEDDING_DIM                  = 768
ENTITY_NAME                    = "Ramesses II"
TTS_CHAR_LIMIT                 = 190


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

search_tool = TavilySearch(max_results=5, search_depth="advanced")
tools       = [search_tool]
tool_node   = ToolNode(tools=tools)

groq_client = Groq(api_key=GROQ_API_KEY1)

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

rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
rewrite_chain = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()

llm_prompt_template = PromptTemplate.from_template(PROMPTS['assistant_persona'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str):
    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced = embeddings[:EMBEDDING_DIM]
    norm = np.linalg.norm(sliced)
    return (sliced / norm if norm > 0 else sliced).tolist()


def chunk_text(text: str, limit: int = TTS_CHAR_LIMIT) -> List[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ""

    for sentence in sentences:
        if len(sentence) > limit:
            words = sentence.split(" ")
            for word in words:
                if len(current) + len(word) + 1 <= limit:
                    current += (" " + word) if current else word
                else:
                    if current:
                        chunks.append(current.strip())
                    current = word
        elif len(current) + len(sentence) + 1 <= limit:
            current += (" " + sentence) if current else sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


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
        "query": state['query'],
        "pharaoh_name": ENTITY_NAME,
        "chat_history": history_str
    }).replace("Search Query:", "").strip()

    print("-" * 50)
    print(search_q)
    print("-" * 60)

    return {
        "messages": [AIMessage(content=search_q, name="search_query")],
        "search_query": search_q
    }


def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['search_query'])

    with Session(engine) as session:
        result = session.execute(
            text(VECTOR_SQL),
            {"pharoah_name": ENTITY_NAME,
             "embedding": str(query_embedding)}
        )
        context = [row[0] for row in result]

    return {"context": context}


def rerank_node(state: AgentState) -> dict:
    docs = [Document(page_content=chunk) for chunk in state['context']]
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

    # 1. ANCHOR TO CURRENT TURN
    last_human_index = -1
    for i, msg in enumerate(state['messages']):
        if isinstance(msg, HumanMessage):
            last_human_index = i

    current_turn_messages = state['messages'][last_human_index:] if last_human_index != -1 else []
    has_searched = any(isinstance(msg, ToolMessage) for msg in current_turn_messages)

    # 2. MERGE CONTEXT
    current_search_results = [
        msg.content for msg in current_turn_messages
        if isinstance(msg, ToolMessage)
    ]
    combined_context = current_search_results + state['context'] if has_searched else state['context']

    # 3. CLEAN HISTORY
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

    # 4. DYNAMIC INSTRUCTION
    extra_instruction = ""
    if has_searched:
        extra_instruction = (
            "\n\nIMPORTANT: You have already consulted the modern scrolls (Search Tool). "
            "Do not call the search tool again. Answer strictly from the context provided. "
            "If the answer is still missing, say: 'The gods have veiled that specific moment from my sight for now.'"
        )

    # 5. PROMPT
    prompt = llm_prompt_template.format(
        pharaoh_name=ENTITY_NAME,
        context="\n\n".join(combined_context),
        query=state['query'],
        chat_history=history_str,
    ) + extra_instruction

    # 6. INVOKE + CIRCUIT BREAKER
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
    # Pause the graph and ask the user
    user_choice = interrupt("Generate speech for this response? (y/n): ")
    tts_enabled = user_choice.strip().lower() == "y"
    return {"tts_enabled": tts_enabled}


def tts_node(state: AgentState) -> dict:
    if not state.get("tts_enabled") or not state.get("response"):
        return {}

    response_text = state["response"]
    chunks = chunk_text(response_text)
    output_dir = Path(__file__).parent / "audio"
    output_dir.mkdir(exist_ok=True)

    print(f"\n[TTS]: Generating speech ({len(chunks)} chunk(s))...")

    for i, chunk in enumerate(chunks):
        output_path = output_dir / f"response_part_{i+1}.wav"
        try:
            audio = groq_client.audio.speech.create(
                model=GROQ_TTS_MODEL,
                voice=GROQ_TTS_VOICE,
                input=chunk,
                response_format="wav"
            )
            audio.write_to_file(str(output_path))
            print(f"[TTS]: Part {i+1}/{len(chunks)} saved → {output_path}")
            if i < len(chunks) - 1:
                time.sleep(0.3)
        except Exception as e:
            print(f"[TTS]: Failed on chunk {i+1}: {e}")

    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

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
        END: "tts_gate"       # after final answer → ask user
    }
)
workflow.add_edge("tools",     "generator")
workflow.add_edge("tts_gate",  "tts_gate")  # re-entry point after interrupt resumes
workflow.add_conditional_edges(
    "tts_gate",
    route_tts,
    {
        "tts": "tts",
        END: END
    }
)
workflow.add_edge("tts", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Pharaoh-RAG Phase 5 — TTS ({ENTITY_NAME}) Ready:")

    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        initial_state = {
            "messages":   [("user", user_input)],
            "query":      user_input,
            "context":    [],
            "tts_enabled": False
        }

        # First invoke — runs until interrupt in tts_gate_node
        result = graph.invoke(initial_state, config=config)

        # Graph is now paused — ask the user and resume
        if result.get("__interrupt__"):
            user_choice = input("\nGenerate speech for this response? (y/n): ").strip()
            graph.invoke(Command(resume=user_choice), config=config)


if __name__ == "__main__":
    main()