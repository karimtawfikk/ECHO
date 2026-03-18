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
from langchain_tavily import TavilySearch

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine

import yaml

warnings.filterwarnings("ignore")
load_dotenv()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def load_resources():
    base_path = Path(__file__).parent.parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_template = f.read()
    with open(base_path / "evaluation_promptnew.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return sql_template, prompts

SQL_TEMPLATE, PROMPTS = load_resources()


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

GROQ_API_KEY1          = os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2          = os.getenv("GROQ_API_KEY2")
GROQ_API_KEY8          = os.getenv("GROQ_API_KEY8")
CF_WORKERSAI_ACCOUNTID = os.getenv("R2_ACCOUNT_ID")
CF_AI_API              = os.getenv("CF_AI_API")
JINA_API_KEY           = os.getenv("JINA_API_KEY")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_GENERATOR_MODEL_NAME      = "openai/gpt-oss-120b"
GROQ_QUERY_REWRITER_MODEL_NAME = "qwen/qwen3-32b"
GROQ_STT_MODEL_NAME            = "whisper-large-v3"
EDGE_TTS_RATE                  = "+9%"
EDGE_TTS_PITCH                 = "-10Hz"
TOP_K                          = 3
EMBEDDING_DIM                  = 768

STT_SAMPLE_RATE                = 16000

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

ENTITY_TYPE          = None
ENTITY_NAME          = None
VECTOR_SQL           = None
rewrite_chain        = None
llm_prompt_template  = None


# State
class AgentState(TypedDict):
    query:        str
    search_query: str
    messages:     Annotated[list, add_messages]
    context:      List[str]
    response:     str
    voice_mode:   bool


# Models & Tools
embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)

search_tool = TavilySearch(max_results=4, search_depth="advanced")
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
    temperature=0.7,
    max_tokens=4096,
    top_p=0.95,
    api_key=GROQ_API_KEY8,
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


def clean_for_tts(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    text = re.sub(r'\n+',           ' ',   text).strip()
    return text


def record_audio() -> np.ndarray:
    import sounddevice as sd

    print("[STT]: Recording... press Enter when done.")

    frames = []
    stop   = threading.Event()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    def wait_for_enter():
        input()
        stop.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    with sd.InputStream(samplerate=STT_SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
        stop.wait()

    return np.concatenate(frames, axis=0)


def transcribe_audio(audio: np.ndarray) -> str:
    from scipy.io.wavfile import write as wav_write

    audio_int16 = (audio * 32767).astype(np.int16)
    buffer      = io.BytesIO()
    wav_write(buffer, STT_SAMPLE_RATE, audio_int16)
    buffer.seek(0)

    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", buffer.read()),
        model=GROQ_STT_MODEL_NAME,
        temperature=0.2
    )
    return transcription.text.strip()


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
    
    

    name_key = ENTITY_CONFIG[ENTITY_TYPE]["name_key"]
    
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

def generate_node(state: AgentState) -> dict:

    combined_context = state.get('context', [])
    print("REREANKED CONTEXT:",len(combined_context))
    # 3. Build Chat History for Persona Continuity
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


    
    name_key = ENTITY_CONFIG[ENTITY_TYPE]["name_key"]

    # 5. Format the Final Prompt
    prompt = llm_prompt_template.format(**{
        name_key:       ENTITY_NAME,
        "context":      "\n\n".join(combined_context),
        "query":        state['query'],
        "chat_history": history_str,
    })

    # 6. Stream the Response (No Tool Calls allowed)
    print(f"\n{ENTITY_NAME}: ", end="", flush=True)
    full_content = ""

    for chunk in generator_llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_content += chunk.content

    print("\n🏛️ [SOURCE]: Final answer generated using RAG context ONLY.")
    
    # 7. Return Final Response
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
workflow.add_node("generator",  generate_node)
workflow.add_node("tools",      tool_node)
workflow.add_node("tts_router", lambda state: {})
workflow.add_node("tts",        tts_node)

workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter",  "retriever")
workflow.add_edge("retriever", "generator")
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

def main():
    global ENTITY_TYPE, ENTITY_NAME, VECTOR_SQL, rewrite_chain, llm_prompt_template

    print("\n╔══════════════════════════════════╗")
    print("║        ANCIENT EGYPT RAG         ║")
    print("╚══════════════════════════════════╝\n")

    while True:
        ENTITY_TYPE = input("Entity type — 'pharaoh' or 'landmark': ").strip().lower()
        if ENTITY_TYPE in ENTITY_CONFIG:
            break
        print("  → Please enter 'pharaoh' or 'landmark'.")

    ENTITY_NAME = input(f"Enter the {ENTITY_TYPE} name: ").strip()

    cfg         = ENTITY_CONFIG[ENTITY_TYPE]
    VECTOR_SQL  = SQL_TEMPLATE.format(
        texts_table=cfg["texts_table"],
        entities_table=cfg["entities_table"],
        entity_id_col=cfg["entity_id_col"]
    )

    prompt_key          = cfg["prompt_key"]
    rewrite_chain       = PromptTemplate.from_template(PROMPTS["rewrite_prompt"][prompt_key]) | query_rewriter_llm | StrOutputParser()
    llm_prompt_template = PromptTemplate.from_template(PROMPTS["assistant_persona"][prompt_key])

    print(f"\nNow speaking with: {ENTITY_NAME} ({ENTITY_TYPE})")
    print("Commands: 'v' or 'voice' → toggle voice mode | 'q' → quit\n")

    config     = {"configurable": {"thread_id": "1"}}
    voice_mode = False

    while True:
        if voice_mode:
            raw = input("\n[Voice Mode ON] — press Enter to speak, or type ('v' to switch back, 'q' to quit)\n> ").strip()
            if raw.lower() in ['v', 'voice']:
                voice_mode = False
                print("[Voice Mode OFF]")
                continue
            elif raw.lower() in ['quit', 'exit', 'q']:
                break
            elif raw == "":
                try:
                    audio      = record_audio()
                    user_input = transcribe_audio(audio)
                    print(f"[STT]: {user_input}")

                    if not user_input or len(user_input.strip()) < 2:
                        print("[STT]: I didn't catch that. Please try again.")
                        continue
                except Exception as e:
                    print(f"[STT]: Failed — {e}")
                    continue
            else:
                user_input = raw
        else:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.lower() in ['v', 'voice']:
                voice_mode = True
                print("[Voice Mode ON] — press Enter with empty input to start speaking")
                continue

        graph.invoke(
            {"messages":   [("user", user_input)],
             "query":      user_input,
             "context":    [],
             "voice_mode": voice_mode},
            config=config
        )


if __name__ == "__main__":
    main()