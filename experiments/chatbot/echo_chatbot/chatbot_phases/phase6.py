from sys import path
from pathlib import Path
path.append(str(Path(__file__).resolve().parents[4]))

from warnings import filterwarnings
from os import getenv
from re import sub
from io import BytesIO
from asyncio import run
from threading import Event, Thread
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

from edge_tts import Communicate
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
#from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_tavily import TavilySearch

from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db.session import engine

from yaml import safe_load

filterwarnings("ignore")
load_dotenv()

def load_resources():
    base_path = Path(__file__).parent.parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_template = f.read()
    with open(base_path / "prompts.yaml", "r", encoding="utf-8") as f:
        prompts = safe_load(f)
    return sql_template, prompts

SQL_TEMPLATE, PROMPTS = load_resources()

GROQ_API_KEY1          = getenv("GROQ_API_KEY1")
GROQ_API_KEY2          = getenv("GROQ_API_KEY2")
CF_WORKERSAI_ACCOUNTID = getenv("R2_ACCOUNT_ID")
CF_AI_API              = getenv("CF_AI_API")
JINA_API_KEY           = getenv("JINA_API_KEY")

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
    "en": {
        "male": "en-CA-LiamNeural",
        "female": "en-US-JennyNeural"
    },
    "es": {
        "male": "es-ES-AlvaroNeural",
        "female": "es-ES-ElviraNeural"
    },
    "fr": {
        "male": "fr-FR-HenriNeural",
        "female": "fr-FR-DeniseNeural"
    },
    "ar": {
        "male": "ar-EG-ShakirNeural",
        "female": "ar-BH-LailaNeural"
    },
    "de": {
        "male": "de-DE-ConradNeural",
        "female": "de-DE-KatjaNeural"
    },
    "it": {
        "male": "it-IT-DiegoNeural",
        "female": "it-IT-ElsaNeural"
    },
    "pt": {
        "male": "pt-BR-AntonioNeural",
        "female": "pt-BR-FranciscaNeural"
    }
}

DEFAULT_VOICE = "en-US-ChristopherNeural"

ENTITY_TYPE          = None
ENTITY_NAME          = None
ENTITY_ID            = None
VECTOR_SQL           = None
PHARAOH_GENDER       = None
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
"""embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)"""
qwen_model=None
def load_model_background():
    global qwen_model,start
    from sentence_transformers import SentenceTransformer
    qwen_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device='cuda',
    tokenizer_kwargs={"padding_side": "left"}
    )
    

reranker = JinaRerank(
    model="jina-reranker-v3",
    top_n=TOP_K,
    jina_api_key=JINA_API_KEY
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
    extra_body={"reasoning_effort": "none", "reasoning_format": "hidden"}
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
# Helpers
# ---------------------------------------------------------------------------
"""
def get_embedding(text: str):
    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced     = embeddings[:EMBEDDING_DIM]
    norm       = np.linalg.norm(sliced)
    return (sliced / norm if norm > 0 else sliced).tolist()"""

def get_embedding(text: str):
    embeddings = qwen_model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    sliced = embeddings[:EMBEDDING_DIM]
    norm = np.linalg.norm(sliced)
    
    return (sliced / norm if norm > 0 else sliced).tolist()


def clean_for_tts(text: str) -> str:
    text = sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = sub(r'\*(.+?)\*',     r'\1', text)
    text = sub(r'#{1,6}\s*',     '',    text)
    text = sub(r'\n+',           ' ',   text).strip()
    return text


def record_audio() -> np.ndarray:
    import sounddevice as sd

    print("[STT]: Recording... press Enter when done.")

    frames = []
    stop   = Event()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    def wait_for_enter():
        input()
        stop.set()

    listener = Thread(target=wait_for_enter, daemon=True)
    listener.start()

    with sd.InputStream(samplerate=STT_SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
        stop.wait()

    return np.concatenate(frames, axis=0)


def transcribe_audio(audio: np.ndarray) -> str:
    from scipy.io.wavfile import write as wav_write

    audio_int16 = (audio * 32767).astype(np.int16)
    buffer      = BytesIO()
    wav_write(buffer, STT_SAMPLE_RATE, audio_int16)
    buffer.seek(0)

    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", buffer.read()),
        model=GROQ_STT_MODEL_NAME,
        temperature=0
    )
    return transcription.text.strip()

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
# Global variable to store user memory
USER_MEMORY = []

def rewrite_node(state: AgentState) -> dict:
    global USER_MEMORY
    
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
    
    # Call rewriter
    response = rewrite_chain.invoke({
        name_key:       ENTITY_NAME,
        "chat_history": history_str,
        "query":        state["query"]
    }).strip()
    
    
    # Check if [MEMORY] tag exists
    if "[MEMORY]:" in response:
        parts = response.split("[MEMORY]:")
        search_q = parts[0].replace("Search Query:", "").strip()
        memory_str = parts[1].strip()
        
        memory_items = [item.strip() for item in memory_str.split(',')]
        
        for item in memory_items:
            if '=' in item:
                key, value = item.split('=', 1)
                memory_entry = f"{key.strip()}={value.strip()}"

                if memory_entry in USER_MEMORY:
                    continue
            
                USER_MEMORY = [m for m in USER_MEMORY if not m.startswith(f"{key.strip()}=")]
                USER_MEMORY.append(memory_entry) 
                           
    else:
        search_q = response.replace("Search Query:", "").strip()
    
    if not search_q:
        search_q = state["query"]
    
    return {
        "messages":     [AIMessage(content=search_q, name="search_query")],
        "search_query": search_q
    }

def retrieve_node(state: AgentState) -> dict:

    query_embedding = get_embedding(state['search_query'])
    
    with Session(engine) as session:
        result = session.execute(
            text(VECTOR_SQL),
            {
                "entity_id": ENTITY_ID,
                "embedding": str(query_embedding)
            }
        )
        context = [row[0] for row in result]
    return {"context": context}


def rerank_node(state: AgentState) -> dict:
    docs     = [Document(page_content=chunk) for chunk in state['context']]
    reranked = reranker.compress_documents(docs, query=state['search_query'])
    return {"context": [doc.page_content for doc in reranked]}

def generate_node(state: AgentState) -> dict:
    global USER_MEMORY
    
    if "OUT_OF_SCOPE" in state.get('search_query', ''):
        response_text = "I'm sorry but you speak of a time that is not mine. My eyes see only the borders of my own reign." \
            if ENTITY_TYPE == "pharaoh" else \
            "I'm sorry, that lies beyond what my stones remember."
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
   
    user_info_str = ""
    if USER_MEMORY:
        for item in USER_MEMORY:
            if '=' in item:
                key, value = item.split('=', 1)
                user_info_str += f"\n- {key.strip()}: {value.strip()}"
    else:
        user_info_str = "No user information available yet."

    extra_instruction = ""
    if has_searched:
        extra_instruction = (
            "\n\nIMPORTANT: You have already consulted the modern scrolls (Search Tool). "
            "Do not call the search tool again. Answer strictly from the context provided. "
            "Only if the answer is still missing, say: 'The gods have veiled that specific moment from my sight for now.'"
        )
    
    name_key = ENTITY_CONFIG[ENTITY_TYPE]["name_key"]

    prompt = llm_prompt_template.format(**{
        name_key:       ENTITY_NAME,
        "context":      "\n\n".join(combined_context),
        "query":        state['query'],
        "chat_history": history_str,
        "user_info":    user_info_str
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

    if response.tool_calls and not has_searched:
        print(f"[DECISION]: Consulting modern scrolls via {response.tool_calls[0]['name']}...")
        return {"messages": [response]}

    return {
        "messages": [AIMessage(content=full_content, name="generator_response")],
        "response": full_content
    }

def tts_node(state: AgentState) -> dict:
    if not state.get("response"):
        return {}

    clean_text = clean_for_tts(state["response"])
    output_dir = Path(__file__).parent.parent / "audio"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "response.mp3")

    try:
        lang = detect(clean_text)
        lang_voices = LANG_TO_VOICE.get(lang, LANG_TO_VOICE["en"])
        voice = lang_voices.get(PHARAOH_GENDER, LANG_TO_VOICE["en"]["male"])
    except Exception:
        voice = DEFAULT_VOICE
    
    print(f"\n[TTS]: Generating speech via edge-tts ({voice})...")
    async def generate():
        communicate = Communicate(clean_text, voice, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH)
        await communicate.save(output_path)

    try:
        run(generate())
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
graph  = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global ENTITY_TYPE, ENTITY_NAME, ENTITY_ID, VECTOR_SQL,PHARAOH_GENDER, rewrite_chain, llm_prompt_template
    Thread(target=load_model_background, daemon=True).start()

    print("\n╔══════════════════════════════════╗")
    print("║        ANCIENT EGYPT RAG         ║")
    print("╚══════════════════════════════════╝\n")

    while True:
        ENTITY_TYPE = input("Entity type — 'pharaoh' or 'landmark': ").strip().lower()
        if ENTITY_TYPE in ENTITY_CONFIG:
            break
        print("  → Please enter 'pharaoh' or 'landmark'.")

    ENTITY_NAME = input(f"Enter the {ENTITY_TYPE} name: ").strip()

    cfg = ENTITY_CONFIG[ENTITY_TYPE]
    
    # NEW: Lookup entity ID ONCE at conversation start
    print(f"  🔍 Looking up {ENTITY_TYPE}...")
    
    id_col = "pharaoh_id" if ENTITY_TYPE == "pharaoh" else "landmark_id"
    table_name = "pharaohs" if ENTITY_TYPE == "pharaoh" else "landmarks"
    
    with Session(engine) as session:
        result = None
        if table_name=="pharaohs":
            result = session.execute(
                text(f"SELECT id, gender FROM {table_name} WHERE name = :name"),
                {"name": ENTITY_NAME}
            ).fetchone()
        else:
            result = session.execute(
                text(f"SELECT id FROM {table_name} WHERE name = :name"),
                {"name": ENTITY_NAME}
            ).fetchone()

        if result:
            ENTITY_ID = result[0]
            PHARAOH_GENDER = result[1] if table_name == "pharaohs" else "male"
            print(f"  ✓ Found {ENTITY_TYPE} (ID: {ENTITY_ID})")
        else:
            print(f"  ✗ Error: '{ENTITY_NAME}' not found in database!")
            print(f"  → Check spelling or verify {ENTITY_TYPE} exists in database.")
            return
    
    # Set up SQL with optimized query
    VECTOR_SQL = SQL_TEMPLATE.format(
        texts_table=cfg["texts_table"],
        entity_id_col=cfg["entity_id_col"]
    )

    prompt_key = cfg["prompt_key"]
    rewrite_chain = PromptTemplate.from_template(PROMPTS["rewrite_prompt"][prompt_key]) | query_rewriter_llm | StrOutputParser()
    llm_prompt_template = PromptTemplate.from_template(PROMPTS["assistant_persona"][prompt_key])

    print(f"\nNow speaking with: {ENTITY_NAME} ({ENTITY_TYPE})")
    print("Commands: 'v' or 'voice' → toggle voice mode | 'q' → quit\n")

    config = {"configurable": {"thread_id": "1"}}
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