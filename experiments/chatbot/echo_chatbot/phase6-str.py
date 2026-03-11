import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import warnings
import os
import re
import io
import asyncio
import threading
import numpy as np
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

import edge_tts

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
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pharaoh-RAG",
    page_icon="𓂀",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --gold:    #c9a84c;
    --gold2:   #e8c97a;
    --dark:    #0d0b08;
    --dark2:   #1a1610;
    --dark3:   #242018;
    --sand:    #f0e6c8;
    --sand2:   #d4c49a;
    --red:     #8b2a2a;
}

html, body, [data-testid="stApp"] {
    background-color: var(--dark) !important;
    color: var(--sand) !important;
    font-family: 'EB Garamond', serif !important;
}

[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(ellipse at 50% 0%, #2a1f0a 0%, transparent 60%),
        var(--dark);
}

h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: var(--gold) !important;
    letter-spacing: 0.08em;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Main container */
.main-title {
    text-align: center;
    padding: 2rem 0 0.5rem;
    font-family: 'Cinzel', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--gold);
    letter-spacing: 0.15em;
    text-shadow: 0 0 40px rgba(201,168,76,0.4);
}

.sub-title {
    text-align: center;
    font-family: 'EB Garamond', serif;
    font-style: italic;
    color: var(--sand2);
    font-size: 1.1rem;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.divider {
    border: none;
    border-top: 1px solid rgba(201,168,76,0.3);
    margin: 1rem 0 1.5rem;
}

/* Chat messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding: 1rem 0;
    min-height: 400px;
}

.msg-row-user {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    align-items: flex-end;
}

.msg-row-pharaoh {
    display: flex;
    justify-content: flex-start;
    gap: 0.75rem;
    align-items: flex-end;
}

.bubble-user {
    background: linear-gradient(135deg, #3d2e10, #2a1f08);
    border: 1px solid rgba(201,168,76,0.4);
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 70%;
    font-family: 'EB Garamond', serif;
    font-size: 1.05rem;
    color: var(--sand);
    line-height: 1.5;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}

.bubble-pharaoh {
    background: linear-gradient(135deg, #1a1208, #241a0a);
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 75%;
    font-family: 'EB Garamond', serif;
    font-size: 1.05rem;
    color: var(--sand);
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}

.avatar-user {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3d2e10, #c9a84c);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    border: 1px solid rgba(201,168,76,0.5);
}

.avatar-pharaoh {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #8b2a2a, #c9a84c);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    border: 1px solid rgba(201,168,76,0.5);
}

.sender-label {
    font-family: 'Cinzel', serif;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--gold);
    opacity: 0.7;
    margin-bottom: 0.2rem;
}

/* Input area */
.input-area {
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, var(--dark) 80%, transparent);
    padding: 1rem 0 1.5rem;
    margin-top: 1rem;
}

[data-testid="stTextInput"] input {
    background: var(--dark2) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 24px !important;
    color: var(--sand) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1.05rem !important;
    padding: 0.6rem 1.2rem !important;
    caret-color: var(--gold) !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(201,168,76,0.7) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: rgba(212,196,154,0.4) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2a1f08, #3d2e10) !important;
    border: 1px solid rgba(201,168,76,0.5) !important;
    color: var(--gold) !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 20px !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3d2e10, #5a4520) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 0 12px rgba(201,168,76,0.3) !important;
}

/* Voice button */
.voice-btn > button {
    background: linear-gradient(135deg, #1a0a0a, #3d1010) !important;
    border-color: rgba(139,42,42,0.6) !important;
    color: #e87a7a !important;
    font-size: 1.1rem !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
}

.voice-btn-active > button {
    background: linear-gradient(135deg, #3d1010, #8b2a2a) !important;
    border-color: #e87a7a !important;
    box-shadow: 0 0 16px rgba(232,122,122,0.4) !important;
    animation: pulse 1s infinite !important;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 16px rgba(232,122,122,0.4); }
    50%       { box-shadow: 0 0 24px rgba(232,122,122,0.7); }
}

/* Thinking indicator */
.thinking {
    display: flex;
    gap: 5px;
    padding: 0.5rem 0;
    align-items: center;
}

.thinking span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--gold);
    animation: bounce 1.2s infinite;
    opacity: 0.6;
}

.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 100% { transform: translateY(0); opacity: 0.4; }
    50%       { transform: translateY(-6px); opacity: 1; }
}

/* Audio player */
[data-testid="stAudio"] {
    background: transparent !important;
}

audio {
    filter: sepia(0.3) !important;
    width: 100% !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--dark2); }
::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.3); border-radius: 2px; }

/* STT status */
.stt-status {
    text-align: center;
    font-family: 'Cinzel', serif;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    color: #e87a7a;
    padding: 0.5rem;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


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
GROQ_STT_MODEL_NAME            = "whisper-large-v3"
EDGE_TTS_VOICE                 = "en-US-SteffanNeural"
EDGE_TTS_RATE                  = "+9%"
EDGE_TTS_PITCH                 = "-10Hz"
TOP_K                          = 3
EMBEDDING_DIM                  = 768
ENTITY_NAME                    = "Ramesses II"

STT_SAMPLE_RATE                = 16000
STT_SILENCE_THRESHOLD          = 0.015
STT_SILENCE_DURATION           = 1.5
STT_MIN_DURATION               = 1.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query:        str
    search_query: str
    messages:     Annotated[list, add_messages]
    context:      List[str]
    response:     str
    voice_mode:   bool


# ---------------------------------------------------------------------------
# Models & Tools (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
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
    rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
    rewrite_chain           = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()
    llm_prompt_template     = PromptTemplate.from_template(PROMPTS['assistant_persona'])
    return (embedding_model, reranker, tool_node, groq_client,
            rewrite_chain, llm_prompt_template,generator_llm, tools)

(embedding_model, reranker, tool_node, groq_client,
 rewrite_chain, llm_prompt_template,generator_llm, tools) = load_models()


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


def generate_tts_bytes(text: str) -> bytes:
    clean_text = clean_for_tts(text)
    output     = io.BytesIO()

    async def _generate():
        communicate = edge_tts.Communicate(clean_text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output.write(chunk["data"])

    asyncio.run(_generate())
    output.seek(0)
    return output.read()


def record_until_silence() -> np.ndarray:
    import sounddevice as sd

    frames        = []
    silence_start = [None]
    done          = threading.Event()

    def callback(indata, frame_count, time_info, status):
        chunk        = indata.copy()
        frames.append(chunk)
        elapsed_secs = len(frames) * frame_count / STT_SAMPLE_RATE
        volume       = np.linalg.norm(chunk)

        if elapsed_secs < STT_MIN_DURATION:
            return

        if volume < STT_SILENCE_THRESHOLD:
            if silence_start[0] is None:
                silence_start[0] = len(frames)
        else:
            silence_start[0] = None

        if silence_start[0] is not None:
            silent_frames = len(frames) - silence_start[0]
            silent_secs   = silent_frames * frame_count / STT_SAMPLE_RATE
            if silent_secs >= STT_SILENCE_DURATION:
                done.set()

    with sd.InputStream(samplerate=STT_SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
        done.wait()

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
# Graph nodes
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
        "query":        state['query'],
        "pharaoh_name": ENTITY_NAME,
        "chat_history": history_str
    }).replace("Search Query:", "").strip()

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
        return {"messages": [response]}

    return {
        "messages": [AIMessage(content=response.content, name="generator_response")],
        "response": response.content
    }


# ---------------------------------------------------------------------------
# Graph (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def build_graph():
    from langgraph.prebuilt import ToolNode
    from langchain_tavily import TavilySearch

    _search_tool = TavilySearch(max_results=4, search_depth="advanced")
    _tools       = [_search_tool]
    _tool_node   = ToolNode(tools=_tools)

    wf = StateGraph(AgentState)
    wf.add_node("rewriter",  rewrite_node)
    wf.add_node("retriever", retrieve_node)
    wf.add_node("reranker",  rerank_node)
    wf.add_node("generator", generate_node)
    wf.add_node("tools",     _tool_node)

    wf.set_entry_point("rewriter")
    wf.add_edge("rewriter",  "retriever")
    wf.add_edge("retriever", "reranker")
    wf.add_edge("reranker",  "generator")
    wf.add_conditional_edges(
        "generator",
        tools_condition,
        {"tools": "tools", END: END}
    )
    wf.add_edge("tools", "generator")

    _memory = MemorySaver()
    return wf.compile(checkpointer=_memory)

graph = build_graph()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history  = []
if "graph_config" not in st.session_state:
    st.session_state.graph_config  = {"configurable": {"thread_id": "1"}}
if "recording" not in st.session_state:
    st.session_state.recording     = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">𓂀 PHARAOH-RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Speak with Ramesses II, Son of Ra, Lord of the Two Lands</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


def render_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row-user">
                <div>
                    <div class="sender-label" style="text-align:right;">YOU</div>
                    <div class="bubble-user">{msg["content"]}</div>
                </div>
                <div class="avatar-user">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            audio_html = ""
            if msg.get("audio"):
                import base64
                audio_b64  = base64.b64encode(msg["audio"]).decode()
                audio_html = f"""
                <audio controls style="width:100%;margin-top:0.5rem;height:32px;">
                    <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
                </audio>"""
            st.markdown(f"""
            <div class="msg-row-pharaoh">
                <div class="avatar-pharaoh">𓂀</div>
                <div>
                    <div class="sender-label">RAMESSES II</div>
                    <div class="bubble-pharaoh">{msg["content"]}{audio_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

render_chat()


def run_graph(user_input: str, voice_mode: bool):
    result = graph.invoke(
        {"messages":   [("user", user_input)],
         "query":      user_input,
         "context":    [],
         "voice_mode": voice_mode},
        config=st.session_state.graph_config
    )
    response_text = result.get("response", "")
    audio_bytes   = None

    if voice_mode and response_text:
        try:
            audio_bytes = generate_tts_bytes(response_text)
        except Exception as e:
            st.warning(f"TTS failed: {e}")

    return response_text, audio_bytes


# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------

st.markdown('<div class="input-area">', unsafe_allow_html=True)

col_input, col_btn = st.columns([10, 1])

with col_input:
    user_text = st.text_input(
        label="message",
        placeholder="Ask the Pharaoh...",
        label_visibility="collapsed",
        key="text_input"
    )

with col_btn:
    if user_text.strip():
        send_clicked = st.button("➤", key="send_btn", help="Send")
        if send_clicked and user_text.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_text.strip()})
            with st.spinner(""):
                response, audio = run_graph(user_text.strip(), voice_mode=False)
            st.session_state.chat_history.append({"role": "pharaoh", "content": response, "audio": audio})
            st.rerun()
    else:
        if st.session_state.recording:
            cancel = st.button("✕", key="cancel_btn", help="Cancel recording")
            if cancel:
                st.session_state.recording = False
                st.rerun()
        else:
            mic = st.button("🎙", key="mic_btn", help="Hold to speak")
            if mic:
                st.session_state.recording = True
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Voice recording flow
# ---------------------------------------------------------------------------

if st.session_state.recording:
    st.markdown('<div class="stt-status">🔴 Listening... (silence auto-stops)</div>', unsafe_allow_html=True)

    try:
        audio      = record_until_silence()
        user_input = transcribe_audio(audio)
        st.session_state.recording = False

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner(""):
                response, audio_bytes = run_graph(user_input, voice_mode=True)
            st.session_state.chat_history.append({"role": "pharaoh", "content": response, "audio": audio_bytes})

    except Exception as e:
        st.session_state.recording = False
        st.error(f"Voice failed: {e}")

    st.rerun()