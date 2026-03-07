import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
import warnings
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
import numpy as np
from sqlalchemy.orm import Session
from src.db.session import engine
from src.models import Pharaoh
from sqlalchemy import text
import yaml
import streamlit as st

warnings.filterwarnings("ignore")
load_dotenv()

# --- RESOURCES ---
def load_resources():
    base_path = Path(__file__).parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()

# --- CONSTANTS ---
ENTITY_NAME = "Ramesses II"
EMBEDDING_DIM = 768
GROQ_GENERATOR_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_QUERY_RERWRITER_MODEL_NAME = "qwen/qwen3-32b"

# --- CORE LOGIC ---
embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=os.getenv("R2_ACCOUNT_ID"),
    api_token=os.getenv("CF_AI_API"),
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)

def get_embedding(text_str: str):
    embeddings = np.array(embedding_model.embed_query(text_str))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced_embedding = embeddings[:EMBEDDING_DIM]
    norm = np.linalg.norm(sliced_embedding)
    final_embedding = sliced_embedding / norm if norm > 0 else sliced_embedding
    return final_embedding.tolist()

query_rewriter_llm = ChatGroq(
    model_name=GROQ_QUERY_RERWRITER_MODEL_NAME,
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY1"),
    extra_body={"reasoning_effort": "default", "reasoning_format": "hidden"} 
)

generator_llm = ChatGroq(
    model_name=GROQ_GENERATOR_MODEL_NAME,
    temperature=0.8,
    api_key=os.getenv("GROQ_API_KEY2"),
    extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"}
)

class AgentState(TypedDict):
    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str
    stream_iterator: object # New key to pass the stream to the UI

# ... (Keep all imports and setup identical)

#  PROMPT & CHAIN
rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
rewrite_chain = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()

llm_prompt_template = PromptTemplate.from_template(PROMPTS['assistant_persona'])
llm_chain = llm_prompt_template | generator_llm | StrOutputParser()

# --- GRAPH NODES ---
def rewrite_node(state: AgentState) -> dict:
    clean_dialouge = [msg for msg in state['messages'][:-1] if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "search_query"]
    history_window = clean_dialouge[-10:] if clean_dialouge else []
    dialogue = [f"{'User ' if isinstance(msg, HumanMessage) else 'Search Query '}: {msg.content}" for msg in history_window]
    history_str = "\n".join(dialogue) if dialogue else "No history yet."
    
    rewrite_prompt = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
    chain = rewrite_prompt | query_rewriter_llm | StrOutputParser()
    search_q = chain.invoke({"query": state['query'], "pharaoh_name": ENTITY_NAME, "chat_history": history_str}).replace("Search Query:", "").strip()
    return {"messages": [AIMessage(content=search_q, name="search_query")], "search_query": search_q}

def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['search_query'])
    with Session(engine) as session:
        pharaoh = session.query(Pharaoh).filter_by(name=ENTITY_NAME).first()
        if not pharaoh: return {"context": []}
        result = session.execute(text(VECTOR_SQL), {"p_id": pharaoh.id, "embedding": str(query_embedding), "limit": 3})
        context = [row[0] for row in result]
    return {"context": context}

def generate_node(state: AgentState) -> dict:
    # We do NOT run the LLM here to avoid serialization errors.
    # We just prepare the history string for the UI to use.
    clean_dialogue = [msg for msg in state['messages'] if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "generator_response"]
    history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else []
    dialogue = [f"{'User ' if isinstance(msg, HumanMessage) else ENTITY_NAME}: {msg.content}" for msg in history_window]
    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."
    
    # Pass the string to the state so the main loop can access it
    return {"response": history_str}

# --- BUILD GRAPH (UNTOUCHED) ---
workflow = StateGraph(AgentState)
workflow.add_node("rewriter", rewrite_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)
workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# --- STREAMLIT UI ---
st.set_page_config(page_title=f"Legacy of {ENTITY_NAME}", page_icon="🔱")

st.markdown("""
    <style>
    .stChatMessage { direction: rtl; text-align: right; }
    .stChatMessage div { direction: ltr; text-align: left; }
    .stChatMessage [data-testid="stMarkdownContainer"] { direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title(f"🔱 Court of {ENTITY_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Speak, traveler..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": "streamlit_session_v2"}}
        
        # 1. Run the Graph UP TO the generator node
        # This gets our context and history string ready
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "query": user_input,
            "context": []
        }
        final_graph_state = graph.invoke(initial_state, config=config)
        
        # 2. Stream directly from the LLM Chain using the state data
        # This keeps the "generator object" out of LangGraph's memory
        stream = llm_chain.stream({
            "pharaoh_name": ENTITY_NAME,
            "context": "\n\n".join(final_graph_state['context']),
            "query": user_input,
            "chat_history": final_graph_state['response'], # This is the history_str we saved
        })
        
        response_text = st.write_stream(stream)
        
        # 3. Manually update LangGraph memory so it remembers the answer
        graph.update_state(config, {
            "messages": [AIMessage(content=response_text, name="generator_response")],
            "response": response_text
        })
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})