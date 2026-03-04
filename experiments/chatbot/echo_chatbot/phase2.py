import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
import gc
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
from langchain_groq import ChatGroq  # New Cloud Provider
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import numpy as np
from sqlalchemy.orm import Session
from src.db.session import engine
from src.models import Pharaoh, PharaohText, Landmark, LandmarkText
from sqlalchemy import text
import yaml
warnings.filterwarnings("ignore")


def load_resources():
    base_path = Path(__file__).parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
        
    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()
COLLECTION_NAME = "pharaohs"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_PATH = "models\Qwen3-Embedding-0.6B"
GROQ_API_KEY1= os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2= os.getenv("GROQ_API_KEY2")
GROQ_GENERATOR_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_QUERY_RERWRITER_MODEL_NAME = "qwen/qwen3-32b"
TOP_K = 3
EMBEDDING_DIM = 768
ENTITY_NAME="Ramesses II"

load_dotenv()

# LangGraph state definition
class AgentState(TypedDict):
    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str

#Embedding model
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    trust_remote_code=True,
    device="cuda"
)

def get_embedding(text: str):
    embeddings = embedding_model.encode([text], normalize_embeddings=True)
    embeddings = embeddings[:, :EMBEDDING_DIM]
    query_embedding = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return query_embedding[0].tolist()


# Cloud LLM
query_rewriter_llm = ChatGroq(
    model_name=GROQ_QUERY_RERWRITER_MODEL_NAME,
    temperature=0.2,
    max_tokens=1024,
    api_key= GROQ_API_KEY1,
     extra_body={
        "reasoning_effort": "default",
        "reasoning_format": "hidden" # Use hidden for the rewriter to keep it clean
    } 
)

generator_llm = ChatGroq(
    model_name=GROQ_GENERATOR_MODEL_NAME,
    temperature=0.4,
    max_tokens=4096,
    top_p=0.95,
    api_key=GROQ_API_KEY2,   
)

#  PROMPT & CHAIN
rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
rewrite_chain = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()

llm_prompt_template = PromptTemplate.from_template(PROMPTS['assistant_persona'])
llm_chain = llm_prompt_template | generator_llm | StrOutputParser()


#LangGraph nodes definition
def rewrite_node(state: AgentState) -> dict:
    dialogue = []
    for msg in state['messages'][:-1]:
        if isinstance(msg, HumanMessage):
            role = "User "
            dialogue.append(f"{role}: {msg.content}")
        elif getattr(msg, "name", None) == "search_query":
            role = "Search Query "
            dialogue.append(f"{role}: {msg.content}")

    history_str = "\n".join(dialogue) if dialogue else "No history yet."

    print("\n\n")
    print("="*50)
    print("Rewrite History",history_str)
    print("="*50)
    print("\n\n")


    search_q = rewrite_chain.invoke({"query": state['query'],
                                     "pharaoh_name": ENTITY_NAME,
                                     "chat_history":history_str}).replace("Search Query:", "").strip()

    return {"messages": [AIMessage(content=search_q, name="search_query")], #add metadata
            "search_query": search_q}

def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['search_query'])
    
    with Session(engine) as session:
        
        pharaoh = session.query(Pharaoh).filter_by(name=ENTITY_NAME).first()
        if not pharaoh: return {"context": []}

        raw_query = text(VECTOR_SQL)
        result = session.execute(raw_query, {
            "p_id": pharaoh.id,
            "embedding": str(query_embedding),
            "limit": TOP_K
        })

        context = [row[0] for row in result]

    return {"context": context}

def generate_node(state: AgentState) -> dict:
    dialogue = []
    for msg in state['messages'][:-1]: #skupping the last message which is the current user query
        if isinstance(msg, HumanMessage) :
            role = "User "
            dialogue.append(f"{role}: {msg.content}")
        elif getattr(msg, "name", None) == "generator_response":
            role = ENTITY_NAME
            dialogue.append(f"{role}: {msg.content}")

    
    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."

    print("\n\n")
    print("="*50)
    print("Generator History",history_str)
    print("="*50)
    print("\n\n")

    print(ENTITY_NAME, ": ")
    response_text=""
    for chunk in llm_chain.stream({ #Token-level LLM stream, not LG
        "pharaoh_name": ENTITY_NAME,
        "context": "\n\n".join(state['context']),
        "query": state['search_query'],
        "chat_history": history_str,
    }):
        print(chunk, end="", flush=True)
        response_text += chunk
    
    print() 
    
    return {
        "messages": [AIMessage(content=response_text,name="generator_response")],
        "response": response_text
    }

# --- GRAPH CONSTRUCTION ---
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

# Visualize graph
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png(output_file_path="graph.png")))

def main():
    print("Agentic RAG Ready (Streaming & Persistent Memory):")
    
    # This ID represents "User 1's Chat Room"
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']: break
        
        inital_state={
            "messages": [("user", user_input)], #automatically converted to HumanMessage by the Annotated type
            "query": user_input,
            "context": []}
        
        graph.invoke(inital_state,config=config)

if __name__ == "__main__":
    main()