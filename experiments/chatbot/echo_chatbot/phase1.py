import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
import gc
import warnings
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
CHROMA_DB_PATH = Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\pharaohs_qwen_MRL_768_db") 
COLLECTION_NAME = "pharaohs"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_PATH = "models\Qwen3-Embedding-0.6B"

GROQ_MODEL_NAME = "llama-3.1-8b-instant"
TOP_K = 3
EMBEDDING_DIM = 768
ENTITY_TYPE=""

load_dotenv()

# LangGraph state definition
class AgentState(TypedDict):
    query: str
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

# ChromaDB client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# Cloud LLM
llm = ChatGroq(
    model_name=GROQ_MODEL_NAME,
    temperature=0.4,
    max_tokens=512
)


# --- PROMPT & CHAIN ---
prompt_template = PromptTemplate.from_template(PROMPTS['assistant_persona'])
chain = prompt_template | llm | StrOutputParser()

#LangGraph nodes definition
def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['query'])
    
    with Session(engine) as session:
        NAME = "Ramesses II"
        pharaoh = session.query(Pharaoh).filter_by(name=NAME).first()
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
    # 1. Build history
    dialogue = []
    for msg in state['messages'][:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        dialogue.append(f"{role}: {msg.content}")
    
    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."

    # --- DEBUG SECTION ---
    print("\n" + "="*30)
    print("DEBUG: CHAT HISTORY PASSED TO LLM")
    print(history_str)
    print("="*30 + "\n")
    # ---------------------

    # 2. Invoke the chain
    response_text = chain.invoke({
        "context": "\n\n".join(state['context']),
        "query": state['query'],
        "chat_history": history_str
    })
    
    return {
        "messages": [AIMessage(content=response_text)],
        "response": response_text
    }

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver() # Initialize the "RAM" storage
graph = workflow.compile(checkpointer=memory)

def main():
    print("Agentic RAG Ready (Streaming & Persistent Memory):")
    
    # This ID represents "User 1's Chat Room"
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']: break
        
        # 'stream' will yield a dictionary every time a NODE finishes.
        for event in graph.stream(
            {"messages": [("user", user_input)], #automatically converted to HumanMessage by the Annotated type
             "query": user_input,
             "context": []}, 
            config=config
        ):
            for node_name, value in event.items():
                if node_name == "generator":
                    print(f"\nAssistant: {value['messages'][-1].content}\n")

if __name__ == "__main__":
    main()