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
from pathlib import Path
import numpy as np
from sqlalchemy.orm import Session
from src.db.session import engine
from sqlalchemy import text
import yaml
warnings.filterwarnings("ignore")

load_dotenv()

def load_resources():
    base_path = Path(__file__).parent / "resources"
    with open(base_path / "queries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
        
    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()
GROQ_API_KEY1= os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2= os.getenv("GROQ_API_KEY2")

CF_WORKERSAI_ACCOUNTID=os.getenv("R2_ACCOUNT_ID")
CF_AI_API=os.getenv("CF_AI_API")

GROQ_GENERATOR_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_QUERY_RERWRITER_MODEL_NAME = "qwen/qwen3-32b"
TOP_K = 3
EMBEDDING_DIM = 768
ENTITY_NAME="Ramesses II"


# LangGraph state definition
class AgentState(TypedDict):
    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str

#Embedding model
embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)

def get_embedding(text: str):
    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced_embedding = embeddings[:EMBEDDING_DIM]

    norm = np.linalg.norm(sliced_embedding)
    final_embedding = sliced_embedding / norm if norm > 0 else sliced_embedding

    return final_embedding.tolist()

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
    temperature=0.8,
    max_tokens=4096,
    top_p=0.95,
    api_key=GROQ_API_KEY2,   
    extra_body={
        "reasoning_effort": "medium",
        "reasoning_format": "hidden"
    }
)

#  PROMPT & CHAIN
rewrite_prompt_template = PromptTemplate.from_template(PROMPTS['rewrite_prompt'])
rewrite_chain = rewrite_prompt_template | query_rewriter_llm | StrOutputParser()

llm_prompt_template = PromptTemplate.from_template(PROMPTS['assistant_persona'])
llm_chain = llm_prompt_template | generator_llm | StrOutputParser()


#LangGraph nodes definition
def rewrite_node(state: AgentState) -> dict:

    clean_dialouge = [
        msg for msg in state['messages'][:-1] # Exclude current user input
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "search_query"
    ]
    
    history_window = clean_dialouge[-10:] if clean_dialouge else []

    dialogue = []
    for msg in history_window:
        if isinstance(msg, HumanMessage):
            role = "User "
            dialogue.append(f"{role}: {msg.content}")
        elif getattr(msg, "name", None) == "search_query":
            role = "Search Query "
            dialogue.append(f"{role}: {msg.content}")

    history_str = "\n".join(dialogue) if dialogue else "No history yet."

    """print("\n\n")
    print("="*50)
    print("Rewrite History: ",history_str)
    print("="*50)
    print("\n\n")"""


    search_q = rewrite_chain.invoke({"query": state['query'],
                                     "pharaoh_name": ENTITY_NAME,
                                     "chat_history":history_str}).replace("Search Query:", "").strip()

    return {"messages": [AIMessage(content=search_q, name="search_query")], #add metadata
            "search_query": search_q}

def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['search_query'])
    
    with Session(engine) as session:
        raw_query = text(VECTOR_SQL)
        result = session.execute(raw_query, {
                "pharoah_name": ENTITY_NAME,
                "embedding": str(query_embedding),
                "limit": TOP_K
                })

        context = [row[0] for row in result]

    return {"context": context}

def generate_node(state: AgentState) -> dict:
    
    clean_dialogue = [
        msg for msg in state['messages'] 
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "generator_response"
    ] #neglect search_query messages

    history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else [] #last 5 turns.(10 so 5 turns)

    dialogue = []
    for msg in history_window: 
        if isinstance(msg, HumanMessage) :
            role = "User "
            dialogue.append(f"{role}: {msg.content}")
        elif getattr(msg, "name", None) == "generator_response":
            role = ENTITY_NAME
            dialogue.append(f"{role}: {msg.content}")

    
    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."

    """print("\n\n")
    print("="*50)
    print("Generator History",history_str)
    print("="*50)
    print("\n\n")"""

    print(ENTITY_NAME, ": ")
    response_text=""
    for chunk in llm_chain.stream({ #Token-level LLM stream, not LG
        "pharaoh_name": ENTITY_NAME,
        "context": "\n\n".join(state['context']),
        "query": state['query'],
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