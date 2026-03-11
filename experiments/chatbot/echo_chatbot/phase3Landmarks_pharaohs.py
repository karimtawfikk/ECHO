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
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from src.db.session import engine
from sqlalchemy import text
import yaml
import numpy as np
warnings.filterwarnings("ignore")

load_dotenv()

ENTITY_TYPE = "landmark"     
ENTITY_NAME = "Pyramids of Giza"  

def load_resources():
    base_path = Path(__file__).parent / "resources"
    with open(base_path / "landpharoqueries.sql", "r") as f:
        sql_query = f.read()
    with open(base_path / "landpharoprompt.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return prompts, sql_query


PROMPTS, VECTOR_SQL = load_resources()
GROQ_API_KEY1 = os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2")

CF_WORKERSAI_ACCOUNTID = os.getenv("R2_ACCOUNT_ID")
CF_AI_API = os.getenv("CF_AI_API")

CF_RERANKER_ACCOUNTID=os.getenv("CF_ACCOUNT_ID")
CF_RERANKER_API=os.getenv("CF_RERANKER_API")
JINA_API_KEY = os.getenv("JINA_API_KEY")

GROQ_GENERATOR_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_QUERY_RERWRITER_MODEL_NAME = "qwen/qwen3-32b"
TOP_K = 3
EMBEDDING_DIM = 768


# LangGraph State
class AgentState(TypedDict):
    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str
    entity_type: str
    entity_name: str


# Embedding Model 
embedding_model = CloudflareWorkersAIEmbeddings(
    account_id=CF_WORKERSAI_ACCOUNTID,
    api_token=CF_AI_API,
    model_name="@cf/qwen/qwen3-embedding-0.6b"
)


# Reranker Model
reranker = JinaRerank(
    model="jina-reranker-v3",
    top_n=TOP_K,
    jina_api_key=JINA_API_KEY
)


def get_embedding(text: str):
    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)
    sliced_embedding = embeddings[:EMBEDDING_DIM]

    norm = np.linalg.norm(sliced_embedding)
    final_embedding = sliced_embedding / norm if norm > 0 else sliced_embedding

    return final_embedding.tolist()


# -------- LLMs --------
query_rewriter_llm = ChatGroq(
    model_name=GROQ_QUERY_RERWRITER_MODEL_NAME,
    temperature=0.2,
    max_tokens=1024,
    api_key=GROQ_API_KEY1,
     extra_body={
        "reasoning_effort": "default",
          "reasoning_format": "hidden"
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
def get_rewrite_chain(entity_type):

    if entity_type == "pharaoh":
        prompt = PROMPTS["rewrite_prompt"]["pharaoh"]
    else:
        prompt = PROMPTS["rewrite_prompt"]["landmark"]

    template = PromptTemplate.from_template(prompt)
    return template | query_rewriter_llm | StrOutputParser()


def get_llm_chain(entity_type):

    if entity_type == "pharaoh":
        prompt = PROMPTS["assistant_persona"]["pharaoh"]
    else:
        prompt = PROMPTS["assistant_persona"]["landmark"]

    template = PromptTemplate.from_template(prompt)
    return template | generator_llm | StrOutputParser()


#LangGraph nodes definition
def rewrite_node(state: AgentState) -> dict:

    rewrite_chain = get_rewrite_chain(state["entity_type"])

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
        "query": state["query"],
        "pharaoh_name": state["entity_name"],
        "landmark_name": state["entity_name"],
        "chat_history": history_str
    }).replace("Search Query:", "").strip()

    return {
        "messages": [AIMessage(content=search_q, name="search_query")],
        "search_query": search_q
    }


# Retrieve Node
def retrieve_node(state: AgentState) -> dict:
    query_embedding = get_embedding(state['search_query'])

    if state["entity_type"] == "pharaoh":
        sql = VECTOR_SQL.format(
            texts_table="pharaohs_texts",
            entities_table="pharaohs",
            entity_id_col="pharaoh_id"
        )
    else:
        sql = VECTOR_SQL.format(
            texts_table="landmarks_texts",
            entities_table="landmarks",
            entity_id_col="landmark_id"
        )

    with Session(engine) as session:
        raw_query = text(sql)

        result = session.execute(raw_query, {
            "entity_name": state["entity_name"],
            "embedding": str(query_embedding)
        })

        context = [row[0] for row in result]

    return {"context": context}


# Rerank Node
def rerank_node(state: AgentState) -> dict:

    docs = [Document(page_content=chunk) for chunk in state['context']]
    reranked = reranker.compress_documents(docs, query=state['search_query'])

    return {"context": [doc.page_content for doc in reranked]}


# Generator Node
def generate_node(state: AgentState) -> dict:

    llm_chain = get_llm_chain(state["entity_type"])

    clean_dialogue = [
        msg for msg in state['messages']
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "generator_response"
    ]

    history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else []

    dialogue = []
    for msg in history_window:
        if isinstance(msg, HumanMessage):
            role = "User "
            dialogue.append(f"{role}: {msg.content}")
        elif getattr(msg, "name", None) == "generator_response":
            role = ENTITY_NAME
            dialogue.append(f"{role}: {msg.content}")
            
    history_str = "\n".join(dialogue) if dialogue else "No previous conversation."

    print(ENTITY_NAME, ": ")
    response_text = ""
    for chunk in llm_chain.stream({
        "pharaoh_name": ENTITY_NAME,
        "landmark_name": ENTITY_NAME,
        "context": "\n\n".join(state['context']),
        "query": state['query'],
        "chat_history": history_str,
    }):
        print(chunk, end="", flush=True)
        response_text += chunk

    print()

    return {
        "messages": [AIMessage(content=response_text, name="generator_response")],
        "response": response_text
    }


#Graph Definition
workflow = StateGraph(AgentState)
workflow.add_node("rewriter", rewrite_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("reranker", rerank_node)
workflow.add_node("generator", generate_node)

workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter", "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_edge("reranker", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


def main():

    print("Agentic RAG Ready (Pharaohs + Landmarks):")
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        initial_state = {
            "messages": [("user", user_input)],
            "query": user_input,
            "context": [],
            "entity_type": ENTITY_TYPE,
            "entity_name": ENTITY_NAME
        }

        graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()