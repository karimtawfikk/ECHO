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
import numpy as np

warnings.filterwarnings("ignore")
load_dotenv()

# CONFIG

ENTITY_TYPE = "landmark"  
ENTITY_NAME = "Pyramids of Giza"

# RESOURCES

def load_resources():
    base_path = Path(__file__).parent / "resources"

    with open(base_path / "landpharoqueries.sql") as f:
        sql_query = f.read()

    with open(base_path / "landpharoprompt.yaml") as f:
        prompts = yaml.safe_load(f)

    return prompts, sql_query

PROMPTS, VECTOR_SQL = load_resources()

# ENV

GROQ_API_KEY1 = os.getenv("GROQ_API_KEY1")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY2")
CF_WORKERSAI_ACCOUNTID = os.getenv("R2_ACCOUNT_ID")
CF_AI_API = os.getenv("CF_AI_API")
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_GENERATOR_MODEL_NAME      = "openai/gpt-oss-120b"
GROQ_QUERY_REWRITER_MODEL_NAME = "qwen/qwen3-32b"
TOP_K = 3
EMBEDDING_DIM = 768

# STATE

class AgentState(TypedDict):

    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str
    entity_type: str
    entity_name: str

# MODELS

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
tools = [search_tool]
tool_node = ToolNode(tools=tools)

# LLMs

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

# CHAINS

def get_rewrite_chain(entity_type):

    prompt = PROMPTS["rewrite_prompt"][entity_type]
    template = PromptTemplate.from_template(prompt)
    return template | query_rewriter_llm | StrOutputParser()


def get_llm_prompt(entity_type):

    return PromptTemplate.from_template(
        PROMPTS["assistant_persona"][entity_type]
    )

# EMBEDDING

def get_embedding(text: str):

    embeddings = np.array(embedding_model.embed_query(text))
    embeddings = embeddings / np.linalg.norm(embeddings)

    sliced = embeddings[:EMBEDDING_DIM]
    norm = np.linalg.norm(sliced)

    return (sliced / norm if norm > 0 else sliced).tolist()

# NODES

def rewrite_node(state: AgentState):

    rewrite_chain = get_rewrite_chain(state["entity_type"])
    clean_dialogue = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, HumanMessage) or getattr(msg, "name", None) == "search_query"
    ]
    history_window = clean_dialogue[-10:]

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

    print("-"*50)
    print(search_q)
    print("-"*50)

    return {
        "messages":[AIMessage(content=search_q,name="search_query")],
        "search_query":search_q
    }

def retrieve_node(state: AgentState):
    query_embedding = get_embedding(state["search_query"])

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
        result = session.execute(
            text(sql),

            {
                "entity_name":state["entity_name"],
                "embedding":str(query_embedding)
            }
        )
        context=[row[0] for row in result]

    return {"context":context}

def rerank_node(state: AgentState):
    docs=[Document(page_content=chunk) for chunk in state["context"]]
    reranked=reranker.compress_documents(docs,query=state["search_query"])
    return {"context": [doc.page_content for doc in reranked]}

def generate_node(state: AgentState):

    if "OUT_OF_SCOPE" in state.get("search_query",""):
        response_text = "I'm sorry but you speak of a time that is not mine. My eyes see only the borders of my own reign."
        return {
            "messages": [AIMessage(content=response_text, name="irrelevant_query")],
            "response": response_text
        }

    llm_prompt=get_llm_prompt(state["entity_type"])
    last_human_index=-1

    for i,msg in enumerate(state["messages"]):
        if isinstance(msg,HumanMessage):
            last_human_index=i

    current_turn_messages=state["messages"][last_human_index:]
    has_searched=any(isinstance(msg,ToolMessage) for msg in current_turn_messages)

    search_results=[
        msg.content for msg in current_turn_messages
        if isinstance(msg,ToolMessage)
    ]

    combined_context=search_results+state["context"] if has_searched else state["context"]

    clean_dialogue=[
        msg for msg in state["messages"]
        if isinstance(msg,HumanMessage)
        or getattr(msg,"name",None)=="generator_response"
    ]
    history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else []

    dialogue=[]
    for msg in history_window:
        if isinstance(msg,HumanMessage):
            dialogue.append(f"User: {msg.content}")
        else:
            dialogue.append(f"{state['entity_name']}: {msg.content}")

    history_str="\n".join(dialogue) if dialogue else "No previous conversation."
    

    extra_instruction = ""
    if has_searched:
        extra_instruction = (
            "\n\nIMPORTANT: You have already consulted the modern scrolls (Search Tool). "
            "Do not call the search tool again. Answer strictly from the context provided. "
            "If the answer is still missing, say: 'The gods have veiled that specific moment from my sight for now.'"
        )


    prompt=llm_prompt.format(
        pharaoh_name=state["entity_name"],
        landmark_name=state["entity_name"],
        context="\n\n".join(combined_context),
        query=state["query"],
        chat_history=history_str
    ) + extra_instruction

    response=generator_llm.invoke(prompt)

    if response.tool_calls and not has_searched:
        print("\nConsulting modern scrolls...")
        return {"messages":[response]}

    print(f"\n{state['entity_name']}: {response.content}")
    return {
        "messages":[AIMessage(content=response.content,name="generator_response")],
        "response":response.content
    }

workflow=StateGraph(AgentState)
workflow.add_node("rewriter",rewrite_node)
workflow.add_node("retriever",retrieve_node)
workflow.add_node("reranker",rerank_node)
workflow.add_node("generator",generate_node)
workflow.add_node("tools",tool_node)

workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter","retriever")
workflow.add_edge("retriever","reranker")
workflow.add_edge("reranker","generator")
workflow.add_conditional_edges(
    "generator",
    tools_condition,
    {
        "tools":"tools",
        END:END
    }
)
workflow.add_edge("tools","generator")

memory=MemorySaver()
graph=workflow.compile(checkpointer=memory)

def main():

    print("Agentic RAG Ready (Pharaohs + Landmarks) - Phase 4:")
    config={"configurable":{"thread_id":"1"}}

    while True:
        user_input=input("\nUser: ").strip()
        if user_input.lower() in ["quit","exit","q"]:
            break
        initial_state={
            "messages":[("user",user_input)],
            "query":user_input,
            "context":[],
            "entity_type":ENTITY_TYPE,
            "entity_name":ENTITY_NAME
        }
        graph.invoke(initial_state,config=config)
if __name__=="__main__":
    main()