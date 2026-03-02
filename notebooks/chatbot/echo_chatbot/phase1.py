import gc
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
import chromadb
from pathlib import Path
import numpy as np
import bitsandbytes

CHROMA_DB_PATH = Path(r"C:\Uni\4th Year\GP\ECHO\data\chatbot\embeddings\pharaohs_qwen_MRL_768_db") 
COLLECTION_NAME = "pharaohs"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
GENERATION_MODEL_NAME ="mistralai/Mistral-7B-Instruct-v0.3"
TOP_K = 3
EMBEDDING_DIM = 768
device = "cuda" if torch.cuda.is_available() else "cpu"


# LangGraph state definition
class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    context: List[str]
    response: str

#Embedding model
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    tokenizer_kwargs={"padding_side": "left"},
    device="cpu"
)

def get_embedding(text: str):
    embeddings = embedding_model.encode([text], normalize_embeddings=True)
    embeddings = embeddings[:, :EMBEDDING_DIM]
    query_embedding = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return query_embedding[0].tolist()
    
#ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# LLM
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model.to(device).eval()

def generate_text(prompt: str):
    prompt = str(prompt)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    ).to(model.device)
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.6,
            min_new_tokens=10,
            max_new_tokens=256,
        )
        input_length = inputs["input_ids"].shape[1]
        generated_ids = out_ids[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


    torch.cuda.empty_cache()
    gc.collect()
    
    return response


prompt_template = PromptTemplate.from_template("""
You are a helpful historical assistant. 
Answer the question based ONLY on the provided context. 
Do not create multiple choice options. 
Do not list steps. 
Do not repeat the context. 
Give a direct, concise answer.

Context:
{context}

Question: {query}

Answer:""")

chain = prompt_template | RunnableLambda(generate_text) | StrOutputParser()

#LangGraph nodes definition
def retrieve_node(state: AgentState) -> dict:    
    query_embedding = get_embedding(state['query'])
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
         where={"entity_name": {"$eq": "Tutankhamun.txt"}}
    )
    
    context = results["documents"][0] if results["documents"] else []
    
    return {"context": context}

def generate_node(state: AgentState) -> dict:
    print("Generating response...")
    
    response = chain.invoke({
        "context": "\n\n".join(state['context']),
        "query": state['query']
    })
    
    ai_message = AIMessage(content=response)
    
    return {
        "messages": [ai_message],
        "response": response
    }

# Constructing the state graph
state_graph = StateGraph(AgentState)
state_graph.add_node("retriever", retrieve_node)
state_graph.add_node("generator",generate_node)

state_graph.set_entry_point("retriever")
state_graph.add_edge("retriever", "generator")
state_graph.add_edge("generator", END)

graph=state_graph.compile()

def main():
    print("RAG Chatbot Ready (Type 'quit' to exit)")
    while True:
        user_query = input("You: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        initial_state = {
            "query": user_query,
            "messages": [HumanMessage(content=user_query)],
            "context": [],
            "response": ""
        }
        
        result = graph.invoke(initial_state)
        
        print(f"\nAssistant: {result['response']}\n")
        
if __name__ == "__main__":
    main() 