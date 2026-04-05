from __future__ import annotations

from asyncio import run
from io import BytesIO
from os import getenv
from pathlib import Path
from re import sub
from time import perf_counter
from typing import Annotated, TypedDict

import numpy as np
from dotenv import load_dotenv
from edge_tts import Communicate
from groq import Groq
from langdetect import detect
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from sqlalchemy import text
from sqlalchemy.orm import Session
from yaml import safe_load

from src.db import engine

load_dotenv()


class AgentState(TypedDict):
    session_id: str
    query: str
    search_query: str
    messages: Annotated[list, add_messages]
    context: list[str]
    response: str


class EchoChatbotRuntime:
    ENTITY_CONFIG = {
        "pharaoh": {
            "texts_table": "pharaohs_texts",
            "entities_table": "pharaohs",
            "entity_id_col": "pharaoh_id",
            "prompt_key": "pharaoh",
            "name_key": "pharaoh_name",
        },
        "landmark": {
            "texts_table": "landmarks_texts",
            "entities_table": "landmarks",
            "entity_id_col": "landmark_id",
            "prompt_key": "landmark",
            "name_key": "landmark_name",
        },
    }

    LANG_TO_VOICE = {
        "en": {"male": "en-CA-LiamNeural", "female": "en-US-JennyNeural"},
        "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
        "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
        "ar": {"male": "ar-EG-ShakirNeural", "female": "ar-BH-LailaNeural"},
        "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
        "it": {"male": "it-IT-DiegoNeural", "female": "it-IT-ElsaNeural"},
        "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
    }

    DEFAULT_VOICE = "en-US-ChristopherNeural"
    GROQ_GENERATOR_MODEL_NAME = "openai/gpt-oss-120b"
    GROQ_QUERY_REWRITER_MODEL_NAME = "qwen/qwen3-32b"
    GROQ_STT_MODEL_NAME = "whisper-large-v3"
    EDGE_TTS_RATE = "+9%"
    EDGE_TTS_PITCH = "-10Hz"
    TOP_K = 3
    EMBEDDING_DIM = 768

    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.resources_dir = (
            self.repo_root / "experiments" / "chatbot" / "echo_chatbot" / "resources"
        )
        self.sql_template, self.prompts = self._load_resources()
        self.sessions: dict[str, dict[str, object]] = {}
        self.qwen_model = None
        self.hf_token = getenv("HF_TOKEN")

        self.groq_client = Groq(api_key=getenv("GROQ_API_KEY1"))
        self.query_rewriter_llm = ChatGroq(
            model_name=self.GROQ_QUERY_REWRITER_MODEL_NAME,
            temperature=0.2,
            max_tokens=1024,
            api_key=getenv("GROQ_API_KEY1"),
            extra_body={"reasoning_effort": "none", "reasoning_format": "hidden"},
        )
        self.search_tool = TavilySearch(max_results=4, search_depth="advanced")
        self.tools = [self.search_tool]
        self.tool_node = ToolNode(tools=self.tools)
        self.generator_llm = ChatGroq(
            model_name=self.GROQ_GENERATOR_MODEL_NAME,
            temperature=0.8,
            max_tokens=4096,
            top_p=0.95,
            api_key=getenv("GROQ_API_KEY2"),
            extra_body={"reasoning_effort": "medium", "reasoning_format": "hidden"},
        ).bind_tools(self.tools)
        self.reranker = JinaRerank(
            model="jina-reranker-v3",
            top_n=self.TOP_K,
            jina_api_key=getenv("JINA_API_KEY"),
        )

        workflow = StateGraph(AgentState)
        workflow.add_node("rewriter", self.rewrite_node)
        workflow.add_node("retriever", self.retrieve_node)
        workflow.add_node("reranker", self.rerank_node)
        workflow.add_node("generator", self.generate_node)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("rewriter")
        workflow.add_edge("rewriter", "retriever")
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("reranker", "generator")
        workflow.add_conditional_edges(
            "generator",
            tools_condition,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "generator")
        self.graph = workflow.compile(checkpointer=MemorySaver())

    def _load_resources(self) -> tuple[str, dict]:
        with open(self.resources_dir / "queries.sql", "r", encoding="utf-8") as file:
            sql_template = file.read()
        with open(self.resources_dir / "prompts.yaml", "r", encoding="utf-8") as file:
            prompts = safe_load(file)
        return sql_template, prompts

    def ensure_models_loaded(self) -> None:
        if self.qwen_model is not None:
            return

        from sentence_transformers import SentenceTransformer
        import torch

        start = perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[chatbot] Loading embedding model on device={device}...", flush=True)
        self.qwen_model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device=device,
            tokenizer_kwargs={"padding_side": "left"},
            token=self.hf_token,
        )
        print(f"[chatbot] Embedding model ready in {perf_counter() - start:.2f}s", flush=True)

    def get_embedding(self, text_value: str) -> list[float]:
        self.ensure_models_loaded()
        embeddings = self.qwen_model.encode(
            text_value,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sliced = embeddings[: self.EMBEDDING_DIM]
        norm = np.linalg.norm(sliced)
        return (sliced / norm if norm > 0 else sliced).tolist()

    def clean_for_tts(self, text_value: str) -> str:
        text_value = sub(r"\*\*(.+?)\*\*", r"\1", text_value)
        text_value = sub(r"\*(.+?)\*", r"\1", text_value)
        text_value = sub(r"#{1,6}\s*", "", text_value)
        return sub(r"\n+", " ", text_value).strip()

    def resolve_entity(self, entity_type: str, entity_name: str) -> tuple[int, str]:
        table_name = "pharaohs" if entity_type == "pharaoh" else "landmarks"

        with Session(engine) as session:
            if entity_type == "pharaoh":
                result = session.execute(
                    text(f"SELECT id, gender FROM {table_name} WHERE name = :name"),
                    {"name": entity_name},
                ).fetchone()
            else:
                result = session.execute(
                    text(f"SELECT id FROM {table_name} WHERE name = :name"),
                    {"name": entity_name},
                ).fetchone()

        if not result:
            raise ValueError(f"{entity_type.title()} '{entity_name}' was not found in the database.")

        entity_id = result[0]
        gender = result[1] if entity_type == "pharaoh" and result[1] else "male"
        return entity_id, gender

    def resolve_optional_gender(self, entity_type: str | None, entity_name: str | None) -> str:
        if entity_type != "pharaoh" or not entity_name:
            return "male"

        with Session(engine) as session:
            result = session.execute(
                text("SELECT gender FROM pharaohs WHERE name = :name"),
                {"name": entity_name},
            ).fetchone()

        if not result or not result[0]:
            return "male"
        return result[0]

    def build_rewrite_chain(self, entity_type: str):
        prompt_key = self.ENTITY_CONFIG[entity_type]["prompt_key"]
        return (
            PromptTemplate.from_template(self.prompts["rewrite_prompt"][prompt_key])
            | self.query_rewriter_llm
            | StrOutputParser()
        )

    def build_persona_template(self, entity_type: str):
        prompt_key = self.ENTITY_CONFIG[entity_type]["prompt_key"]
        return PromptTemplate.from_template(self.prompts["assistant_persona"][prompt_key])

    def get_vector_sql(self, entity_type: str) -> str:
        cfg = self.ENTITY_CONFIG[entity_type]
        return self.sql_template.format(
            texts_table=cfg["texts_table"],
            entity_id_col=cfg["entity_id_col"],
        )

    def _get_session_context(self, session_id: str) -> dict[str, object]:
        session_context = self.sessions.get(session_id)
        if not session_context:
            raise RuntimeError(f"Session '{session_id}' is not initialized.")
        return session_context

    def _format_memory_block(self, session_context: dict[str, object]) -> str:
        session_memory = session_context.get("user_memory", [])
        if session_memory:
            user_info_str = ""
            for item in session_memory:
                if "=" in item:
                    key, value = item.split("=", 1)
                    user_info_str += f"\n- {key.strip()}: {value.strip()}"
            return user_info_str
        return "No user information available yet."

    def rewrite_node(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        session_context = self._get_session_context(session_id)
        entity_type = str(session_context["entity_type"])
        entity_name = str(session_context["entity_name"])
        session_memory = session_context.setdefault("user_memory", [])
        rewrite_chain = self.build_rewrite_chain(entity_type)

        clean_dialogue = [
            msg
            for msg in state["messages"][:-1]
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

        name_key = self.ENTITY_CONFIG[entity_type]["name_key"]
        response = rewrite_chain.invoke(
            {
                name_key: entity_name,
                "chat_history": history_str,
                "query": state["query"],
            }
        ).strip()

        if "[MEMORY]:" in response:
            parts = response.split("[MEMORY]:", 1)
            search_query = parts[0].replace("Search Query:", "").strip()
            memory_items = [item.strip() for item in parts[1].strip().split(",")]
            for item in memory_items:
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                memory_entry = f"{key.strip()}={value.strip()}"
                session_memory = [m for m in session_memory if not m.startswith(f"{key.strip()}=")]
                session_memory.append(memory_entry)
            session_context["user_memory"] = session_memory
        else:
            search_query = response.replace("Search Query:", "").strip()

        if not search_query:
            search_query = state["query"]

        return {
            "messages": [AIMessage(content=search_query, name="search_query")],
            "search_query": search_query,
        }

    def retrieve_node(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        session_context = self._get_session_context(session_id)
        query_embedding = self.get_embedding(state["search_query"])
        vector_sql = self.get_vector_sql(str(session_context["entity_type"]))

        with Session(engine) as session:
            result = session.execute(
                text(vector_sql),
                {"entity_id": int(session_context["entity_id"]), "embedding": str(query_embedding)},
            )
            context = [row[0] for row in result]

        return {"context": context}

    def rerank_node(self, state: AgentState) -> dict:
        docs = [Document(page_content=chunk) for chunk in state["context"]]
        reranked = self.reranker.compress_documents(docs, query=state["search_query"])
        return {"context": [doc.page_content for doc in reranked]}

    def generate_node(self, state: AgentState) -> dict:
        response_text, tool_calls = self._generate_response(state)
        if tool_calls:
            return {"messages": [AIMessage(content=response_text, tool_calls=tool_calls)]}
        return {
            "messages": [AIMessage(content=response_text, name="generator_response")],
            "response": response_text,
        }

    def _build_generation_payload(self, state: AgentState) -> tuple[str | None, str, bool]:
        session_id = state["session_id"]
        session_context = self._get_session_context(session_id)
        entity_type = str(session_context["entity_type"])
        entity_name = str(session_context["entity_name"])

        if "OUT_OF_SCOPE" in state.get("search_query", ""):
            out_of_scope = (
                "I'm sorry but you speak of a time that is not mine. My eyes see only the borders of my own reign."
                if entity_type == "pharaoh"
                else "I'm sorry, that lies beyond what my stones remember."
            )
            return out_of_scope, "", False

        last_human_index = -1
        for index, message in enumerate(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_index = index

        current_turn_messages = state["messages"][last_human_index:] if last_human_index != -1 else []
        has_searched = any(isinstance(message, ToolMessage) for message in current_turn_messages)
        current_search_results = [
            message.content for message in current_turn_messages if isinstance(message, ToolMessage)
        ]
        combined_context = current_search_results + state["context"] if has_searched else state["context"]

        clean_dialogue = [
            message
            for message in state["messages"]
            if isinstance(message, HumanMessage) or getattr(message, "name", None) == "generator_response"
        ]
        history_window = clean_dialogue[-11:-1] if len(clean_dialogue) > 1 else []

        dialogue = []
        for message in history_window:
            if isinstance(message, HumanMessage):
                dialogue.append(f"User: {message.content}")
            elif getattr(message, "name", None) == "generator_response":
                dialogue.append(f"{entity_name}: {message.content}")

        history_str = "\n".join(dialogue) if dialogue else "No previous conversation."
        user_info_str = self._format_memory_block(session_context)

        extra_instruction = ""
        if has_searched:
            extra_instruction = (
                "\n\nIMPORTANT: You have already consulted the modern scrolls (Search Tool). "
                "Do not call the search tool again. Answer strictly from the context provided. "
                "Only if the answer is still missing, say: 'The gods have veiled that specific moment from my sight for now.'"
            )

        name_key = self.ENTITY_CONFIG[entity_type]["name_key"]
        persona_template = self.build_persona_template(entity_type)
        prompt = persona_template.format(
            **{
                name_key: entity_name,
                "context": "\n\n".join(combined_context),
                "query": state["query"],
                "chat_history": history_str,
                "user_info": user_info_str,
            }
        ) + extra_instruction

        return None, prompt, has_searched

    def _generate_response(self, state: AgentState) -> tuple[str, list]:
        out_of_scope_text, prompt, has_searched = self._build_generation_payload(state)
        if out_of_scope_text is not None:
            return out_of_scope_text, []

        response = self.generator_llm.invoke(prompt)
        if response.tool_calls and not has_searched:
            return response.content, response.tool_calls
        return response.content, []

    def _stream_generation(self, state: AgentState):
        out_of_scope_text, prompt, has_searched = self._build_generation_payload(state)
        if out_of_scope_text is not None:
            yield {"type": "token", "content": out_of_scope_text}
            return out_of_scope_text, []

        full_content = ""
        tool_calls_buf = None

        for chunk in self.generator_llm.stream(prompt):
            if chunk.content:
                full_content += chunk.content
                yield {"type": "token", "content": chunk.content}
            if chunk.tool_calls:
                tool_calls_buf = chunk.tool_calls

        if tool_calls_buf and not has_searched:
            return full_content, tool_calls_buf
        return full_content, []

    def _invoke_search_tool(self, tool_call: dict) -> ToolMessage:
        result = self.search_tool.invoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

    def _persist_session_messages(self, session_id: str, messages: list) -> None:
        session_context = self._get_session_context(session_id)
        session_context["messages"] = messages

    def stream_chat(
        self,
        *,
        session_id: str,
        entity_type: str,
        entity_name: str,
        message: str,
    ):
        total_start = perf_counter()
        print(
            f"[chatbot] /chat start session={session_id} entity_type={entity_type} entity_name={entity_name}",
            flush=True,
        )

        step_start = perf_counter()
        entity_id, gender = self.resolve_entity(entity_type, entity_name)
        print(f"[chatbot] resolve_entity: {perf_counter() - step_start:.2f}s", flush=True)

        session_context = self.sessions.get(session_id)
        if (
            session_context is None
            or session_context.get("entity_type") != entity_type
            or session_context.get("entity_name") != entity_name
        ):
            self.sessions[session_id] = {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "entity_id": entity_id,
                "gender": gender,
                "user_memory": [],
                "messages": [],
            }
        else:
            session_context["entity_id"] = entity_id
            session_context["gender"] = gender

        session_context = self._get_session_context(session_id)
        messages = list(session_context.get("messages", []))
        messages.append(HumanMessage(content=message))

        state: AgentState = {
            "session_id": session_id,
            "messages": messages,
            "query": message,
            "search_query": "",
            "context": [],
            "response": "",
        }

        step_start = perf_counter()
        rewrite_result = self.rewrite_node(state)
        print(f"[chatbot] rewrite_node: {perf_counter() - step_start:.2f}s", flush=True)
        state["messages"] = state["messages"] + rewrite_result["messages"]
        state["search_query"] = rewrite_result["search_query"]

        step_start = perf_counter()
        retrieve_result = self.retrieve_node(state)
        print(f"[chatbot] retrieve_node: {perf_counter() - step_start:.2f}s", flush=True)
        state["context"] = retrieve_result["context"]

        step_start = perf_counter()
        rerank_result = self.rerank_node(state)
        print(f"[chatbot] rerank_node: {perf_counter() - step_start:.2f}s", flush=True)
        state["context"] = rerank_result["context"]

        print("[chatbot] generation stream starting...", flush=True)
        stream_start = perf_counter()
        stream_generator = self._stream_generation(state)
        try:
            while True:
                event = next(stream_generator)
                if event["type"] == "token":
                    yield f"data: {event['content']}\n\n"
        except StopIteration as stop:
            stream_result = stop.value
        print(f"[chatbot] generation stream pass: {perf_counter() - stream_start:.2f}s", flush=True)
        if isinstance(stream_result, tuple):
            final_text, streamed_tool_calls = stream_result
        else:
            final_text, streamed_tool_calls = "", []

        if streamed_tool_calls:
            print("[chatbot] streamed generation requested tool call", flush=True)
            tool_message = self._invoke_search_tool(streamed_tool_calls[0])
            state["messages"] = state["messages"] + [
                AIMessage(content=final_text, tool_calls=streamed_tool_calls),
                tool_message,
            ]
            stream_start = perf_counter()
            stream_generator = self._stream_generation(state)
            try:
                while True:
                    event = next(stream_generator)
                    if event["type"] == "token":
                        yield f"data: {event['content']}\n\n"
            except StopIteration as stop:
                stream_result = stop.value
            print(
                f"[chatbot] generation stream second pass: {perf_counter() - stream_start:.2f}s",
                flush=True,
            )
            if isinstance(stream_result, tuple):
                final_text, _ = stream_result

        state["messages"] = state["messages"] + [AIMessage(content=final_text, name="generator_response")]
        self._persist_session_messages(session_id, state["messages"])
        print(f"[chatbot] /chat done in {perf_counter() - total_start:.2f}s", flush=True)
        yield "event: done\ndata: [DONE]\n\n"

    def chat(
        self,
        *,
        session_id: str,
        entity_type: str,
        entity_name: str,
        message: str,
    ) -> tuple[int, str]:
        entity_id, gender = self.resolve_entity(entity_type, entity_name)
        session_context = self.sessions.get(session_id)
        if (
            session_context is None
            or session_context.get("entity_type") != entity_type
            or session_context.get("entity_name") != entity_name
        ):
            self.sessions[session_id] = {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "entity_id": entity_id,
                "gender": gender,
                "user_memory": [],
            }
        else:
            session_context["entity_id"] = entity_id
            session_context["gender"] = gender

        result = self.graph.invoke(
            {
                "session_id": session_id,
                "messages": [("user", message)],
                "query": message,
                "search_query": "",
                "context": [],
                "response": "",
            },
            config={"configurable": {"thread_id": session_id}},
        )

        reply_text = result.get("response")
        if not reply_text:
            raise RuntimeError("Chatbot runtime returned no response text.")
        return entity_id, reply_text

    def transcribe_audio(self, filename: str, audio_bytes: bytes) -> str:
        transcription = self.groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=self.GROQ_STT_MODEL_NAME,
            temperature=0,
        )
        text_value = transcription.text.strip()
        if not text_value:
            raise RuntimeError("Transcription returned empty text.")
        return text_value

    def synthesize_speech(
        self, text_value: str, entity_type: str | None = None, entity_name: str | None = None
    ) -> tuple[bytes, str, str]:
        clean_text = self.clean_for_tts(text_value)
        gender = self.resolve_optional_gender(entity_type, entity_name)

        try:
            language = detect(clean_text)
            lang_voices = self.LANG_TO_VOICE.get(language, self.LANG_TO_VOICE["en"])
            voice = lang_voices.get(gender, self.LANG_TO_VOICE["en"]["male"])
        except Exception:
            language = "en"
            voice = self.DEFAULT_VOICE

        audio_buffer = BytesIO()

        async def generate() -> None:
            communicate = Communicate(
                clean_text,
                voice,
                rate=self.EDGE_TTS_RATE,
                pitch=self.EDGE_TTS_PITCH,
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

        run(generate())
        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            raise RuntimeError("Speech synthesis produced no audio.")

        return audio_bytes, language, voice


chatbot_runtime = EchoChatbotRuntime()
