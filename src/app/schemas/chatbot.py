from typing import Optional
from pydantic import BaseModel

class ChatHistoryMessage(BaseModel):
    role: str
    content: str

class InitRequest(BaseModel):
    thread_id: str
    user_id: Optional[str] = None
    entity: str
    entity_type: str
    context: Optional[str] = None
    history: Optional[list[ChatHistoryMessage]] = None
    rewriter_history: Optional[list[ChatHistoryMessage]] = None

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: Optional[str] = None
    voice_mode: bool
    entity: str 
    entity_type: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    entity_name: str

class TranscribeResponse(BaseModel):
    text: str
