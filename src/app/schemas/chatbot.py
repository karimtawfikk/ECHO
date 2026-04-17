from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    voice_mode: bool
    entity: str 
    entity_type: str

class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    entity_name: str

class TranscribeResponse(BaseModel):
    text: str
