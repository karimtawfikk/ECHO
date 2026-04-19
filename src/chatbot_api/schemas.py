from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    entity_type: str = Field(..., pattern="^(pharaoh|landmark)$")
    entity_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class InitSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    entity_type: str = Field(..., pattern="^(pharaoh|landmark)$")
    entity_name: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str


class TranscriptionResponse(BaseModel):
    transcription: str


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1)
    entity_type: str | None = Field(default=None, pattern="^(pharaoh|landmark)$")
    entity_name: str | None = None


class SpeechMetadata(BaseModel):
    language: str
    voice: str
