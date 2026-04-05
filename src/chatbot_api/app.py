from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SpeechRequest,
    TranscriptionResponse,
)
from .service import chatbot_service


app = FastAPI(title="ECHO Chatbot API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return chatbot_service.chat(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voice/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)) -> TranscriptionResponse:
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        transcription = chatbot_service.transcribe_audio(audio.filename or "audio.wav", audio_bytes)
        return TranscriptionResponse(transcription=transcription)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voice/speak", responses={200: {"content": {"audio/mpeg": {}}}})
def synthesize_speech(request: SpeechRequest) -> StreamingResponse:
    try:
        audio_bytes, metadata = chatbot_service.synthesize_speech(request)
        headers = {
            "X-ECHO-Voice": metadata.voice,
            "X-ECHO-Language": metadata.language,
        }
        return StreamingResponse(BytesIO(audio_bytes), media_type="audio/mpeg", headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
