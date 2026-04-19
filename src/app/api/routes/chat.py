import base64
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

CHATBOT_API_URL = "http://127.0.0.1:8000"

from src.app.schemas.chatbot import ChatRequest, ChatResponse, TranscribeResponse, InitRequest

@router.post("/init")
async def init_chat(req: InitRequest):
    payload = {
        "session_id": req.thread_id,
        "entity_type": req.entity_type,
        "entity_name": req.entity
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(f"{CHATBOT_API_URL}/init", json=payload)
            if res.status_code == 200:
                return {"status": "success"}
            else:
                raise HTTPException(status_code=res.status_code, detail=res.text)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Init failed: {exc}")

@router.get("/info")
async def info():
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(f"{CHATBOT_API_URL}/health")
            status = res.json().get("status", "unknown")
            return {"entity_name": "Docker Chatbot API Linked", "status": status}
        except httpx.RequestError as exc:
            return {"entity_name": "Docker Chatbot API Unreachable", "status": "error", "error": str(exc)}

import json

@router.post("/chat")
async def chat(req: ChatRequest):
    payload = {
        "session_id": req.thread_id,
        "entity_type": req.entity_type,
        "entity_name": req.entity,
        "message": req.message
    }
    
    async def event_generator():
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{CHATBOT_API_URL}/chat", json=payload) as response:
                    if response.status_code != 200:
                        err = await response.aread()
                        yield f"data: {json.dumps({'error': err.decode('utf-8', errors='ignore')})}\n\n"
                        return
                        
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                continue
                                
                            try:
                                json_data = json.loads(data_str)
                                if isinstance(json_data, dict) and any(k in json_data for k in ["tool", "search", "event", "tool_calls"]):
                                    yield f"data: {data_str}\n\n"
                                    continue
                            except:
                                pass
                                
                            full_text += data_str
                            yield f"data: {json.dumps({'text': data_str})}\n\n"
                            
        except httpx.RequestError as exc:
            yield f"data: {json.dumps({'error': f'Failed to connect to chatbot API: {exc}'})}\n\n"
            return

        audio_url = None
        if req.voice_mode and full_text.strip():
            try:
                speech_payload = {
                    "text": full_text,
                    "entity_type": req.entity_type,
                    "entity_name": req.entity
                }
                async with httpx.AsyncClient(timeout=60.0) as client:
                    res = await client.post(f"{CHATBOT_API_URL}/voice/speak", json=speech_payload)
                    if res.status_code == 200:
                        b64_str = base64.b64encode(res.content).decode("utf-8")
                        audio_url = f"data:audio/mp3;base64,{b64_str}"
                        yield f"data: {json.dumps({'audio_url': audio_url})}\n\n"
            except Exception as e:
                print(f"[PHARAOH TTS] Failed: {e}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:
        try:
            files = {"audio": (audio.filename or "recording.webm", await audio.read(), audio.content_type)}
            res = await client.post(f"{CHATBOT_API_URL}/voice/transcribe", files=files)
            if res.status_code == 200:
                return TranscribeResponse(text=res.json().get("transcription", ""))
            else:
                raise HTTPException(status_code=res.status_code, detail=res.text)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Transcription failed: {exc}")

