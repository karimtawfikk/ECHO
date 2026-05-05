import httpx
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from src.app.schemas.video import VideoGenerationRequest

router = APIRouter()

VIDEO_API_URL = "http://127.0.0.1:8005"

@router.post("/generate")
async def generate_video(request: VideoGenerationRequest):
    payload = {"entity_name": request.entity_name, "is_landmark": request.is_landmark}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # The Video API will return immediately with {"status": "started"}
            resp = await client.post(f"{VIDEO_API_URL}/generate", json=payload)
            return resp.json()
        except Exception as e:
            raise HTTPException(500, str(e))

@router.get("/status/{entity_name}")
async def get_status(entity_name: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{VIDEO_API_URL}/status/{entity_name}")
            return resp.json()
        except Exception as e:
            raise HTTPException(500, str(e))

@router.get("/stream/{entity_name}")
async def stream_video(entity_name: str):
    filename = f"{entity_name.replace(' ', '_')}_final_video.mp4"
    filepath = os.path.join("tts_Outputs", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not ready")
