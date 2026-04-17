import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

VIDEO_API_URL = "http://localhost:8001"

from src.app.schemas.video import VideoGenerationRequest

@router.post("/generate")
async def generate_video(request: VideoGenerationRequest):
    payload = {
        "entity_name": request.entity_name,
        "is_landmark": request.is_landmark
    }
    
    async def iter_video():
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream("POST", f"{VIDEO_API_URL}/generate", json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        raise HTTPException(status_code=response.status_code, detail=f"Video generation failed: {err_text.decode('utf-8', errors='ignore')}")
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.RequestError as exc:
                raise HTTPException(status_code=503, detail=f"Video API unreachable: {exc}")

    return StreamingResponse(iter_video(), media_type="video/mp4")
