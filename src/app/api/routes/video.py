import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import os
from fastapi.responses import FileResponse

router = APIRouter()

VIDEO_API_URL = "http://127.0.0.1:8005"

@router.post("/generate")
async def generate_video(request: VideoGenerationRequest):
    return StreamingResponse(iter_video(), media_type="video/mp4")
