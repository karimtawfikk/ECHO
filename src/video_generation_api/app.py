from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .schemas import HealthResponse, VideoGenerationRequest
from .runtime import video_generation_runtime
from .service import video_generation_service


from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="ECHO Video Generation API", version="0.1.0")

# Serve the outputs folder so the frontend can download files directly
os.makedirs("tts_Outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="tts_Outputs"), name="outputs")


from fastapi import BackgroundTasks

job_status = {}

@app.on_event("startup")
def preload_models() -> None:
    video_generation_runtime.ensure_models_loaded()

@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/generate")
def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    job_status[request.entity_name] = "processing"
    
    def run_generation():
        try:
            print(f"[video_api] Background generation started for {request.entity_name}")
            video_generation_service.generate(request)
            job_status[request.entity_name] = "ready"
            print(f"[video_api] Background generation complete for {request.entity_name}")
        except Exception as e:
            print(f"[video_api] Error: {e}")
            job_status[request.entity_name] = "failed"
            
    background_tasks.add_task(run_generation)
    return {"status": "started"}

@app.get("/status/{entity_name}")
def get_status(entity_name: str):
    status = job_status.get(entity_name, "unknown")
    filename = f"{entity_name.replace(' ', '_')}_final_video.mp4"
    if os.path.exists(f"tts_Outputs/{filename}"):
        status = "ready"
    return {"status": status}
