from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .schemas import HealthResponse, VideoGenerationRequest
from .service import video_generation_service


app = FastAPI(title="ECHO Video Generation API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/generate", responses={200: {"content": {"video/mp4": {}}}})
def generate_video(request: VideoGenerationRequest) -> FileResponse:
    try:
        output_path, metadata = video_generation_service.generate(request)
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Video generation completed but no file was found.")

        headers = {
            "X-ECHO-Entity-Name": metadata.entity_name,
            "X-ECHO-Entity-Type": metadata.entity_type,
            "X-ECHO-Output-Path": metadata.output_path,
        }
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=metadata.filename,
            headers=headers,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
