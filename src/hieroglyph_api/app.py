import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .schemas import (
    HealthResponse,
    HieroglyphTranslationRequest,
    TranslationResult,
)
from .runtime import hieroglyph_runtime
from .service import hieroglyph_service


app = FastAPI(title="ECHO Hieroglyph Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
def preload_models() -> None:
    hieroglyph_runtime.ensure_models_loaded()


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/translate", response_model=TranslationResult)
def detect_hieroglyphs(request: HieroglyphTranslationRequest) -> TranslationResult:
    try:
        result, _metadata = hieroglyph_service.detect(request)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/translate/stream")
async def detect_hieroglyphs_stream(request: HieroglyphTranslationRequest):
    """
    Streaming version of the translate endpoint.
    Sends progress updates (phase 1-4) as they happen.
    """
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_step(step: int):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "step": step})

    def run_pipeline():
        try:
            result, _metadata = hieroglyph_service.detect(request, on_step=on_step)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "data": result.dict()})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

    # Start pipeline in a background thread
    loop.run_in_executor(executor, run_pipeline)

    async def event_generator():
        while True:
            message = await queue.get()
            if message["type"] == "error":
                yield f"data: {json.dumps(message)}\n\n"
                break
            
            yield f"data: {json.dumps(message)}\n\n"
            
            if message["type"] == "result":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/translate/upload", response_model=TranslationResult)
async def detect_hieroglyphs_upload(image: UploadFile = File(...)) -> TranslationResult:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        request = HieroglyphTranslationRequest(image_base64=image_b64)
        
        result, _metadata = hieroglyph_service.detect(request)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
