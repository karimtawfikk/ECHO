import base64

from fastapi import FastAPI, File, HTTPException, UploadFile

from .schemas import (
    HealthResponse,
    HieroglyphTranslationRequest,
    TranslationResult,
)
from .runtime import hieroglyph_runtime
from .service import hieroglyph_service


app = FastAPI(title="ECHO Hieroglyph Detection API", version="0.1.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
