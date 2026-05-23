import base64
from fastapi import FastAPI, File, HTTPException, UploadFile
from .schemas import HealthResponse, RecognitionRequest, RecognitionResult
from .runtime import recognition_inference
from .service import recognition_service

app = FastAPI(title="ECHO Artifact Recognition API", version="0.1.0")

@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    if all([recognition_inference.binary_model, recognition_inference.pharaoh_model, recognition_inference.landmark_model]):
        return HealthResponse(status="ok")
    return HealthResponse(status="models_not_loaded")

@app.post("/recognize", response_model=RecognitionResult)
async def recognize(request: RecognitionRequest) -> RecognitionResult:
    try:
        return await recognition_service.recognize(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/recognize/upload", response_model=RecognitionResult)
async def recognize_upload(image: UploadFile = File(...)) -> RecognitionResult:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image.")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        request = RecognitionRequest(image_base64=image_b64)
        return await recognition_service.recognize(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
