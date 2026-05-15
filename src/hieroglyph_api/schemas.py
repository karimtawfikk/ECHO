from __future__ import annotations
from pydantic import BaseModel, Field

class HieroglyphTranslationRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, description="Base64-encoded image (PNG/JPEG).")

class ClassifiedSymbol(BaseModel):
    gardiner_code: str = Field(..., min_length=1)
    classification_confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[float] = Field(..., description="[x_min, y_min, x_max, y_max] in pixel coords.")
    detection_confidence: float = Field(..., ge=0.0, le=1.0)

class TranslationResult(BaseModel):
    symbols: list[ClassifiedSymbol] = Field(..., description="Symbols in reading order.")
    num_symbols_detected: int
    num_clusters: int = Field(..., description="Number of DBSCAN clusters (rows/columns).")
    translation_text: str = Field("", description="English translation from Gardiner code sequence.")
    annotated_image_base64: str | None = Field(None, description="Image with bounding boxes drawn (Base64).")

class HieroglyphTranslationMetadata(BaseModel):
    num_symbols_detected: int
    num_clusters: int
    pipeline_time_ms: int

class HealthResponse(BaseModel):
    status: str
