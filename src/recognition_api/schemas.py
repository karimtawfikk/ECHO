from pydantic import BaseModel, Field

class RecognitionRequest(BaseModel):
    image_base64: str = Field(..., min_length=1, description="Base64-encoded image (PNG/JPEG).")

class RecognitionResult(BaseModel):
    type: str = Field(..., description="Predicted type (pharaoh or landmark)")
    name: str = Field(..., description="Predicted raw name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    binary_confidence: float = Field(..., ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str
