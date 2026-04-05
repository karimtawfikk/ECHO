from pydantic import BaseModel, Field


class VideoGenerationRequest(BaseModel):
    entity_name: str = Field(..., min_length=1)
    is_landmark: bool = False


class VideoGenerationMetadata(BaseModel):
    entity_name: str
    entity_type: str
    filename: str
    output_path: str


class HealthResponse(BaseModel):
    status: str

