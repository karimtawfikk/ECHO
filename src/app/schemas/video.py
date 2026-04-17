from pydantic import BaseModel

class VideoGenerationRequest(BaseModel):
    entity_name: str
    is_landmark: bool
