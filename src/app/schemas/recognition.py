from pydantic import BaseModel
from typing import Optional, List

class SubEntityResponse(BaseModel):
    name: str
    type: Optional[str] = None
    dynasty: Optional[str] = None
    period: Optional[str] = None

class EntityResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    dynasty: Optional[str] = None
    period: Optional[str] = None
    location: Optional[str] = None
    composite_entity: Optional[str] = None
    composite_entities_data: Optional[List[SubEntityResponse]] = None

class RecognitionResponse(BaseModel):
    source: str = "recognition"
    type: str # "pharaoh" | "landmark"
    name: str
    confidence: float
    binary_confidence: float
    entity: Optional[EntityResponse] = None # Optional in case DB is missing the entity
