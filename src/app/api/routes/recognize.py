from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.db import get_db
from src.app.schemas.recognition import RecognitionResponse

router = APIRouter()

@router.post("/", response_model=RecognitionResponse)
async def recognize_artifact(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            return RecognitionResponse(
                source="error", type="error", name="Invalid file type", 
                confidence=0.0, binary_confidence=0.0
            )
        
        image_data = await file.read()

        # Run inference using standalone recognition microservice
        import os
        import httpx
        RECOGNITION_API_URL = os.environ.get("RECOGNITION_API_URL", "http://localhost:8002")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"image": (file.filename, image_data, file.content_type)}
            response = await client.post(f"{RECOGNITION_API_URL}/recognize/upload", files=files)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Recognition microservice error: {response.text}"
                )
            inference_result = response.json()
        
        predicted_type = inference_result["type"]
        predicted_name = inference_result["name"]
        final_conf = inference_result["confidence"]
        binary_conf = inference_result["binary_confidence"]

        # Load entity metadata from DB
        from src.app.services.entity_loader import load_entity
        entity = load_entity(predicted_type, predicted_name, db)

        # Clean name for display (remove brackets and replace underscores)
        display_name = predicted_name.split('(')[0].replace('_', ' ').strip()

        entity_data = None
        if entity:
            entity_data = {
                "id": entity.id,
                "name": entity.name,
                "description": entity.description,
            }

            if predicted_type == "pharaoh":
                entity_data["type"] = getattr(entity, "type", None)
                entity_data["dynasty"] = getattr(entity, "dynasty", None)
                entity_data["period"] = getattr(entity, "period", None)
                composite_raw = getattr(entity, "composite_entity", None)
                entity_data["composite_entity"] = composite_raw

                # Look up sub-entity metadata for composite entities
                if composite_raw:
                    from src.db_models import Pharaoh
                    sub_names = [s.strip() for s in composite_raw.split(",") if s.strip()]
                    entity_data["composite_entities_data"] = [
                        {
                            "name": sn,
                            "type": getattr(row, "type", None) if row else None,
                            "dynasty": getattr(row, "dynasty", None) if row else None,
                            "period": getattr(row, "period", None) if row else None,
                        }
                        for sn in sub_names
                        for row in [db.query(Pharaoh).filter(Pharaoh.name.ilike(sn)).first()]
                    ]
            else:
                entity_data["location"] = getattr(entity, "location", None)

        return {
            "source": "recognition",
            "type": predicted_type,
            "name": display_name,
            "raw_name": predicted_name,
            "confidence": final_conf,
            "binary_confidence": binary_conf,
            "entity": entity_data,
        }

    except Exception as e:
        print(f"[RECOGNIZE ERROR]: {e}") # <--- ADD THIS LINE
        import traceback
        traceback.print_exc()           # <--- ADD THIS LINE TOO

        
        return RecognitionResponse(
            source="error",
            type="error",
            name="recognition_failed",
            confidence=0.0,
            binary_confidence=0.0,
            entity=None
        )

