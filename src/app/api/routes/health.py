import os
from fastapi import APIRouter
from sqlalchemy.orm import Session
from fastapi import Depends
from src.db import get_db
from src.app.services.recognition_inference import recognition_inference
from sqlalchemy import text

router = APIRouter()


@router.get("/")
def health_unified(db: Session = Depends(get_db)):
    """Unified health check — one request to verify the entire stack."""
    # Database
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
        db_err = None
    except Exception as e:
        db_ok = False
        db_err = str(e)

    # Models
    svc = recognition_inference
    models_ok = all([svc.binary_model, svc.pharaoh_model, svc.landmark_model])

    overall = db_ok and models_ok

    return {
        "ok": overall,
        "database": {"connected": db_ok, "error": db_err},
        "models": {"loaded": models_ok},
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

@router.get("/db")
def health_db(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"database": "connected"}
    except Exception as e:
        return {"database": "disconnected", "error": str(e)}

@router.get("/app")
def health_app():
    return {"status": "active", "mode": os.getenv("ENVIRONMENT", "development")}

@router.get("/models")
def health_models():
    svc = recognition_inference
    
    return {
        "status": "active",
        "models": {
            "binary": svc.binary_model is not None,
            "pharaoh": svc.pharaoh_model is not None,
            "landmark": svc.landmark_model is not None,
        },
        "encoders": {
            "binary": svc.binary_encoder is not None,
            "pharaoh": svc.pharaoh_encoder is not None,
            "landmark": svc.landmark_encoder is not None,
        }
    }
