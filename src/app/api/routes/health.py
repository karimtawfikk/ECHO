import os
from fastapi import APIRouter
from sqlalchemy.orm import Session
from fastapi import Depends
from src.db import get_db
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
    import httpx
    RECOGNITION_API_URL = os.environ.get("RECOGNITION_API_URL", "http://localhost:8002")
    try:
        r = httpx.get(f"{RECOGNITION_API_URL}/health", timeout=2.0)
        models_ok = r.json().get("status") == "ok"
    except Exception:
        models_ok = False

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
    import httpx
    RECOGNITION_API_URL = os.environ.get("RECOGNITION_API_URL", "http://localhost:8002")
    try:
        r = httpx.get(f"{RECOGNITION_API_URL}/health", timeout=2.0)
        rec_data = r.json()
        rec_ok = rec_data.get("status") == "ok"
    except Exception:
        rec_ok = False
    
    return {
        "status": "active" if rec_ok else "degraded",
        "models": {
            "binary": rec_ok,
            "pharaoh": rec_ok,
            "landmark": rec_ok,
        },
        "encoders": {
            "binary": rec_ok,
            "pharaoh": rec_ok,
            "landmark": rec_ok,
        }
    }
