import os
from pathlib import Path
from dotenv import load_dotenv

_ECHO_ROOT = Path(__file__).resolve().parents[3]  
load_dotenv(_ECHO_ROOT / ".env")

class Settings:
    BASE_DIR: str = Path(__file__).resolve().parent.parent.parent

    CORS_ORIGINS: list = [
        "https://echo-eg.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    MODEL_PATH: str = os.path.join(BASE_DIR, "ml_models", "recognition_models")
    
settings = Settings()
