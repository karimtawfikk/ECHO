import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the ECHO root (not echo-backend's local copy)
_ECHO_ROOT = Path(__file__).resolve().parents[3]  # config.py -> core/ -> app/ -> echo-backend/ -> ECHO/
load_dotenv(_ECHO_ROOT / ".env")

class Settings:
    BASE_DIR: str = Path(__file__).resolve().parent.parent.parent

    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # Path to static folder (absolute)
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    MODEL_PATH: str = os.path.join(BASE_DIR, "recognition_models")
    
settings = Settings()
