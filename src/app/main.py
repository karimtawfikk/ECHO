import sys
from pathlib import Path

# Add ECHO root to Python path so we can import shared code from src/
# parents[2]: main.py -> app/ -> echo-backend/ -> ECHO/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from src.app.core.config import settings
from src.app.api.routes import recognize, health, trending_entities, chat, video
from src.app.services.recognition_inference import recognition_inference

app = FastAPI(
    title="E.C.H.O — Every Capture Has Origins",
    description="Egyptian artifact and landmark recognition with origins exploration.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root → redirect to Swagger UI
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Include Routers
app.include_router(recognize.router,      prefix="/api/v1/recognize", tags=["recognition"])
app.include_router(health.router,         prefix="/api/v1/health",    tags=["health"])
app.include_router(trending_entities.router,    prefix="/api/v1/entities",  tags=["entities"])
app.include_router(chat.router,           prefix="/api/v1/chat",      tags=["chat"])
app.include_router(video.router,          prefix="/api/v1/video",     tags=["video"])

# Mount Static Files — MUST come AFTER all routers so it doesn't shadow /docs
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    print(f"\n[E.C.H.O] Online\n")
