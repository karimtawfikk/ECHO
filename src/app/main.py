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
from src.app.api.routes import recognize, health, trending_entities, chat, video, assets, hieroglyphs

app = FastAPI(
    title="E.C.H.O — Every Capture Has Origins",
    description="Egyptian artifact and landmark recognition with origins exploration.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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
app.include_router(assets.router,         prefix="/api/v1/assets",    tags=["assets"])
app.include_router(hieroglyphs.router,   prefix="/api/v1/hieroglyphs", tags=["hieroglyphs"])


@app.on_event("startup")
async def startup_event():
    print(f"\n[E.C.H.O] Online\n")
