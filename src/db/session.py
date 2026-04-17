import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load .env from the ECHO root
_ECHO_ROOT = Path(__file__).resolve().parents[2]  # session.py -> db/ -> src/ -> ECHO/
load_dotenv(_ECHO_ROOT / ".env")

# Get DATABASE_URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,              # Keep 5 connections open
    max_overflow=10,          # Allow 10 extra during peak
    pool_pre_ping=True,       # Verify connection is alive before use
    pool_recycle=3600,        # Recycle connections after 1 hour
    echo=False                # Set to True to see SQL queries (debug)
)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Helper function to get DB session (for FastAPI or notebooks)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()