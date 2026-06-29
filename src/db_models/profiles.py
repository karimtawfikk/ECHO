from sqlalchemy import Column, String, DateTime, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from src.db import Base

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(UUID(as_uuid=True), primary_key=True)
    username = Column(String, unique=True, nullable=True)
    full_name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    user_metadata = Column(JSONB, nullable=True, server_default=text("'{}'::jsonb"))
    favorites = Column(JSONB, nullable=True, server_default=text("'[]'::jsonb"))
    created_at = Column(DateTime, nullable=False, server_default=text("timezone('utc'::text, now())"))
    updated_at = Column(DateTime, nullable=True)

    recognition_history = relationship("RecognitionHistory", back_populates="user", cascade="all, delete-orphan")
    translation_history = relationship("TranslationHistory", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
