from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.db import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    entity_name = Column(String, nullable=False)
    entity_type = Column(String, nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=True, server_default=text("now()"))
    entity_location = Column(String, nullable=True)
    is_pinned = Column(Boolean, nullable=True, server_default=text("false"))

    # Relationships
    user = relationship("Profile", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")
    rewritten_messages = relationship("ChatMessageRewriter", back_populates="conversation", cascade="all, delete-orphan")
