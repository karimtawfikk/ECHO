from sqlalchemy import Column, String, DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.db import Base

class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    created_at = Column(DateTime, nullable=False, server_default=text("now()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(String, nullable=False)
    translation = Column(String, nullable=False)

    user = relationship("Profile", back_populates="translation_history")
