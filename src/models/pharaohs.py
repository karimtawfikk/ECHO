from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from app.db.session import Base

class Pharaoh(Base):
    __tablename__ = "pharaohs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)                # e.g., "Ramesses II"
    dynasty = Column(String)                             # dynasty info
    description = Column(String)                         # metadata
    period = Column(String)
    images = relationship("PharaohImage", back_populates="pharaoh")

    #text_embedding  = Column(Vector(768))               # store text embeddings