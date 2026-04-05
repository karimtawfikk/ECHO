from sqlalchemy import Column, Integer, ForeignKey, String, Index, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.db import Base

class PharaohText(Base):
    __tablename__ = "pharaohs_texts"

    id = Column(Integer, primary_key=True, index=True)
    pharaoh_id = Column(Integer, ForeignKey("pharaohs.id"))
    text_chunk = Column(Text)     
    text_embedding = Column(Vector(768))
    
    pharaoh = relationship("Pharaoh", back_populates="texts")

    __table_args__ = (
        Index(
            "hnsw_idx_pharaoh_text_embedding",
            text_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"text_embedding": "vector_cosine_ops"},
        ),
    )
