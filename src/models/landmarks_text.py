from sqlalchemy import Column, Integer, ForeignKey, String, Index, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.db.session import Base

class LandmarkText(Base):
    __tablename__ = "landmarks_texts"

    id = Column(Integer, primary_key=True, index=True)
    landmark_id = Column(Integer, ForeignKey("landmarks.id"))
    text_chunk = Column(Text)     
    text_embedding = Column(Vector(768))
    
    landmark = relationship("Landmark", back_populates="texts")

    __table_args__ = (
        Index(
            "hnsw_idx_landmark_text_embedding",
            text_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"text_embedding": "vector_cosine_ops"},
        ),
    )