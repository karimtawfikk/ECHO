from sqlalchemy import Column, Index, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.db import Base

class PharaohImage(Base):
    __tablename__ = "pharaohs_images"

    id = Column(Integer, primary_key=True, index=True)
    pharaoh_id = Column(Integer, ForeignKey("pharaohs.id"))
    image_path = Column(String)            # optional: file path or URL
    image_embedding = Column(Vector(1024)) # embedding of this image
    image_description = Column(String)       # optional metadata about the image
    
    pharaoh = relationship("Pharaoh", back_populates="images")

    __table_args__ = (
        Index(
            "hnsw_idx_pharaoh_image_embedding",
            image_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"image_embedding": "vector_cosine_ops"},
        ),
    )
