from sqlalchemy import Column, Index, Integer, ForeignKey, String, Boolean
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.db import Base

class LandmarkImage(Base):
    __tablename__ = "landmark_images"

    id = Column(Integer, primary_key=True, index=True)
    landmark_id = Column(Integer, ForeignKey("landmarks.id"))
    image_path = Column(String)
    image_embedding = Column(Vector(1024))
    is_builder = Column(Boolean, default=False)       
    is_plan = Column(Boolean, default=False)           
    is_location = Column(Boolean, default=False)   
    landmark = relationship("Landmark", back_populates="images")

    __table_args__ = (
        Index(
            "hnsw_idx_landmark_image_embedding",
            image_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"image_embedding": "vector_cosine_ops"},
        ),
    )
