from sqlalchemy import Column, ForeignKey, Index, Integer, Text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from src.db.session import Base

class LandmarkScript(Base):
    __tablename__ = "landmarks_scripts"

    id = Column(Integer, primary_key=True, index=True)
    landmark_id = Column(Integer, ForeignKey("landmarks.id"))
    landmark_script = Column(Text)
    landmark_script_embedding = Column(Vector(1024))  

    landmark = relationship("Landmark", back_populates="scripts")

    __table_args__ = (
        Index(
            "hnsw_idx_landmark_script_embedding",
            landmark_script_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"landmark_script_embedding": "vector_cosine_ops"},
        ),
    )
