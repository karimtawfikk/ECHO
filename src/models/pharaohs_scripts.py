from sqlalchemy import Column, Index, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.db.session import Base

class PharaohScript(Base):
    __tablename__ = "pharaohs_scripts"

    id = Column(Integer, primary_key=True, index=True)
    pharaoh_id = Column(Integer, ForeignKey("pharaohs.id"))
    pharaoh_script = Column(Text)   
    pharaoh_script_embedding = Column(Vector(1024))  
    
    pharaoh = relationship("Pharaoh", back_populates="scripts")

    
    __table_args__ = (
        Index(
            "hnsw_idx_pharaoh_script_embedding",
            pharaoh_script_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"pharaoh_script_embedding": "vector_cosine_ops"},
        ),
    )