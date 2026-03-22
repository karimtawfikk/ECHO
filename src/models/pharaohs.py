from sqlalchemy import Column, Integer, String,Index
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from src.db.session import Base

class Pharaoh(Base):
    __tablename__ = "pharaohs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)                # e.g., "Ramesses II"
    dynasty = Column(String,nullable=True)                             # dynasty info
    description = Column(String,nullable=True)                         # metadata
    period = Column(String,nullable=True)
    composite_entity = Column(String,nullable=True)   
   
    images = relationship("PharaohImage", back_populates="pharaoh")
    texts = relationship("PharaohText", back_populates="pharaoh")
    scripts = relationship("PharaohScript", back_populates="pharaoh")
                 
    __table_args__ = (
        Index('idx_pharaohs_name', 'name'),
    )