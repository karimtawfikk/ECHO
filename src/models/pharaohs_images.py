from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.db.session import Base

class PharaohImage(Base):
    __tablename__ = "pharaohs_images"

    id = Column(Integer, primary_key=True, index=True)
    pharaoh_id = Column(Integer, ForeignKey("pharaohs.id"))
    image_path = Column(String)            # optional: file path or URL
    image_embedding = Column(Vector(1024)) # embedding of this image
    image_description = Column(String)       # optional metadata about the image
    
    pharaoh = relationship("Pharaoh", back_populates="images")