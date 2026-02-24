from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from app.db.session import Base

class Landmark(Base):
    __tablename__ = "landmarks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)                         
    location = Column(String)                             
    images = relationship("LandmarkImage", back_populates="landmark")

    #text_embedding  = Column(Vector(768))                # store text embeddings