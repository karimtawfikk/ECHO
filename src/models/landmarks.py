from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from src.db.session import Base

class Landmark(Base):
    __tablename__ = "landmarks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)                         
    location = Column(String, nullable=True)                             
    images = relationship("LandmarkImage", back_populates="landmark")
    texts = relationship("LandmarkText", back_populates="landmark")
