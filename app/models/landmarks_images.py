from sqlalchemy import Column, Integer, ForeignKey, String, Boolean
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.db.session import Base

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