import json
from pathlib import Path
import chromadb
from sqlalchemy.orm import Session
from app.db.session import engine, Base
from app.models.landmarks import Landmark
from app.models.landmarks_images import LandmarkImage
from chromadb import Client

Base.metadata.create_all(bind=engine)

landmarks_json_path = Path("data/video_generation/outputs/landmarks.json")
with open(landmarks_json_path, "r", encoding="utf-8") as f:
    landmarks_data = json.load(f)

with Session(engine) as session:
    for lm in landmarks_data:
        landmark = Landmark(
            id=lm["id"],
            name=lm["name"],
            description=lm.get("description", ""),
            location=lm.get("location", "")
        )
        session.merge(landmark)
    session.commit()


chroma_db_path = Path("data/video_generation/embeddings/chroma_db_landmarks")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("landmarks_images")

all_items = collection.get(include=["metadatas", "embeddings"])
id=0
landmark_count=0
with Session(engine) as session:
    for metadata, emb in zip(all_items["metadatas"], all_items["embeddings"]):
        
        img_path = metadata.get("path")
        
        landmark_name = metadata["landmark"]
        landmark_obj = session.query(Landmark).filter_by(name=landmark_name).first()
        if not landmark_obj:
            skipped=metadata["landmark"]
            landmark_count+=1   
            print(f"⚠️ Landmark not found in DB for image: {img_path}")
            continue
        landmark_id = landmark_obj.id

        is_builder = metadata.get("is_builder", "no") == "yes"
        is_plan = metadata.get("is_plan", "no") == "yes"
        is_location = metadata.get("is_location", "no") == "yes"
    
        lm_image = LandmarkImage(
            id=id,
            landmark_id=landmark_id,
            image_path=img_path,
            image_embedding=emb,
            is_builder=is_builder,
            is_plan=is_plan,
            is_location=is_location
        )
        id+=1
        session.merge(lm_image)

    session.commit()

print("Landmark images synced from Chroma to PostgreSQL!")
print(f"Skipped {landmark_count} items due to missing landmarks in DB.")
