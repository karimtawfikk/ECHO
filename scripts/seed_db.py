import json
from pathlib import Path
import chromadb
from sqlalchemy.orm import Session
from src.db.session import engine, Base
from src.models.landmarks import Landmark
from src.models.landmarks_images import LandmarkImage
from src.models.landmarks_text import LandmarkText
from src.models.landmarks_scripts import LandmarkScript 
from src.models.pharaohs import Pharaoh
from src.models.pharaohs_images import PharaohImage
from src.models.pharaohs_text import PharaohText
from src.models.pharaohs_scripts import PharaohScript



Base.metadata.create_all(bind=engine)

#Landmarks
landmarks_json_path = Path(r"D:\GP\ECHO\data\data\video_generation\outputs\landmarks.json")
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

#Landmark Images
chroma_db_path = Path(r"D:\GP\ECHO\data\data\video_generation\embeddings\chroma_db_landmarks")
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
            print(f" Landmark not found in DB for image: {img_path}")
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

#Landmarks Text
chroma_db_path = Path(r"D:\GP\ECHO\data\data\chatbot\embeddings\landmarks_qwen_MRL_768_db")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("landmarks")

all_items = collection.get(include=["metadatas", "embeddings","documents"])
id=0
landmark_count=0
skipped=[]
with Session(engine) as session:
    for metadata, emb, docs in zip(all_items["metadatas"], all_items["embeddings"], all_items["documents"]):
        text_chunk = docs
        
        landmark_name = metadata["entity_name"].replace(".txt", "").strip()
        landmark_obj = session.query(Landmark).filter_by(name=landmark_name).first()
        if not landmark_obj:
            skipped.append(metadata["entity_name"])
            landmark_count+=1   
            print(f" Landmark not found in DB for text chunk: {text_chunk}")
            continue
        landmark_id = landmark_obj.id

        lm_text = LandmarkText(
            id=id,
            landmark_id=landmark_id,
            text_chunk=text_chunk,
            text_embedding=emb,
        )
        
        id+=1
        session.merge(lm_text)

    session.commit()


print(f"{landmark_count} Landmark texts skipped due to missing landmarks in DB.")
print(skipped)
print("Landmark texts synced from Chroma to PostgreSQL!")


#Landmarks Scripts
chroma_db_path = Path(r"D:\GP\ECHO\data\data\video_generation\embeddings\chroma_db_scripts_landmarks")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("landmarks_scripts")
all_items = collection.get(include=["metadatas", "embeddings"])

SCRIPTS_DIR = Path(r"D:\GP\ECHO\data\data\video_generation\outputs\landmarks_scripts")

id=0
landmark_count=0
skipped=[]

with Session(engine) as session:
    for metadata, emb in zip(all_items["metadatas"], all_items["embeddings"]):
        
        landmark_name = metadata.get("landmark_name")
        script_path = Path("data") / Path(metadata.get("path"))
        

        landmark_script = script_path.read_text(encoding="utf-8", errors="ignore")
        
        landmark_obj = session.query(Landmark).filter_by(name=landmark_name).first()
        if not landmark_obj:
            skipped.append(landmark_name)
            landmark_count+=1   
            print(f" Landmark script not found in DB : {landmark_name}")
            continue

        lm_script = LandmarkScript(
            id=id,
            landmark_id=landmark_obj.id,
            landmark_script=landmark_script,
            landmark_script_embedding=emb,
        )
        
        id+=1
        session.merge(lm_script)

    session.commit()

print(f"{landmark_count} Landmark scripts skipped due to missing landmarks in DB.")
print(skipped)
print("Landmark scripts synced from Chroma to PostgreSQL!")


#Pharaohs
pharaohs_json_path = Path(r"D:\GP\ECHO\data\data\video_generation\outputs\pharaohs.json")
with open(pharaohs_json_path, "r", encoding="utf-8") as f:
    pharaohs_data = json.load(f)

with Session(engine) as session:
    for ph in pharaohs_data:
        pharaoh = Pharaoh(
            id=ph["id"],
            name=ph["name"],
            period=ph["period"],
            dynasty=ph["dynasty"],
            description=ph["description"],
            composite_entity=ph["composite_entity"]
        )
        session.merge(pharaoh)
    session.commit()

#Pharaoh Images
chroma_db_path = Path(r"D:\GP\ECHO\data\data\video_generation\embeddings\chroma_db_pharaohs")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("pharaohs_images")

all_items = collection.get(include=["metadatas", "embeddings"])
id=0
pharaoh_count=0
with Session(engine) as session:
    for metadata, emb in zip(all_items["metadatas"], all_items["embeddings"]):
        
        img_path = metadata.get("path")
        
        pharaoh_name = metadata["pharaoh_name"]
        pharaoh_obj = session.query(Pharaoh).filter_by(name=pharaoh_name).first()
        if not pharaoh_obj:
            skipped=metadata["pharaoh_name"]
            pharaoh_count+=1   
            print(f" Pharaoh not found in DB for image: {img_path}")
            continue
        pharaoh_id = pharaoh_obj.id
        image_description = metadata.get("image_description", "").split('.')[0]
     
        ph_image = PharaohImage(
            id=id,
            pharaoh_id=pharaoh_id,
            image_path=img_path,
            image_embedding=emb,
            image_description=image_description
        )
        id+=1
        session.merge(ph_image)

    session.commit()

print("Pharaoh images synced from Chroma to PostgreSQL!")

#Pharaohs Text
chroma_db_path = Path(r"D:\GP\ECHO\data\data\chatbot\embeddings\pharaohs_qwen_MRL_768_db")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("pharaohs")

all_items = collection.get(include=["metadatas", "embeddings","documents"])
id=0
pharoah_count=0
skipped=[]
with Session(engine) as session:
    for metadata, emb, docs in zip(all_items["metadatas"], all_items["embeddings"], all_items["documents"]):
        text_chunk = docs
        
        pharaoh_name = metadata["entity_name"].replace(".txt", "").strip()
        pharaoh_obj = session.query(Pharaoh).filter_by(name=pharaoh_name).first()
        if not pharaoh_obj:
            skipped.append(metadata["entity_name"])
            pharoah_count+=1   
            continue
        pharaoh_id = pharaoh_obj.id

        ph_text = PharaohText(
            id=id,
            pharaoh_id=pharaoh_id,
            text_chunk=text_chunk,
            text_embedding=emb
        )
        
        id+=1
        session.merge(ph_text)

    session.commit()
print("Pharaohs texts synced from Chroma to PostgreSQL!")


#Pharaohs Scripts
chroma_db_path = Path(r"D:\GP\ECHO\data\data\video_generation\embeddings\chroma_db_scripts_pharaohs")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("pharaohs_scripts")
all_items = collection.get(include=["metadatas", "embeddings"])



id=0
pharaoh_count=0
skipped=[]

with Session(engine) as session:
    for metadata, emb in zip(all_items["metadatas"], all_items["embeddings"]):
        
        pharaoh_name = metadata.get("pharaoh_name")
        script_path = Path("data") / Path(metadata.get("path"))

        pharaoh_script = script_path.read_text(encoding="utf-8", errors="ignore")
        
        pharaoh_obj = session.query(Pharaoh).filter_by(name=pharaoh_name).first()
        if not pharaoh_obj:
            skipped.append(pharaoh_name)
            pharaoh_count+=1   
            print(f" Pharaoh not found in DB for script: {pharaoh_name}")
            continue

        ph_script = PharaohScript(
            id=id,
            pharaoh_id=pharaoh_obj.id,
            pharaoh_script=pharaoh_script,
            pharaoh_script_embedding=emb,
        )
        
        id+=1
        session.merge(ph_script)

    session.commit()

print(f"{pharaoh_count} Pharaoh scripts skipped due to missing pharaohs in DB.")
print(skipped)
print("Pharaoh scripts synced from Chroma to PostgreSQL!")

#run using python -m scripts.seed_db
