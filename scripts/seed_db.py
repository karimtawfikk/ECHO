from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
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

#Pharaohs
pharaohs_json_path = Path(r"C:\Uni\4th Year\GP\ECHO\data\video_generation\outputs\pharaohs.json")
with open(pharaohs_json_path, "r", encoding="utf-8") as f:
    pharaohs_data = json.load(f)

with Session(engine) as session:
    for ph in pharaohs_data:
        pharaoh = Pharaoh(
            id=ph["id"],
            name=ph["name"],
            period=ph["period"],
            dynasty=ph["dynasty"],
            type=ph["type"],
            description=ph["description"],
            composite_entity=ph["composite_entity"]
        )
        session.merge(pharaoh)
    session.commit()

#run using python -m scripts.seed_db
