from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from src.db import get_db
from src.db_models import Pharaoh, Landmark

from src.app.services.r2_image_resolver import R2ImageResolver

router = APIRouter()

def _serialize_pharaoh(p: Pharaoh) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "type": getattr(p, "type", None),
        "dynasty": p.dynasty,
        "period": p.period,
        "location": None,
        "image": R2ImageResolver.get_pharaoh_image(p.name),
    }


def _serialize_landmark(l: Landmark) -> dict:
    return {
        "id": l.id,
        "name": l.name,
        "description": l.description,
        "dynasty": None,
        "period": None,
        "location": l.location,
        "image": R2ImageResolver.get_landmark_image(l.name),
    }


# ── Fixed lists — always shown in this order ────────────────────────────
PHARAOH_NAMES = [
    "Akhenaton",
    "Cleopatra VII Philopator",
    "Hatshepsut",
    "Ramesses II",
    "Tutankhamun",
]

LANDMARK_NAMES = [
    "Pyramids of Giza",
    "Sphinx",
    "Temple of Karnak",
    "Temple of Luxor",
    "The Great Temple of Ramesses II at Abu Simbel",
]


@router.get("/trending")
def get_trending_entities(db: Session = Depends(get_db)):
    """
    Returns the fixed set of featured pharaohs and landmarks from the DB,
    in the exact display order defined above.
    """
    try:
        # Fetch pharaohs by name, preserve order
        pharaaoh_rows = db.execute(
            select(Pharaoh)
            .where(Pharaoh.name.in_(PHARAOH_NAMES))
        ).scalars().all()

        pharaoh_map = {p.name: p for p in pharaaoh_rows}
        pharaohs = [_serialize_pharaoh(pharaoh_map[n]) for n in PHARAOH_NAMES if n in pharaoh_map]

        # Fetch landmarks by name, preserve order
        landmark_rows = db.execute(
            select(Landmark)
            .where(Landmark.name.in_(LANDMARK_NAMES))
        ).scalars().all()

        landmark_map = {l.name: l for l in landmark_rows}
        landmarks = [_serialize_landmark(landmark_map[n]) for n in LANDMARK_NAMES if n in landmark_map]

        return {
            "pharaohs": pharaohs,
            "landmarks": landmarks,
        }
    except Exception as e:
        return {"pharaohs": [], "landmarks": [], "error": str(e)}


@router.get("/all")
def get_all_entities(db: Session = Depends(get_db), search: str = ""):
    """
    Returns ALL pharaohs and landmarks from the DB.
    Optional `search` query param filters by name (case-insensitive).
    """
    try:
        pharaoh_query = select(Pharaoh).order_by(Pharaoh.name)
        landmark_query = select(Landmark).order_by(Landmark.name)

        if search:
            pharaoh_query = pharaoh_query.where(Pharaoh.name.ilike(f"%{search}%"))
            landmark_query = landmark_query.where(Landmark.name.ilike(f"%{search}%"))

        pharaoh_rows = db.execute(pharaoh_query).scalars().all()
        landmark_rows = db.execute(landmark_query).scalars().all()

        pharaohs = [_serialize_pharaoh(p) for p in pharaoh_rows]
        landmarks = [_serialize_landmark(l) for l in landmark_rows]

        return {
            "pharaohs": pharaohs,
            "landmarks": landmarks,
        }
    except Exception as e:
        return {"pharaohs": [], "landmarks": [], "error": str(e)}


@router.get("/details")
def get_entity_details(name: str, type: str, db: Session = Depends(get_db)):
    """
    Fetches the full entity metadata from the database by exact/similar name match,
    including composite entity sub-fields and lists.
    """
    try:
        from src.app.services.entity_loader import load_entity
        entity = load_entity(type, name, db)
        if not entity:
            return {"error": "Entity not found"}

        entity_data = {
            "id": entity.id,
            "name": entity.name,
            "description": entity.description,
        }

        if type == "pharaoh":
            entity_data["type"] = getattr(entity, "type", None)
            entity_data["dynasty"] = getattr(entity, "dynasty", None)
            entity_data["period"] = getattr(entity, "period", None)
            entity_data["image"] = R2ImageResolver.get_pharaoh_image(entity.name)
            composite_raw = getattr(entity, "composite_entity", None)
            entity_data["composite_entity"] = composite_raw
            
            # Load full metadata for nested composite sub-entities
            if composite_raw:
                sub_names = [s.strip() for s in composite_raw.split(",") if s.strip()]
                entity_data["composite_entities_data"] = [
                    {
                        "name": sn,
                        "type": getattr(row, "type", None) if row else None,
                        "dynasty": getattr(row, "dynasty", None) if row else None,
                        "period": getattr(row, "period", None) if row else None,
                        "image": R2ImageResolver.get_pharaoh_image(sn) if row else None,
                    }
                    for sn in sub_names
                    for row in [db.query(Pharaoh).filter(Pharaoh.name.ilike(sn)).first()]
                ]
        else:
            entity_data["location"] = getattr(entity, "location", None)
            entity_data["image"] = R2ImageResolver.get_landmark_image(entity.name)

        return {
            "source": "explore",
            "type": type,
            "name": entity.name,
            "confidence": 1.0,
            "binary_confidence": 1.0,
            "entity": entity_data,
        }
    except Exception as e:
        return {"error": str(e)}
