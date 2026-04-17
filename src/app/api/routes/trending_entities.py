from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from src.db import get_db
from src.models import Pharaoh, Landmark

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
    }


def _serialize_landmark(l: Landmark) -> dict:
    return {
        "id": l.id,
        "name": l.name,
        "description": l.description,
        "dynasty": None,
        "period": None,
        "location": l.location,
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
