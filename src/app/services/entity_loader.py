from sqlalchemy import select
from sqlalchemy.orm import Session
from src.models import Pharaoh, Landmark


def load_entity(category: str, predicted_name: str, db: Session):
    """Fetch entity from DB by exact name match."""
    model_class = Pharaoh if category == "pharaoh" else Landmark

    # Handle underscores from model labels to match DB spaces
    search_name = predicted_name.replace("_", " ")

    stmt = select(model_class).where(model_class.name.ilike(search_name))
    return db.execute(stmt).scalars().first()
