from sqlalchemy import select
from sqlalchemy.orm import Session
from src.models import Pharaoh, Landmark


def load_entity(category: str, predicted_name: str, db: Session):
    """Fetch entity from DB by exact name match."""
    model_class = Pharaoh if category == "pharaoh" else Landmark

    stmt = select(model_class).where(model_class.name == predicted_name)
    return db.execute(stmt).scalars().first()
