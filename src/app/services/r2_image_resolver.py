import os
import logging
from src.app.api.routes.assets import get_r2_client

logger = logging.getLogger(__name__)

class R2ImageResolver:
    _pharaoh_map = None
    _landmark_map = None
    _initialized = False

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        
        cls._pharaoh_map = {}
        cls._landmark_map = {}
        
        client = get_r2_client()
        if not client:
            logger.warning("[R2ImageResolver] R2 client not available. Cannot fetch dynamic images.")
            cls._initialized = True
            return
            
        bucket_name = os.getenv("R2_BUCKET_NAME", "echo-data")
        paginator = client.get_paginator("list_objects_v2")
        
        try:
            # 1. Fetch pharaohs images
            for page in paginator.paginate(Bucket=bucket_name, Prefix="data/video_generation/raw/pharaohs_images/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    parts = key.split("/")
                    if len(parts) >= 6:
                        entity_name = parts[4].strip().lower()
                        filename = parts[5]
                        name_no_ext, _ = os.path.splitext(filename)
                        if name_no_ext.strip().lower() == "statue 1":
                            cls._pharaoh_map[entity_name] = key
                            
            # 2. Fetch landmarks images
            for page in paginator.paginate(Bucket=bucket_name, Prefix="data/video_generation/raw/landmarks_images/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    parts = key.split("/")
                    if len(parts) >= 6:
                        entity_name = parts[4].strip().lower()
                        filename = parts[5]
                        name_no_ext, _ = os.path.splitext(filename)
                        if name_no_ext.strip().lower() == "1":
                            cls._landmark_map[entity_name] = key
                            
            logger.info(f"[R2ImageResolver] Successfully mapped {len(cls._pharaoh_map)} pharaohs and {len(cls._landmark_map)} landmarks.")
        except Exception as e:
            logger.error(f"[R2ImageResolver] Error initializing maps: {e}")
            
        cls._initialized = True

    @classmethod
    def get_pharaoh_image(cls, name: str) -> str:
        cls.initialize()
        if not cls._pharaoh_map:
            return None
        return cls._pharaoh_map.get(name.strip().lower())

    @classmethod
    def get_landmark_image(cls, name: str) -> str:
        cls.initialize()
        if not cls._landmark_map:
            return None
        return cls._landmark_map.get(name.strip().lower())
