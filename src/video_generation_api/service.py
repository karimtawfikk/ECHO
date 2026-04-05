from __future__ import annotations

from pathlib import Path
from threading import Lock

from .runtime import video_generation_runtime
from .schemas import VideoGenerationMetadata, VideoGenerationRequest


class VideoGenerationService:
    def __init__(self) -> None:
        self._lock = Lock()

    def generate(self, request: VideoGenerationRequest) -> tuple[Path, VideoGenerationMetadata]:
        entity_type = "landmark" if request.is_landmark else "pharaoh"

        with self._lock:
            output_path = video_generation_runtime.generate_video(
                entity_name=request.entity_name,
                is_landmark=request.is_landmark,
            )

        metadata = VideoGenerationMetadata(
            entity_name=request.entity_name,
            entity_type=entity_type,
            filename=output_path.name,
            output_path=str(output_path),
        )
        return output_path, metadata


video_generation_service = VideoGenerationService()
