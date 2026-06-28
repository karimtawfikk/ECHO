from __future__ import annotations

from threading import Semaphore
from typing import Any
from .runtime import hieroglyph_runtime
from .schemas import (
    ClassifiedSymbol,
    HieroglyphTranslationMetadata,
    HieroglyphTranslationRequest,
    TranslationResult,
)

class HieroglyphDetectionService:

    def __init__(self, max_concurrent_gpu: int = 1) -> None:
        self._gpu_semaphore = Semaphore(max_concurrent_gpu)

    def detect(
        self,
        request: HieroglyphTranslationRequest,
        on_step: Any | None = None
    ) -> tuple[TranslationResult, HieroglyphTranslationMetadata]:

        image_bgr = hieroglyph_runtime.decode_image(request.image_base64)

        with self._gpu_semaphore:
            raw_result, raw_metadata = hieroglyph_runtime.run_pipeline(image_bgr, on_step=on_step)

        symbols = [
            ClassifiedSymbol(
                gardiner_code=s["gardiner_code"],
                classification_confidence=s["classification_confidence"],
                bbox=s["bbox"],
                detection_confidence=s["detection_confidence"],
            )
            for s in raw_result["symbols"]
        ]

        result = TranslationResult(
            symbols=symbols,
            num_symbols_detected=raw_result["num_symbols_detected"],
            num_clusters=raw_result["num_clusters"],
            translation_text=raw_result.get("translation_text", ""),
            annotated_image_base64=raw_result.get("annotated_image_base64"),
        )

        metadata = HieroglyphTranslationMetadata(
            num_symbols_detected=raw_metadata["num_symbols_detected"],
            num_clusters=raw_metadata["num_clusters"],
            pipeline_time_ms=raw_metadata["pipeline_time_ms"],
        )

        return result, metadata

hieroglyph_service = HieroglyphDetectionService()
