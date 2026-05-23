from __future__ import annotations
import base64
from threading import Semaphore
from .runtime import recognition_inference
from .schemas import RecognitionRequest, RecognitionResult

class RecognitionService:
    def __init__(self, max_concurrent_gpu: int = 1) -> None:
        self._gpu_semaphore = Semaphore(max_concurrent_gpu)

    async def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        image_bytes = base64.b64decode(request.image_base64)
        with self._gpu_semaphore:
            raw_result = await recognition_inference.run_hierarchical_inference(image_bytes)
        
        return RecognitionResult(
            type=raw_result["type"],
            name=raw_result["name"],
            confidence=raw_result["confidence"],
            binary_confidence=raw_result["binary_confidence"]
        )

recognition_service = RecognitionService()
