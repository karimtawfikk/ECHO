from threading import Lock

from .schemas import ChatRequest, SpeechRequest, SpeechMetadata
from .runtime import chatbot_runtime


class ChatbotService:
    def __init__(self) -> None:
        self._lock = Lock()

    def stream_chat(self, request: ChatRequest):
        def event_stream():
            with self._lock:
                yield from chatbot_runtime.stream_chat(
                    session_id=request.session_id,
                    entity_type=request.entity_type,
                    entity_name=request.entity_name,
                    message=request.message,
                )

        return event_stream()

    def transcribe_audio(self, filename: str, audio_bytes: bytes) -> str:
        with self._lock:
            return chatbot_runtime.transcribe_audio(filename, audio_bytes)

    def synthesize_speech(self, request: SpeechRequest) -> tuple[bytes, SpeechMetadata]:
        audio_bytes, language, voice = chatbot_runtime.synthesize_speech(
            request.text,
            entity_type=request.entity_type,
            entity_name=request.entity_name,
        )
        return audio_bytes, SpeechMetadata(language=language, voice=voice)


chatbot_service = ChatbotService()
