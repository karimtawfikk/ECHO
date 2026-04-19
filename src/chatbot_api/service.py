

from .schemas import ChatRequest, SpeechRequest, SpeechMetadata, InitSessionRequest
from .runtime import chatbot_runtime

class ChatbotService:
    def __init__(self) -> None:
        pass

    def init_session(self, request: InitSessionRequest) -> None:
        chatbot_runtime.init_session(
            session_id=request.session_id,
            entity_type=request.entity_type,
            entity_name=request.entity_name,
        )

    def stream_chat(self, request: ChatRequest):
        def event_stream():
            yield from chatbot_runtime.stream_chat(
                    session_id=request.session_id,
                    entity_type=request.entity_type,
                    entity_name=request.entity_name,
                    message=request.message,
                )

        return event_stream()

    def transcribe_audio(self, filename: str, audio_bytes: bytes) -> str:
        return chatbot_runtime.transcribe_audio(filename, audio_bytes)

    def synthesize_speech(self, request: SpeechRequest) -> tuple[bytes, SpeechMetadata]:
        audio_bytes, language, voice = chatbot_runtime.synthesize_speech(
            request.text,
            entity_type=request.entity_type,
            entity_name=request.entity_name,
        )
        return audio_bytes, SpeechMetadata(language=language, voice=voice)


chatbot_service = ChatbotService()
