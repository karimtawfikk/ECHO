from .schemas import ChatRequest, SpeechRequest, SpeechMetadata, InitSessionRequest
from .runtime import chatbot_runtime

class ChatbotService:
    def __init__(self) -> None:
        pass

    def init_session(self, session_id: str, entity_type: str, entity_name: str, user_id: str | None = None, context: str | None = None, history: list | None = None, rewriter_history: list | None = None) -> None:
        chatbot_runtime.init_session(
            session_id=session_id,
            user_id=user_id,
            entity_type=entity_type,
            entity_name=entity_name,
            context=context,
            history=history,
            rewriter_history=rewriter_history,
        )

    def stream_chat(self, request: ChatRequest):
        def event_stream():
            yield from chatbot_runtime.stream_chat(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    entity_type=request.entity_type,
                    entity_name=request.entity_name,
                    message=request.message,
                    context=request.context,
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
