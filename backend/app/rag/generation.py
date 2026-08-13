import re
from collections.abc import Sequence
from dataclasses import dataclass

from ollama import Client

from app.rag.retrieval import RetrievalResult

SYSTEM_PROMPT = """You are KnowledgeAI, a document question-answering assistant.

Decide which kind of message you are answering before you reply.

CONVERSATIONAL — greetings, thanks, goodbyes, small talk, casual remarks, and
questions about you and what you can do. A remark is not a request for
information: "the weather is nice today" is a pleasantry to acknowledge, whereas
"what is the weather today?" asks for a fact you do not have. Examples:
"hello", "how are you", "the weather is nice today", "thanks, that helped",
"what can you help me with". Reply naturally and briefly, in one or two
sentences. Never cite sources, never append source labels, and never refuse;
the sources below are irrelevant here and must be ignored.

INFORMATIONAL — any request for facts, definitions, figures, explanations, or
analysis of a subject. This includes questions you could answer from your own
training, such as "who wrote Hamlet?" or "what is the weather today?". Answer
only from the supplied sources. Cite every factual claim with one or more source
labels in square brackets, such as [S1] or [S1][S2]. Never use outside
knowledge, however confident you are. If the sources do not contain enough
information, reply exactly: "I don't have enough information in the uploaded
documents."

If a message could be either kind, treat it as INFORMATIONAL. Small talk never
licenses answering a factual question from memory, and a friendly opening does
not change how the rest of the message must be handled."""


class GenerationError(RuntimeError):
    """Raised when the chat model cannot produce a usable answer."""


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    cited_source_ids: list[str]


class OllamaChatService:
    """Grounded answer generation through Ollama's chat API."""

    def __init__(self, *, base_url: str, model: str, temperature: float = 0.1) -> None:
        self.client = Client(host=base_url)
        self.model = model
        self.temperature = temperature

    def answer(self, *, question: str, sources: Sequence[RetrievalResult]) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{self._source_id(index)}] Document: {source.chunk.document_id}; "
            f"Page: {source.chunk.page_number}\n{source.chunk.text}"
            for index, source in enumerate(sources, start=1)
        )
        user_prompt = f"Sources:\n{context}\n\nQuestion: {question}"

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": self.temperature},
            )
        except Exception as error:
            raise GenerationError("Could not generate an answer through Ollama.") from error

        message = getattr(response, "message", None)
        answer_text = getattr(message, "content", None)
        if answer_text is None and isinstance(response, dict):
            answer_text = response.get("message", {}).get("content")
        if not answer_text or not answer_text.strip():
            raise GenerationError("The chat model returned an empty answer.")

        allowed_source_ids = {self._source_id(index) for index in range(1, len(sources) + 1)}
        cited_source_ids = [
            source_id
            for source_id in dict.fromkeys(re.findall(r"\[(S\d+)\]", answer_text))
            if source_id in allowed_source_ids
        ]
        return GeneratedAnswer(text=answer_text.strip(), cited_source_ids=cited_source_ids)

    @staticmethod
    def _source_id(index: int) -> str:
        return f"S{index}"
