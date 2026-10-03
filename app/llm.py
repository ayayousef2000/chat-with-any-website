"""Step 8: generate an answer grounded in the retrieved chunks, using a GPT model served by Groq."""

from collections.abc import Sequence

from groq import Groq

from app.citations import normalize_citations
from app.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about a single web page.
Use only the numbered excerpts from that page provided by the user.
Do not make up facts.

Citations:
- After each statement taken from the page, cite the excerpts it comes from using plain square brackets with
  the excerpt number only, for example [1] or [2][3].
- Never use any other citation style, such as 【1†L1-L3】, and never mention line numbers.

If the excerpts do not contain the answer, say only that the page does not seem to cover it, and cite nothing.
Answer in the same language as the question."""


class GroqChat:
    """Generates answers with a chat model served by Groq."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = Groq(api_key=api_key)
        self._model = model

    def answer(self, question: str, title: str, chunks: Sequence[RetrievedChunk]) -> str:
        """Answer a question using only the given excerpts.

        Args:
            question: The user's question.
            title: The title of the page the excerpts come from.
            chunks: The retrieved excerpts, most relevant first.

        Returns:
            The model's answer, citing excerpts with plain ``[n]`` markers.
        """
        excerpts = "\n\n".join(
            f"[{number}] (section: {chunk.heading})\n{chunk.text}" if chunk.heading else f"[{number}]\n{chunk.text}"
            for number, chunk in enumerate(chunks, start=1)
        )
        user_message = f"Page title: {title}\n\nExcerpts:\n{excerpts}\n\nQuestion: {question}"

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        return normalize_citations((completion.choices[0].message.content or "").strip())
