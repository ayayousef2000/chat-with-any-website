"""Step 8: generate an answer grounded in the retrieved chunks, using a GPT model served by Groq."""

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import groq
from groq import Groq
from groq.types.chat import ChatCompletion, ChatCompletionMessageParam

from app.citations import normalize_citations
from app.upstream import Retrier, retry_after_seconds
from app.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

# How long a key is left alone after its limit was reached, when Groq does not say when it will work again.
DEFAULT_REST_SECONDS = 60.0

# Chinese, Japanese and Korean letters. The model now and then answers in Chinese to a question that is not.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

# How many more times to ask when the answer came in the wrong language.
LANGUAGE_RETRIES = 2


def _answered_in_cjk_by_mistake(question: str, answer: str) -> bool:
    """Tell whether the answer uses Chinese, Japanese or Korean letters although the question does not."""
    return _CJK.search(answer) is not None and _CJK.search(question) is None


SYSTEM_PROMPT = """You answer questions about a single web page.
Use only the numbered excerpts from that page provided by the user.
Do not make up facts. State only what the excerpts say: do not add reasons, explanations or background of your
own, even when they seem obvious or are common knowledge.

Citations:
- After each statement taken from the page, cite the excerpts it comes from using plain square brackets with
  the excerpt number only, for example [1] or [2][3].
- Never use any other citation style, such as 【1†L1-L3】, and never mention line numbers.

If the excerpts do not contain the answer, say only that the page does not seem to cover it, and cite nothing.
Language: write the whole answer in the language of the question, never in the language of the excerpts or
of these instructions. A question in English always gets an answer in English. Never answer in Chinese unless
the question itself is written in Chinese."""


class GroqChat:
    """Generates answers with a chat model served by Groq.

    Several keys can be given and are used in a circle. The first is used until Groq says its limit is reached,
    then the second, and so on; after the last, the first is used again. A key whose limit was reached rests for
    as long as Groq asked and is skipped until it has recovered. The key in use stays in use until its own limit
    is reached.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        retrier: Retrier | None = None,
        backup_keys: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._retrier = retrier or Retrier()
        # The client's own retries are off: retrying is done here, once, with a policy that suits a person waiting.
        self._clients = [Groq(api_key=key, max_retries=0) for key in dict.fromkeys([api_key, *backup_keys])]
        self._model = model
        self._clock = clock
        self._lock = threading.Lock()
        self._current = 0  # the key in use
        self._rested_until = [0.0] * len(self._clients)  # per key: the time from which it may be used again

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

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(LANGUAGE_RETRIES + 1):
            completion = self._retrier.call(lambda: self._complete(messages, temperature=0.2), name="Groq answer")
            answer = (completion.choices[0].message.content or "").strip()
            if attempt == LANGUAGE_RETRIES or not _answered_in_cjk_by_mistake(question, answer):
                break
            logger.warning("Groq answered in the wrong language; asking again (attempt %d)", attempt + 2)
        return normalize_citations(answer)

    def complete(self, messages: Sequence[ChatCompletionMessageParam], **options: Any) -> str:
        """Send messages to the model and return its reply, with the same key rotation and retries as answers.

        Args:
            messages: The conversation to send.
            **options: Further request options, such as ``temperature`` or ``response_format``.

        Returns:
            The text of the model's reply.
        """
        completion = self._retrier.call(lambda: self._complete(messages, **options), name="Groq request")
        return (completion.choices[0].message.content or "").strip()

    def _complete(self, messages: Sequence[ChatCompletionMessageParam], **options: Any) -> ChatCompletion:
        """Make one request, moving on to the next key each time a key's limit is reached.

        Raises:
            groq.RateLimitError: When no key can be used. It is the error of the key that recovers soonest,
                so the wait reported to the person is the shortest one.
        """
        now = self._clock()
        with self._lock:
            count = len(self._clients)
            circle = [(self._current + step) % count for step in range(count)]  # the key in use, then the next ones
            ready = [index for index in circle if self._rested_until[index] <= now]
            # With every key resting, try the one that recovers first: its limit may have been lifted already.
            candidates = ready or [min(circle, key=self._rested_until.__getitem__)]

        soonest: tuple[float, groq.RateLimitError] | None = None
        for position, index in enumerate(candidates):
            try:
                completion: ChatCompletion = self._clients[index].chat.completions.create(
                    model=self._model, messages=list(messages), **options
                )
                with self._lock:
                    self._current = index
                return completion
            except groq.RateLimitError as error:
                asked = retry_after_seconds(error)
                rest = DEFAULT_REST_SECONDS if asked is None else asked
                with self._lock:
                    self._rested_until[index] = now + rest
                if soonest is None or rest < soonest[0]:
                    soonest = (rest, error)
                if position + 1 < len(candidates):
                    logger.warning(
                        "Groq key %d of %d reached its limit; using key %d",
                        index + 1,
                        len(self._clients),
                        candidates[position + 1] + 1,
                    )
        if soonest is None:  # pragma: no cover - there is always a candidate, and only a limit error leaves the loop
            raise AssertionError("unreachable")
        raise soonest[1]
