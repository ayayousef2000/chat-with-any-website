"""Step 5: turn text into vectors with Cohere."""

from collections.abc import Sequence
from typing import Literal

import cohere

from app.upstream import Retrier

_MAX_TEXTS_PER_CALL = 96

InputType = Literal["search_document", "search_query"]


class CohereEmbedder:
    """Creates text embeddings with the Cohere API."""

    def __init__(self, api_key: str, model: str, dimension: int, retrier: Retrier | None = None) -> None:
        self._retrier = retrier or Retrier()
        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model
        self._dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts that will be stored and searched later.

        Args:
            texts: The texts to embed. They are sent in batches that respect the API limit.

        Returns:
            One vector per input text, in the same order.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_TEXTS_PER_CALL):
            vectors.extend(self._embed(texts[start : start + _MAX_TEXTS_PER_CALL], "search_document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for searching stored documents.

        Args:
            text: The question.

        Returns:
            The query vector.
        """
        return self._embed([text], "search_query")[0]

    def _embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        # The client's own retries are off: retrying is done here, once, with a policy that suits a person waiting.
        response = self._retrier.call(
            lambda: self._client.embed(
                model=self._model,
                texts=list(texts),
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self._dimension,
                request_options={"max_retries": 0},
            ),
            name="Cohere embeddings",
        )
        vectors = response.embeddings.float_
        if not vectors or len(vectors) != len(texts):
            raise RuntimeError("Cohere returned an unexpected number of embeddings.")
        return vectors
