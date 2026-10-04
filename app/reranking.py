"""Step 7b: reorder retrieved chunks with a Cohere reranker and keep the best ones."""

from dataclasses import replace

import cohere

from app.upstream import Retrier
from app.vector_store import RetrievedChunk


class CohereReranker:
    """Reorders retrieved chunks by relevance to a question using the Cohere rerank API."""

    def __init__(self, api_key: str, model: str, retrier: Retrier | None = None) -> None:
        self._retrier = retrier or Retrier()
        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Keep the chunks most relevant to the query.

        Args:
            query: The user's question.
            chunks: Candidate chunks from retrieval.
            top_n: How many chunks to keep.

        Returns:
            Up to ``top_n`` chunks, most relevant first, with their score replaced by the rerank score.
        """
        if len(chunks) <= 1:
            return chunks
        documents = [f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text for chunk in chunks]
        response = self._retrier.call(
            lambda: self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(chunks)),
                request_options={"max_retries": 0},
            ),
            name="Cohere rerank",
        )
        return [replace(chunks[result.index], score=result.relevance_score) for result in response.results]
