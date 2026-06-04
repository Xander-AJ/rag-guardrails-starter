"""Vector store abstraction.

Defines the retriever contract and two implementations:

- :class:`BaseRetriever`: the abstract async interface the RAG pipeline depends
  on, plus the :class:`RetrievedChunk` result shape and :class:`RetrieverError`.
- :class:`QdrantRetriever`: a Qdrant-backed retriever. To keep the library
  lightweight at this stage, the query embedding is produced *without* an
  external embedding model — text is tokenized with ``tiktoken`` and reduced to
  a fixed-size, L2-normalized term-frequency vector via the hashing trick. The
  Qdrant collection must be populated with vectors built the same way and of
  the same dimension (:data:`DEFAULT_VECTOR_SIZE`).
- :class:`InMemoryRetriever`: a dependency-free retriever that scores chunks by
  token overlap with the query, for tests and examples without a live Qdrant.

Connection details come from :class:`~rag_guardrails.config.Settings`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

import tiktoken
from qdrant_client import AsyncQdrantClient

from rag_guardrails.config import Settings, get_config
from rag_guardrails.guardrails._text import content_tokens

# Dimension of the lightweight term-frequency embedding. Qdrant collections
# used with :class:`QdrantRetriever` must be created with this vector size.
DEFAULT_VECTOR_SIZE: int = 256

# tiktoken encoding used to tokenize text for the term-frequency embedding.
DEFAULT_ENCODING: str = "cl100k_base"


@dataclass
class RetrievedChunk:
    """A single chunk returned from a retrieval query.

    Attributes:
        content: The chunk's text content.
        source: An identifier for where the chunk came from (document id, URL,
            file path, or the vector point id as a fallback).
        score: Relevance score for the query; higher means more relevant.
        metadata: Arbitrary additional payload associated with the chunk.
    """

    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrieverError(Exception):
    """Raised when retrieval cannot be performed or configured correctly."""


class BaseRetriever(ABC):
    """Abstract contract for a document retriever.

    Implementations fetch the most relevant chunks for a query from some
    backing store and return them as :class:`RetrievedChunk` objects ordered by
    descending relevance.
    """

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Retrieve the ``top_k`` most relevant chunks for ``query``.

        Args:
            query: The user query to retrieve context for.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` objects, most relevant
            first.

        Raises:
            RetrieverError: If retrieval cannot be performed.
        """
        raise NotImplementedError


def _term_frequency_embedding(
    tokens: list[int], vector_size: int
) -> list[float]:
    """Build an L2-normalized term-frequency vector from token ids.

    Token ids are hashed into ``vector_size`` buckets (the hashing trick) and
    counted, then the resulting vector is L2-normalized so similarity reduces
    to cosine similarity. A returned all-zero vector indicates empty input.

    Args:
        tokens: Token ids produced by the tokenizer.
        vector_size: Target dimensionality of the embedding.

    Returns:
        A list of ``vector_size`` floats.
    """
    vector = [0.0] * vector_size
    for token_id in tokens:
        vector[token_id % vector_size] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        vector = [value / norm for value in vector]
    return vector


class QdrantRetriever(BaseRetriever):
    """Retriever backed by a Qdrant vector collection.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. Falls back to
            the shared singleton from
            :func:`~rag_guardrails.config.get_config`. ``qdrant_url`` and
            ``qdrant_collection`` are read from it.
        vector_size: Dimensionality of the query embedding; must match the
            collection's vector size. Defaults to :data:`DEFAULT_VECTOR_SIZE`.

    Raises:
        RetrieverError: If ``qdrant_url`` is not configured.
    """

    def __init__(
        self,
        config: Settings | None = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        self._config: Settings = config or get_config()
        if self._config.qdrant_url is None:
            raise RetrieverError(
                "Qdrant URL is not configured. Set QDRANT_URL in the "
                "environment or .env file, or use InMemoryRetriever for local "
                "testing."
            )

        self._collection: str = self._config.qdrant_collection
        self._vector_size: int = vector_size
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            url=self._config.qdrant_url
        )
        self._encoder: tiktoken.Encoding | None = None

    def _embed(self, text: str) -> list[float]:
        """Embed ``text`` as a normalized term-frequency vector.

        The tiktoken encoder is loaded lazily on first use to avoid paying for
        it at construction time.
        """
        if self._encoder is None:
            self._encoder = tiktoken.get_encoding(DEFAULT_ENCODING)
        tokens = self._encoder.encode(text)
        return _term_frequency_embedding(tokens, self._vector_size)

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Embed ``query`` and run a vector similarity search in Qdrant.

        Args:
            query: The user query.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` objects, most relevant
            first.

        Raises:
            RetrieverError: If the Qdrant search fails.
        """
        query_vector = self._embed(query)
        try:
            points = await self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize all backend errors
            raise RetrieverError(
                f"Qdrant search failed for collection "
                f"'{self._collection}': {exc}"
            ) from exc

        return [self._to_chunk(point) for point in points]

    @staticmethod
    def _to_chunk(point: Any) -> RetrievedChunk:
        """Map a Qdrant scored point to a :class:`RetrievedChunk`."""
        payload: dict[str, Any] = dict(point.payload or {})
        content = payload.pop("content", None) or payload.pop("text", "")
        source = payload.pop("source", None) or str(getattr(point, "id", ""))
        return RetrievedChunk(
            content=content,
            source=source,
            score=float(getattr(point, "score", 0.0)),
            metadata=payload,
        )

    async def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        await self._client.close()


class InMemoryRetriever(BaseRetriever):
    """In-memory retriever scoring chunks by token overlap with the query.

    Intended for tests and examples that need a working retriever without a
    live Qdrant instance. Scoring uses the same content-token extraction as the
    guardrails (:func:`rag_guardrails.guardrails._text.content_tokens`).

    Args:
        chunks: The corpus of chunks to search over.
    """

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks: list[RetrievedChunk] = list(chunks)

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks with the highest token overlap.

        The relevance score is the fraction of the query's content tokens that
        appear in the chunk. Chunks with no overlap are excluded. The returned
        chunks are copies with their ``score`` set to the computed relevance.

        Args:
            query: The user query.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` objects, most relevant
            first.
        """
        query_tokens = set(content_tokens(query))
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            chunk_tokens = set(content_tokens(chunk.content))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            if overlap > 0.0:
                scored.append(replace(chunk, score=round(overlap, 4)))

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]
