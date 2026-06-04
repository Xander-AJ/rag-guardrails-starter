"""Optional reranking layer.

Sits between retrieval and generation. Given a query and the candidate
documents returned by the retriever(s), reorders them so the most relevant
context is prioritized for the LLM.

This module provides:

- :class:`BaseReranker`: the abstract async interface every reranker satisfies.
- :class:`LexicalReranker`: a dependency-free BM25-inspired lexical reranker
  that rescores an existing candidate set.
- :class:`ReciprocalRankFusion`: a reranker that queries several retrievers in
  parallel and fuses their rankings with Reciprocal Rank Fusion (RRF).
- :class:`RerankerError`: the common error type for reranking failures.

Reranking is entirely optional: the pipeline must work without a reranker
configured.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace

from rag_guardrails.guardrails._text import content_tokens
from rag_guardrails.rag.retriever import BaseRetriever, RetrievedChunk

# Standard RRF damping constant; larger values flatten the contribution of top
# ranks. 60 is the value from the original Cormack et al. RRF paper.
DEFAULT_RRF_K: int = 60


class RerankerError(Exception):
    """Raised when reranking cannot be performed."""


class BaseReranker(ABC):
    """Abstract contract for a reranker.

    Implementations reorder candidate chunks for a query and truncate to the
    most relevant ``top_k``. All reranking is asynchronous so implementations
    may call out to retrievers, cross-encoders, or LLM scorers.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Reorder ``chunks`` by relevance to ``query`` and keep the top ``k``.

        Args:
            query: The user query to rank chunks against.
            chunks: Candidate chunks to rerank.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` objects, most relevant
            first, with ``score`` set to the reranker's relevance score.

        Raises:
            RerankerError: If reranking cannot be performed.
        """
        raise NotImplementedError


class LexicalReranker(BaseReranker):
    """BM25-inspired lexical reranker with no external dependencies.

    Each chunk is scored by summing, over the query's content terms, the term's
    normalized frequency in the chunk weighted by an inverse-document-frequency
    factor computed across the candidate set::

        score(chunk) = sum over query terms t of
            (tf(t, chunk) / len(chunk)) * log(1 + 1 / df(t))

    where ``tf(t, chunk)`` is the number of times ``t`` appears in the chunk,
    ``len(chunk)`` is the chunk's content-token count, and ``df(t)`` is the
    number of candidate chunks containing ``t``. Rare terms (low ``df``) are
    weighted more heavily, and longer chunks are length-normalized so they do
    not win on sheer size. Tokenization reuses
    :func:`rag_guardrails.guardrails._text.content_tokens`, so stopwords and
    very short tokens are ignored.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Rescore ``chunks`` with BM25-inspired lexical scoring.

        Args:
            query: The user query to rank chunks against.
            chunks: Candidate chunks to rerank.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to ``top_k`` chunks (copies with ``score`` set to the lexical
            score), most relevant first. Returns an empty list if the query has
            no content tokens or ``chunks`` is empty.
        """
        if not chunks:
            return []

        query_terms = set(content_tokens(query))
        if not query_terms:
            return []

        # Tokenize each chunk once; keep term counts and lengths.
        chunk_tokens: list[list[str]] = [
            content_tokens(chunk.content) for chunk in chunks
        ]
        chunk_counts: list[Counter[str]] = [
            Counter(tokens) for tokens in chunk_tokens
        ]

        # Document frequency: how many chunks contain each query term.
        document_frequency: Counter[str] = Counter()
        for counts in chunk_counts:
            for term in query_terms:
                if counts.get(term, 0) > 0:
                    document_frequency[term] += 1

        scored: list[RetrievedChunk] = []
        for chunk, tokens, counts in zip(
            chunks, chunk_tokens, chunk_counts, strict=True
        ):
            length = len(tokens)
            if length == 0:
                continue

            score = 0.0
            for term in query_terms:
                term_frequency = counts.get(term, 0)
                if term_frequency == 0:
                    continue
                df = document_frequency[term]
                idf_weight = math.log(1.0 + 1.0 / df)
                score += (term_frequency / length) * idf_weight

            if score > 0.0:
                scored.append(replace(chunk, score=round(score, 6)))

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]


class ReciprocalRankFusion(BaseReranker):
    """Reranker that fuses several retrievers' rankings via RRF.

    On each call the configured retrievers are queried in parallel and their
    ranked result lists are combined with Reciprocal Rank Fusion::

        score(chunk) = sum over ranked lists of 1 / (k + rank)

    where ``rank`` is the chunk's 1-based position in a given list and ``k`` is
    the damping constant (:data:`DEFAULT_RRF_K`, 60 by default). Any candidate
    chunks passed to :meth:`rerank` are treated as an additional ranked list, so
    the reranker honors the :class:`BaseReranker` contract while still drawing on
    its own retrievers. Chunks are deduplicated by content, accumulating each
    occurrence's RRF contribution.

    Args:
        retrievers: Retrievers to query and fuse. Must be non-empty.
        k: RRF damping constant. Defaults to :data:`DEFAULT_RRF_K`.
        fetch_k: Number of chunks to request from each retriever. Deeper lists
            give RRF more signal; defaults to 10.

    Raises:
        RerankerError: If ``retrievers`` is empty.
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        k: int = DEFAULT_RRF_K,
        fetch_k: int = 10,
    ) -> None:
        if not retrievers:
            raise RerankerError(
                "ReciprocalRankFusion requires at least one retriever."
            )
        self._retrievers: list[BaseRetriever] = list(retrievers)
        self._k: int = k
        self._fetch_k: int = fetch_k

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Query all retrievers in parallel and fuse their rankings via RRF.

        Args:
            query: The user query to retrieve and rank against.
            chunks: Optional pre-retrieved candidates, included as an additional
                ranked list in the fusion.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to ``top_k`` deduplicated chunks (copies with ``score`` set to the
            fused RRF score), most relevant first.

        Raises:
            RerankerError: If any retriever fails.
        """
        try:
            retrieved: list[list[RetrievedChunk]] = await asyncio.gather(
                *(
                    retriever.retrieve(query, top_k=self._fetch_k)
                    for retriever in self._retrievers
                )
            )
        except Exception as exc:  # noqa: BLE001 - normalize retriever failures
            raise RerankerError(f"Retriever failed during fusion: {exc}") from exc

        ranked_lists: list[list[RetrievedChunk]] = list(retrieved)
        if chunks:
            ranked_lists.append(chunks)

        fused_scores: dict[str, float] = {}
        representatives: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                key = chunk.content
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (
                    self._k + rank
                )
                # Keep the first-seen chunk as the representative for its content.
                representatives.setdefault(key, chunk)

        fused: list[RetrievedChunk] = [
            replace(representatives[key], score=round(score, 6))
            for key, score in fused_scores.items()
        ]
        fused.sort(key=lambda chunk: chunk.score, reverse=True)
        return fused[:top_k]
