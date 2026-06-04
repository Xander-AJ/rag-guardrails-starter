"""Abstract base provider.

Defines the contract every LLM backend must satisfy so the rest of the system
(notably ``rag.pipeline``) can remain provider-agnostic.

This module declares:

- :class:`ProviderResponse`: the shared response shape returned by every
  provider's :meth:`BaseProvider.complete` call.
- :class:`ProviderError`: a common exception type that concrete providers wrap
  backend-specific failures in.
- :class:`BaseProvider`: the abstract interface for chat/text generation,
  covering buffered completion, streaming, and model introspection.

Concrete backends live in sibling modules (e.g. ``openai_provider``) and must
subclass :class:`BaseProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Normalized response returned by a provider's completion call.

    Attributes:
        content: The generated text content.
        model: Identifier of the model that produced the response.
        input_tokens: Number of prompt/input tokens consumed.
        output_tokens: Number of completion/output tokens produced.
        raw_response: The provider's raw response payload, preserved for
            debugging and access to backend-specific fields.
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    """Raised when an LLM provider fails to produce a response.

    Concrete providers wrap backend-specific errors (authentication failures,
    rate limits, transport/API errors) in this exception so callers can handle
    a single, provider-agnostic error type.
    """


class BaseProvider(ABC):
    """Abstract contract for an LLM backend.

    Implementations translate the system's provider-agnostic message format
    into backend-specific calls and map the results onto
    :class:`ProviderResponse`. All generation methods are asynchronous.

    The ``messages`` argument follows the common chat format: a list of
    mappings with at least ``"role"`` and ``"content"`` keys, e.g.
    ``[{"role": "user", "content": "Hello"}]``.
    """

    @abstractmethod
    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single, buffered completion for ``messages``.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Backend-specific generation parameters (e.g.
                ``temperature``, ``max_tokens``).

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the backend fails to produce a response.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion for ``messages`` as text chunks.

        Implementations are async generators; each yielded value is an
        incremental piece of the generated text. Callers consume the result
        with ``async for chunk in provider.stream(...)``.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Backend-specific generation parameters.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the backend fails during streaming.
        """
        raise NotImplementedError
        yield ""  # pragma: no cover - marks this coroutine as an async generator

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the identifier of the model this provider will use."""
        raise NotImplementedError
