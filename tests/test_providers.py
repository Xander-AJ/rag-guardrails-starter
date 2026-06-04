"""Tests for rag_guardrails.providers.

Exercises the three concrete providers — OpenAIProvider, OpenRouterProvider, and
LangChainProvider — with no real API calls. The OpenAI-compatible clients are
replaced with :class:`unittest.mock.MagicMock` via :func:`unittest.mock.patch`,
and their ``chat.completions.create`` is an :class:`~unittest.mock.AsyncMock`.
LangChain is driven through an injected fake ``BaseChatModel``.

Covered behavior:

- Missing API key raises :class:`ProviderError` at construction.
- ``complete()`` maps the backend response onto :class:`ProviderResponse`.
- ``stream()`` yields successive (non-empty) string chunks.
- Backend ``APIError`` / ``LangChainException`` is wrapped in
  :class:`ProviderError`.
- ``LangChainProvider`` accepts a pre-built ``llm`` directly (the escape hatch).
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, AIMessageChunk

from rag_guardrails.config import Settings
from rag_guardrails.providers.anthropic_provider import AnthropicProvider
from rag_guardrails.providers.base import ProviderError, ProviderResponse
from rag_guardrails.providers.cohere_provider import CohereProvider
from rag_guardrails.providers.gemini_provider import GeminiProvider
from rag_guardrails.providers.langchain_provider import LangChainProvider
from rag_guardrails.providers.mistral_provider import MistralProvider
from rag_guardrails.providers.ollama_provider import OllamaProvider
from rag_guardrails.providers.openai_provider import OpenAIProvider
from rag_guardrails.providers.openrouter_provider import OpenRouterProvider


# --- OpenAI-compatible fakes ------------------------------------------------
def fake_chat_response(content: str = "Hello from mock") -> object:
    """An object shaped like an OpenAI chat-completion response."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=5)
    response = types.SimpleNamespace(
        choices=[choice], model="gpt-4o-mini", usage=usage
    )
    response.model_dump = lambda: {"model": "gpt-4o-mini", "object": "chat"}
    return response


async def fake_chat_stream():
    """An async iterator of OpenAI-style streaming chunks (with an empty one)."""
    for piece in ["Hel", "lo", "", " world"]:
        delta = types.SimpleNamespace(content=piece)
        choice = types.SimpleNamespace(delta=delta)
        yield types.SimpleNamespace(choices=[choice])


def openai_api_error() -> openai.APIError:
    """A minimally-constructed openai.APIError for side_effect raising."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APIError("boom", request=request, body=None)


# ===========================================================================
# OpenAIProvider
# ===========================================================================
def test_openai_missing_key_raises() -> None:
    """Constructing without an API key raises ProviderError."""
    with pytest.raises(ProviderError):
        OpenAIProvider(config=Settings(openai_api_key=None))


@pytest.mark.asyncio
async def test_openai_complete_maps_fields() -> None:
    """complete() maps the backend response onto ProviderResponse fields."""
    with patch(
        "rag_guardrails.providers.openai_provider.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.chat.completions.create = AsyncMock(
            return_value=fake_chat_response("Mapped content")
        )
        provider = OpenAIProvider(config=Settings(openai_api_key="sk-test"))

        result = await provider.complete([{"role": "user", "content": "hi"}])

    assert isinstance(result, ProviderResponse)
    assert result.content == "Mapped content"
    assert result.model == "gpt-4o-mini"
    assert result.input_tokens == 11
    assert result.output_tokens == 5
    assert result.raw_response["object"] == "chat"


@pytest.mark.asyncio
async def test_openai_stream_yields_chunks() -> None:
    """stream() yields the non-empty string deltas from the backend."""
    with patch(
        "rag_guardrails.providers.openai_provider.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.chat.completions.create = AsyncMock(return_value=fake_chat_stream())
        provider = OpenAIProvider(config=Settings(openai_api_key="sk-test"))

        chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Hel", "lo", " world"]
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.asyncio
async def test_openai_api_error_wrapped() -> None:
    """An openai.APIError is wrapped in ProviderError."""
    with patch(
        "rag_guardrails.providers.openai_provider.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.chat.completions.create = AsyncMock(side_effect=openai_api_error())
        provider = OpenAIProvider(config=Settings(openai_api_key="sk-test"))

        with pytest.raises(ProviderError):
            await provider.complete([{"role": "user", "content": "hi"}])


# ===========================================================================
# OpenRouterProvider
# ===========================================================================
def test_openrouter_missing_key_raises() -> None:
    """Constructing without an OpenRouter API key raises ProviderError."""
    with pytest.raises(ProviderError):
        OpenRouterProvider(config=Settings(openrouter_api_key=None))


@pytest.mark.asyncio
async def test_openrouter_complete_maps_fields() -> None:
    """complete() maps the backend response onto ProviderResponse fields."""
    with patch(
        "rag_guardrails.providers.openrouter_provider.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.chat.completions.create = AsyncMock(
            return_value=fake_chat_response("Routed content")
        )
        provider = OpenRouterProvider(
            config=Settings(openrouter_api_key="sk-or-test")
        )

        result = await provider.complete([{"role": "user", "content": "hi"}])

    assert isinstance(result, ProviderResponse)
    assert result.content == "Routed content"
    assert result.input_tokens == 11
    assert result.output_tokens == 5


@pytest.mark.asyncio
async def test_openrouter_stream_yields_chunks() -> None:
    """stream() yields the non-empty string deltas from the backend."""
    with patch(
        "rag_guardrails.providers.openrouter_provider.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.chat.completions.create = AsyncMock(return_value=fake_chat_stream())
        provider = OpenRouterProvider(
            config=Settings(openrouter_api_key="sk-or-test")
        )

        chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Hel", "lo", " world"]


@pytest.mark.asyncio
async def test_openrouter_api_error_wrapped() -> None:
    """An openai.APIError is wrapped in ProviderError."""
    with patch(
        "rag_guardrails.providers.openrouter_provider.AsyncOpenAI"
    ) as mock_client_cls:
        client = mock_client_cls.return_value
        client.chat.completions.create = AsyncMock(side_effect=openai_api_error())
        provider = OpenRouterProvider(
            config=Settings(openrouter_api_key="sk-or-test")
        )

        with pytest.raises(ProviderError):
            await provider.complete([{"role": "user", "content": "hi"}])


# ===========================================================================
# LangChainProvider
# ===========================================================================
class FakeChatModel:
    """A minimal stand-in for a LangChain BaseChatModel."""

    model_name = "fake/model-1"

    def __init__(self, content: str = "LangChain answer") -> None:
        self._content = content

    async def ainvoke(self, messages: object, **kwargs: object) -> AIMessage:
        message = AIMessage(content=self._content)
        message.usage_metadata = {"input_tokens": 7, "output_tokens": 3}
        message.response_metadata = {"model_name": "fake/model-1"}
        return message

    async def astream(self, messages: object, **kwargs: object):
        for piece in ["Lang", "Chain", "", " stream"]:
            yield AIMessageChunk(content=piece)


class BoomChatModel(FakeChatModel):
    """A chat model whose ainvoke raises a LangChainException."""

    async def ainvoke(self, messages: object, **kwargs: object) -> AIMessage:
        raise LangChainException("backend exploded")


def test_langchain_missing_key_and_no_llm_raises() -> None:
    """Without an injected llm and no OpenAI key, construction raises."""
    with pytest.raises(ProviderError):
        LangChainProvider(config=Settings(openai_api_key=None))


def test_langchain_accepts_injected_llm() -> None:
    """An injected llm is used directly (no API key required)."""
    provider = LangChainProvider(
        config=Settings(openai_api_key=None), llm=FakeChatModel()
    )
    assert provider.get_model_name() == "fake/model-1"


@pytest.mark.asyncio
async def test_langchain_complete_maps_fields() -> None:
    """complete() maps the LangChain response and usage onto ProviderResponse."""
    provider = LangChainProvider(
        config=Settings(openai_api_key=None),
        llm=FakeChatModel("Mapped LC content"),
    )

    result = await provider.complete([{"role": "user", "content": "hi"}])

    assert isinstance(result, ProviderResponse)
    assert result.content == "Mapped LC content"
    assert result.model == "fake/model-1"
    assert result.input_tokens == 7
    assert result.output_tokens == 3


@pytest.mark.asyncio
async def test_langchain_stream_yields_chunks() -> None:
    """stream() yields the non-empty string chunks from the LangChain model."""
    provider = LangChainProvider(
        config=Settings(openai_api_key=None), llm=FakeChatModel()
    )

    chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

    assert chunks == ["Lang", "Chain", " stream"]
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.asyncio
async def test_langchain_exception_wrapped() -> None:
    """A LangChainException raised by the model is wrapped in ProviderError."""
    provider = LangChainProvider(
        config=Settings(openai_api_key=None), llm=BoomChatModel()
    )

    with pytest.raises(ProviderError):
        await provider.complete([{"role": "user", "content": "hi"}])


_MESSAGES = [{"role": "user", "content": "hi"}]


def _ns(**kwargs: object) -> types.SimpleNamespace:
    """Shorthand for building a namespace stand-in for an SDK object."""
    return types.SimpleNamespace(**kwargs)


async def _aiter(items: list[object]):
    """Wrap a list as an async iterator."""
    for item in items:
        yield item


# ===========================================================================
# AnthropicProvider
# ===========================================================================
def test_anthropic_missing_key_raises() -> None:
    """Constructing without an Anthropic API key raises ProviderError."""
    with pytest.raises(ProviderError):
        AnthropicProvider(config=Settings(anthropic_api_key=None))


@pytest.mark.asyncio
async def test_anthropic_complete_maps_fields() -> None:
    """complete() maps the Anthropic message onto ProviderResponse."""
    response = _ns(
        content=[_ns(text="Claude content")],
        model="claude-3-5-haiku-20241022",
        usage=_ns(input_tokens=10, output_tokens=4),
    )
    response.model_dump = lambda: {"id": "msg"}
    with patch(
        "rag_guardrails.providers.anthropic_provider.AsyncAnthropic"
    ) as mock_cls:
        client = mock_cls.return_value
        client.messages.create = AsyncMock(return_value=response)
        provider = AnthropicProvider(
            config=Settings(anthropic_api_key="sk-ant")
        )
        result = await provider.complete(_MESSAGES)

    assert result.content == "Claude content"
    assert result.model == "claude-3-5-haiku-20241022"
    assert result.input_tokens == 10
    assert result.output_tokens == 4


@pytest.mark.asyncio
async def test_anthropic_stream_yields_chunks() -> None:
    """stream() yields the text deltas from the Anthropic text stream."""

    class FakeStreamCtx:
        async def __aenter__(self) -> FakeStreamCtx:
            self.text_stream = _aiter(["Cla", "ude", ""])
            return self

        async def __aexit__(self, *args: object) -> bool:
            return False

    with patch(
        "rag_guardrails.providers.anthropic_provider.AsyncAnthropic"
    ) as mock_cls:
        client = mock_cls.return_value
        client.messages.stream = MagicMock(return_value=FakeStreamCtx())
        provider = AnthropicProvider(
            config=Settings(anthropic_api_key="sk-ant")
        )
        chunks = [c async for c in provider.stream(_MESSAGES)]

    assert chunks == ["Cla", "ude"]


# ===========================================================================
# OllamaProvider (no API key)
# ===========================================================================
@pytest.mark.asyncio
async def test_ollama_complete_maps_fields() -> None:
    """complete() maps the Ollama chat response onto ProviderResponse."""
    response = _ns(
        message=_ns(content="Ollama content"),
        model="llama3.2",
        prompt_eval_count=8,
        eval_count=3,
    )
    response.model_dump = lambda: {"model": "llama3.2"}
    with patch(
        "rag_guardrails.providers.ollama_provider.AsyncClient"
    ) as mock_cls:
        client = mock_cls.return_value
        client.chat = AsyncMock(return_value=response)
        provider = OllamaProvider(config=Settings())
        result = await provider.complete(_MESSAGES)

    assert result.content == "Ollama content"
    assert result.model == "llama3.2"
    assert result.input_tokens == 8
    assert result.output_tokens == 3


@pytest.mark.asyncio
async def test_ollama_stream_yields_chunks() -> None:
    """stream() yields the content deltas from the Ollama stream."""
    parts = [_ns(message=_ns(content=t)) for t in ["Oll", "ama"]]
    with patch(
        "rag_guardrails.providers.ollama_provider.AsyncClient"
    ) as mock_cls:
        client = mock_cls.return_value
        client.chat = AsyncMock(return_value=_aiter(parts))
        provider = OllamaProvider(config=Settings())
        chunks = [c async for c in provider.stream(_MESSAGES)]

    assert chunks == ["Oll", "ama"]


@pytest.mark.asyncio
async def test_ollama_connection_error_wrapped() -> None:
    """A transport error is wrapped in ProviderError with a docs pointer."""
    with patch(
        "rag_guardrails.providers.ollama_provider.AsyncClient"
    ) as mock_cls:
        client = mock_cls.return_value
        client.chat = AsyncMock(side_effect=ConnectionError("refused"))
        provider = OllamaProvider(config=Settings())
        with pytest.raises(ProviderError, match="ollama"):
            await provider.complete(_MESSAGES)


# ===========================================================================
# GeminiProvider
# ===========================================================================
def test_gemini_missing_key_raises() -> None:
    """Constructing without a Gemini API key raises ProviderError."""
    with pytest.raises(ProviderError):
        GeminiProvider(config=Settings(gemini_api_key=None))


@pytest.mark.asyncio
async def test_gemini_complete_maps_fields() -> None:
    """complete() maps the Gemini response onto ProviderResponse."""
    response = _ns(
        text="Gemini content",
        usage_metadata=_ns(prompt_token_count=9, candidates_token_count=5),
    )
    with patch("rag_guardrails.providers.gemini_provider.genai") as mock_genai:
        model = mock_genai.GenerativeModel.return_value
        model.generate_content_async = AsyncMock(return_value=response)
        provider = GeminiProvider(config=Settings(gemini_api_key="sk-gem"))
        result = await provider.complete(_MESSAGES)

    assert result.content == "Gemini content"
    assert result.input_tokens == 9
    assert result.output_tokens == 5


@pytest.mark.asyncio
async def test_gemini_stream_yields_chunks() -> None:
    """stream() yields the text from each Gemini chunk."""
    chunks_in = [_ns(text=t) for t in ["Gem", "ini"]]
    with patch("rag_guardrails.providers.gemini_provider.genai") as mock_genai:
        model = mock_genai.GenerativeModel.return_value
        model.generate_content_async = AsyncMock(return_value=_aiter(chunks_in))
        provider = GeminiProvider(config=Settings(gemini_api_key="sk-gem"))
        chunks = [c async for c in provider.stream(_MESSAGES)]

    assert chunks == ["Gem", "ini"]


# ===========================================================================
# CohereProvider
# ===========================================================================
def test_cohere_missing_key_raises() -> None:
    """Constructing without a Cohere API key raises ProviderError."""
    with pytest.raises(ProviderError):
        CohereProvider(config=Settings(cohere_api_key=None))


@pytest.mark.asyncio
async def test_cohere_complete_maps_fields() -> None:
    """complete() maps the Cohere v2 chat response onto ProviderResponse."""
    response = _ns(
        message=_ns(content=[_ns(text="Cohere content")]),
        usage=_ns(tokens=_ns(input_tokens=12, output_tokens=6)),
    )
    response.dict = lambda: {"id": "x"}
    with patch("rag_guardrails.providers.cohere_provider.cohere") as mock_cohere:
        client = mock_cohere.AsyncClientV2.return_value
        client.chat = AsyncMock(return_value=response)
        provider = CohereProvider(config=Settings(cohere_api_key="sk-co"))
        result = await provider.complete(_MESSAGES)

    assert result.content == "Cohere content"
    assert result.input_tokens == 12
    assert result.output_tokens == 6


@pytest.mark.asyncio
async def test_cohere_stream_yields_chunks() -> None:
    """stream() yields text from content-delta events."""
    events = [
        _ns(
            type="content-delta",
            delta=_ns(message=_ns(content=_ns(text=t))),
        )
        for t in ["Co", "here"]
    ]
    with patch("rag_guardrails.providers.cohere_provider.cohere") as mock_cohere:
        client = mock_cohere.AsyncClientV2.return_value
        client.chat_stream = MagicMock(return_value=_aiter(events))
        provider = CohereProvider(config=Settings(cohere_api_key="sk-co"))
        chunks = [c async for c in provider.stream(_MESSAGES)]

    assert chunks == ["Co", "here"]


# ===========================================================================
# MistralProvider
# ===========================================================================
def test_mistral_missing_key_raises() -> None:
    """Constructing without a Mistral API key raises ProviderError."""
    with pytest.raises(ProviderError):
        MistralProvider(config=Settings(mistral_api_key=None))


@pytest.mark.asyncio
async def test_mistral_complete_maps_fields() -> None:
    """complete() maps the Mistral response onto ProviderResponse."""
    response = _ns(
        choices=[_ns(message=_ns(content="Mistral content"))],
        usage=_ns(prompt_tokens=11, completion_tokens=4),
        model="mistral-small-latest",
    )
    response.model_dump = lambda: {"id": "x"}
    with patch("rag_guardrails.providers.mistral_provider.Mistral") as mock_cls:
        client = mock_cls.return_value
        client.chat.complete_async = AsyncMock(return_value=response)
        provider = MistralProvider(config=Settings(mistral_api_key="sk-mi"))
        result = await provider.complete(_MESSAGES)

    assert result.content == "Mistral content"
    assert result.model == "mistral-small-latest"
    assert result.input_tokens == 11
    assert result.output_tokens == 4


@pytest.mark.asyncio
async def test_mistral_stream_yields_chunks() -> None:
    """stream() yields the delta content from each Mistral event."""
    events = [
        _ns(data=_ns(choices=[_ns(delta=_ns(content=t))]))
        for t in ["Mis", "tral"]
    ]
    with patch("rag_guardrails.providers.mistral_provider.Mistral") as mock_cls:
        client = mock_cls.return_value
        client.chat.stream_async = AsyncMock(return_value=_aiter(events))
        provider = MistralProvider(config=Settings(mistral_api_key="sk-mi"))
        chunks = [c async for c in provider.stream(_MESSAGES)]

    assert chunks == ["Mis", "tral"]
