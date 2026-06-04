"""Central configuration for rag-guardrails-starter.

Defines the type-safe :class:`Settings` model consumed across the guardrails,
RAG, provider, and memory layers. Configuration is loaded once from the process
environment (and an optional ``.env`` file) and exposed as a cached singleton
via :func:`get_config`, so every subsystem reads from one place rather than
reaching into ``os.environ`` directly.

Environment variables are matched to fields case-insensitively (e.g. the
``openai_api_key`` field is populated from ``OPENAI_API_KEY``). See
``.env.example`` for the supported variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal[
    "openai",
    "openrouter",
    "langchain",
    "anthropic",
    "ollama",
    "gemini",
    "cohere",
    "mistral",
]


class Settings(BaseSettings):
    """Type-safe application settings loaded from the environment.

    Values are read from environment variables (case-insensitive) and from a
    ``.env`` file if present. Unknown environment variables are ignored so the
    same ``.env`` can be shared with other tooling.

    Attributes:
        provider: Which LLM backend to use. One of ``"openai"``,
            ``"openrouter"``, ``"langchain"``, ``"anthropic"``, ``"ollama"``,
            ``"gemini"``, ``"cohere"``, or ``"mistral"``.
        model_name: Default model identifier passed to the active provider.
        openai_api_key: API key for the OpenAI provider, if configured.
        openrouter_api_key: API key for the OpenRouter provider, if configured.
        openrouter_base_url: Base URL for the OpenRouter-compatible API.
        anthropic_api_key: API key for the Anthropic provider, if configured.
        ollama_base_url: Base URL of the local Ollama server.
        gemini_api_key: API key for the Google Gemini provider, if configured.
        cohere_api_key: API key for the Cohere provider, if configured.
        mistral_api_key: API key for the Mistral provider, if configured.
        qdrant_url: URL of the Qdrant vector store, if configured.
        qdrant_collection: Name of the Qdrant collection used for retrieval.
        memory_window: Number of recent conversation turns to retain in
            multi-turn memory.
        input_guard_enabled: Whether the input guard runs on incoming queries.
        output_guard_enabled: Whether the output guard runs on draft answers.
        emergency_detection_enabled: Whether the emergency detector runs and
            may short-circuit the pipeline to crisis handling.
        hallucination_threshold: Minimum grounding score (0.0-1.0) a response
            must meet before it is considered sufficiently supported by the
            retrieved context.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Provider selection ---
    provider: Provider = Field(
        default="openai",
        description="Active LLM backend: openai, openrouter, or langchain.",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Default model identifier for the active provider.",
    )

    # --- Provider credentials ---
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the OpenAI provider.",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the OpenRouter provider.",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter-compatible API.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the Anthropic provider.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the Google Gemini provider.",
    )
    cohere_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the Cohere provider.",
    )
    mistral_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the Mistral provider.",
    )

    # --- Vector store (Qdrant) ---
    qdrant_url: str | None = Field(
        default=None,
        description="URL of the Qdrant vector store.",
    )
    qdrant_collection: str = Field(
        default="rag_documents",
        description="Name of the Qdrant collection used for retrieval.",
    )

    # --- Memory ---
    memory_window: int = Field(
        default=10,
        ge=0,
        description="Number of recent conversation turns to retain in memory.",
    )

    # --- Guardrail toggles ---
    input_guard_enabled: bool = Field(
        default=True,
        description="Run the input guard on incoming queries.",
    )
    output_guard_enabled: bool = Field(
        default=True,
        description="Run the output guard on draft answers.",
    )
    emergency_detection_enabled: bool = Field(
        default=True,
        description="Run the emergency detector before retrieval/generation.",
    )
    hallucination_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum grounding score a response must meet.",
    )


@lru_cache(maxsize=1)
def get_config() -> Settings:
    """Return the cached, process-wide :class:`Settings` singleton.

    The first call constructs a :class:`Settings` instance (loading from the
    environment and ``.env``); subsequent calls return the same instance via
    :func:`functools.lru_cache`. To pick up changed environment variables in a
    long-running process or in tests, call ``get_config.cache_clear()`` first.

    Returns:
        The shared :class:`Settings` instance.
    """
    return Settings()
