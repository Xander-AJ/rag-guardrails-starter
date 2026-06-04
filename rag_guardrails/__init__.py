"""rag-guardrails-starter.

A starter library for building Retrieval-Augmented Generation (RAG) pipelines
with first-class safety guardrails and pluggable LLM providers.

The package is organized into four areas:

- ``guardrails``: pre/post-LLM safety (input filtering, output validation,
  emergency routing, hallucination hooks).
- ``rag``: the retrieve -> (optional) rerank -> generate pipeline.
- ``providers``: pluggable LLM backends behind a shared base interface.
- ``memory``: multi-turn conversation state and context management.

See ``config`` for shared settings and ``PROJECT_STRUCTURE.md`` for the
intended architecture and request flow.
"""

__version__ = "0.1.0"

from .config import get_config
from .guardrails.emergency_detector import EmergencyDetector
from .guardrails.hallucination import HallucinationDetector
from .guardrails.input_guard import InputGuard
from .guardrails.output_guard import OutputGuard
from .memory.multi_turn import ConversationMemory
from .providers.anthropic_provider import AnthropicProvider
from .providers.cohere_provider import CohereProvider
from .providers.gemini_provider import GeminiProvider
from .providers.langchain_provider import LangChainProvider
from .providers.mistral_provider import MistralProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.openrouter_provider import OpenRouterProvider
from .rag.pipeline import PipelineResponse, RAGPipeline
from .rag.reranker import (
    BaseReranker,
    LexicalReranker,
    ReciprocalRankFusion,
)
from .rag.retriever import InMemoryRetriever, QdrantRetriever

__all__ = [
    "__version__",
    "RAGPipeline",
    "PipelineResponse",
    "InputGuard",
    "OutputGuard",
    "EmergencyDetector",
    "HallucinationDetector",
    "OpenAIProvider",
    "OpenRouterProvider",
    "LangChainProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "GeminiProvider",
    "CohereProvider",
    "MistralProvider",
    "InMemoryRetriever",
    "QdrantRetriever",
    "BaseReranker",
    "LexicalReranker",
    "ReciprocalRankFusion",
    "ConversationMemory",
    "get_config",
]
