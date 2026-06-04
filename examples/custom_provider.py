"""Example: provider-agnostic generation.

The key architectural feature of this library is that every LLM backend
implements the same :class:`~rag_guardrails.providers.base.BaseProvider`
contract (``complete`` / ``stream`` / ``get_model_name``). Retrieval, prompting,
and the guardrails never need to know which backend is behind them — providers
are swappable.

This example builds one prompt from a shared retriever and runs it through five
provider configurations:

  1. OpenAIProvider     — default settings.
  2. OpenRouterProvider — model chosen per call via a ``model=`` kwarg.
  3. LangChainProvider  — wrapping a custom ChatOpenAI (temperature=0.9).
  4. AnthropicProvider  — a different vendor, same contract (Claude).
  5. OllamaProvider     — a local model, no API key required.

Each provider is constructed independently; one missing API key only skips that
provider rather than stopping the demo.

Requirements (set the keys for whichever providers you want to run):
``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``, ``ANTHROPIC_API_KEY``. Ollama needs a
local server (see https://github.com/ollama/ollama). Run:
``python examples/custom_provider.py``
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from langchain_openai import ChatOpenAI

from rag_guardrails import (
    AnthropicProvider,
    InMemoryRetriever,
    LangChainProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from rag_guardrails.providers.base import BaseProvider, ProviderError
from rag_guardrails.rag.retriever import RetrievedChunk

CHUNKS = [
    "The Great Rift Valley runs through Kenya from north to south.",
    "Lake Nakuru is famous for the flamingos that gather on its shores.",
    "Mount Kenya is the highest mountain in Kenya and the second-highest in Africa.",
]

QUERY = "What is Mount Kenya known for?"


def build_messages(sources: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Assemble a shared system+context prompt for every provider."""
    context = "\n".join(f"[{i}] {c.content}" for i, c in enumerate(sources, start=1))
    return [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUERY}"},
    ]


async def main() -> None:
    retriever = InMemoryRetriever(
        [RetrievedChunk(content=text, source=f"geo-{i}", score=0.0)
         for i, text in enumerate(CHUNKS, start=1)]
    )
    messages = build_messages(await retriever.retrieve(QUERY))

    # (label, provider factory, per-call kwargs) — same prompt, five backends.
    configs: list[tuple[str, Callable[[], BaseProvider], dict[str, str]]] = [
        ("OpenAI (default)", OpenAIProvider, {}),
        ("OpenRouter (claude-3-haiku)", OpenRouterProvider,
         {"model": "anthropic/claude-3-haiku"}),
        ("LangChain (ChatOpenAI temp=0.9)",
         lambda: LangChainProvider(llm=ChatOpenAI(temperature=0.9)), {}),
        ("Anthropic (Claude)", AnthropicProvider, {}),
        ("Ollama (local llama3.2)", OllamaProvider, {}),
    ]

    print(f"Query: {QUERY}\n")
    for label, factory, kwargs in configs:
        try:
            provider = factory()
            response = await provider.complete(messages, **kwargs)
        except ProviderError as exc:
            print(f"[{label}] skipped: {exc}\n")
            continue
        print(f"[{label}] (model: {response.model})")
        print(f"  {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
