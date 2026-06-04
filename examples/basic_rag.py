"""Quickstart: the simplest working guarded RAG pipeline.

This example wires an in-memory corpus of facts about registering a business in
Kenya to the default OpenAI provider and asks a single question. It shows how to
build a :class:`~rag_guardrails.RAGPipeline`, run a query, and read the result:
the answer, whether it was blocked (and why), and which sources were used.

Requirements:
    - Set ``OPENAI_API_KEY`` in your environment (or a ``.env`` file).
    - Install the package: ``pip install -e .``

Run:
    python examples/basic_rag.py
"""

from __future__ import annotations

import asyncio

from rag_guardrails import InMemoryRetriever, OpenAIProvider, RAGPipeline
from rag_guardrails.rag.retriever import RetrievedChunk

CHUNKS = [
    "Registering a company in Kenya is done online through the eCitizen BRS portal.",
    "A private limited company in Kenya requires at least one director and one "
    "shareholder.",
    "Name reservation with the Registrar of Companies costs KES 150 and lasts 30 days.",
    "Every Kenyan business must obtain a KRA PIN for tax purposes before it can trade.",
    "A single business permit is issued by the county government where the "
    "business operates.",
]


async def main() -> None:
    retriever = InMemoryRetriever(
        [RetrievedChunk(content=text, source=f"doc-{i}", score=0.0)
         for i, text in enumerate(CHUNKS, start=1)]
    )
    pipeline = RAGPipeline(provider=OpenAIProvider(), retriever=retriever)

    response = await pipeline.query("How do I register a company in Kenya?")

    print(f"Answer:\n{response.answer}\n")
    if response.was_blocked:
        print(f"Blocked: {response.block_reason}")
    else:
        print("Blocked: no")
    print("\nSources used:")
    for source in response.sources:
        print(f"  - [{source.source}] {source.content}")


if __name__ == "__main__":
    asyncio.run(main())
