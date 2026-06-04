"""Advanced example: a healthcare-tuned guarded RAG assistant.

This is the showcase example. It reflects patterns from real health-AI work
(Tabibu Health): a pipeline tuned for a medical setting, where being wrong is
worse than being silent. It demonstrates:

  - A healthcare configuration: emergency detection on, a strict hallucination
    threshold (0.5 instead of the 0.7 default), and medical disclaimers via the
    output guard.
  - Conversation memory carrying context across a follow-up question.
  - A critical-emergency input that short-circuits to crisis guidance *before*
    any retrieval or LLM call.
  - A locally-relevant question about Kenyan medicine availability.

For each query it prints the question, the answer (or block reason), the sources
used, the hallucination verdict, and whether conversation memory was in play.

Requirements:
    - Set ``OPENAI_API_KEY`` in your environment (or a ``.env`` file).
    - Install the package: ``pip install -e .``

Run:
    python examples/healthcare_domain.py
"""

from __future__ import annotations

import asyncio

from rag_guardrails import (
    ConversationMemory,
    InMemoryRetriever,
    OpenAIProvider,
    RAGPipeline,
    get_config,
)
from rag_guardrails.config import Settings
from rag_guardrails.rag.retriever import RetrievedChunk

# Realistic malaria knowledge — a high-burden condition in Kenya.
CHUNKS = [
    "Malaria is caused by Plasmodium parasites transmitted through the bite of "
    "infected female Anopheles mosquitoes.",
    "Common malaria symptoms include fever, chills, headache, muscle aches, and "
    "fatigue, usually appearing 10 to 15 days after the infective bite.",
    "Artemisinin-based combination therapy (ACT) is the recommended first-line "
    "treatment for uncomplicated malaria.",
    "In Kenya, ACT medicines such as artemether-lumefantrine (Coartem) are "
    "available at public hospitals and licensed pharmacies, often subsidized.",
    "Severe malaria with difficulty breathing or convulsions is a medical "
    "emergency requiring immediate hospital care.",
    "Insecticide-treated mosquito nets are a key malaria prevention measure "
    "recommended across Kenya.",
]


def build_pipeline() -> tuple[RAGPipeline, ConversationMemory]:
    """Construct a healthcare-tuned pipeline and return it with its memory."""
    base = get_config()
    config = Settings(
        openai_api_key=base.openai_api_key,
        emergency_detection_enabled=True,
        output_guard_enabled=True,  # medical disclaimers are injected here
        hallucination_threshold=0.5,  # stricter grounding for clinical content
    )
    memory = ConversationMemory(config=config)
    retriever = InMemoryRetriever(
        [RetrievedChunk(content=text, source=f"kb-{i}", score=0.0)
         for i, text in enumerate(CHUNKS, start=1)]
    )
    pipeline = RAGPipeline(
        config=config,
        provider=OpenAIProvider(config=config),
        retriever=retriever,
        memory=memory,
    )
    return pipeline, memory


async def ask(pipeline: RAGPipeline, memory: ConversationMemory, question: str) -> None:
    """Run one query and print the full healthcare-relevant breakdown."""
    had_history = len(memory) > 0
    response = await pipeline.query(question)
    memory_used = had_history and not response.was_blocked

    print(f"Q: {question}")
    if response.was_blocked:
        print(f"   BLOCKED: {response.block_reason}")
    print(f"   Answer: {response.answer}")
    print(f"   Sources: {[s.source for s in response.sources] or 'none'}")
    print(f"   Hallucination verdict: {response.hallucination_result.verdict}")
    print(f"   Memory used: {memory_used}\n")


async def main() -> None:
    pipeline, memory = build_pipeline()

    # (1) A normal health question — grounded retrieval + generation.
    await ask(pipeline, memory, "What are the common symptoms of malaria?")

    # (2) A follow-up that only makes sense with memory ("it" = malaria).
    await ask(pipeline, memory, "What is the recommended treatment for it?")

    # (3) A critical emergency — short-circuits to crisis guidance, no LLM call.
    await ask(pipeline, memory, "Help, I'm having a seizure and can't breathe!")

    # (4) A locally-relevant question about Kenyan medicine availability.
    await ask(pipeline, memory, "Where can I get Coartem in Kenya?")


if __name__ == "__main__":
    asyncio.run(main())
