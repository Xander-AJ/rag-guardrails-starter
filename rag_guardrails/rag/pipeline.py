"""Core RAG orchestration.

Ties the whole system together. :class:`RAGPipeline` implements the target
request flow:

    input_guard
      -> (critical emergency short-circuits to crisis handling)
      -> retriever
      -> provider (LLM)
      -> hallucination hooks (on retrieved context + draft answer)
      -> output_guard
      -> response (+ memory update)

The pipeline holds orchestration logic only: safety decisions live in the
``guardrails`` package, retrieval in ``rag.retriever``, and generation in the
``providers`` package. All collaborators are injectable for testing and default
to sensible production choices.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from rag_guardrails.config import Settings, get_config
from rag_guardrails.guardrails.emergency_detector import EmergencyResult
from rag_guardrails.guardrails.hallucination import (
    HallucinationDetector,
    HallucinationResult,
)
from rag_guardrails.guardrails.input_guard import InputGuard, InputGuardResult
from rag_guardrails.guardrails.output_guard import OutputGuard, OutputGuardResult
from rag_guardrails.memory.multi_turn import ConversationMemory
from rag_guardrails.providers.base import BaseProvider
from rag_guardrails.providers.openai_provider import OpenAIProvider
from rag_guardrails.rag.retriever import (
    BaseRetriever,
    InMemoryRetriever,
    RetrievedChunk,
)

# Canned responses used when a request or response is blocked.
_SAFE_BLOCK_MESSAGE = (
    "I'm sorry, but I can't help with that request."
)
_SAFE_OUTPUT_BLOCK_MESSAGE = (
    "I'm sorry, but I'm unable to provide a response to that."
)

# System prompt defining the assistant role and grounding instructions.
_SYSTEM_PROMPT = (
    "You are a helpful, careful assistant. Answer the user's question using "
    "only the information in the provided context. If the context does not "
    "contain the answer, say you don't have enough information rather than "
    "guessing. Be concise and refer to sources by their bracket number when "
    "relevant."
)


@dataclass
class PipelineResponse:
    """The full result of a pipeline query.

    Attributes:
        answer: The final answer text returned to the user (a safe canned
            message when blocked).
        sources: The retrieved chunks used as context (may be empty).
        input_guard_result: The input guard's verdict for the query.
        output_guard_result: The output guard's verdict for the answer. For a
            request blocked at the input stage this reflects the canned safe
            message (which is itself safe).
        hallucination_result: The grounding/hallucination assessment of the
            answer. Neutral (grounded, empty) when the LLM was not called.
        emergency: Any detected emergency, or ``None``.
        was_blocked: Whether the request or response was blocked.
        block_reason: Why it was blocked, or ``None``.
    """

    answer: str
    sources: list[RetrievedChunk]
    input_guard_result: InputGuardResult
    output_guard_result: OutputGuardResult
    hallucination_result: HallucinationResult
    emergency: EmergencyResult | None
    was_blocked: bool
    block_reason: str | None


class RAGPipeline:
    """Orchestrates guardrails, retrieval, generation, and memory.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`; falls back to
            :func:`~rag_guardrails.config.get_config`.
        provider: Optional LLM provider; defaults to :class:`OpenAIProvider`
            built from ``config``.
        retriever: Optional retriever; defaults to an empty
            :class:`InMemoryRetriever`.
        memory: Optional conversation memory; defaults to a new
            :class:`ConversationMemory` built from ``config``.

    The :class:`InputGuard`, :class:`OutputGuard`, and
    :class:`HallucinationDetector` are always constructed internally from
    ``config``.
    """

    def __init__(
        self,
        config: Settings | None = None,
        provider: BaseProvider | None = None,
        retriever: BaseRetriever | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        # Note: use explicit ``is None`` checks rather than ``or`` — an empty
        # ConversationMemory is falsy (it defines ``__len__``), so ``or`` would
        # silently discard an injected-but-empty memory.
        self._config: Settings = config if config is not None else get_config()
        self._provider: BaseProvider = (
            provider if provider is not None
            else OpenAIProvider(config=self._config)
        )
        self._retriever: BaseRetriever = (
            retriever if retriever is not None else InMemoryRetriever([])
        )
        self._memory: ConversationMemory = (
            memory if memory is not None
            else ConversationMemory(config=self._config)
        )

        self._input_guard = InputGuard(config=self._config)
        self._output_guard = OutputGuard(config=self._config)
        self._hallucinator = HallucinationDetector(config=self._config)

    async def query(
        self, user_input: str, top_k: int = 5
    ) -> PipelineResponse:
        """Run the full guarded RAG pipeline for ``user_input``.

        Args:
            user_input: The raw user query.
            top_k: Number of context chunks to retrieve.

        Returns:
            A :class:`PipelineResponse` capturing the answer and every stage's
            result.
        """
        # (1) Input guard — block or critical-emergency short-circuit.
        input_result = await self._input_guard.check(user_input)
        if not input_result.is_safe:
            return self._input_blocked_response(input_result)

        safe_query = input_result.sanitized_text

        # (2) Retrieve context.
        sources = await self._retriever.retrieve(safe_query, top_k=top_k)
        context = [chunk.content for chunk in sources]

        # (3) Build the prompt + (4) call the provider.
        messages = self._build_messages(safe_query, sources)
        response = await self._provider.complete(messages)
        draft = response.content

        # (5) Hallucination check against retrieved context.
        hallucination_result = await self._hallucinator.score(draft, context)

        # (6) Output guard.
        output_result = await self._output_guard.check(
            draft, safe_query, context
        )

        # (7) Output blocked -> safe message, do not store to memory.
        if not output_result.is_safe:
            return PipelineResponse(
                answer=_SAFE_OUTPUT_BLOCK_MESSAGE,
                sources=sources,
                input_guard_result=input_result,
                output_guard_result=output_result,
                hallucination_result=hallucination_result,
                emergency=input_result.emergency,
                was_blocked=True,
                block_reason=output_result.blocked_reason,
            )

        final_answer = output_result.sanitized_output

        # (8) Record the exchange (sanitized) in memory.
        self._memory.add_turn(safe_query, final_answer)

        # (9) Full response.
        return PipelineResponse(
            answer=final_answer,
            sources=sources,
            input_guard_result=input_result,
            output_guard_result=output_result,
            hallucination_result=hallucination_result,
            emergency=input_result.emergency,
            was_blocked=False,
            block_reason=None,
        )

    async def stream_query(
        self, user_input: str, top_k: int = 5
    ) -> AsyncIterator[str]:
        """Stream a guarded RAG answer for ``user_input``.

        Runs the same input-guard short-circuit as :meth:`query`, then streams
        the provider's output chunks. Output validation is applied *after*
        streaming completes (it cannot retract already-streamed text): if the
        completed answer fails the output guard, a trailing safety notice is
        emitted and the exchange is not stored in memory.

        Args:
            user_input: The raw user query.
            top_k: Number of context chunks to retrieve.

        Yields:
            Successive chunks of the answer (or a single safe message when the
            request is blocked at the input stage).
        """
        # (1) Input guard — block or critical-emergency short-circuit.
        input_result = await self._input_guard.check(user_input)
        if not input_result.is_safe:
            emergency = input_result.emergency
            if emergency is not None and emergency.severity == "critical":
                yield self._emergency_message(emergency)
            else:
                yield _SAFE_BLOCK_MESSAGE
            return

        safe_query = input_result.sanitized_text

        # (2) Retrieve + (3) build prompt.
        sources = await self._retriever.retrieve(safe_query, top_k=top_k)
        context = [chunk.content for chunk in sources]
        messages = self._build_messages(safe_query, sources)

        # (4) Stream the provider output, accumulating for post-hoc checks.
        pieces: list[str] = []
        async for piece in self._provider.stream(messages):
            pieces.append(piece)
            yield piece
        draft = "".join(pieces)

        # (6) Output guard on the completed answer (post-hoc for streaming).
        output_result = await self._output_guard.check(
            draft, safe_query, context
        )
        if not output_result.is_safe:
            yield (
                "\n\n[Notice: part of this response was withheld for safety "
                "reasons.]"
            )
            return

        # (8) Record the exchange in memory.
        self._memory.add_turn(safe_query, output_result.sanitized_output)

    def _build_messages(
        self, query: str, sources: list[RetrievedChunk]
    ) -> list[dict[str, Any]]:
        """Assemble system prompt, history, and the context-laden user turn."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        messages.extend(self._memory.get_messages())
        messages.append(
            {"role": "user", "content": self._format_user_turn(query, sources)}
        )
        return messages

    @staticmethod
    def _format_user_turn(query: str, sources: list[RetrievedChunk]) -> str:
        """Format retrieved context and the question into a user message."""
        if sources:
            blocks = "\n\n".join(
                f"[{i}] (source: {chunk.source})\n{chunk.content}"
                for i, chunk in enumerate(sources, start=1)
            )
            context_section = f"Context:\n{blocks}"
        else:
            context_section = "Context:\n(No relevant context was retrieved.)"
        return f"{context_section}\n\nQuestion: {query}"

    def _input_blocked_response(
        self, input_result: InputGuardResult
    ) -> PipelineResponse:
        """Build the response for a request blocked at the input stage."""
        emergency = input_result.emergency
        if emergency is not None and emergency.severity == "critical":
            answer = self._emergency_message(emergency)
        else:
            answer = _SAFE_BLOCK_MESSAGE

        # The LLM was not called, so the output/hallucination stages did not
        # run; report neutral results describing the canned safe answer.
        neutral_output = OutputGuardResult(
            is_safe=True,
            blocked_reason=None,
            sanitized_output=answer,
            confidence_score=1.0,
            warnings=[],
        )
        neutral_hallucination = HallucinationResult(
            score=0.0,
            grounded_claims=[],
            ungrounded_claims=[],
            verdict="grounded",
        )
        return PipelineResponse(
            answer=answer,
            sources=[],
            input_guard_result=input_result,
            output_guard_result=neutral_output,
            hallucination_result=neutral_hallucination,
            emergency=emergency,
            was_blocked=True,
            block_reason=input_result.blocked_reason,
        )

    @staticmethod
    def _emergency_message(emergency: EmergencyResult) -> str:
        """Compose a crisis response from an emergency result."""
        message = emergency.message
        if emergency.emergency_contacts:
            contacts = ", ".join(emergency.emergency_contacts)
            message = f"{message}\n\nEmergency contacts: {contacts}"
        return message
