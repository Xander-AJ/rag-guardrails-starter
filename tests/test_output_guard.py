"""Tests for rag_guardrails.guardrails.output_guard.

Exercises the real :class:`~rag_guardrails.guardrails.output_guard.OutputGuard`
pipeline end to end — no mocking of the guard internals. Covers the clean
grounded path, toxicity blocking, the no-context confidence shortcut, low
grounding warnings, PII redaction, medical-disclaimer injection (and its
de-duplication), and the ``output_guard_enabled=False`` bypass.

Each guard is built with an explicit :class:`~rag_guardrails.config.Settings`
instance so the tests are deterministic regardless of ambient environment
variables / ``.env`` files.
"""

from __future__ import annotations

import pytest

from rag_guardrails.config import Settings
from rag_guardrails.guardrails.output_guard import OutputGuard, OutputGuardResult


@pytest.fixture
def guard() -> OutputGuard:
    """A default output guard with the output guard explicitly enabled."""
    return OutputGuard(config=Settings(output_guard_enabled=True))


@pytest.fixture
def context_chunks() -> list[str]:
    """Sample retrieved context about seasonal flu."""
    return [
        "Seasonal influenza is a viral infection. Common symptoms include "
        "fever, cough, sore throat, and fatigue.",
        "Rest and fluids help recovery from the flu. Antiviral medication may "
        "be prescribed in some cases.",
    ]


@pytest.mark.asyncio
async def test_clean_grounded_output_passes(
    guard: OutputGuard, context_chunks: list[str]
) -> None:
    """An answer supported by the context is safe with positive confidence."""
    result = await guard.check(
        "The flu causes fever, cough, and sore throat. Rest and fluids help "
        "recovery.",
        "What are the symptoms of the flu?",
        context_chunks,
    )

    assert isinstance(result, OutputGuardResult)
    assert result.is_safe is True
    assert result.blocked_reason is None
    assert result.confidence_score > 0
    assert result.sanitized_output


@pytest.mark.asyncio
async def test_toxic_output_blocked_and_emptied(guard: OutputGuard) -> None:
    """Toxic output is blocked: unsafe with an empty sanitized output."""
    result = await guard.check("You should harm yourself", "anything", [])

    assert result.is_safe is False
    assert result.blocked_reason is not None
    assert result.sanitized_output == ""
    assert result.confidence_score == 0.0


@pytest.mark.asyncio
async def test_toxic_output_hurt_yourself(guard: OutputGuard) -> None:
    """'You should hurt yourself' is blocked just like the 'harm' variant."""
    result = await guard.check("You should hurt yourself", "anything", [])

    assert result.is_safe is False
    assert result.sanitized_output == ""


@pytest.mark.asyncio
async def test_no_context_yields_full_confidence(guard: OutputGuard) -> None:
    """With no context, confidence is 1.0 and no grounding warning is added."""
    result = await guard.check(
        "Here is a general answer with several content words.",
        "A general question",
        [],
    )

    assert result.is_safe is True
    assert result.confidence_score == 1.0
    assert not any("grounding" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_ungrounded_output_warns(
    guard: OutputGuard, context_chunks: list[str]
) -> None:
    """Output unsupported by the context gets a low-grounding warning."""
    result = await guard.check(
        "Quantum chromodynamics describes gluon interactions in particle "
        "physics.",
        "What are the symptoms of the flu?",
        context_chunks,
    )

    assert result.is_safe is True
    assert result.confidence_score < guard._config.hallucination_threshold
    assert any("grounding" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_pii_in_output_redacted_and_warned(
    guard: OutputGuard, context_chunks: list[str]
) -> None:
    """Leaked email/phone are redacted and a PII warning is added."""
    result = await guard.check(
        "You can reach the clinic at admin@clinic.com or 0712345678.",
        "How do I contact the clinic?",
        context_chunks,
    )

    assert result.is_safe is True
    assert "admin@clinic.com" not in result.sanitized_output
    assert "0712345678" not in result.sanitized_output
    assert "[EMAIL]" in result.sanitized_output
    assert "[PHONE]" in result.sanitized_output
    assert any("pii" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_medical_query_appends_disclaimer(
    guard: OutputGuard, context_chunks: list[str]
) -> None:
    """A medical-domain query triggers a disclaimer appended to the output."""
    result = await guard.check(
        "Rest and fluids help.",
        "What treatment helps my symptoms and illness?",
        context_chunks,
    )

    assert result.is_safe is True
    assert "disclaimer" in result.sanitized_output.lower()
    assert "healthcare provider" in result.sanitized_output.lower()


@pytest.mark.asyncio
async def test_disclaimer_not_duplicated(
    guard: OutputGuard, context_chunks: list[str]
) -> None:
    """An output that already contains a disclaimer is not given another."""
    output = "Rest helps. Disclaimer: consult a doctor."
    result = await guard.check(
        output,
        "What medical treatment helps my symptoms?",
        context_chunks,
    )

    assert result.sanitized_output.lower().count("disclaimer") == 1
    assert result.sanitized_output == output


@pytest.mark.asyncio
async def test_disabled_guard_passes_toxic_unchanged(
    context_chunks: list[str],
) -> None:
    """With output_guard_enabled=False, toxic content passes through as-is."""
    guard = OutputGuard(config=Settings(output_guard_enabled=False))

    toxic = "You should harm yourself"
    result = await guard.check(toxic, "anything", context_chunks)

    assert result.is_safe is True
    assert result.blocked_reason is None
    assert result.sanitized_output == toxic
    assert result.confidence_score == 1.0
    assert result.warnings == []
