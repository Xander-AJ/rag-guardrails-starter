"""Tests for rag_guardrails.guardrails.emergency_detector.

Exercises the real
:class:`~rag_guardrails.guardrails.emergency_detector.EmergencyDetector` end to
end — no mocking of internals. Covers clean input, critical medical and
mental-health detection (with the correct Kenyan emergency contacts), negation
handling, the soft-keyword corroboration rule, urgency escalation, and the
``emergency_detection_enabled=False`` bypass.

Each detector is built with an explicit
:class:`~rag_guardrails.config.Settings` instance so the tests are deterministic
regardless of ambient environment variables / ``.env`` files.
"""

from __future__ import annotations

import pytest

from rag_guardrails.config import Settings
from rag_guardrails.guardrails.emergency_detector import (
    EmergencyDetector,
    EmergencyResult,
)


@pytest.fixture
def detector() -> EmergencyDetector:
    """A detector with emergency detection explicitly enabled."""
    return EmergencyDetector(config=Settings(emergency_detection_enabled=True))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "What time does the pharmacy open?",
        "Tell me about the history of the Roman Empire.",
        "How do I bake sourdough bread?",
        "",
    ],
)
async def test_clean_input_is_not_emergency(
    detector: EmergencyDetector, text: str
) -> None:
    """Benign (or empty) input is not an emergency and has severity 'none'."""
    result = await detector.detect(text)

    assert isinstance(result, EmergencyResult)
    assert result.is_emergency is False
    assert result.severity == "none"
    assert result.category == "none"
    assert result.emergency_contacts == []


@pytest.mark.asyncio
async def test_critical_medical_emergency(detector: EmergencyDetector) -> None:
    """'I am not breathing' is a critical medical emergency with KE contacts."""
    result = await detector.detect("I am not breathing")

    assert result.is_emergency is True
    assert result.severity == "critical"
    assert result.category == "medical"
    # Kenyan general emergency numbers.
    assert "999" in result.emergency_contacts
    assert "112" in result.emergency_contacts


@pytest.mark.asyncio
async def test_critical_mental_health_emergency(
    detector: EmergencyDetector,
) -> None:
    """'I want to kill myself' is critical mental_health with Befrienders KE."""
    result = await detector.detect("I want to kill myself")

    assert result.is_emergency is True
    assert result.severity == "critical"
    assert result.category == "mental_health"
    assert any(
        "Befrienders" in contact for contact in result.emergency_contacts
    ), f"Befrienders Kenya missing from contacts: {result.emergency_contacts!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "I am not suicidal",
        "I have no chest pain",
        "I don't want to die",
    ],
)
async def test_negation_suppresses_emergency(
    detector: EmergencyDetector, text: str
) -> None:
    """A negated crisis phrase does not trip the detector."""
    result = await detector.detect(text)

    assert result.is_emergency is False
    assert result.severity == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "I have a headache",
        "I feel a bit nauseous",
        "I am feeling worried",
    ],
)
async def test_lone_soft_keyword_does_not_escalate(
    detector: EmergencyDetector, text: str
) -> None:
    """A single soft keyword stays at 'none' or 'low' — never high/critical."""
    result = await detector.detect(text)

    assert result.severity in ("none", "low")
    assert result.severity not in ("high", "critical")


@pytest.mark.asyncio
async def test_urgency_cue_escalates_soft_keyword(
    detector: EmergencyDetector,
) -> None:
    """An urgency cue alongside a soft keyword escalates it to a low emergency."""
    baseline = await detector.detect("I have a headache")
    escalated = await detector.detect("I have a headache right now")

    # Without urgency the lone soft keyword does not trip the detector...
    assert baseline.is_emergency is False
    assert baseline.severity == "none"
    # ...but the urgency cue corroborates it into a low-severity emergency.
    assert escalated.is_emergency is True
    assert escalated.severity == "low"
    assert escalated.category == "medical"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "I am not breathing",
        "I want to kill myself",
        "I have a headache right now",
    ],
)
async def test_disabled_detector_never_flags(text: str) -> None:
    """With emergency_detection_enabled=False, nothing is ever an emergency."""
    detector = EmergencyDetector(
        config=Settings(emergency_detection_enabled=False)
    )

    result = await detector.detect(text)

    assert result.is_emergency is False
    assert result.severity == "none"
    assert result.category == "none"
    assert result.emergency_contacts == []
