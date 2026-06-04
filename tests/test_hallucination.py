"""Tests for rag_guardrails.guardrails.hallucination.

Exercises the real
:class:`~rag_guardrails.guardrails.hallucination.HallucinationDetector` — no
mocking. Covers fully-grounded and fully-ungrounded outputs, the empty-context
and empty-output shortcuts, partial grounding, and how a custom
``hallucination_threshold`` shifts the verdict boundaries.

Detectors are built with explicit :class:`~rag_guardrails.config.Settings` so
the tests are deterministic regardless of ambient configuration.
"""

from __future__ import annotations

import pytest

from rag_guardrails.config import Settings
from rag_guardrails.guardrails.hallucination import (
    HallucinationDetector,
    HallucinationResult,
)

_CONTEXT = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower and "
    "the Louvre museum.",
]


@pytest.fixture
def detector() -> HallucinationDetector:
    """A detector using the default hallucination threshold (0.7)."""
    return HallucinationDetector(config=Settings())


@pytest.mark.asyncio
async def test_fully_grounded_output(detector: HallucinationDetector) -> None:
    """Every claim supported by context -> grounded verdict, low score."""
    result = await detector.score(
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        _CONTEXT,
    )

    assert isinstance(result, HallucinationResult)
    assert result.verdict == "grounded"
    assert result.score == 0.0
    assert result.ungrounded_claims == []


@pytest.mark.asyncio
async def test_fully_ungrounded_output(detector: HallucinationDetector) -> None:
    """No claim supported by context -> hallucinated verdict, high score."""
    result = await detector.score(
        "Quantum gluons bind quarks together. Photosynthesis converts "
        "sunlight into glucose.",
        _CONTEXT,
    )

    assert result.verdict == "hallucinated"
    assert result.score == 1.0
    assert result.grounded_claims == []


@pytest.mark.asyncio
async def test_empty_context_scores_fully_ungrounded(
    detector: HallucinationDetector,
) -> None:
    """With no context, a contentful answer cannot be supported -> score 1.0."""
    result = await detector.score("Paris is the capital of France.", [])

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_empty_output_scores_grounded(
    detector: HallucinationDetector,
) -> None:
    """An empty (contentless) output is treated as fully grounded -> score 0.0."""
    result = await detector.score("", _CONTEXT)

    assert result.score == 0.0
    assert result.verdict == "grounded"


@pytest.mark.asyncio
async def test_partial_grounding(detector: HallucinationDetector) -> None:
    """One grounded and one ungrounded claim -> partial verdict at score 0.5."""
    result = await detector.score(
        "Paris is the capital of France. Photosynthesis converts sunlight "
        "into glucose.",
        _CONTEXT,
    )

    assert result.score == 0.5
    assert result.verdict == "partial"
    assert len(result.grounded_claims) == 1
    assert len(result.ungrounded_claims) == 1


@pytest.mark.asyncio
async def test_custom_threshold_shifts_verdict_boundary() -> None:
    """The same score yields different verdicts under different thresholds.

    A score of 0.5 is:
      - 'hallucinated' when threshold <= 0.5 (0.5 >= threshold), and
      - 'partial' under the default threshold of 0.7 (0.35 <= 0.5 < 0.7).
    """
    output = (
        "Paris is the capital of France. Photosynthesis converts sunlight "
        "into glucose."
    )

    strict = HallucinationDetector(config=Settings(hallucination_threshold=0.4))
    lenient = HallucinationDetector(config=Settings(hallucination_threshold=0.7))

    strict_result = await strict.score(output, _CONTEXT)
    lenient_result = await lenient.score(output, _CONTEXT)

    # Same underlying score...
    assert strict_result.score == lenient_result.score == 0.5
    # ...but the threshold moves the verdict boundary.
    assert strict_result.verdict == "hallucinated"
    assert lenient_result.verdict == "partial"
