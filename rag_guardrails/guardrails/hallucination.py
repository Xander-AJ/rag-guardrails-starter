"""Hallucination detection hooks.

Runs after generation, comparing the draft answer against the retrieved
context to estimate whether the response is grounded in the supplied sources.

:class:`HallucinationDetector` performs sentence-level *claim extraction* — it
splits the output into atomic claims — then scores each claim by token-level
overlap against the context chunks. No external ML model is required; the
detector is intentionally dependency-free and structured so an
entailment/embedding model can be swapped in later.

The aggregate :class:`HallucinationResult.score` runs from ``0.0`` (fully
grounded) to ``1.0`` (fully hallucinated) and is mapped to a verdict using the
``hallucination_threshold`` from configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rag_guardrails.config import Settings, get_config
from rag_guardrails.guardrails._text import content_tokens

Verdict = Literal["grounded", "partial", "hallucinated"]


@dataclass
class HallucinationResult:
    """Outcome of scoring a draft answer against its retrieved context.

    Attributes:
        score: Hallucination score in ``[0.0, 1.0]`` where ``0.0`` means fully
            grounded (every claim supported by the context) and ``1.0`` means
            fully hallucinated (no claim supported).
        grounded_claims: Atomic claims judged to be supported by the context.
        ungrounded_claims: Atomic claims not sufficiently supported.
        verdict: ``"grounded"``, ``"partial"``, or ``"hallucinated"`` derived
            from ``score`` and the configured threshold.
    """

    score: float
    grounded_claims: list[str]
    ungrounded_claims: list[str]
    verdict: Verdict = field(default="grounded")


# A claim is considered grounded when at least this fraction of its content
# tokens also appear in the context. Documented constant rather than a config
# knob to keep per-claim classification stable across deployments.
_CLAIM_GROUNDING_THRESHOLD: float = 0.5

# Sentence/clause boundaries used to break the output into atomic claims.
_CLAIM_SPLIT = re.compile(r"[.!?;\n]+")


class HallucinationDetector:
    """Scores how well a generated answer is grounded in retrieved context.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. Falls back to
            the shared singleton from
            :func:`~rag_guardrails.config.get_config`. The
            ``hallucination_threshold`` determines the verdict boundaries.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

    async def score(
        self, output: str, context: list[str]
    ) -> HallucinationResult:
        """Score ``output`` for grounding against ``context``.

        Splits the output into atomic claims, classifies each as grounded or
        ungrounded by token overlap with the context, and aggregates the result
        into a hallucination score and verdict.

        Args:
            output: The generated answer to evaluate.
            context: The retrieved context chunks the answer should rely on.

        Returns:
            A :class:`HallucinationResult`. An empty/contentless output is
            treated as fully grounded (``score=0.0``). When ``context`` is
            empty, no claim can be supported, so every claim is ungrounded.
        """
        claims = self._extract_claims(output)
        if not claims:
            return HallucinationResult(
                score=0.0,
                grounded_claims=[],
                ungrounded_claims=[],
                verdict=self._verdict(0.0),
            )

        context_tokens = set(content_tokens(" ".join(context)))

        grounded: list[str] = []
        ungrounded: list[str] = []
        for claim in claims:
            if self._is_grounded(claim, context_tokens):
                grounded.append(claim)
            else:
                ungrounded.append(claim)

        total = len(grounded) + len(ungrounded)
        score = round(len(ungrounded) / total, 4) if total else 0.0

        return HallucinationResult(
            score=score,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            verdict=self._verdict(score),
        )

    @staticmethod
    def _extract_claims(output: str) -> list[str]:
        """Split ``output`` into atomic claims with factual content.

        Sentences/clauses are split on terminal punctuation and newlines.
        Fragments with no content tokens (e.g. "Okay.") are dropped.
        """
        claims: list[str] = []
        for fragment in _CLAIM_SPLIT.split(output):
            claim = fragment.strip()
            if claim and content_tokens(claim):
                claims.append(claim)
        return claims

    @staticmethod
    def _is_grounded(claim: str, context_tokens: set[str]) -> bool:
        """Return ``True`` if enough of ``claim``'s tokens appear in context."""
        claim_tokens = set(content_tokens(claim))
        if not claim_tokens:
            return True
        overlap = len(claim_tokens & context_tokens) / len(claim_tokens)
        return overlap >= _CLAIM_GROUNDING_THRESHOLD

    def _verdict(self, score: float) -> Verdict:
        """Map a hallucination ``score`` to a verdict using the threshold.

        ``score < threshold * 0.5`` -> grounded; ``score < threshold`` ->
        partial; otherwise hallucinated.
        """
        threshold = self._config.hallucination_threshold
        if score < threshold * 0.5:
            return "grounded"
        if score < threshold:
            return "partial"
        return "hallucinated"
