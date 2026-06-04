"""Output validation.

Final safety stage in the request flow. :class:`OutputGuard` inspects the
model's draft answer before it is returned to the user and produces a
structured :class:`OutputGuardResult`.

The check pipeline runs, in order:

1. **Toxicity screening** — flags harmful, discriminatory, or dangerous output
   and blocks it.
2. **Context grounding** — estimates how well the output is supported by the
   retrieved context chunks (a hallucination proxy) and produces a
   ``confidence_score``; low grounding adds a warning.
3. **PII-leak detection** — redacts any email/phone/national-ID that leaked
   into the output and adds a warning.
4. **Disclaimer injection** — appends a medical/legal/financial disclaimer when
   the content is in one of those domains and no disclaimer is already present.

When ``output_guard_enabled`` is ``False`` in configuration, the guard passes
the output through unchanged with ``is_safe=True`` and ``confidence_score=1.0``.

Grounding and toxicity use lightweight, dependency-free lexical heuristics so
the guard works without extra model calls; both are structured so a model-based
scorer can be swapped in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern

from rag_guardrails.config import Settings, get_config
from rag_guardrails.guardrails._text import content_tokens, detect_pii, redact_pii


@dataclass
class OutputGuardResult:
    """Outcome of running the output guard over a draft answer.

    Attributes:
        is_safe: Whether the output is safe to return to the user. ``False``
            when toxicity screening blocks it.
        blocked_reason: Human-readable reason the output was blocked, or
            ``None`` when ``is_safe`` is ``True``.
        sanitized_output: The cleaned output to return — PII-redacted and with
            any disclaimer appended. Empty string when blocked.
        confidence_score: Grounding confidence in ``[0.0, 1.0]`` estimating how
            well the output is supported by the provided context. ``1.0`` when
            no context was supplied or when the guard is disabled.
        warnings: Non-blocking advisories (e.g. low grounding, redacted PII).
    """

    is_safe: bool
    blocked_reason: str | None
    sanitized_output: str
    confidence_score: float
    warnings: list[str] = field(default_factory=list)


# --- Toxicity patterns ------------------------------------------------------
# Compact, dependency-free lexicon. Real deployments should back this with a
# dedicated moderation model; the structure here is meant to be swappable.
_TOXICITY_PATTERNS: list[Pattern[str]] = [
    re.compile(
        r"\b(you should|go|why don't you)\s+(kill yourself|end your life|"
        r"harm yourself|hurt yourself)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\ball\s+\w+\s+(people|men|women|muslims|christians|jews|blacks|"
        r"whites|asians|immigrants)\s+(are|should be)\s+"
        r"(inferior|stupid|subhuman|killed|eliminated|deported)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(here'?s how to|steps to|instructions to)\b[^.]{0,40}?\b"
        r"(make|build|synthesi[sz]e)\b[^.]{0,40}?\b(bomb|explosive|nerve agent|"
        r"ricin|sarin|chemical weapon)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(incite|encourage|promote)\b[^.]{0,30}?\b(violence|genocide|"
        r"terrorism)\b",
        re.IGNORECASE,
    ),
]

# --- Domain detection for disclaimer injection -----------------------------
_DOMAIN_PATTERNS: dict[str, Pattern[str]] = {
    "medical": re.compile(
        r"\b(symptom|symptoms|diagnos\w+|treatment|medication|dosage|disease|"
        r"prescription|therapy|patient|medical|drug|side effect|illness|"
        r"clinical)\b",
        re.IGNORECASE,
    ),
    "legal": re.compile(
        r"\b(lawsuit|legal|attorney|lawyer|contract|liability|court|statute|"
        r"litigation|defendant|plaintiff|sue|legally)\b",
        re.IGNORECASE,
    ),
    "financial": re.compile(
        r"\b(invest\w*|stocks?|shares?|portfolio|tax\w*|financial|returns?|"
        r"dividend|interest rate|mortgage|loan|securities)\b",
        re.IGNORECASE,
    ),
}

_DISCLAIMERS: dict[str, str] = {
    "medical": (
        "Disclaimer: This information is for general educational purposes only "
        "and is not a substitute for professional medical advice. Please "
        "consult a qualified healthcare provider."
    ),
    "legal": (
        "Disclaimer: This information is for general informational purposes "
        "only and does not constitute legal advice. Please consult a licensed "
        "attorney for your specific situation."
    ),
    "financial": (
        "Disclaimer: This information is for general educational purposes only "
        "and is not financial advice. Please consult a licensed financial "
        "advisor before making decisions."
    ),
}

_DISCLAIMER_MARKER = re.compile(r"\bdisclaimer\b", re.IGNORECASE)


class OutputGuard:
    """Validates and sanitizes generated answers before they reach the user.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. Falls back to
            the shared singleton from
            :func:`~rag_guardrails.config.get_config`. The
            ``hallucination_threshold`` is used as the minimum grounding score
            below which a low-grounding warning is emitted.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

    async def check(
        self, output: str, query: str, context: list[str]
    ) -> OutputGuardResult:
        """Run the full output-guard pipeline over a draft answer.

        Args:
            output: The model's draft answer.
            query: The originating user query (used for domain detection).
            context: The retrieved context chunks the answer should be grounded
                in. May be empty for non-RAG flows.

        Returns:
            An :class:`OutputGuardResult` describing safety, the sanitized
            output, the grounding confidence, and any non-blocking warnings.
        """
        if not self._config.output_guard_enabled:
            return OutputGuardResult(
                is_safe=True,
                blocked_reason=None,
                sanitized_output=output,
                confidence_score=1.0,
                warnings=[],
            )

        warnings: list[str] = []

        # (1) Toxicity screening — blocks immediately.
        toxic_reason = self._screen_toxicity(output)
        if toxic_reason is not None:
            return OutputGuardResult(
                is_safe=False,
                blocked_reason=toxic_reason,
                sanitized_output="",
                confidence_score=0.0,
                warnings=warnings,
            )

        # (2) Context grounding check.
        confidence_score = self._grounding_score(output, context)
        if context and confidence_score < self._config.hallucination_threshold:
            warnings.append(
                "Low grounding: the response may not be fully supported by the "
                f"retrieved context (score {confidence_score:.2f} < threshold "
                f"{self._config.hallucination_threshold:.2f})."
            )

        sanitized = output

        # (3) PII-leak detection — redact and warn (non-blocking).
        sanitized, leaked = self._redact_pii(sanitized)
        if leaked:
            warnings.append(
                "Redacted potential PII leaked into the output: "
                f"{', '.join(leaked)}."
            )

        # (4) Disclaimer injection for sensitive domains.
        sanitized = self._inject_disclaimers(sanitized, query)

        return OutputGuardResult(
            is_safe=True,
            blocked_reason=None,
            sanitized_output=sanitized,
            confidence_score=confidence_score,
            warnings=warnings,
        )

    @staticmethod
    def _screen_toxicity(output: str) -> str | None:
        """Return a reason string if the output matches a toxicity pattern."""
        for pattern in _TOXICITY_PATTERNS:
            if pattern.search(output):
                return "Output failed toxicity screening."
        return None

    @staticmethod
    def _grounding_score(output: str, context: list[str]) -> float:
        """Estimate grounding as content-word overlap with the context.

        Returns the fraction of the output's content words (stopwords removed)
        that also appear in the concatenated context. Returns ``1.0`` when
        there is no context to check against or the output has no content
        words.
        """
        if not context:
            return 1.0

        context_tokens = set(content_tokens(" ".join(context)))
        output_tokens = set(content_tokens(output))
        if not output_tokens:
            return 1.0

        grounded = sum(1 for tok in output_tokens if tok in context_tokens)
        return round(grounded / len(output_tokens), 4)

    @staticmethod
    def _redact_pii(text: str) -> tuple[str, list[str]]:
        """Redact PII spans, returning the cleaned text and kinds redacted."""
        leaked = list(detect_pii(text).keys())
        return redact_pii(text), leaked

    @staticmethod
    def _inject_disclaimers(output: str, query: str) -> str:
        """Append disclaimers for any sensitive domain detected.

        A disclaimer is skipped if the output already contains the word
        "disclaimer" (case-insensitive), to avoid duplication.
        """
        if _DISCLAIMER_MARKER.search(output):
            return output

        haystack = f"{query}\n{output}"
        additions: list[str] = []
        for domain, pattern in _DOMAIN_PATTERNS.items():
            if pattern.search(haystack):
                additions.append(_DISCLAIMERS[domain])

        if not additions:
            return output

        return output.rstrip() + "\n\n" + "\n\n".join(additions)
