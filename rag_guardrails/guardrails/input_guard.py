"""Input safety filtering.

First stage in the request flow. :class:`InputGuard` inspects the raw user
query before any retrieval or generation happens and decides whether it is safe
to proceed, returning a structured :class:`InputGuardResult`.

The check pipeline runs, in order:

1. **Emergency detection** — delegates to
   :class:`~rag_guardrails.guardrails.emergency_detector.EmergencyDetector`.
   A *critical* emergency short-circuits immediately so the caller can route to
   crisis handling without hitting an LLM. Non-critical emergencies are
   attached to the result but do not block.
2. **Prompt-injection detection** — jailbreak patterns, role overrides, and
   instruction-injection attempts.
3. **Content safety** — PII detection (email, phone, national ID) and
   harmful-intent classification.
4. **Sanitization** — whitespace/control-character cleanup and PII redaction,
   producing a safe version of the input.

When ``input_guard_enabled`` is ``False`` in configuration, the guard passes
the input through unchanged with ``is_safe=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

from rag_guardrails.config import Settings, get_config
from rag_guardrails.guardrails._text import detect_pii, redact_pii
from rag_guardrails.guardrails.emergency_detector import (
    EmergencyDetector,
    EmergencyResult,
)


@dataclass
class InputGuardResult:
    """Outcome of running the input guard over a user query.

    Attributes:
        is_safe: Whether the input is safe to proceed to the normal RAG flow.
            ``False`` when blocked or when a critical emergency short-circuits.
        blocked_reason: Human-readable reason the input was blocked, or
            ``None`` when ``is_safe`` is ``True``.
        emergency: The :class:`EmergencyResult` if an emergency was detected
            (any severity), otherwise ``None``. Always populated for critical
            short-circuits so callers can route to crisis handling.
        sanitized_text: A cleaned, PII-redacted version of the input. Provided
            even when the input is blocked.
    """

    is_safe: bool
    blocked_reason: str | None
    emergency: EmergencyResult | None
    sanitized_text: str


# --- Prompt-injection patterns ---------------------------------------------
# Each entry is (compiled_pattern, human-readable reason).
_INJECTION_PATTERNS: list[tuple[Pattern[str], str]] = [
    (
        re.compile(
            r"\b(ignore|disregard|forget)\b[^.]{0,40}?\b"
            r"(previous|prior|above|earlier|all)\b[^.]{0,40}?\b"
            r"(instructions?|prompts?|rules?|context|messages?)\b",
            re.IGNORECASE,
        ),
        "Instruction-override attempt detected.",
    ),
    (
        re.compile(
            r"\b(you are now|from now on,? you|act as|pretend (to be|you are)|"
            r"role[\s-]?play as)\b",
            re.IGNORECASE,
        ),
        "Role-override attempt detected.",
    ),
    (
        re.compile(
            r"\b(reveal|show|print|repeat|expose|leak)\b[^.]{0,30}?\b"
            r"(system|developer)?\s?(prompt|instructions?|message)\b",
            re.IGNORECASE,
        ),
        "Attempt to extract system prompt detected.",
    ),
    (
        re.compile(
            r"\b(jailbreak|developer mode|do anything now|\bdan\b|"
            r"unrestricted mode)\b",
            re.IGNORECASE,
        ),
        "Jailbreak attempt detected.",
    ),
    (
        re.compile(
            r"\b(bypass|disable|turn off|circumvent|ignore)\b[^.]{0,30}?\b"
            r"(filter|guardrail|safety|restrictions?|moderation|rules?)\b",
            re.IGNORECASE,
        ),
        "Attempt to bypass safety controls detected.",
    ),
    (
        re.compile(
            r"(<\|?(im_start|im_end|system|assistant|user)\|?>|\[/?INST\]|"
            r"###\s*(system|instruction)|^\s*system\s*:)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "Injected role/instruction markup detected.",
    ),
]

# --- Harmful-intent patterns -----------------------------------------------
_HARMFUL_PATTERNS: list[Pattern[str]] = [
    re.compile(
        r"\bhow (to|do i|can i|could i)\b[^.]{0,40}?\b(make|build|create|"
        r"construct|assemble)\b[^.]{0,40}?\b(bombs?|explosives?|grenades?|"
        r"firearms?|guns?|weapons?|molotovs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow (to|do i|can i|could i)\b[^.]{0,40}?\b(kill|murder|poison|"
        r"harm|hurt|attack)\b[^.]{0,30}?\b(someone|somebody|a person|people|"
        r"him|her|them|my)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow (to|do i|can i|could i)\b[^.]{0,40}?\b(synthesi[sz]e|make|"
        r"cook|produce|manufacture)\b[^.]{0,40}?\b(meth|methamphetamine|"
        r"cocaine|heroin|fentanyl|ricin|sarin|nerve agents?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow (to|do i|can i|could i)\b[^.]{0,40}?\b(hack|break)\b[^.]{0,20}?"
        r"\binto\b[^.]{0,30}?\b(account|system|network|computer|phone|email|"
        r"server|database)\b",
        re.IGNORECASE,
    ),
]

# Control characters to strip during sanitization (keep tab/newline/carriage
# return; drop other C0/C1 control characters).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")


class InputGuard:
    """Filters and sanitizes incoming user queries before retrieval/generation.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. Falls back to
            the shared singleton from
            :func:`~rag_guardrails.config.get_config`.
        emergency_detector: Optional
            :class:`~rag_guardrails.guardrails.emergency_detector.EmergencyDetector`.
            One is constructed internally (sharing ``config``) when not given.
    """

    def __init__(
        self,
        config: Settings | None = None,
        emergency_detector: EmergencyDetector | None = None,
    ) -> None:
        self._config: Settings = config or get_config()
        self._emergency_detector: EmergencyDetector = (
            emergency_detector or EmergencyDetector(config=self._config)
        )

    async def check(self, text: str) -> InputGuardResult:
        """Run the full input-guard pipeline over ``text``.

        Args:
            text: The raw user query.

        Returns:
            An :class:`InputGuardResult` describing whether the input is safe,
            why it was blocked (if so), any detected emergency, and the
            sanitized text.
        """
        if not self._config.input_guard_enabled:
            return InputGuardResult(
                is_safe=True,
                blocked_reason=None,
                emergency=None,
                sanitized_text=text,
            )

        # (1) Emergency detection — critical short-circuits immediately.
        emergency = await self._emergency_detector.detect(text)
        if emergency.is_emergency and emergency.severity == "critical":
            return InputGuardResult(
                is_safe=False,
                blocked_reason=(
                    "Critical emergency detected; routing to crisis handling."
                ),
                emergency=emergency,
                sanitized_text=self._sanitize(text),
            )
        emergency_payload = emergency if emergency.is_emergency else None

        # (2) Prompt-injection detection.
        injection_reason = self._detect_injection(text)
        if injection_reason is not None:
            return InputGuardResult(
                is_safe=False,
                blocked_reason=injection_reason,
                emergency=emergency_payload,
                sanitized_text=self._sanitize(text),
            )

        # (3) Content safety — harmful intent blocks; PII is redacted in (4).
        harmful_reason = self._detect_harmful_intent(text)
        if harmful_reason is not None:
            return InputGuardResult(
                is_safe=False,
                blocked_reason=harmful_reason,
                emergency=emergency_payload,
                sanitized_text=self._sanitize(text),
            )

        # (4) Sanitization (whitespace/control cleanup + PII redaction).
        return InputGuardResult(
            is_safe=True,
            blocked_reason=None,
            emergency=emergency_payload,
            sanitized_text=self._sanitize(text),
        )

    @staticmethod
    def _detect_injection(text: str) -> str | None:
        """Return a reason string if a prompt-injection pattern matches."""
        for pattern, reason in _INJECTION_PATTERNS:
            if pattern.search(text):
                return reason
        return None

    @staticmethod
    def _detect_harmful_intent(text: str) -> str | None:
        """Return a reason string if harmful intent is detected."""
        for pattern in _HARMFUL_PATTERNS:
            if pattern.search(text):
                return "Potentially harmful intent detected."
        return None

    @staticmethod
    def detect_pii(text: str) -> list[str]:
        """Return the kinds of PII (``email``/``phone``/``national_id``) present.

        Exposed as a helper so callers can inspect PII without sanitizing.
        """
        return list(detect_pii(text).keys())

    @staticmethod
    def _sanitize(text: str) -> str:
        """Clean ``text`` and redact detected PII.

        Strips control characters, collapses runs of horizontal whitespace,
        redacts email/phone/national-ID spans with placeholders, and trims the
        result.
        """
        cleaned = _CONTROL_CHARS.sub("", text)
        cleaned = redact_pii(cleaned)
        cleaned = _WHITESPACE_RUN.sub(" ", cleaned)
        return cleaned.strip()
