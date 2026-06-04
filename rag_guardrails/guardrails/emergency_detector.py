"""Crisis/emergency routing.

Runs early in the request flow (after input filtering) to detect queries that
signal an emergency or crisis situation requiring special handling.

The :class:`EmergencyDetector` classifies an input across four categories
(medical, mental health, physical danger, and general crisis) and assigns a
severity. When an emergency is detected the pipeline can short-circuit the
normal retrieve -> generate path and surface the crisis message and contacts
carried on the returned :class:`EmergencyResult`.

Detection is intentionally lightweight and dependency-free: it combines
word-boundary keyword/phrase matching with a small lexical *scoring* layer that
weights matches by tier, accounts for negation (e.g. "no chest pain", "I am
not suicidal"), and considers urgency cues. This is deliberately more robust
than naive substring matching, while remaining swappable for a model-based
scorer later.

Emergency contacts use Kenyan numbers (999, 911, 112) and Befrienders Kenya
(+254 722 178 177) for mental-health crises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rag_guardrails.config import Settings, get_config

Severity = Literal["none", "low", "high", "critical"]
Category = Literal["none", "medical", "mental_health", "physical_danger", "crisis"]


@dataclass
class EmergencyResult:
    """Outcome of an emergency-detection pass over a piece of text.

    Attributes:
        is_emergency: Whether the text was classified as an emergency
            (``True`` for any severity above ``"none"``).
        severity: Escalation level. ``"low"`` for soft/early-warning keywords,
            ``"high"`` for explicit crisis language, ``"critical"`` for an
            immediate threat to life. ``"none"`` for clean inputs.
        category: The dominant emergency category, or ``"none"``.
        message: A human-facing message appropriate to the category and
            severity. Empty for clean inputs.
        emergency_contacts: Relevant emergency contact strings. Empty for clean
            inputs.
    """

    is_emergency: bool
    severity: Severity
    category: Category
    message: str
    emergency_contacts: list[str] = field(default_factory=list)


# Kenyan emergency contacts.
_GENERAL_EMERGENCY: list[str] = ["999", "911", "112"]
_BEFRIENDERS: str = "Befrienders Kenya: +254 722 178 177"

# Severity ordering used for escalation comparisons.
_SEVERITY_RANK: dict[Severity, int] = {
    "none": 0,
    "low": 1,
    "high": 2,
    "critical": 3,
}

# Words/contractions that, when they immediately precede a match, indicate the
# matched phrase is being negated and should not count as a signal.
_NEGATIONS: frozenset[str] = frozenset(
    {"no", "not", "never", "without", "denies", "deny", "none", "nobody"}
)

# Signal lexicon: category -> tier -> phrases. Multi-word phrases are matched
# with word boundaries so "pain" does not match "painting".
_SIGNALS: dict[Category, dict[Severity, list[str]]] = {
    "medical": {
        "critical": [
            "cardiac arrest",
            "heart attack",
            "not breathing",
            "stopped breathing",
            "can't breathe",
            "cannot breathe",
            "unconscious",
            "unresponsive",
            "overdose",
            "overdosed",
            "choking",
            "severe bleeding",
            "bleeding out",
            "anaphylaxis",
            "anaphylactic",
            "seizure",
        ],
        "high": [
            "chest pain",
            "stroke",
            "collapsed",
            "severe pain",
            "can't move",
            "blood everywhere",
            "vomiting blood",
            "broken bone",
        ],
        "low": [
            "dizzy",
            "nausea",
            "nauseous",
            "fever",
            "headache",
            "injured",
            "wound",
            "bleeding",
            "sick",
        ],
    },
    "mental_health": {
        "critical": [
            "kill myself",
            "killing myself",
            "end my life",
            "ending my life",
            "take my own life",
            "want to die",
            "hang myself",
            "suicide",
            "suicidal",
        ],
        "high": [
            "self harm",
            "self-harm",
            "hurt myself",
            "harm myself",
            "cut myself",
            "no reason to live",
            "can't go on",
            "don't want to live",
        ],
        "low": [
            "depressed",
            "hopeless",
            "worthless",
            "can't cope",
            "overwhelmed",
            "give up",
            "empty inside",
        ],
    },
    "physical_danger": {
        "critical": [
            "being attacked",
            "trying to kill me",
            "going to kill me",
            "he has a gun",
            "she has a gun",
            "being stabbed",
            "shooting",
        ],
        "high": [
            "attacked",
            "assault",
            "being abused",
            "threatened",
            "in danger",
            "trapped",
            "held against",
        ],
        "low": [
            "unsafe",
            "afraid",
            "scared for my",
        ],
    },
    "crisis": {
        "critical": [
            "about to die",
            "going to die",
            "dying",
        ],
        "high": [
            "call an ambulance",
            "call the police",
            "need help now",
            "this is an emergency",
            "life threatening",
        ],
        "low": [
            "urgent",
            "please help",
            "scared",
            "worried",
        ],
    },
}

# Urgency cues that strengthen otherwise-soft signals.
_URGENCY = re.compile(
    r"\b(right now|immediately|asap|hurry|urgent|emergency)\b", re.IGNORECASE
)

# Per-category contact lists for detected emergencies.
_CONTACTS: dict[Category, list[str]] = {
    "medical": list(_GENERAL_EMERGENCY),
    "physical_danger": list(_GENERAL_EMERGENCY),
    "mental_health": [_BEFRIENDERS, "999", "112"],
    "crisis": [*_GENERAL_EMERGENCY, _BEFRIENDERS],
    "none": [],
}


def _compile_signals() -> list[tuple[Category, Severity, str, re.Pattern[str]]]:
    """Compile the signal lexicon into word-boundary regex patterns.

    Returns:
        A list of ``(category, tier, phrase, compiled_pattern)`` tuples.
    """
    compiled: list[tuple[Category, Severity, str, re.Pattern[str]]] = []
    for category, tiers in _SIGNALS.items():
        for tier, phrases in tiers.items():
            for phrase in phrases:
                pattern = re.compile(
                    r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE
                )
                compiled.append((category, tier, phrase, pattern))
    return compiled


_COMPILED_SIGNALS = _compile_signals()


class EmergencyDetector:
    """Detects emergency/crisis intent in user text.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. When omitted,
            the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used. The
            ``emergency_detection_enabled`` flag is honored: when disabled,
            :meth:`detect` always returns a clean (non-emergency) result.
    """

    # Tier weights used when aggregating per-category scores.
    _TIER_WEIGHT: dict[Severity, float] = {"low": 1.0, "high": 5.0, "critical": 10.0}
    # Category preference for tie-breaking (specific categories beat "crisis").
    _CATEGORY_PRIORITY: list[Category] = [
        "medical",
        "mental_health",
        "physical_danger",
        "crisis",
    ]

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

    async def detect(self, text: str) -> EmergencyResult:
        """Classify ``text`` for emergency/crisis content.

        Args:
            text: The user-supplied text to inspect.

        Returns:
            An :class:`EmergencyResult`. For clean inputs (or when detection is
            disabled), ``is_emergency`` is ``False`` and ``severity`` /
            ``category`` are ``"none"``.
        """
        if not self._config.emergency_detection_enabled or not text.strip():
            return self._clean()

        normalized = text.lower()
        matches = self._find_matches(normalized)
        if not matches:
            return self._clean()

        has_urgency = bool(_URGENCY.search(normalized))
        severity, category = self._score(matches, has_urgency)

        if severity == "none":
            return self._clean()

        return EmergencyResult(
            is_emergency=True,
            severity=severity,
            category=category,
            message=self._message(category, severity),
            emergency_contacts=list(_CONTACTS[category]),
        )

    def _find_matches(
        self, normalized: str
    ) -> list[tuple[Category, Severity, str]]:
        """Return non-negated signal matches found in ``normalized`` text."""
        found: list[tuple[Category, Severity, str]] = []
        for category, tier, phrase, pattern in _COMPILED_SIGNALS:
            for match in pattern.finditer(normalized):
                if not self._is_negated(normalized, match.start()):
                    found.append((category, tier, phrase))
        return found

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        """Return ``True`` if a match at ``start`` is preceded by negation.

        Looks at the two words immediately before the match (e.g. "no chest
        pain", "I am not suicidal", "don't want to live").
        """
        preceding = re.findall(r"[a-z']+", text[:start])[-2:]
        return any(
            word in _NEGATIONS or word.endswith("n't") for word in preceding
        )

    def _score(
        self, matches: list[tuple[Category, Severity, str]], has_urgency: bool
    ) -> tuple[Severity, Category]:
        """Reduce raw matches to a single ``(severity, category)`` verdict.

        Critical or high matches escalate directly. Soft (low) matches require
        corroboration — at least two soft signals, or one soft signal alongside
        an urgency cue — so that a lone benign keyword does not trip the
        detector.
        """
        tiers_present = {tier for _, tier, _ in matches}

        if "critical" in tiers_present:
            return "critical", self._dominant_category(matches, "critical")
        if "high" in tiers_present:
            return "high", self._dominant_category(matches, "high")

        low_matches = [m for m in matches if m[1] == "low"]
        # Multi-word soft phrases count for more than single keywords.
        low_score = sum(2.0 if " " in phrase else 1.0 for _, _, phrase in low_matches)
        if low_score >= 2.0 or (low_matches and has_urgency):
            return "low", self._dominant_category(matches, "low")

        return "none", "none"

    def _dominant_category(
        self, matches: list[tuple[Category, Severity, str]], tier: Severity
    ) -> Category:
        """Pick the strongest category among matches at ``tier``."""
        scores: dict[Category, float] = {}
        for category, match_tier, _ in matches:
            if match_tier == tier:
                scores[category] = scores.get(category, 0.0) + self._TIER_WEIGHT[tier]

        best_score = max(scores.values())
        # Tie-break by category priority (specific categories beat "crisis").
        for category in self._CATEGORY_PRIORITY:
            if scores.get(category, 0.0) == best_score:
                return category
        return "crisis"

    @staticmethod
    def _clean() -> EmergencyResult:
        """Return the canonical non-emergency result."""
        return EmergencyResult(
            is_emergency=False,
            severity="none",
            category="none",
            message="",
            emergency_contacts=[],
        )

    @staticmethod
    def _message(category: Category, severity: Severity) -> str:
        """Build a human-facing message for a detected emergency."""
        messages: dict[Category, dict[Severity, str]] = {
            "medical": {
                "critical": (
                    "This may be a life-threatening medical emergency. Call "
                    "emergency services immediately."
                ),
                "high": (
                    "This may be a medical emergency. Please seek medical "
                    "attention urgently or call emergency services."
                ),
                "low": (
                    "These symptoms may need medical attention. If they worsen "
                    "or feel severe, contact a medical professional or "
                    "emergency services."
                ),
            },
            "mental_health": {
                "critical": (
                    "It sounds like you may be thinking about harming yourself "
                    "or ending your life. You are not alone and your life "
                    "matters — please reach out to a crisis line right now."
                ),
                "high": (
                    "It sounds like you're in severe emotional distress. "
                    "Please consider reaching out to a crisis line — you don't "
                    "have to face this alone."
                ),
                "low": (
                    "It sounds like you may be struggling emotionally. If "
                    "things feel overwhelming, please consider talking to "
                    "someone you trust or a support line."
                ),
            },
            "physical_danger": {
                "critical": (
                    "You may be in immediate physical danger. If you can do so "
                    "safely, call emergency services right now."
                ),
                "high": (
                    "It sounds like you may be in danger. Please contact "
                    "emergency services or get to a safe place."
                ),
                "low": (
                    "If you feel unsafe, consider contacting someone you trust "
                    "or emergency services."
                ),
            },
            "crisis": {
                "critical": (
                    "This sounds like an emergency. Please contact emergency "
                    "services immediately."
                ),
                "high": (
                    "This sounds urgent. Please contact emergency services or "
                    "a support line."
                ),
                "low": "If this is urgent, please reach out for help.",
            },
        }
        return messages.get(category, {}).get(severity, "")
