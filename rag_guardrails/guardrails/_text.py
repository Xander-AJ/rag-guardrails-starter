"""Private shared text utilities for the guardrails package.

Centralizes lexical helpers used by more than one guard so they do not drift
out of sync:

- **PII** detection and redaction (:data:`PII_PATTERNS`, :func:`detect_pii`,
  :func:`redact_pii`) — shared by ``input_guard`` (sanitization) and
  ``output_guard`` (leak detection).
- **Tokenization** (:data:`STOPWORDS`, :func:`content_tokens`) — shared by
  ``output_guard`` (grounding confidence) and ``hallucination`` (claim
  scoring).

This module is private to the package (underscore-prefixed); guards import the
names they need rather than re-declaring local copies.
"""

from __future__ import annotations

import re

# --- PII patterns (detection + redaction) ----------------------------------
# Insertion order matters for redaction: email and phone are redacted before
# national-ID so digit runs inside a phone number are not re-matched as an ID.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"(?<!\w)(?:\+?254|0)\s?7\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\w)"
        r"|(?<!\w)\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\w)"
    ),
    "national_id": re.compile(r"(?<!\d)\d{7,8}(?!\d)"),
}

# Placeholder substituted for each PII kind during redaction.
_PII_PLACEHOLDERS: dict[str, str] = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "national_id": "[ID]",
}


def detect_pii(text: str) -> dict[str, list[str]]:
    """Return the PII found in ``text``, keyed by kind.

    Args:
        text: The text to scan.

    Returns:
        A mapping of PII kind (``"email"``/``"phone"``/``"national_id"``) to the
        list of matched substrings. Kinds with no matches are omitted, so an
        empty dict means no PII was found.
    """
    found: dict[str, list[str]] = {}
    for kind, pattern in PII_PATTERNS.items():
        matches = [match.group(0) for match in pattern.finditer(text)]
        if matches:
            found[kind] = matches
    return found


def redact_pii(text: str) -> str:
    """Replace any PII in ``text`` with placeholder tokens.

    Email, phone, and national-ID spans are replaced with ``[EMAIL]``,
    ``[PHONE]``, and ``[ID]`` respectively (in that order).

    Args:
        text: The text to redact.

    Returns:
        The text with PII spans replaced by placeholders.
    """
    for kind, pattern in PII_PATTERNS.items():
        text = pattern.sub(_PII_PLACEHOLDERS[kind], text)
    return text


# --- Tokenization -----------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")

# Stopwords excluded from content-token overlap so common filler/acknowledgement
# words do not dominate grounding or hallucination scores.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "this", "that",
        "these", "those", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "with", "as", "by", "at", "from",
        "it", "its", "you", "your", "they", "their", "we", "our", "i", "he",
        "she", "his", "her", "can", "may", "might", "will", "would", "should",
        "could", "have", "has", "had", "do", "does", "did", "not", "no",
        "yes", "there", "here", "what", "which", "who", "when", "where",
        "how", "why", "about", "into", "than", "also", "more", "most", "some",
        "any", "all", "such", "so", "very", "just", "only", "out", "up",
        # Conversational fillers / acknowledgements: low factual content, so an
        # acknowledgement-only reply is not treated as a fabricated claim.
        "okay", "sure", "yeah", "thanks", "thank", "hello", "hey", "please",
        "well", "actually", "really", "sorry", "great",
    }
)


def content_tokens(text: str) -> list[str]:
    """Tokenize ``text`` into lowercased content words.

    Drops stopwords and tokens of two characters or fewer. Returns a list
    (preserving repeats and order); callers wanting set semantics should wrap
    the result in :class:`set`.

    Args:
        text: The text to tokenize.

    Returns:
        The list of content tokens.
    """
    return [
        token
        for token in _WORD.findall(text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
