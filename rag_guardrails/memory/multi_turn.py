"""Multi-turn context management.

Tracks conversation history across turns and prepares it for inclusion in
prompts. :class:`ConversationMemory` keeps a rolling window of the most recent
exchanges (sized by ``memory_window`` in configuration) and can render them as
chat messages for a provider call.

The memory stores complete user/assistant exchanges as
:class:`ConversationTurn` records and trims to the configured window as new
turns are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_guardrails.config import Settings, get_config


@dataclass
class ConversationTurn:
    """A single user/assistant exchange.

    Attributes:
        user: The user's message for this turn.
        assistant: The assistant's response for this turn.
    """

    user: str
    assistant: str


class ConversationMemory:
    """Rolling-window store of conversation turns.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. Falls back to
            the shared singleton from
            :func:`~rag_guardrails.config.get_config`. ``memory_window``
            controls how many of the most recent turns are retained.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()
        self._turns: list[ConversationTurn] = []

    @property
    def window(self) -> int:
        """Maximum number of turns retained (from ``config.memory_window``)."""
        return self._config.memory_window

    def add_turn(self, user: str, assistant: str) -> None:
        """Record a user/assistant exchange, trimming to the window.

        Args:
            user: The user's message.
            assistant: The assistant's response.
        """
        self._turns.append(ConversationTurn(user=user, assistant=assistant))
        if self.window >= 0 and len(self._turns) > self.window:
            self._turns = self._turns[-self.window :] if self.window else []

    def get_turns(self) -> list[ConversationTurn]:
        """Return the retained turns, oldest first."""
        return list(self._turns)

    def get_messages(self) -> list[dict[str, Any]]:
        """Render retained turns as chat messages for a provider call.

        Returns:
            A flat list of ``{"role": ..., "content": ...}`` messages, with
            each turn expanded into a user message followed by an assistant
            message, oldest first.
        """
        messages: list[dict[str, Any]] = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def clear(self) -> None:
        """Remove all stored turns."""
        self._turns.clear()

    def __len__(self) -> int:
        """Return the number of retained turns."""
        return len(self._turns)
