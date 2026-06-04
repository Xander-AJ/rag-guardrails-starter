"""Tests for rag_guardrails.memory.multi_turn.

Exercises the real :class:`~rag_guardrails.memory.multi_turn.ConversationMemory`
— recording turns, rendering them as chat messages, rolling-window truncation,
clearing, and length reporting. Memory is built with explicit
:class:`~rag_guardrails.config.Settings` so the window size is deterministic.
"""

from __future__ import annotations

import pytest

from rag_guardrails.config import Settings
from rag_guardrails.memory.multi_turn import ConversationMemory, ConversationTurn


@pytest.fixture
def memory() -> ConversationMemory:
    """A conversation memory with a generous window (10)."""
    return ConversationMemory(config=Settings(memory_window=10))


def test_add_turn_and_get_turns(memory: ConversationMemory) -> None:
    """Recorded turns are returned in order as ConversationTurn objects."""
    memory.add_turn("hello", "hi there")
    memory.add_turn("how are you?", "doing well")

    turns = memory.get_turns()
    assert len(turns) == 2
    assert all(isinstance(t, ConversationTurn) for t in turns)
    assert turns[0] == ConversationTurn(user="hello", assistant="hi there")
    assert turns[1].user == "how are you?"
    assert turns[1].assistant == "doing well"


def test_get_messages_format(memory: ConversationMemory) -> None:
    """get_messages renders each turn as a user then assistant message."""
    memory.add_turn("question one", "answer one")
    memory.add_turn("question two", "answer two")

    messages = memory.get_messages()

    assert messages == [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]


def test_window_truncation_drops_oldest() -> None:
    """Adding beyond memory_window keeps only the most recent turns."""
    memory = ConversationMemory(config=Settings(memory_window=2))

    memory.add_turn("turn 1", "a1")
    memory.add_turn("turn 2", "a2")
    memory.add_turn("turn 3", "a3")

    turns = memory.get_turns()
    assert len(turns) == 2
    # Oldest ("turn 1") was dropped; the two most recent remain, in order.
    assert [t.user for t in turns] == ["turn 2", "turn 3"]


def test_clear_empties_memory(memory: ConversationMemory) -> None:
    """clear() removes all stored turns."""
    memory.add_turn("hello", "hi")
    memory.add_turn("again", "yes")
    assert len(memory) == 2

    memory.clear()

    assert len(memory) == 0
    assert memory.get_turns() == []
    assert memory.get_messages() == []


def test_len_returns_turn_count(memory: ConversationMemory) -> None:
    """len() reflects the number of retained turns."""
    assert len(memory) == 0
    memory.add_turn("a", "1")
    assert len(memory) == 1
    memory.add_turn("b", "2")
    assert len(memory) == 2
