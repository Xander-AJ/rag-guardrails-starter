"""Memory package: conversation state.

Manages multi-turn conversation context so the pipeline can ground responses
in prior exchanges.

- ``multi_turn``: stores turn history and manages the context window.
"""
