"""Providers package: pluggable LLM backends.

All LLM backends live here behind the shared interface defined in ``base``:

- ``base``: abstract provider contract every backend must implement.
- ``openai_provider``: default backend, talking to the OpenAI API directly.
- ``openrouter_provider``: backend for the OpenRouter aggregator.
- ``langchain_provider``: adapter wrapping a LangChain chat model.

New backends should subclass the base provider rather than being wired into
the pipeline directly.
"""
