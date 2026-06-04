"""Guardrails package: the safety core of rag-guardrails-starter.

Houses pre- and post-LLM safety components:

- ``input_guard``: filters and sanitizes incoming user queries.
- ``output_guard``: validates generated responses before they reach the user.
- ``emergency_detector``: detects crisis/emergency intent and may short-circuit
  the normal pipeline to a crisis-handling path.
- ``hallucination``: hooks for checking draft answers against retrieved
  context.

Guard logic lives here and is invoked by ``rag.pipeline`` as orchestration;
safety checks should not be embedded directly inside the pipeline.
"""
