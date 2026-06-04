# Contributing

Thanks for your interest in rag-guardrails-starter.

## Development setup

```bash
pip install -e ".[dev]"
```

## Project layout

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for the intended layout and
module responsibilities. A few conventions to keep in mind:

- Prefer adding code under the existing packages rather than inventing new
  top-level packages.
- Keep guard logic in `rag_guardrails/guardrails/`; do not embed safety checks
  inside `pipeline.py` beyond orchestration calls.
- New LLM backends belong in `rag_guardrails/providers/` and must implement the
  base provider contract in `providers/base.py`.
- Domain-specific demos (e.g. healthcare) live in `examples/`, not in the core
  library.

## Tooling

This project uses:

- **black** — formatting
- **ruff** — linting and import sorting
- **mypy** — static type checking
- **pytest** (+ **pytest-asyncio**) — testing

Run them before opening a pull request:

```bash
black .
ruff check .
mypy rag_guardrails
pytest tests/
```

The test suite must stay green at **65+ passing tests**. The tests exercise the
real guards, detectors, and pipeline (the LLM provider is mocked, so no API key
is required to run them). If you change guard behavior, update or add the tests
that pin it.

## Suggested implementation order

1. `pyproject.toml`, `config.py`, `providers/base.py` + `openai_provider.py`
2. `rag/retriever.py`, `rag/pipeline.py` (minimal RAG without reranker)
3. `guardrails/input_guard.py`, `output_guard.py`, `emergency_detector.py`
4. `memory/multi_turn.py`, `guardrails/hallucination.py`, `rag/reranker.py`
5. Additional providers, then `examples/` and `tests/` alongside `docs/`

## Pull requests

- Keep changes focused and scoped to the relevant package.
- Add or update tests under `tests/` mirroring the module you change.
- Make sure formatting, linting, types, and tests pass.
