# RAG Guardrails Starter — Planned Project Structure

Reference document for implementation. **Do not treat this as “create every file now.”** Use it to guide incremental builds and to keep Claude Code aligned with the intended layout.

## Directory tree

```
rag-guardrails-starter/
│
├── rag_guardrails/               # Core library
│   ├── __init__.py
│   │
│   ├── guardrails/               # The heart of the project
│   │   ├── __init__.py
│   │   ├── input_guard.py        # Input safety filtering
│   │   ├── output_guard.py       # Output validation
│   │   ├── emergency_detector.py # Crisis/emergency routing
│   │   └── hallucination.py      # Hallucination detection hooks
│   │
│   ├── rag/                      # RAG pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Core RAG orchestration
│   │   ├── retriever.py          # Vector store abstraction
│   │   └── reranker.py           # Optional reranking layer
│   │
│   ├── providers/                # LLM provider abstraction
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base provider
│   │   ├── openai_provider.py    # OpenAI (default)
│   │   ├── openrouter_provider.py
│   │   └── langchain_provider.py # LangChain wrapper
│   │
│   ├── memory/                   # Conversation memory
│   │   ├── __init__.py
│   │   └── multi_turn.py         # Multi-turn context management
│   │
│   └── config.py                 # Central configuration
│
├── examples/                     # Runnable examples
│   ├── basic_rag.py
│   ├── with_guardrails.py
│   ├── healthcare_domain.py      # Tabibu-style domain patterns
│   └── custom_provider.py
│
├── tests/
│   ├── test_input_guard.py
│   ├── test_output_guard.py
│   ├── test_emergency_detector.py
│   ├── test_pipeline.py
│   └── test_providers.py
│
├── docs/
│   ├── guardrails.md
│   ├── providers.md
│   └── examples.md
│
├── pyproject.toml                # Modern Python packaging
├── README.md
├── CONTRIBUTING.md
└── .env.example
```

## Module responsibilities

| Path | Role |
|------|------|
| `rag_guardrails/guardrails/` | Pre/post-LLM safety: input filters, output checks, emergency routing, hallucination hooks |
| `rag_guardrails/rag/` | Retrieve → (optional) rerank → generate; orchestrated in `pipeline.py` |
| `rag_guardrails/providers/` | Pluggable LLM backends behind a shared `base.py` interface |
| `rag_guardrails/memory/` | Multi-turn conversation state and context window management |
| `rag_guardrails/config.py` | Env vars, defaults, and shared settings for guards + RAG + providers |
| `examples/` | Small, runnable scripts demonstrating each major capability |
| `tests/` | Unit tests mirroring guardrails, pipeline, and provider modules |
| `docs/` | Human-facing guides (not API reference generated from code) |

## Request flow (target architecture)

```
User query
    → input_guard
    → emergency_detector (may short-circuit to crisis handling)
    → retriever (+ optional reranker)
    → provider (LLM)
    → hallucination hooks (on retrieved context + draft answer)
    → output_guard
    → response (+ memory update via multi_turn)
```

## Implementation order (suggested)

1. `pyproject.toml`, `config.py`, `providers/base.py` + `openai_provider.py`
2. `rag/retriever.py`, `rag/pipeline.py` (minimal RAG without reranker)
3. `guardrails/input_guard.py`, `output_guard.py`, `emergency_detector.py`
4. `memory/multi_turn.py`, `guardrails/hallucination.py`, `rag/reranker.py`
5. Additional providers, then `examples/` and `tests/` in parallel with `docs/`

## Notes for agents (Claude Code, Cursor, etc.)

- Prefer adding code under the paths above rather than inventing new top-level packages.
- Keep guard logic in `guardrails/`; do not embed safety checks inside `pipeline.py` beyond orchestration calls.
- New LLM backends belong in `providers/` and must implement the base provider contract.
- Domain-specific demos (e.g. healthcare) live in `examples/`, not in the core library.
- When scaffolding, create only the files needed for the current task; this document is the full target shape, not a bulk file-generation checklist.
