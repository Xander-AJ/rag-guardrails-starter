# rag-guardrails-starter

A Python library for building RAG pipelines with input/output guardrails, emergency detection, and hallucination scoring in the critical path.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-65%20passing-brightgreen)

## The problem this solves

Most RAG tutorials show you how to retrieve documents and generate an answer. None of them show you what happens when a user submits a prompt injection, types "I can't breathe" into your medical chatbot, or when the model returns a fluent answer that isn't supported by any of your sources. In production, those cases are not edge cases — they are the reason a RAG feature gets pulled. This library is the infrastructure layer that sits between the user and the model: it filters input, routes emergencies, scores answers against their context, and validates output before any of it reaches the user.

## Features

**Safety layer**
- **Input guard** — detects prompt injection, jailbreak attempts, harmful intent, and redacts PII before the query reaches the model.
- **Output guard** — toxicity screening, PII-leak redaction, and domain disclaimer injection (medical/legal/financial) on the draft answer.
- **Emergency detection** — classifies medical, mental-health, and physical-danger language by severity; a critical emergency short-circuits the pipeline to crisis guidance without calling the LLM.
- **Hallucination scoring** — breaks the answer into claims and scores each against the retrieved context, producing a grounded/partial/hallucinated verdict.

**RAG pipeline**
- **Retriever abstraction** — a single `BaseRetriever` interface with a dependency-free `InMemoryRetriever` and a `QdrantRetriever` behind it.
- **Reranker** — optional lexical (BM25-style) reranking and reciprocal rank fusion across multiple retrievers.
- **Multi-turn memory** — a rolling window of conversation turns, rendered into the prompt automatically.

**Provider agnostic**
- OpenAI, OpenRouter, and any LangChain chat model implement the same `BaseProvider` contract. Switching backends is a one-line change; the guardrails and retrieval code never change.

## Installation

```bash
pip install rag-guardrails-starter
```

For local development (tests, linting, type checking):

```bash
git clone https://github.com/Xander-AJ/rag-guardrails-starter.git
cd rag-guardrails-starter
pip install -e ".[dev]"
```

### Environment variables

Copy `.env.example` to `.env` and fill in the keys for the providers you use:

```bash
# .env.example
OPENAI_API_KEY=sk-...                 # required for OpenAIProvider (the default)
OPENROUTER_API_KEY=sk-or-...          # required only for OpenRouterProvider
QDRANT_URL=http://localhost:6333      # required only for QdrantRetriever
QDRANT_COLLECTION=rag_documents       # optional; defaults to rag_documents
MODEL_NAME=gpt-4o-mini                # optional; default model identifier
```

`InMemoryRetriever` needs no Qdrant instance, so the quickstart below runs with only `OPENAI_API_KEY` set.

## Quickstart

```python
import asyncio
from rag_guardrails import InMemoryRetriever, OpenAIProvider, RAGPipeline
from rag_guardrails.rag.retriever import RetrievedChunk

chunks = [
    "Registering a company in Kenya is done online through the eCitizen BRS portal.",
    "Every Kenyan business must obtain a KRA PIN before it can trade.",
    "A single business permit is issued by the county where the business operates.",
]

async def main():
    retriever = InMemoryRetriever(
        [RetrievedChunk(content=c, source=f"doc-{i}", score=0.0) for i, c in enumerate(chunks)]
    )
    pipeline = RAGPipeline(provider=OpenAIProvider(), retriever=retriever)
    response = await pipeline.query("How do I register a company in Kenya?")
    print(response.answer)
    print("Sources:", [s.source for s in response.sources])

asyncio.run(main())
```

## The guardrails pipeline

```
User query
    │
    ▼
┌─────────────┐   blocked / critical emergency
│ InputGuard  │ ───────────────────────────────►  return safe message
└─────────────┘                                    (LLM never called)
    │ safe
    ▼
┌─────────────┐
│  Retriever  │   (+ optional reranker)
└─────────────┘
    │ context
    ▼
┌─────────────┐
│  Provider   │   (OpenAI / OpenRouter / LangChain)
└─────────────┘
    │ draft answer
    ▼
┌──────────────────────┐
│ HallucinationDetector│   scores answer against retrieved context
└──────────────────────┘
    │
    ▼
┌─────────────┐   blocked (toxicity)
│ OutputGuard │ ───────────────────────────────►  return safe message
└─────────────┘
    │ safe (PII redacted, disclaimers added)
    ▼
  Response  (+ conversation memory updated)
```

**InputGuard** runs first. It checks for emergencies, prompt injection, and harmful intent, and redacts PII. A critical emergency or a blocked query returns immediately — the retriever and LLM are never reached. **Retriever** fetches the most relevant chunks for the sanitized query, optionally reordered by a **reranker**. **Provider** generates a draft answer from the retrieved context. **HallucinationDetector** scores that draft against the context so a fluent-but-unsupported answer is visible in the result. **OutputGuard** screens the draft for toxicity (blocking if found), redacts any leaked PII, and appends a disclaimer for medical/legal/financial content. Only then is the answer returned and the exchange recorded in memory.

## Provider configuration

All three providers implement `BaseProvider` and are passed to `RAGPipeline` the same way.

```python
# OpenAI (default)
from rag_guardrails import OpenAIProvider
provider = OpenAIProvider()
```

```python
# OpenRouter, choosing a model per call via kwargs
from rag_guardrails import OpenRouterProvider
provider = OpenRouterProvider()
await provider.complete(messages, model="anthropic/claude-3-haiku")
```

```python
# LangChain, wrapping any chat model you construct
from rag_guardrails import LangChainProvider
from langchain_openai import ChatOpenAI
provider = LangChainProvider(llm=ChatOpenAI(temperature=0.9))
```

## Configuration reference

All settings are read from the environment (or a `.env` file) via `rag_guardrails.config.get_config()`. Field names map to upper-case environment variables (e.g. `hallucination_threshold` → `HALLUCINATION_THRESHOLD`).

| Field | Type | Default | Controls |
|-------|------|---------|----------|
| `provider` | `"openai" \| "openrouter" \| "langchain"` | `"openai"` | Active LLM backend. |
| `model_name` | `str` | `"gpt-4o-mini"` | Default model identifier for the active provider. |
| `openai_api_key` | `SecretStr \| None` | `None` | API key for `OpenAIProvider`. |
| `openrouter_api_key` | `SecretStr \| None` | `None` | API key for `OpenRouterProvider`. |
| `openrouter_base_url` | `str` | `"https://openrouter.ai/api/v1"` | Base URL for the OpenRouter-compatible API. |
| `qdrant_url` | `str \| None` | `None` | URL of the Qdrant vector store (required for `QdrantRetriever`). |
| `qdrant_collection` | `str` | `"rag_documents"` | Qdrant collection used for retrieval. |
| `memory_window` | `int` | `10` | Number of recent conversation turns retained. |
| `input_guard_enabled` | `bool` | `True` | Run the input guard on incoming queries. |
| `output_guard_enabled` | `bool` | `True` | Run the output guard on draft answers. |
| `emergency_detection_enabled` | `bool` | `True` | Run the emergency detector before retrieval/generation. |
| `hallucination_threshold` | `float` | `0.7` | Minimum grounding score; also sets the verdict boundaries. |

## Examples

Each is a self-contained script under `examples/`. Run with `python examples/<name>.py`.

- **`basic_rag.py`** — the minimal pipeline: load chunks, build the pipeline, run one query, print the answer and sources.
- **`with_guardrails.py`** — the guardrail layer made visible: a clean query, a prompt injection that is blocked before the LLM, and a medical query that gets a disclaimer appended.
- **`healthcare_domain.py`** — a healthcare-tuned configuration (strict hallucination threshold, emergency detection) running a normal question, a memory-backed follow-up, a critical emergency, and a local-availability question.
- **`custom_provider.py`** — the same query run through OpenAI, OpenRouter (with a model override), and LangChain (with an injected `ChatOpenAI`) to show provider swapping.

## Project structure

```
rag_guardrails/
├── __init__.py                      # public API: pipeline, guards, providers, retrievers
├── config.py                        # Settings model + get_config(); reads env / .env
├── guardrails/
│   ├── _text.py                     # shared PII patterns and tokenization helpers
│   ├── input_guard.py               # injection / harmful-intent detection, PII redaction
│   ├── output_guard.py              # toxicity screen, PII-leak redaction, disclaimers
│   ├── emergency_detector.py        # severity-tiered emergency classification
│   └── hallucination.py             # claim extraction and grounding score
├── rag/
│   ├── retriever.py                 # BaseRetriever, InMemoryRetriever, QdrantRetriever
│   ├── reranker.py                  # lexical reranker + reciprocal rank fusion
│   └── pipeline.py                  # RAGPipeline orchestration + PipelineResponse
├── providers/
│   ├── base.py                      # BaseProvider contract + ProviderResponse
│   ├── openai_provider.py           # OpenAI backend
│   ├── openrouter_provider.py       # OpenRouter backend (OpenAI-compatible)
│   └── langchain_provider.py        # adapter for any LangChain chat model
└── memory/
    └── multi_turn.py                # ConversationMemory rolling-window store
```

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the suggested build order and module conventions. The test suite runs with `pytest tests/` and exercises the real guards, detectors, and pipeline (the provider is mocked, so no API key is needed to run it). All pull requests must keep the suite green at 65+ passing tests; if you change guard behavior, update or add the tests that pin it. Run `pip install -e ".[dev]"` to get pytest, ruff, black, and mypy.

## License

MIT © John Alexander Kamau. See [`LICENSE`](LICENSE).
