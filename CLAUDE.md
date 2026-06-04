# Claude Code — project notes

## Resume the main dev session

From this repo root, continue the rag-guardrails-starter implementation session:

```bash
claude --resume 8fcbbcea-e65a-4520-b537-033bbb3a0d4a
```

Run after `cd` into `/Users/xander/code/rag-guardrails-starter` (or any path where you normally start Claude for this project).

## Project context

- Target layout and module roles: [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)
- Architecture and install: [`README.md`](README.md)
- Suggested build order: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Session recap (last known state)

Guardrails, config, providers (incl. OpenAI), retriever, memory, and RAG pipeline are implemented and manually verified. Remaining work: reranker, OpenRouter/LangChain providers, and fleshing out examples, tests, and docs.
