"""RAG package: retrieval-augmented generation pipeline.

Implements the retrieve -> (optional) rerank -> generate flow:

- ``retriever``: vector-store abstraction for fetching candidate documents.
- ``reranker``: optional layer that reorders retrieved candidates by relevance.
- ``pipeline``: orchestrates retrieval, reranking, provider generation, and the
  guardrail hooks defined in ``rag_guardrails.guardrails``.

The pipeline is responsible for orchestration only; safety logic lives in the
guardrails package.
"""
