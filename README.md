# Semantic Retrieval Pipeline (retrieval only — no generation)

FastAPI service that ingests documents, chunks them, embeds them, and retrieves
semantically relevant chunks over a **Qdrant hybrid index** (dense + sparse BM25
with RRF fusion) followed by a **cross-encoder reranker**.

## Design documentation

This README covers running the service and using the API day to day. For
the actual design reasoning:

- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) — architecture and request flow, API
  specification, database schema, scaling strategy, assumptions and
  tradeoffs.
- [SEMANTIC_RETRIEVAL_PIPELINE_DESIGN.md](SEMANTIC_RETRIEVAL_PIPELINE_DESIGN.md) —
  chunking strategy, embedding lifecycle, hybrid search, vector database
  choice, reranking, failure handling.

## Layout

```
app/
  main.py                  FastAPI app + lifespan init
  models.py                pydantic request/response schemas
  db/                      documents + jobs metadata (SQLAlchemy)
    models.py              ORM models (Document, Job)
    queries.py             CRUD functions, operate on an open Session
    session.py             engine/session factory (SQLite or Postgres, per config)
    store.py               DocumentStore — session-per-call facade over queries.py
  core/                    cross-cutting app infra (not RAG-specific)
    config.py              pydantic-settings (RAG_* env vars)
    logging_config.py      JSON logging + per-request context
    middleware.py          request id, access logging
    rate_limiter.py        token-bucket rate limiting (local or Redis)
    lua/token_bucket.lua   atomic Redis rate-limit script
  pipeline/                retrieval pipeline logic
    ingest.py              async extract→chunk→embed→upsert pipeline
    embeddings.py          sentence-transformers dense + FastEmbed BM25 sparse
    vectorstore.py         Qdrant hybrid search (RRF, w/ client-side fallback)
    retrieval.py           hybrid+rerank core used by /query
    reranker.py            sentence-transformers CrossEncoder
    chunking/
      markdown_code.py       LangChain structure-based splitters
      unstructured_chunker.py unstructured partition_* + chunk_by_title
  api/
    documents.py           POST/GET/DELETE /documents
    query.py               POST /query

alembic/                   schema migrations (alembic upgrade head to apply)
  env.py                   resolves DB URL from app.core.config at runtime
  versions/0001_initial.py documents + jobs tables (mirrors app/db/models.py)
```
