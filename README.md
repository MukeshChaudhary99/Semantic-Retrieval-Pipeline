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

## Architecture

```
POST /documents ──► save file ──► SQLite (pending) ──► BackgroundTask
                                                        │
                                            extract → chunk → embed → upsert
                                                        │
                                            .md/.code → LangChain structure splitters
                                            .pdf/.txt → unstructured.io (chunk_by_title,
                                                        tables, images)
                                                        │
                                            dense: sentence-transformers
                                            sparse: FastEmbed BM25
                                                        ▼
                                            Qdrant server (docker-compose, port 6333)

POST /query ──► dense+sparse embed query ──► hybrid RRF search (top-N candidates)
            ──► cross-encoder rerank ──► ranked results
```

## Setup

```bash
docker compose up -d qdrant   # vector DB server, http://localhost:6333

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional, defaults work
alembic upgrade head   # creates rag.db (documents + jobs tables)
uvicorn app.main:app --reload --port 8000
```

Qdrant runs as its own server (not embedded mode) so the query service isn't
limited to a single process — multiple `uvicorn` workers or app replicas can
share the same collection. Data persists in the `qdrant_storage` Docker
volume across restarts. Point at a different instance with `RAG_QDRANT_URL`
(and `RAG_QDRANT_API_KEY` if auth is enabled on the server).

Document/job metadata is SQLAlchemy + Alembic-managed (`app/db/`, `alembic/`) —
SQLite locally by default, or Postgres in production via `RAG_ENVIRONMENT=production`
(+ `RAG_DB_HOST`/`RAG_DB_USER`/`RAG_DB_PASSWORD`/…, or a single `RAG_DATABASE_URL`).
The app never creates/alters tables itself; run `alembic upgrade head` after
pulling changes that add a new revision under `alembic/versions/`.

First run downloads the embedding/reranker models (~a few hundred MB, cached afterwards).

> **PDF notes:** `strategy=auto` (default) uses hi_res layout detection for table/image
> extraction — needs `poppler-utils`, `tesseract-ocr` and `unstructured-inference`.
> If hi_res is unavailable, the pipeline automatically falls back to the
> lightweight `fast` strategy (text-only, no table structure).
> Set `RAG_UNSTRUCTURED_STRATEGY` to control this.

> **Environment quirks seen on this machine** (behind a Sophos TLS-inspecting
> proxy — first-run model downloads fail without these):
> - `HF_HUB_DISABLE_XET=1` — the hf-xet CDN backend returns 0-byte downloads.
> - `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` and
>   `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` — points downloads
>   at the **system** CA bundle (which has the proxy's root CA merged in via
>   `update-ca-certificates`) instead of `certifi`'s bundled one, which
>   doesn't. Fixes `CERTIFICATE_VERIFY_FAILED: self-signed certificate in
>   certificate chain`.
> - `RAG_SSL_STRICT_VERIFY=false` (default) relaxes Python 3.13's
>   `VERIFY_X509_STRICT` (`app/core/ssl_compat.py`) — needed in addition to
>   the CA bundle above, since this proxy's CA doesn't mark Basic
>   Constraints critical. Chain/signature validation stays fully enforced;
>   only that one overly-strict check is relaxed. Set to `true` on a
>   compliant network.

## APIs

### 1. Upload — `POST /documents` (async, returns `202`)

Accepts one or more files in a single request (`files`, repeated). `metadata`,
if given, is a single JSON object applied to every file in the batch. Each
file is validated and stored independently — one rejected file (bad
extension, empty body) doesn't fail the others; see `results` per file.

```bash
curl -X POST http://localhost:8000/documents \
  -F "files=@report.pdf" \
  -F "files=@notes.md" \
  -F 'metadata={"team": "platform", "year": 2026}'
# {"accepted": 2, "rejected": 0, "results": [
#   {"filename": "report.pdf", "status": "pending", "document_id": "...", "job_id": "...", ...},
#   {"filename": "notes.md", "status": "pending", "document_id": "...", "job_id": "...", ...}
# ]}
```

Up to `RAG_MAX_UPLOAD_FILES` files per request (default 20).

Poll status: `GET /documents/{document_id}` → `status: pending|processing|indexed|failed`.
List docs: `GET /documents`.

Supported: `.pdf`, `.md`/`.markdown`/`.mdx`, `.txt`, and code
(`.py .js .ts .java .go .rs .cpp .c .cs .rb .php .swift .kt .scala .lua .pl .hs .ex .ps1 .sol .html .rst .proto` …).

### 2. Query — `POST /query`

```bash
curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{
  "query": "how does authentication work",
  "top_k": 5,
  "filters": {"file_type": "markdown"},
  "rerank": true
}'
```

- `filters` — exact-match payload fields (`document_id`, `filename`, `file_type`,
  `element_type`, `language`, `page_number`, or list values for OR).
- `rerank=false` returns raw RRF-fusion ordering.
- Soft-deleted documents are always excluded.

### 3. Delete — `DELETE /documents/{id}`

```bash
curl -X DELETE "http://localhost:8000/documents/{id}"           # soft delete (default)
curl -X DELETE "http://localhost:8000/documents/{id}?hard=true" # hard delete
```

**Soft vs hard delete**

| | Soft (default) | Hard (`?hard=true`) |
|---|---|---|
| Metadata | kept, `status=deleted`, `deleted_at` set | removed |
| Vectors/chunks | kept in Qdrant, excluded at query time | deleted by `document_id` filter |
| File on disk | kept | deleted |
| Reversible | yes (re-ingest / clear flag) | no |

**Partial-failure handling (hard delete):** each step — `vectors`, `file`,
`metadata` — is isolated and reported in `steps`/`errors`. If vector deletion
fails, the metadata record is kept (`status=delete_failed`) so the operation can
be retried and no chunks are orphaned silently.

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
