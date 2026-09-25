from typing import Any, Literal

from pydantic import BaseModel, Field

# what Qdrant's MatchValue/MatchAny actually accept — anything else (a
# nested object, null, ...) can't be turned into a filter condition
FilterValue = str | int | float | bool | list[str | int | float | bool]


# ---------- Upload ----------

class UploadItemResult(BaseModel):
    filename: str
    status: Literal["pending", "rejected"]
    document_id: str | None = None
    job_id: str | None = None
    message: str


class UploadBatchResponse(BaseModel):
    accepted: int
    rejected: int
    results: list[UploadItemResult]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    metadata: dict[str, Any]
    num_chunks: int
    error: str | None = None
    created_at: str
    updated_at: str
    deleted_at: str | None = None


class DocumentStatusResponse(BaseModel):
    document: DocumentInfo
    job: dict[str, Any] | None = None


# ---------- Query ----------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    prefetch_limit: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Candidates fetched from hybrid search before reranking",
    )
    filters: dict[str, FilterValue] = Field(
        default_factory=dict,
        description="Exact-match metadata filters, e.g. "
        '{"document_id": "...", "file_type": "pdf", "filename": "a.md", '
        '"element_type": "Table", "language": "python"}. '
        "Values must be a string, number, boolean, or a list of those "
        "(list = OR) — not an object.",
        examples=[{"file_type": "pdf"}],
    )
    rerank: bool = True


class QueryResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float = Field(description="Reranker score (or RRF fusion score if rerank disabled)")
    rrf_score: float | None = None
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    top_k: int
    num_candidates: int
    reranked: bool
    results: list[QueryResult]


# ---------- Delete ----------

class DeleteResponse(BaseModel):
    document_id: str
    mode: Literal["soft", "hard"]
    success: bool
    steps: dict[str, str]  # step -> ok | failed | skipped
    errors: list[str]
    message: str


class ErrorResponse(BaseModel):
    detail: str
