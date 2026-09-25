"""Retrieval core used by POST /query.

hybrid (dense + BM25 sparse) search -> RRF fusion -> optional cross-encoder rerank
"""

from dataclasses import dataclass, field
from typing import Any

from ..core.config import get_settings
from ..db import DocumentStore
from .embeddings import embed_dense, embed_sparse
from .reranker import rerank
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    rrf_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def retrieve(
    store: DocumentStore,
    vectorstore: VectorStore,
    query: str,
    top_k: int,
    prefetch_limit: int | None = None,
    filters: dict[str, Any] | None = None,
    do_rerank: bool = True,
) -> tuple[list[RetrievedChunk], int, bool]:
    """Return (ranked chunks, num_candidates, reranked_flag)."""
    settings = get_settings()
    top_k = min(top_k, settings.max_top_k)
    do_rerank = do_rerank and settings.rerank_enabled
    prefetch_limit = prefetch_limit or max(settings.prefetch_limit, top_k)

    dense_query = embed_dense([query])[0]
    sparse_query = embed_sparse([query])[0]

    # soft-deleted docs are filtered out here, not deleted from the index
    candidates = vectorstore.hybrid_search(
        dense_query=dense_query,
        sparse_query=sparse_query,
        limit=prefetch_limit,
        prefetch_limit=prefetch_limit,
        user_filters=filters,
        excluded_document_ids=store.deleted_document_ids(),
    )
    rrf_scores = {str(p.id): float(p.score or 0.0) for p in candidates}

    if do_rerank and candidates:
        # cross-encoder scores (query, chunk) pairs — more accurate than fusion order
        texts = [p.payload.get("text", "") for p in candidates]
        scores = rerank(query, texts)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    else:
        scores = [rrf_scores[str(p.id)] for p in candidates]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

    results: list[RetrievedChunk] = []
    for i in order[:top_k]:
        p = candidates[i]
        payload = dict(p.payload or {})
        text = payload.pop("text", "")
        results.append(
            RetrievedChunk(
                chunk_id=str(p.id),
                document_id=payload.get("document_id", ""),
                text=text,
                score=float(scores[i]),
                rrf_score=rrf_scores.get(str(p.id), 0.0),
                metadata=payload,
            )
        )
    return results, len(candidates), do_rerank
