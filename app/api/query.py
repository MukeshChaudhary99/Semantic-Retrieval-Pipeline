import logging

from fastapi import APIRouter, Depends, Request

from ..core.rate_limiter import rate_limit
from ..models import QueryRequest, QueryResponse, QueryResult
from ..pipeline.retrieval import retrieve

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(rate_limit())],
)
async def query_documents(body: QueryRequest, request: Request):
    """Semantic search: hybrid (dense + BM25 sparse) retrieval -> RRF -> rerank."""
    chunks, num_candidates, reranked = retrieve(
        store=request.app.state.store,
        vectorstore=request.app.state.vectorstore,
        query=body.query,
        top_k=body.top_k,
        prefetch_limit=body.prefetch_limit,
        filters=body.filters,
        do_rerank=body.rerank,
    )
    return QueryResponse(
        query=body.query,
        top_k=min(body.top_k, len(chunks)),
        num_candidates=num_candidates,
        reranked=reranked,
        results=[
            QueryResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=c.score,
                rrf_score=c.rrf_score if reranked else None,
                metadata=c.metadata,
            )
            for c in chunks
        ],
    )
