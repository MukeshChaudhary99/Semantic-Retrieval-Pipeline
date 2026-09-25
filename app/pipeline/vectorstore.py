import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from ..core.config import get_settings

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
RRF_K = 60  # standard RRF constant


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.collection = settings.collection_name
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._ensure_collection(settings.dense_dim)

    def _ensure_collection(self, dense_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_NAME: models.VectorParams(
                    size=dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Created Qdrant collection %r", self.collection)

    # ---------- write ----------

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list,
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict],
        base_payload: dict[str, Any],
    ) -> int:
        points = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            # deterministic ids → re-ingesting the same doc overwrites instead of duplicating
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk.index}"))
            payload = {
                **base_payload,
                **chunk.metadata,
                "document_id": document_id,
                "chunk_index": chunk.index,
                "text": chunk.text,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_NAME: dense,
                        SPARSE_NAME: models.SparseVector(
                            indices=sparse["indices"], values=sparse["values"]
                        ),
                    },
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def delete_document_vectors(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def count_document_vectors(self, document_id: str) -> int:
        result = self.client.count(
            collection_name=self.collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        return result.count

    # ---------- read ----------

    def _build_filter(
        self,
        user_filters: dict[str, Any] | None,
        excluded_document_ids: list[str],
    ) -> models.Filter | None:
        must = []
        for key, value in (user_filters or {}).items():
            if isinstance(value, list):
                must.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=value))
                )
            else:
                must.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        must_not = []
        if excluded_document_ids:
            must_not.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=excluded_document_ids),
                )
            )
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

    def hybrid_search(
        self,
        dense_query: list[float],
        sparse_query: dict,
        limit: int,
        prefetch_limit: int,
        user_filters: dict[str, Any] | None = None,
        excluded_document_ids: list[str] | None = None,
    ) -> list[models.ScoredPoint]:
        query_filter = self._build_filter(user_filters, excluded_document_ids or [])
        sparse_vec = models.SparseVector(
            indices=sparse_query["indices"], values=sparse_query["values"]
        )
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using=DENSE_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=sparse_vec,
                        using=SPARSE_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return list(response.points)
        except Exception as exc:
            logger.warning(
                "Native RRF fusion failed (%s); falling back to client-side RRF", exc
            )
            return self._manual_hybrid_search(
                dense_query, sparse_vec, limit, prefetch_limit, query_filter
            )

    def _manual_hybrid_search(
        self,
        dense_query: list[float],
        sparse_vec: models.SparseVector,
        limit: int,
        prefetch_limit: int,
        query_filter: models.Filter | None,
    ) -> list[models.ScoredPoint]:
        dense_hits = self.client.query_points(
            collection_name=self.collection,
            query=dense_query,
            using=DENSE_NAME,
            query_filter=query_filter,
            limit=prefetch_limit,
            with_payload=True,
        ).points
        sparse_hits = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vec,
            using=SPARSE_NAME,
            query_filter=query_filter,
            limit=prefetch_limit,
            with_payload=True,
        ).points

        # classic RRF: score = sum over legs of 1/(k + rank)
        scores: dict[Any, float] = {}
        points: dict[Any, models.ScoredPoint] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits):
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank + 1)
                points.setdefault(hit.id, hit)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        results = []
        for pid, score in ranked:
            p = points[pid]
            results.append(
                models.ScoredPoint(
                    id=p.id,
                    version=p.version,
                    score=score,
                    payload=p.payload,
                    vector=p.vector,
                )
            )
        return results
