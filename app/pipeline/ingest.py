"""Async document ingestion pipeline (runs as a FastAPI BackgroundTask).

extract -> chunk -> embed (dense + sparse) -> upsert into Qdrant -> update status
"""

import logging
from pathlib import Path

from ..core.config import get_settings
from ..db import DocumentStore
from .chunking.base import Chunk
from .chunking.markdown_code import split_code, split_markdown
from .chunking.unstructured_chunker import chunk_pdf_file, chunk_text_file
from .embeddings import embed_dense, embed_sparse
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


def _make_chunks(doc: dict, path: Path, settings) -> list[Chunk]:
    file_type = doc["file_type"]
    if file_type == "markdown":
        return split_markdown(
            path.read_text(encoding="utf-8", errors="replace"),
            settings.chunk_size,
            settings.chunk_overlap,
        )
    if file_type == "code":
        return split_code(
            path.read_text(encoding="utf-8", errors="replace"),
            doc["filename"],
            settings.chunk_size,
            settings.chunk_overlap,
        )
    if file_type == "pdf":
        return chunk_pdf_file(
            path,
            document_id=doc["id"],
            chunk_size=settings.chunk_size,
            strategy=settings.unstructured_strategy,
            extract_images=settings.extract_images,
            image_output_dir=settings.image_extract_dir / doc["id"],
        )
    if file_type == "text":
        return chunk_text_file(path, document_id=doc["id"], chunk_size=settings.chunk_size)
    raise ValueError(f"unknown file_type {file_type!r}")


def ingest_document(document_id: str, job_id: str, store: DocumentStore, vectorstore: VectorStore) -> None:
    settings = get_settings()
    try:
        store.update_job(job_id, "running")
        store.update_document(document_id, status="processing")

        doc = store.get_document(document_id)
        if doc is None:
            raise RuntimeError(f"document {document_id} not found")
        path = Path(doc["file_path"])

        chunks = _make_chunks(doc, path, settings)
        if not chunks:
            raise RuntimeError("no extractable content found in document")

        texts = [c.text for c in chunks]
        dense_vectors = embed_dense(texts)
        sparse_vectors = embed_sparse(texts)

        # user-supplied metadata is namespaced to avoid clashing with chunk fields
        base_payload = {
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            **{f"meta_{k}": v for k, v in (doc["metadata"] or {}).items()},
        }
        n = vectorstore.upsert_chunks(
            document_id, chunks, dense_vectors, sparse_vectors, base_payload
        )
        store.update_document(
            document_id, status="indexed", num_chunks=n, error=None
        )
        store.update_job(job_id, "done")
        logger.info("Ingested document %s (%d chunks)", document_id, n)
    except Exception as exc:  # noqa: BLE001 - record failure and continue
        logger.exception("Ingestion failed for document %s", document_id)
        store.update_document(document_id, status="failed", error=str(exc))
        store.update_job(job_id, "failed", error=str(exc))
