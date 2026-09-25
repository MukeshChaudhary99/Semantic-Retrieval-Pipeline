import json
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from ..core.config import get_settings
from ..core.rate_limiter import rate_limit
from ..models import (
    DeleteResponse,
    DocumentInfo,
    DocumentStatusResponse,
    UploadBatchResponse,
    UploadItemResult,
)
from ..pipeline.chunking.base import SUPPORTED_EXTS, detect_file_type
from ..pipeline.ingest import ingest_document

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])


def _doc_info(doc: dict) -> DocumentInfo:
    return DocumentInfo(**doc)


@router.post(
    "/documents",
    response_model=UploadBatchResponse,
    status_code=202,
    # heavier than a read — extraction/chunking/embedding all run off this
    dependencies=[Depends(rate_limit(requests=10, window_seconds=60))],
)
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()],
    metadata: Annotated[str | None, Form()] = None,
):
    """Upload one or more documents (.pdf, .md, .txt, code files) for async indexing.
    """
    settings = get_settings()

    if len(files) > settings.max_upload_files:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"too many files ({len(files)}); max {settings.max_upload_files} per request"
            },
        )

    try:
        extra_metadata = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "metadata must be valid JSON"})

    store = request.app.state.store
    results: list[UploadItemResult] = []

    for file in files:
        filename = file.filename or "unnamed"

        try:
            file_type = detect_file_type(filename)
        except ValueError as exc:
            results.append(
                UploadItemResult(filename=filename, status="rejected", message=str(exc))
            )
            continue

        content = await file.read()
        if not content:
            results.append(
                UploadItemResult(filename=filename, status="rejected", message="empty file")
            )
            continue

        safe_name = Path(filename).name
        stored_path = settings.upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
        stored_path.write_bytes(content)

        doc = store.create_document(
            filename=safe_name,
            file_type=file_type,
            file_path=str(stored_path),
            size_bytes=len(content),
            metadata=extra_metadata,
        )
        job_id = store.create_job(doc["id"], "ingest")
        # heavy work (partition/chunk/embed) runs after the 202 response is sent
        background_tasks.add_task(
            ingest_document, doc["id"], job_id, store, request.app.state.vectorstore
        )
        results.append(
            UploadItemResult(
                filename=filename,
                status="pending",
                document_id=doc["id"],
                job_id=job_id,
                message="Document accepted for processing",
            )
        )

    accepted = sum(1 for r in results if r.status == "pending")
    return UploadBatchResponse(accepted=accepted, rejected=len(results) - accepted, results=results)


@router.get(
    "/documents",
    response_model=list[DocumentInfo],
    dependencies=[Depends(rate_limit())],
)
async def list_documents(
    request: Request,
    include_deleted: Annotated[bool, Query()] = False,
):
    return [_doc_info(d) for d in request.app.state.store.list_documents(include_deleted)]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentStatusResponse,
    dependencies=[Depends(rate_limit())],
)
async def get_document_status(document_id: str, request: Request):
    store = request.app.state.store
    doc = store.get_document(document_id)
    if doc is None:
        return JSONResponse(status_code=404, content={"detail": "document not found"})
    return DocumentStatusResponse(
        document=_doc_info(doc), job=store.latest_job_for_document(document_id)
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(rate_limit())],
)
async def delete_document(
    document_id: str,
    request: Request,
    hard: Annotated[bool, Query(description="soft delete by default; hard=true purges vectors, file and metadata")] = False,
):
    store = request.app.state.store
    doc = store.get_document(document_id)
    if doc is None:
        return JSONResponse(status_code=404, content={"detail": "document not found"})

    # ---- soft delete: flag the record; chunks are filtered out at query time ----
    if not hard:
        if doc["status"] == "deleted":
            return DeleteResponse(
                document_id=document_id, mode="soft", success=True,
                steps={"metadata": "skipped", "vectors": "skipped", "file": "skipped"},
                errors=[], message="Document is already soft-deleted",
            )
        store.mark_deleted(document_id)
        return DeleteResponse(
            document_id=document_id, mode="soft", success=True,
            steps={"metadata": "ok", "vectors": "skipped", "file": "skipped"},
            errors=[],
            message="Document soft-deleted (chunks excluded from search; data retained). "
                    "Repeat with ?hard=true to purge permanently.",
        )

    # ---- hard delete: vectors -> file -> metadata; each step failure-isolated ----
    steps: dict[str, str] = {}
    errors: list[str] = []

    try:
        request.app.state.vectorstore.delete_document_vectors(document_id)
        steps["vectors"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.exception("vector deletion failed for %s", document_id)
        steps["vectors"] = "failed"
        errors.append(f"vectors: {exc}")

    try:
        file_path = Path(doc["file_path"])
        if file_path.exists():
            file_path.unlink()
        steps["file"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.exception("file deletion failed for %s", document_id)
        steps["file"] = "failed"
        errors.append(f"file: {exc}")

    # keep the metadata record only when vectors could not be removed,
    # so the operation can be retried and no vectors are orphaned silently
    if steps["vectors"] == "ok":
        try:
            store.remove_document(document_id)
            steps["metadata"] = "ok"
        except Exception as exc:  # noqa: BLE001
            steps["metadata"] = "failed"
            errors.append(f"metadata: {exc}")
    else:
        store.update_document(
            document_id, status="delete_failed", delete_error="; ".join(errors)
        )
        steps["metadata"] = "skipped"

    success = all(v == "ok" for v in steps.values())
    return DeleteResponse(
        document_id=document_id,
        mode="hard",
        success=success,
        steps=steps,
        errors=errors,
        message="Document permanently deleted"
        if success
        else "Partial failure — see 'steps'/'errors'; retry to finish cleanup",
    )
