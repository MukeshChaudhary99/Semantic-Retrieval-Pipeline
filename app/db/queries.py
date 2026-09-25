from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Document, Job

ALLOWED_DOCUMENT_UPDATE_FIELDS = {
    "status", "metadata", "num_chunks", "error", "delete_error", "deleted_at",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- documents ----------


def create_document(
    session: Session,
    filename: str,
    file_type: str,
    file_path: str,
    size_bytes: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _now()
    doc = Document(
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        size_bytes=size_bytes,
        status="pending",
        doc_metadata=metadata or {},
        created_at=now,
        updated_at=now,
    )
    session.add(doc)
    session.flush()  # assigns doc.id
    return doc.to_dict()


def get_document(session: Session, doc_id: str) -> dict[str, Any] | None:
    doc = session.get(Document, doc_id)
    return doc.to_dict() if doc else None


def list_documents(session: Session, include_deleted: bool = False) -> list[dict[str, Any]]:
    stmt = select(Document)
    if not include_deleted:
        stmt = stmt.where(Document.deleted_at.is_(None))
    stmt = stmt.order_by(Document.created_at.desc())
    return [doc.to_dict() for doc in session.scalars(stmt)]


def deleted_document_ids(session: Session) -> list[str]:
    # query side filters these out of vector search
    stmt = select(Document.id).where(Document.deleted_at.is_not(None))
    return list(session.scalars(stmt))


def update_document(session: Session, doc_id: str, **fields: Any) -> None:
    for key in fields:
        if key not in ALLOWED_DOCUMENT_UPDATE_FIELDS:
            raise ValueError(f"cannot update field {key}")
    doc = session.get(Document, doc_id)
    if doc is None:
        return
    for key, value in fields.items():
        setattr(doc, "doc_metadata" if key == "metadata" else key, value)
    doc.updated_at = _now()


def mark_deleted(session: Session, doc_id: str) -> None:
    update_document(session, doc_id, status="deleted", deleted_at=_now())


def remove_document(session: Session, doc_id: str) -> None:
    session.execute(sa_delete(Job).where(Job.document_id == doc_id))
    session.execute(sa_delete(Document).where(Document.id == doc_id))


# ---------- jobs ----------


def create_job(session: Session, document_id: str, job_type: str) -> str:
    job = Job(document_id=document_id, type=job_type, status="pending", created_at=_now())
    session.add(job)
    session.flush()  # assigns job.id
    return job.id


def update_job(session: Session, job_id: str, status: str, error: str | None = None) -> None:
    job = session.get(Job, job_id)
    if job is None:
        return
    job.status = status
    job.error = error
    job.finished_at = _now() if status in ("done", "failed") else None


def get_job(session: Session, job_id: str) -> dict[str, Any] | None:
    job = session.get(Job, job_id)
    return job.to_dict() if job else None


def latest_job_for_document(session: Session, doc_id: str) -> dict[str, Any] | None:
    stmt = (
        select(Job)
        .where(Job.document_id == doc_id)
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    job = session.scalars(stmt).first()
    return job.to_dict() if job else None
