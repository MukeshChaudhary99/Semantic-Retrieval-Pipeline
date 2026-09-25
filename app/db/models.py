import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    # pending | processing | indexed | failed | deleted | delete_failed
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    # mapped attribute can't be named `metadata` — collides with Base.metadata
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    delete_error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=_now, index=True)
    updated_at: Mapped[str] = mapped_column(String, nullable=False, default=_now)
    deleted_at: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    jobs: Mapped[list["Job"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "metadata": self.doc_metadata or {},
            "num_chunks": self.num_chunks,
            "error": self.error,
            "delete_error": self.delete_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted_at": self.deleted_at,
        }


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String, nullable=False)  # ingest | delete
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=_now)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="jobs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "type": self.type,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }
