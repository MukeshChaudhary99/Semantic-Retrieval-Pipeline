from typing import Any

from . import queries
from .session import create_session_factory, session_scope


class DocumentStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._session_factory = create_session_factory(database_url)

    # ---------- documents ----------

    def create_document(
        self,
        filename: str,
        file_type: str,
        file_path: str,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with session_scope(self._session_factory) as session:
            return queries.create_document(
                session, filename, file_type, file_path, size_bytes, metadata
            )

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with session_scope(self._session_factory) as session:
            return queries.get_document(session, doc_id)

    def list_documents(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        with session_scope(self._session_factory) as session:
            return queries.list_documents(session, include_deleted)

    def deleted_document_ids(self) -> list[str]:
        with session_scope(self._session_factory) as session:
            return queries.deleted_document_ids(session)

    def update_document(self, doc_id: str, **fields: Any) -> None:
        with session_scope(self._session_factory) as session:
            queries.update_document(session, doc_id, **fields)

    def mark_deleted(self, doc_id: str) -> None:
        with session_scope(self._session_factory) as session:
            queries.mark_deleted(session, doc_id)

    def remove_document(self, doc_id: str) -> None:
        with session_scope(self._session_factory) as session:
            queries.remove_document(session, doc_id)

    # ---------- jobs ----------

    def create_job(self, document_id: str, job_type: str) -> str:
        with session_scope(self._session_factory) as session:
            return queries.create_job(session, document_id, job_type)

    def update_job(self, job_id: str, status: str, error: str | None = None) -> None:
        with session_scope(self._session_factory) as session:
            queries.update_job(session, job_id, status, error)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with session_scope(self._session_factory) as session:
            return queries.get_job(session, job_id)

    def latest_job_for_document(self, doc_id: str) -> dict[str, Any] | None:
        with session_scope(self._session_factory) as session:
            return queries.latest_job_for_document(session, doc_id)
