"""initial schema: documents + jobs

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("file_type", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("num_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("delete_error", sa.String(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("deleted_at", sa.String(), nullable=True),
    )
    op.create_index("ix_documents_created_at", "documents", ["created_at"])
    op.create_index("ix_documents_deleted_at", "documents", ["deleted_at"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("document_id", sa.String(), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("finished_at", sa.String(), nullable=True),
    )
    op.create_index("ix_jobs_document_id", "jobs", ["document_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_document_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_documents_deleted_at", table_name="documents")
    op.drop_index("ix_documents_created_at", table_name="documents")
    op.drop_table("documents")
