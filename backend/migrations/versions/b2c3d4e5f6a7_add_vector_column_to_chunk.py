"""Add vector column to chunk

Revision ID: b2c3d4e5f6a7
Revises: 9aaf509f8bd2
Create Date: 2026-11-17 01:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    from sqlalchemy.types import UserDefinedType

    class Vector(UserDefinedType):  # type: ignore
        def __init__(self, dim: int):
            self.dim = dim

        def get_col_spec(self, **kw):
            return f"vector({self.dim})"


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = '9aaf509f8bd2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    # Enable vector extension on PostgreSQL
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Add 384-dimensional embedding column for sentence-transformers/all-MiniLM-L6-v2
    op.add_column(
        'chunks',
        sa.Column('embedding', Vector(384), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('chunks', 'embedding')
