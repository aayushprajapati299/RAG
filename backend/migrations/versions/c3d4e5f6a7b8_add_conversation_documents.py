"""Add conversation_documents association table and selected_document_id to conversation

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-11-17 06:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create conversation_documents association table
    op.create_table(
        'conversation_documents',
        sa.Column('conversation_id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('(CURRENT_TIMESTAMP)'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('conversation_id', 'document_id'),
    )
    with op.batch_alter_table('conversation_documents', schema=None) as batch_op:
        batch_op.create_index('ix_conversation_documents_conversation_id', ['conversation_id'], unique=False)
        batch_op.create_index('ix_conversation_documents_document_id', ['document_id'], unique=False)

    # 2. Add selected_document_id column to conversations table
    with op.batch_alter_table('conversations', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('selected_document_id', sa.Uuid(), nullable=True)
        )
        batch_op.create_foreign_key(
            'fk_conversations_selected_document_id',
            'documents',
            ['selected_document_id'],
            ['id'],
            ondelete='SET NULL',
        )
        batch_op.create_index('ix_conversations_selected_document_id', ['selected_document_id'], unique=False)

    # 3. Preserve existing conversations and documents:
    # Associate user documents with existing user conversations
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        op.execute(
            """
            INSERT INTO conversation_documents (conversation_id, document_id, created_at)
            SELECT c.id, d.id, CURRENT_TIMESTAMP
            FROM conversations c
            JOIN documents d ON c.user_id = d.user_id
            ON CONFLICT DO NOTHING;
            """
        )
        op.execute(
            """
            UPDATE conversations
            SET selected_document_id = sub.document_id
            FROM (
                SELECT DISTINCT ON (conversation_id) conversation_id, document_id
                FROM conversation_documents
                ORDER BY conversation_id, created_at DESC
            ) sub
            WHERE conversations.id = sub.conversation_id AND conversations.selected_document_id IS NULL;
            """
        )


def downgrade() -> None:
    with op.batch_alter_table('conversations', schema=None) as batch_op:
        batch_op.drop_constraint('fk_conversations_selected_document_id', type_='foreignkey')
        batch_op.drop_index('ix_conversations_selected_document_id')
        batch_op.drop_column('selected_document_id')

    op.drop_table('conversation_documents')
