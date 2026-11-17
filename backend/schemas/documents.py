from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class ChunkResponse(BaseModel):
    """Schema representing an extracted document chunk/page."""
    id: UUID
    document_id: UUID
    page_number: int
    chunk_index: int
    content: str
    char_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    """Schema representing an uploaded document metadata."""
    id: UUID
    filename: str
    file_type: str
    file_size_bytes: Optional[int] = None
    page_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestionJobResponse(BaseModel):
    """Schema representing an ingestion tracking job."""
    id: UUID
    document_id: UUID
    status: str
    error_message: Optional[str] = None
    pages_processed: int
    total_pages: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    """Document metadata including latest ingestion job and prompt identifiers."""
    latest_job: Optional[IngestionJobResponse] = None
    job_id: Optional[UUID] = None
    document_id: Optional[UUID] = None


class IngestionResult(BaseModel):
    """Result summary of a document ingestion run."""
    document_id: UUID
    file_type: str
    page_count: int
    chunks_created: int
