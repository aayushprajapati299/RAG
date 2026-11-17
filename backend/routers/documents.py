"""
Document management router.

Endpoints:
- POST /api/documents: Upload a PDF or TXT document (schedules background ingestion)
- GET /api/documents: List the authenticated user's documents
- GET /api/documents/{document_id}: Retrieve an individual document owned by the user
- GET /api/documents/{document_id}/status: Retrieve ingestion status for an owned document

Architecture Note:
In-process background tasks (via FastAPI BackgroundTasks) are suitable for this
project's scope and lightweight single-instance design, but they execute in the
current application process and are not durable across server restarts.
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import SessionLocal, get_db
from backend.models import Chunk, Conversation, Document, IngestionJob, User
from backend.schemas.documents import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentResponse,
    IngestionJobResponse,
)
from backend.security import get_current_user, verify_csrf_token
from backend.services.document_ingestion import (
    DocumentProcessingError,
    DocumentValidationError,
    OCRUnavailableError,
    generate_storage_path,
    ingest_document,
    validate_file_metadata,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])
chunks_router = APIRouter(prefix="/api/chunks", tags=["chunks"])


def _mark_job_failed(db: Session, job_id: UUID, error_message: str) -> None:
    """Safely mark an IngestionJob as failed with a sanitized error message."""
    try:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = error_message
            db.commit()
    except Exception:
        db.rollback()


def run_background_ingestion(job_id: UUID, document_id: UUID) -> None:
    """
    Background worker function to extract text chunks and persist them.
    Creates and closes its own independent database session.

    Note: In-process background tasks are suitable for this project's requirements,
    but they are not durable across server restarts.
    """
    db = SessionLocal()
    try:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job:
            return

        # Transition status from pending to processing
        job.status = "processing"
        db.commit()

        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            _mark_job_failed(db, job_id, "Document record not found")
            return

        # Perform extraction and chunk generation
        ingest_document(db, document)

        # Mark job completed
        job.status = "completed"
        job.pages_processed = document.page_count
        job.total_pages = document.page_count
        job.error_message = None
        db.commit()

    except DocumentValidationError as ve:
        db.rollback()
        _mark_job_failed(db, job_id, f"Validation error: {str(ve)[:200]}")
    except OCRUnavailableError as ocr_err:
        db.rollback()
        _mark_job_failed(db, job_id, f"OCR unavailable: {str(ocr_err)[:200]}")
    except Exception as e:
        db.rollback()
        _mark_job_failed(db, job_id, "Document processing failed. Please verify file integrity.")
    finally:
        db.close()


@router.post(
    "",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document and schedule background ingestion",
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (.pdf or .txt)"),
    conversation_id: Optional[UUID] = Form(None, description="Optional conversation to associate this document with"),
    x_csrf_token: str = Header(
        ...,
        alias="X-CSRF-Token",
        description="CSRF token for state-changing requests",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Accept a PDF or TXT upload, store under a generated safe filename, create a
    user-owned Document and pending IngestionJob, return identifiers promptly,
    and schedule extraction in the background.
    """
    # Enforce CSRF token validation
    if not x_csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required CSRF token in X-CSRF-Token header",
        )
    if not verify_csrf_token(x_csrf_token, expected_user_id=current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired CSRF token",
        )

    # If conversation_id is provided, verify user ownership upfront
    target_conversation = None
    if conversation_id:
        target_conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
            .first()
        )
        if not target_conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Target conversation not found or not owned by user",
            )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload",
        )

    # Read uploaded file content directly from underlying file stream
    try:
        content = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    file_size = len(content)

    # Validate file extension, MIME type, and size
    try:
        ext = validate_file_metadata(
            filename=file.filename,
            content_type=file.content_type,
            file_size_bytes=file_size,
        )
    except DocumentValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )

    # Generate a collision-resistant safe storage path
    safe_storage_name, full_storage_path = generate_storage_path(file.filename)

    try:
        with open(full_storage_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist uploaded file: {str(e)}",
        )

    # Create user-owned Document record
    file_type = "pdf" if ext == ".pdf" else "txt"
    document = Document(
        id=uuid.uuid4(),
        user_id=current_user.id,
        filename=Path(file.filename).name,  # Clean sanitized basename for display
        file_path=full_storage_path,
        file_type=file_type,
        file_size_bytes=file_size,
        page_count=0,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Create associated IngestionJob record with pending status
    job = IngestionJob(
        id=uuid.uuid4(),
        document_id=document.id,
        user_id=current_user.id,
        status="pending",
        pages_processed=0,
        total_pages=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Associate with target conversation if one was specified
    if target_conversation:
        target_conversation.documents.append(document)
        target_conversation.selected_document_id = document.id
        db.commit()

    # Schedule background extraction task
    background_tasks.add_task(
        run_background_ingestion,
        job_id=job.id,
        document_id=document.id,
    )

    # Return prompt response with document and job identifiers
    return DocumentDetailResponse(
        id=document.id,
        document_id=document.id,
        job_id=job.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size_bytes=document.file_size_bytes,
        page_count=document.page_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
        latest_job=job,
    )


@router.get(
    "",
    response_model=List[DocumentResponse],
    summary="List current user's documents",
)
def list_documents(
    conversation_id: Optional[UUID] = Query(None, description="Optional conversation ID to filter associated documents"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List only documents owned by the currently authenticated user.
    If conversation_id is provided, returns documents associated with that conversation.
    """
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
            .first()
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        return conversation.documents

    documents = (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return documents


@router.delete(
    "/{document_id}",
    summary="Delete a document and its associated chunks and storage",
)
def delete_document(
    document_id: UUID,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a document owned by current_user:
    - Verifies ownership (404 if not found or unauthorized).
    - Unsets selected_document_id in any conversations referencing this document.
    - Removes associated chunks, jobs, and conversation associations via cascade.
    - Safely removes the physical file from disk storage.
    """
    if x_csrf_token and not verify_csrf_token(x_csrf_token, expected_user_id=current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired CSRF token",
        )

    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # 1. Update any conversations that had this document selected
    conversations_with_doc = (
        db.query(Conversation)
        .filter(Conversation.selected_document_id == document.id)
        .all()
    )
    for conv in conversations_with_doc:
        conv.selected_document_id = None

    # 2. Safely remove physical file from disk
    file_path = document.file_path
    if file_path:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning("Failed to remove file on disk %s: %s", file_path, str(e))

    # 3. Delete document record (cascades chunks, ingestion jobs, conversation_documents)
    db.delete(document)
    db.commit()

    return {"status": "ok", "message": "Document deleted successfully"}


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details by ID",
)
def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve an individual document owned by the authenticated user.
    Never accepts a document ID as proof of ownership: returns 404 if
    the document does not exist or belongs to another user.
    """
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    latest_job = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == document.id)
        .order_by(IngestionJob.created_at.desc())
        .first()
    )

    return DocumentDetailResponse(
        id=document.id,
        document_id=document.id,
        job_id=latest_job.id if latest_job else None,
        filename=document.filename,
        file_type=document.file_type,
        file_size_bytes=document.file_size_bytes,
        page_count=document.page_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
        latest_job=latest_job,
    )


@router.get(
    "/{document_id}/status",
    response_model=IngestionJobResponse,
    summary="Get ingestion job status for an owned document",
)
def get_document_status(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve the ingestion status of a document owned by the authenticated user.
    Enforces document ownership: returns 404 if the document does not exist
    or belongs to another user.
    """
    # Enforce ownership check
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    job = (
        db.query(IngestionJob)
        .filter(IngestionJob.document_id == document.id)
        .order_by(IngestionJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found for document",
        )

    return job


@router.get(
    "/{document_id}/chunks/{chunk_id}",
    response_model=ChunkResponse,
    summary="Get a specific document chunk by ID",
)
def get_document_chunk(
    document_id: UUID,
    chunk_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve an individual chunk for an owned document.
    Enforces document ownership; returns 404 if unauthorized or not found.
    """
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    chunk = (
        db.query(Chunk)
        .filter(Chunk.id == chunk_id, Chunk.document_id == document.id)
        .first()
    )
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found",
        )

    return chunk


@chunks_router.get(
    "/{chunk_id}",
    response_model=ChunkResponse,
    summary="Get a chunk by ID ensuring user ownership of the document",
)
def get_chunk(
    chunk_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve an individual chunk by ID, verifying that the authenticated user owns
    the associated document. Returns 404 if not found or unauthorized.
    """
    chunk = (
        db.query(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .filter(Chunk.id == chunk_id, Document.user_id == current_user.id)
        .first()
    )
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunk not found or not owned by user",
        )
    return chunk

