"""
Document ingestion service.

Handles:
- Validation of uploaded files (MIME type, file extension, file size limits)
- Collision-resistant storage filename generation
- PDF and TXT extraction into Chunk models associated with Document
- Dynamic Tesseract OCR resolution with clear failure reporting
- Reliable resource closure and bounded page processing
"""
import io
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import pytesseract
from sqlalchemy.orm import Session

# Import PyMuPDF with backwards-compatibility fallback
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from backend.config import settings
from backend.models import Chunk, Document, IngestionJob
from backend.schemas.documents import IngestionResult
from backend.services.embeddings import generate_embeddings_batch
from backend.services.retrieval import chunk_text

logger = logging.getLogger(__name__)

# Constants & Policy Limits
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "text/plain",
    "application/octet-stream",  # accepted if extension and content inspect as valid
}
MAX_PDF_PAGES_LIMIT = 500  # Guard against denial-of-service from massive documents
TXT_CHUNK_SIZE_WORDS = 500  # Matches existing 500-word virtual page chunking
OCR_TRIGGER_CHAR_THRESHOLD = 20  # Minimum characters to consider page text extractable without OCR


class DocumentValidationError(ValueError):
    """Raised when an uploaded document fails validation."""
    pass


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is required but the Tesseract executable cannot be located."""
    pass


class DocumentProcessingError(Exception):
    """Raised when an error occurs during document extraction or persistence."""
    pass


def resolve_tesseract_cmd(configured_cmd: Optional[str] = None) -> Optional[str]:
    """
    Resolve the path to the Tesseract executable.
    Resolution order:
    1. Explicit configuration (settings.TESSERACT_CMD)
    2. System PATH via shutil.which('tesseract')
    3. Platform-specific default install paths
    """
    candidate = configured_cmd or settings.TESSERACT_CMD
    if candidate and os.path.isfile(candidate):
        return candidate

    which_path = shutil.which("tesseract")
    if which_path and os.path.isfile(which_path):
        return which_path

    if sys.platform == "win32":
        win_candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for c in win_candidates:
            if os.path.isfile(c):
                return c
    elif sys.platform.startswith("linux"):
        linux_candidates = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/usr/bin/tesseract-ocr",
        ]
        for c in linux_candidates:
            if os.path.isfile(c):
                return c
    elif sys.platform == "darwin":
        mac_candidates = [
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
        for c in mac_candidates:
            if os.path.isfile(c):
                return c

    return None


def get_tesseract_cmd() -> str:
    """
    Return the resolved Tesseract executable path.
    Raises OCRUnavailableError if no valid executable is found.
    """
    cmd = resolve_tesseract_cmd()
    if not cmd:
        raise OCRUnavailableError(
            "Tesseract OCR executable could not be found on the server. "
            "Please install Tesseract and either add it to the system PATH "
            "or configure TESSERACT_CMD in your environment settings."
        )
    return cmd


def extract_text_with_ocr(page: fitz.Page) -> str:
    """
    Render a PDF page to a 200 DPI image and perform OCR using Tesseract.
    Releases pixel buffer and PIL image resources reliably.
    """
    tesseract_path = get_tesseract_cmd()
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")

    img = Image.open(io.BytesIO(img_bytes))
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    finally:
        img.close()


def validate_file_metadata(
    filename: str,
    content_type: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> str:
    """
    Validate file extension, content type, and file size.
    Returns the normalized extension (e.g. '.pdf' or '.txt').
    """
    if not filename or not filename.strip():
        raise DocumentValidationError("Filename must not be empty.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentValidationError(
            f"Unsupported file extension '{ext}'. Only .pdf and .txt documents are supported."
        )

    if content_type:
        normalized_ct = content_type.split(";")[0].strip().lower()
        if normalized_ct not in ALLOWED_CONTENT_TYPES:
            raise DocumentValidationError(
                f"Unsupported Content-Type '{content_type}'. Expected PDF or text."
            )

    max_bytes = settings.MAX_UPLOAD_SIZE_BYTES
    if file_size_bytes is not None and file_size_bytes > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise DocumentValidationError(
            f"File size ({file_size_bytes} bytes) exceeds the maximum allowed limit of {max_mb} MB."
        )

    return ext


def generate_storage_path(original_filename: str) -> Tuple[str, str]:
    """
    Generate a collision-resistant storage filename and absolute destination path.
    Prevents path traversal, special character injection, and file collisions.
    Returns: (safe_storage_filename, absolute_storage_path)
    """
    ext = Path(original_filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_dir = os.path.abspath(settings.UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)
    full_path = os.path.join(upload_dir, unique_name)
    return unique_name, full_path


def extract_pdf(
    db: Session,
    document: Document,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Extract text chunks from a PDF document using bounded page processing.
    Falls back to OCR if page text length is below OCR_TRIGGER_CHAR_THRESHOLD.
    Applies configurable chunking while preserving 1-based original page numbers.
    """
    if not os.path.isfile(document.file_path):
        raise DocumentProcessingError(f"Document file not found on disk: {document.file_path}")

    # Validate PDF magic header bytes
    with open(document.file_path, "rb") as f:
        header = f.read(5)
        if not header.startswith(b"%PDF-"):
            raise DocumentValidationError("Invalid file: File header does not match PDF specification.")

    chunks: List[Chunk] = []
    doc = None
    try:
        doc = fitz.open(document.file_path)

        if doc.is_encrypted:
            raise DocumentValidationError("Encrypted or password-protected PDFs are not supported.")

        total_pages = len(doc)
        if total_pages > MAX_PDF_PAGES_LIMIT:
            raise DocumentValidationError(
                f"PDF has {total_pages} pages, which exceeds the maximum limit of {MAX_PDF_PAGES_LIMIT} pages."
            )

        for page_index, page in enumerate(doc):
            page_number = page_index + 1  # 1-based page number for citation
            text = page.get_text().strip()

            if len(text) < OCR_TRIGGER_CHAR_THRESHOLD:
                try:
                    ocr_text = extract_text_with_ocr(page)
                    if ocr_text:
                        text = ocr_text
                except OCRUnavailableError as ocr_err:
                    if not text:
                        raise OCRUnavailableError(
                            f"Page {page_number} appears to be a scanned image, but OCR is unavailable: {ocr_err}"
                        ) from ocr_err

            if text:
                # Split page text into chunks using configured size and overlap
                page_chunks = chunk_text(
                    text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                for chunk_idx, chunk_content in enumerate(page_chunks):
                    chunk = Chunk(
                        id=uuid.uuid4(),
                        document_id=document.id,
                        page_number=page_number,  # Preserves physical PDF page number
                        chunk_index=chunk_idx,
                        content=chunk_content,
                        char_count=len(chunk_content),
                    )
                    chunks.append(chunk)

        document.page_count = total_pages
        return chunks

    except (DocumentValidationError, OCRUnavailableError):
        raise
    except Exception as e:
        raise DocumentProcessingError(f"Failed to process PDF: {str(e)}") from e
    finally:
        if doc is not None:
            doc.close()


def extract_txt(
    db: Session,
    document: Document,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Extract text chunks from a TXT document using configurable chunk size and overlap.
    Preserves 1-based virtual page numbering for citation.
    """
    if not os.path.isfile(document.file_path):
        raise DocumentProcessingError(f"Document file not found on disk: {document.file_path}")

    chunks: List[Chunk] = []
    try:
        max_bytes = settings.MAX_UPLOAD_SIZE_BYTES
        file_size = os.path.getsize(document.file_path)
        if file_size > max_bytes:
            max_mb = max_bytes // (1024 * 1024)
            raise DocumentValidationError(f"Text file exceeds maximum limit of {max_mb} MB.")

        with open(document.file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Reject binary null bytes
        if "\x00" in content:
            raise DocumentValidationError("Invalid text file: binary data detected.")

        if not content.strip():
            document.page_count = 0
            return []

        # Split text into chunks using configured size and overlap
        txt_chunks = chunk_text(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for i, chunk_text_content in enumerate(txt_chunks):
            page_number = i + 1  # 1-based virtual page number
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=document.id,
                page_number=page_number,
                chunk_index=0,
                content=chunk_text_content,
                char_count=len(chunk_text_content),
            )
            chunks.append(chunk)

        document.page_count = len(txt_chunks)
        return chunks

    except DocumentValidationError:
        raise
    except Exception as e:
        raise DocumentProcessingError(f"Failed to process text file: {str(e)}") from e


def ingest_document(
    db: Session,
    document: Document,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Ingest a Document by extracting text and persisting Chunk rows in the database.
    Replaces any existing chunks for this document.
    Updates document.page_count in the database.
    Does NOT use or clear any global PAGE_INDEX.
    """
    file_type = (document.file_type or "").lower()
    if file_type not in ("pdf", "txt"):
        ext = Path(document.file_path).suffix.lower()
        if ext == ".pdf":
            file_type = "pdf"
        elif ext == ".txt":
            file_type = "txt"
        else:
            raise DocumentValidationError(f"Unsupported document file type: {document.file_type}")

    # Remove previous chunks if re-ingesting
    db.query(Chunk).filter(Chunk.document_id == document.id).delete()

    if file_type == "pdf":
        chunks = extract_pdf(
            db,
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        chunks = extract_txt(
            db,
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if chunks:
        # Generate dense embeddings using local sentence-transformers model
        try:
            texts = [c.content for c in chunks]
            embeddings = generate_embeddings_batch(texts)
            if embeddings and len(embeddings) == len(chunks):
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
        except Exception as e:
            logger.warning(
                "Embedding generation during ingestion failed: %s; saving chunks without vectors.",
                str(e),
            )

        db.add_all(chunks)

    db.commit()
    db.refresh(document)
    return chunks


def process_ingestion_job(db: Session, job: IngestionJob) -> IngestionResult:
    """
    Execute an IngestionJob, tracking status, pages processed, and errors.
    """
    document = db.query(Document).filter(Document.id == job.document_id).first()
    if not document:
        job.status = "failed"
        job.error_message = "Associated document record not found"
        db.commit()
        raise DocumentProcessingError(job.error_message)

    job.status = "processing"
    db.commit()

    try:
        chunks = ingest_document(db, document)
        job.status = "completed"
        job.pages_processed = document.page_count
        job.total_pages = document.page_count
        job.error_message = None
        db.commit()

        return IngestionResult(
            document_id=document.id,
            file_type=document.file_type,
            page_count=document.page_count,
            chunks_created=len(chunks),
        )

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise
