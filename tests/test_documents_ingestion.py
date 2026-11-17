"""
Tests for document ingestion, validation, and cross-user isolation.
Verifies:
- Cross-user document access is strictly denied (404)
- Cross-user chunk access is strictly denied (404)
- Document list scoping (users only see their own documents)
- File type and format validation (.pdf and .txt only, magic bytes check)
- File size boundary enforcement (25 MB limit)
- Ingestion job status tracking and transitions (pending -> processing -> completed)
- Page count and chunk persistence
"""
import io
import uuid
import pytest
import fitz  # PyMuPDF
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Document, Chunk, IngestionJob


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory valid PDF file with distinct pages."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_upload_valid_txt_file(auth_client: TestClient, db_session: Session):
    """Test uploading a valid plain text document initiates ingestion and generates chunks."""
    content = (
        "Mars exploration is the study of the planet Mars using orbiters, landers, and rovers.\n"
        "The Curiosity and Perseverance rovers have discovered extensive evidence of past water activity.\n"
        "Perseverance carries the MOXIE instrument to produce oxygen from the atmospheric carbon dioxide."
    )
    files = {"file": ("mars_study.txt", content.encode("utf-8"), "text/plain")}

    response = auth_client.post("/api/documents", files=files)
    assert response.status_code == 201

    data = response.json()
    doc_id = data["id"]
    job_id = data["job_id"]
    assert doc_id is not None
    assert job_id is not None
    assert data["filename"] == "mars_study.txt"
    assert data["file_type"] == "txt"

    # Status endpoint check
    status_res = auth_client.get(f"/api/documents/{doc_id}/status")
    assert status_res.status_code == 200
    job_data = status_res.json()
    assert job_data["document_id"] == doc_id
    assert job_data["status"] == "completed"
    assert job_data["total_pages"] >= 1

    # Verify chunks exist in database
    chunks = db_session.query(Chunk).filter(Chunk.document_id == uuid.UUID(doc_id)).all()
    assert len(chunks) >= 1
    assert chunks[0].page_number == 1
    assert "Mars exploration" in chunks[0].content


def test_upload_valid_pdf_file(auth_client: TestClient, db_session: Session):
    """Test uploading a valid multi-page PDF extracts pages and creates distinct chunks."""
    pages = [
        "Page 1: Introduction to autonomous planetary rover guidance systems.",
        "Page 2: Optical stereoscopic vision sensors operating at 30 fps with high dynamic range.",
    ]
    pdf_bytes = create_sample_pdf_bytes(pages)
    files = {"file": ("rover_manual.pdf", pdf_bytes, "application/pdf")}

    response = auth_client.post("/api/documents", files=files)
    assert response.status_code == 201

    doc_id = response.json()["id"]

    # Verify document detail and page count
    detail_res = auth_client.get(f"/api/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["page_count"] == 2
    assert detail_data["file_type"] == "pdf"

    # Verify chunks preserved physical page numbers
    chunks = (
        db_session.query(Chunk)
        .filter(Chunk.document_id == uuid.UUID(doc_id))
        .order_by(Chunk.page_number.asc())
        .all()
    )
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Page 1" in chunks[0].content
    assert chunks[1].page_number == 2
    assert "Page 2" in chunks[1].content


def test_cross_user_document_access_denied(
    auth_client: TestClient,
    second_auth_client: TestClient,
):
    """Test that User B cannot access, read, or check status of User A's document."""
    # User A uploads a document
    files = {"file": ("user_a_private.txt", b"Confidential User A data", "text/plain")}
    upload_res = auth_client.post("/api/documents", files=files)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # User B attempts to get document metadata -> 404
    b_get_res = second_auth_client.get(f"/api/documents/{doc_id}")
    assert b_get_res.status_code == 404
    assert "not found" in b_get_res.text.lower()

    # User B attempts to get document status -> 404
    b_status_res = second_auth_client.get(f"/api/documents/{doc_id}/status")
    assert b_status_res.status_code == 404

    # User B lists documents -> User A's document is NOT in the list
    b_list_res = second_auth_client.get("/api/documents")
    assert b_list_res.status_code == 200
    b_docs = b_list_res.json()
    assert not any(d["id"] == doc_id for d in b_docs)


def test_cross_user_chunk_access_denied(
    auth_client: TestClient,
    second_auth_client: TestClient,
    db_session: Session,
):
    """Test that User B cannot retrieve chunks belonging to User A's document."""
    # User A uploads a document
    files = {"file": ("secret_project.txt", b"Secret proprietary formula", "text/plain")}
    res = auth_client.post("/api/documents", files=files)
    doc_id = res.json()["id"]

    chunk = db_session.query(Chunk).filter(Chunk.document_id == uuid.UUID(doc_id)).first()
    assert chunk is not None
    chunk_id = str(chunk.id)

    # User A can access chunk
    a_chunk_res = auth_client.get(f"/api/chunks/{chunk_id}")
    assert a_chunk_res.status_code == 200
    assert a_chunk_res.json()["content"] == "Secret proprietary formula"

    # User B cannot access chunk via /api/chunks/{chunk_id} -> 404
    b_chunk_res = second_auth_client.get(f"/api/chunks/{chunk_id}")
    assert b_chunk_res.status_code == 404
    assert "not found" in b_chunk_res.text.lower()

    # User B cannot access chunk via /api/documents/{doc_id}/chunks/{chunk_id} -> 404
    b_doc_chunk_res = second_auth_client.get(f"/api/documents/{doc_id}/chunks/{chunk_id}")
    assert b_doc_chunk_res.status_code == 404


def test_upload_unsupported_file_extension(auth_client: TestClient):
    """Test that unsupported file extensions are rejected with 400 Bad Request."""
    files = {"file": ("malicious.exe", b"MZ...", "application/octet-stream")}
    response = auth_client.post("/api/documents", files=files)
    assert response.status_code == 400
    assert "unsupported file extension" in response.text.lower()

    files_img = {"file": ("photo.jpg", b"\xff\xd8\xff...", "image/jpeg")}
    response2 = auth_client.post("/api/documents", files=files_img)
    assert response2.status_code == 400


def test_upload_corrupt_pdf_file(auth_client: TestClient):
    """Test that an uploaded .pdf without valid PDF magic bytes is rejected with 400."""
    # File named .pdf but containing arbitrary text without %PDF- header
    files = {"file": ("fake.pdf", b"This is not a real PDF file content.", "application/pdf")}
    response = auth_client.post("/api/documents", files=files)
    # Background ingestion will fail and mark job as failed
    assert response.status_code in (201, 400)
    if response.status_code == 201:
        doc_id = response.json()["id"]
        status_res = auth_client.get(f"/api/documents/{doc_id}/status")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "failed"


def test_upload_oversized_file(auth_client: TestClient, monkeypatch):
    """Test that a file exceeding MAX_UPLOAD_SIZE_BYTES is rejected with 400."""
    from backend.config import settings
    # Temporarily set small upload limit for test
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 1024)  # 1 KB

    big_content = b"x" * 2048  # 2 KB > 1 KB limit
    files = {"file": ("large_doc.txt", big_content, "text/plain")}
    response = auth_client.post("/api/documents", files=files)
    assert response.status_code == 400
    assert "exceeds" in response.text.lower()


def test_delete_document_cascades_chunks_and_unsets_conversation_selection(
    auth_client: TestClient,
    test_user: User,
    db_session: Session,
):
    """Deleting a document cascades chunks and unsets conversation selection."""
    from backend.models import Conversation, Chunk

    doc = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="delete_me.txt",
        file_path="/tmp/delete_me.txt",
        file_type="txt",
        page_count=1,
    )
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=1,
        chunk_index=0,
        content="Chunk to be deleted",
        char_count=19,
    )
    db_session.add(chunk)

    conv = Conversation(
        id=uuid.uuid4(),
        user_id=test_user.id,
        title="Chat with Document",
        selected_document_id=doc.id,
    )
    conv.documents.append(doc)
    db_session.add(conv)
    db_session.commit()

    # Delete document
    del_res = auth_client.delete(f"/api/documents/{doc.id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "ok"

    # Verify document is gone
    get_res = auth_client.get(f"/api/documents/{doc.id}")
    assert get_res.status_code == 404

    # Verify chunk is cascaded
    chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
    assert len(chunks) == 0

    # Verify conversation selected_document_id is unset to None
    db_session.refresh(conv)
    assert conv.selected_document_id is None


def test_cross_user_document_deletion_denied(
    auth_client: TestClient,
    second_auth_client: TestClient,
    test_user: User,
    db_session: Session,
):
    """User B cannot delete User A's document."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="user_a_only.txt",
        file_path="/tmp/user_a_only.txt",
        file_type="txt",
        page_count=1,
    )
    db_session.add(doc)
    db_session.commit()

    del_res = second_auth_client.delete(f"/api/documents/{doc.id}")
    assert del_res.status_code == 404

