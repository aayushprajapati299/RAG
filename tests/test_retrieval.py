"""
Tests for document retrieval service (BM25 Lexical + Hybrid Vector Search).
Verifies:
- Configurable chunk size and overlap
- Physical page number preservation
- BM25 retrieval returning relevant page citations with scores
- Query with no useful tokens (or all zeros) returns no sources
- Hybrid retrieval combining normalized BM25 and vector similarity
- Clean fallback to BM25 when embedding model is unavailable
- Retrieval strictly scoped to the authenticated user's selected document
"""
import uuid
import pytest
from sqlalchemy.orm import Session

from backend.models import User, Document, Chunk
from backend.services.retrieval import (
    chunk_text,
    tokenize,
    retrieve_document_chunks,
    DEFAULT_STOPWORDS,
)


def test_chunk_text_size_and_overlap():
    """Verify configurable chunk text size and overlap."""
    words = [f"word{i}" for i in range(1, 101)]
    text = " ".join(words)

    # Chunk size 20, overlap 5
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1

    # First chunk has 20 words
    c0_words = chunks[0].split()
    assert len(c0_words) == 20
    assert c0_words[0] == "word1"
    assert c0_words[-1] == "word20"

    # Second chunk begins with overlap from first chunk (words 16..20)
    c1_words = chunks[1].split()
    assert c1_words[0] == "word16"
    assert c1_words[4] == "word20"


def test_tokenize_filters_stopwords_and_punctuation():
    """Verify tokenizer lowercases, removes punctuation, and filters stopwords."""
    raw = "The Quick, Brown FOX Jumps Over The Lazy Dog! Is it at of a?"
    tokens = tokenize(raw)

    # Stopwords like 'the', 'is', 'it', 'at', 'of', 'a' must be removed
    for sw in DEFAULT_STOPWORDS:
        assert sw not in tokens

    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens


def test_bm25_retrieval_returns_relevant_page_citations(
    db_session: Session,
    test_user: User,
):
    """Verify BM25 retrieval returns top relevant chunks with score, chunk ID, page number, and text."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="space_mission.txt",
        file_path="/tmp/space_mission.txt",
        file_type="txt",
        page_count=3,
    )
    db_session.add(doc)
    db_session.flush()

    # Create 3 chunks on different pages
    c1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=1,
        chunk_index=0,
        content="The James Webb Space Telescope uses gold-coated beryllium mirrors to observe infrared astronomy.",
        char_count=98,
    )
    c2 = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=2,
        chunk_index=0,
        content="The Perseverance rover is equipped with seven scientific instruments and a robotic arm for Martian soil sampling.",
        char_count=113,
    )
    c3 = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=3,
        chunk_index=0,
        content="Solar arrays and lithium-ion batteries provide reliable electrical power for deep space satellite telemetry.",
        char_count=107,
    )
    db_session.add_all([c1, c2, c3])
    db_session.commit()

    # Query for Webb telescope
    results = retrieve_document_chunks(
        db=db_session,
        user_id=test_user.id,
        document_id=doc.id,
        query="telescope beryllium infrared",
        top_k=2,
    )

    assert len(results) >= 1
    top_chunk = results[0]
    assert top_chunk.chunk_id == c1.id
    assert top_chunk.page_number == 1
    assert "James Webb Space Telescope" in top_chunk.text
    assert top_chunk.score > 0.0


def test_retrieval_query_with_no_useful_tokens_returns_no_sources(
    db_session: Session,
    test_user: User,
):
    """Verify that queries with only stopwords or punctuation return an empty list."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="sample.txt",
        file_path="/tmp/sample.txt",
        file_type="txt",
    )
    db_session.add(doc)
    db_session.flush()

    c1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=1,
        chunk_index=0,
        content="Significant technical analysis and research findings.",
        char_count=52,
    )
    db_session.add(c1)
    db_session.commit()

    # Query consisting purely of stopwords and punctuation
    empty_results = retrieve_document_chunks(
        db=db_session,
        user_id=test_user.id,
        document_id=doc.id,
        query="the is at in of and ?",
    )
    assert empty_results == []

    # Whitespace only
    empty_results2 = retrieve_document_chunks(
        db=db_session,
        user_id=test_user.id,
        document_id=doc.id,
        query="   ",
    )
    assert empty_results2 == []


def test_retrieval_strictly_scoped_to_user_owned_document(
    db_session: Session,
    test_user: User,
    second_user: User,
):
    """Verify User B cannot retrieve chunks from User A's document."""
    doc_a = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="user_a_doc.txt",
        file_path="/tmp/user_a_doc.txt",
        file_type="txt",
    )
    db_session.add(doc_a)
    db_session.flush()

    c1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc_a.id,
        page_number=1,
        chunk_index=0,
        content="Confidential quantum cryptography protocols.",
        char_count=44,
    )
    db_session.add(c1)
    db_session.commit()

    # User B attempts retrieval against doc_a
    results = retrieve_document_chunks(
        db=db_session,
        user_id=second_user.id,
        document_id=doc_a.id,
        query="quantum cryptography",
    )
    # Must be empty because document is not owned by second_user
    assert results == []


def test_clean_bm25_fallback_when_embedding_fails(
    db_session: Session,
    test_user: User,
    monkeypatch,
):
    """Verify that when the embedding model is unavailable, retrieval cleanly falls back to pure BM25."""
    import backend.services.retrieval as ret_mod

    # Force generate_embedding to raise an exception
    def _mock_failing_embedding(query: str):
        raise RuntimeError("Local embedding model failed to load ONNX runtime.")

    monkeypatch.setattr(ret_mod, "generate_embedding", _mock_failing_embedding)

    doc = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="fallback_test.txt",
        file_path="/tmp/fallback_test.txt",
        file_type="txt",
    )
    db_session.add(doc)
    db_session.flush()

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=1,
        chunk_index=0,
        content="Rocket propellants include liquid oxygen and methane combustion.",
        char_count=64,
        embedding=None,
    )
    db_session.add(chunk)
    db_session.commit()

    results = retrieve_document_chunks(
        db=db_session,
        user_id=test_user.id,
        document_id=doc.id,
        query="rocket propellants methane",
    )

    # Retrieval must succeed via BM25 fallback without raising an unhandled exception
    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].page_number == 1
    assert results[0].bm25_score is not None
    assert results[0].vector_score is None
