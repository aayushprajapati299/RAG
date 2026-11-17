"""
Tests for chat router and conversation services.
Verifies:
- Cross-user conversation access is denied (404)
- Cross-user document referencing in messages is denied (404)
- Bounded context and citation extraction (chunk ID and page number)
- Chat completions API error / timeout responses do NOT appear as successful answers
- Persisted messages exist only upon successful generation
- Streaming SSE sequence and error handling
"""
import json
import uuid
import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Document, Chunk, Conversation, Message
from backend.services.chat import ProviderAPIError


@pytest.fixture
def user_a_document(db_session: Session, test_user: User) -> Document:
    """Create a sample document and chunk for User A."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="mars_atmosphere.txt",
        file_path="/tmp/mars_atmosphere.txt",
        file_type="txt",
        page_count=1,
    )
    db_session.add(doc)
    db_session.flush()

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        page_number=1,
        chunk_index=0,
        content="The atmosphere of Mars consists of 95 percent carbon dioxide, 2.6 percent nitrogen, and 1.9 percent argon.",
        char_count=107,
    )
    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(doc)
    return doc


def test_create_and_list_conversations(auth_client: TestClient, test_user: User):
    """Test creating a conversation and listing current user's conversations."""
    create_res = auth_client.post(
        "/api/chat/conversations",
        json={"title": "Rover Exploration Chat"},
    )
    assert create_res.status_code == 201
    conv_data = create_res.json()
    assert conv_data["title"] == "Rover Exploration Chat"
    assert conv_data["user_id"] == str(test_user.id)
    conv_id = conv_data["id"]

    list_res = auth_client.get("/api/chat/conversations")
    assert list_res.status_code == 200
    convs = list_res.json()
    assert any(c["id"] == conv_id for c in convs)


def test_cross_user_conversation_access_denied(
    auth_client: TestClient,
    second_auth_client: TestClient,
    user_a_document: Document,
):
    """Test that User B cannot view or post messages to User A's conversation."""
    # User A creates a conversation
    create_res = auth_client.post(
        "/api/chat/conversations",
        json={"title": "User A Private Chat"},
    )
    assert create_res.status_code == 201
    conv_id = create_res.json()["id"]

    # User B attempts to fetch User A's conversation detail -> 404
    b_get_res = second_auth_client.get(f"/api/chat/conversations/{conv_id}")
    assert b_get_res.status_code == 404
    assert "not found" in b_get_res.text.lower()

    # User B attempts to post message to User A's conversation -> 404
    b_post_res = second_auth_client.post(
        f"/api/chat/conversations/{conv_id}/messages",
        json={
            "document_id": str(user_a_document.id),
            "content": "What is on Mars?",
        },
    )
    assert b_post_res.status_code == 404


def test_cross_user_document_in_message_denied(
    auth_client: TestClient,
    second_auth_client: TestClient,
    user_a_document: Document,
):
    """Test that User B cannot query User A's document in their own conversation."""
    # User B creates their own conversation
    b_conv_res = second_auth_client.post(
        "/api/chat/conversations",
        json={"title": "User B Chat"},
    )
    assert b_conv_res.status_code == 201
    b_conv_id = b_conv_res.json()["id"]

    # User B attempts to send message referencing User A's document -> 404
    b_msg_res = second_auth_client.post(
        f"/api/chat/conversations/{b_conv_id}/messages",
        json={
            "document_id": str(user_a_document.id),
            "content": "Tell me about the atmosphere.",
        },
    )
    assert b_msg_res.status_code == 404
    assert "document not found" in b_msg_res.text.lower()


def test_chat_successful_answer_with_citations(
    auth_client: TestClient,
    user_a_document: Document,
    db_session: Session,
    monkeypatch,
):
    """Test successful chat message returns answer and provenance citations with chunk ID and page number."""
    # Mock Groq completions to return a clean assistant answer
    mock_answer = "Mars's atmosphere is primarily carbon dioxide (95%) with nitrogen and argon."

    def _mock_call_groq(messages, **kwargs):
        return mock_answer

    import backend.services.chat as chat_module
    monkeypatch.setattr(chat_module, "call_groq_completions", _mock_call_groq)

    # Create conversation
    conv_res = auth_client.post("/api/chat/conversations", json={"title": "Mars Atmosphere"})
    conv_id = conv_res.json()["id"]

    # Post message
    msg_res = auth_client.post(
        f"/api/chat/conversations/{conv_id}/messages",
        json={
            "document_id": str(user_a_document.id),
            "content": "What are the components of the Martian atmosphere?",
        },
    )
    assert msg_res.status_code == 200

    data = msg_res.json()
    assert data["conversation_id"] == conv_id
    assert data["answer"] == mock_answer
    assert len(data["citations"]) >= 1

    first_citation = data["citations"][0]
    assert "chunk_id" in first_citation
    assert first_citation["page_number"] == 1

    # Verify messages persisted in database
    messages = (
        db_session.query(Message)
        .filter(Message.conversation_id == uuid.UUID(conv_id))
        .order_by(Message.created_at.asc())
        .all()
    )
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert "components of the Martian atmosphere" in messages[0].content
    assert messages[1].role == "assistant"
    assert messages[1].content == mock_answer
    assert messages[1].sources is not None


def test_groq_timeout_returns_structured_error_not_200(
    auth_client: TestClient,
    user_a_document: Document,
    db_session: Session,
    monkeypatch,
):
    """
    Test that a Groq provider timeout returns a 504 status code with structured error details,
    never returns an exception as a 200 answer, and never persists a corrupt assistant message.
    """
    # Mock call_groq_completions to raise ProviderAPIError (504 timeout)
    def _mock_timeout(*args, **kwargs):
        raise ProviderAPIError(
            message="LLM provider request timed out after 30.0 seconds.",
            status_code=504,
            details={"provider": "groq", "timeout": 30.0},
        )

    import backend.services.chat as chat_module
    monkeypatch.setattr(chat_module, "call_groq_completions", _mock_timeout)

    conv_res = auth_client.post("/api/chat/conversations", json={"title": "Timeout Test"})
    conv_id = conv_res.json()["id"]

    # Send message
    res = auth_client.post(
        f"/api/chat/conversations/{conv_id}/messages",
        json={
            "document_id": str(user_a_document.id),
            "content": "Will this request time out?",
        },
    )

    # Must NOT be 200 OK
    assert res.status_code == 504

    # Structured error payload
    data = res.json()
    assert "detail" in data
    assert data["detail"]["error"] == "ProviderError"
    assert "timed out" in data["detail"]["message"].lower()

    # Verify NO assistant message was persisted in database
    assistant_msgs = (
        db_session.query(Message)
        .filter(Message.conversation_id == uuid.UUID(conv_id), Message.role == "assistant")
        .all()
    )
    assert len(assistant_msgs) == 0


def test_groq_provider_failure_returns_structured_error(
    auth_client: TestClient,
    user_a_document: Document,
    monkeypatch,
):
    """Test that upstream LLM 502/503 errors return structured non-2xx API errors."""
    def _mock_failure(*args, **kwargs):
        raise ProviderAPIError(
            message="LLM provider rate limit exceeded.",
            status_code=503,
            details={"provider": "groq", "reason": "rate_limit_exceeded"},
        )

    import backend.services.chat as chat_module
    monkeypatch.setattr(chat_module, "call_groq_completions", _mock_failure)

    conv_res = auth_client.post("/api/chat/conversations", json={"title": "Failure Test"})
    conv_id = conv_res.json()["id"]

    res = auth_client.post(
        f"/api/chat/conversations/{conv_id}/messages",
        json={
            "document_id": str(user_a_document.id),
            "content": "Test rate limit error handling",
        },
    )

    assert res.status_code == 503
    data = res.json()
    assert data["detail"]["error"] == "ProviderError"
    assert "rate limit" in data["detail"]["message"].lower()


def test_streaming_chat_timeout_emits_error_event(
    auth_client: TestClient,
    user_a_document: Document,
    db_session: Session,
    monkeypatch,
):
    """Test that streaming SSE emits event: error when Groq times out, and does not persist assistant message."""
    def _mock_stream_timeout(*args, **kwargs):
        yield "Initial token "
        raise ProviderAPIError(
            message="Stream request timed out.",
            status_code=504,
            details={"provider": "groq", "timeout": 30.0},
        )

    import backend.services.chat as chat_module
    monkeypatch.setattr(chat_module, "stream_groq_completions", _mock_stream_timeout)

    conv_res = auth_client.post("/api/chat/conversations", json={"title": "Stream Timeout"})
    conv_id = conv_res.json()["id"]

    stream_res = auth_client.post(
        f"/api/chat/conversations/{conv_id}/messages/stream",
        json={
            "document_id": str(user_a_document.id),
            "content": "Stream question about Mars",
        },
    )

    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers["content-type"]

    body = stream_res.text
    assert "event: text" in body
    assert "event: error" in body
    assert "timed out" in body.lower()
    # Must NOT have complete event
    assert "event: complete" not in body

    # Assistant message must NOT be saved in DB
    assistant_msgs = (
        db_session.query(Message)
        .filter(Message.conversation_id == uuid.UUID(conv_id), Message.role == "assistant")
        .all()
    )
    assert len(assistant_msgs) == 0


def test_delete_conversation_cascades_messages_preserves_document(
    auth_client: TestClient,
    db_session: Session,
    user_a_document: Document,
):
    """Deleting a conversation removes its messages but preserves the underlying document."""
    # Create conversation with document attached
    conv_res = auth_client.post(
        "/api/chat/conversations",
        json={"title": "To Be Deleted", "document_ids": [str(user_a_document.id)]},
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    # Post message
    msg = Message(
        id=uuid.uuid4(),
        conversation_id=uuid.UUID(conv_id),
        role="user",
        content="Hello Mars",
    )
    db_session.add(msg)
    db_session.commit()

    # Delete conversation
    del_res = auth_client.delete(f"/api/chat/conversations/{conv_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "ok"

    # Verify conversation is gone
    get_res = auth_client.get(f"/api/chat/conversations/{conv_id}")
    assert get_res.status_code == 404

    # Verify message is cascaded
    deleted_msgs = db_session.query(Message).filter(Message.conversation_id == uuid.UUID(conv_id)).all()
    assert len(deleted_msgs) == 0

    # Verify document is PRESERVED
    doc = db_session.query(Document).filter(Document.id == user_a_document.id).first()
    assert doc is not None


def test_cross_user_conversation_deletion_denied(
    auth_client: TestClient,
    second_auth_client: TestClient,
):
    """User B cannot delete User A's conversation."""
    conv_res = auth_client.post("/api/chat/conversations", json={"title": "User A Chat"})
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    del_res = second_auth_client.delete(f"/api/chat/conversations/{conv_id}")
    assert del_res.status_code == 404


def test_conversation_document_association_and_scoping(
    auth_client: TestClient,
    test_user: User,
    db_session: Session,
):
    """Each conversation maintains its own associated documents and selection independently."""
    doc1 = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="doc1.txt",
        file_path="/tmp/doc1.txt",
        file_type="txt",
        page_count=1,
    )
    doc2 = Document(
        id=uuid.uuid4(),
        user_id=test_user.id,
        filename="doc2.txt",
        file_path="/tmp/doc2.txt",
        file_type="txt",
        page_count=1,
    )
    db_session.add_all([doc1, doc2])
    db_session.commit()

    # Create Chat 1 with doc1 selected
    c1_res = auth_client.post(
        "/api/chat/conversations",
        json={"title": "Chat 1", "document_ids": [str(doc1.id)], "selected_document_id": str(doc1.id)},
    )
    assert c1_res.status_code == 201
    c1_id = c1_res.json()["id"]

    # Create Chat 2 with doc2 selected
    c2_res = auth_client.post(
        "/api/chat/conversations",
        json={"title": "Chat 2", "document_ids": [str(doc2.id)], "selected_document_id": str(doc2.id)},
    )
    assert c2_res.status_code == 201
    c2_id = c2_res.json()["id"]

    # Verify Chat 1 state
    c1_detail = auth_client.get(f"/api/chat/conversations/{c1_id}").json()
    assert c1_detail["selected_document_id"] == str(doc1.id)
    assert any(d["id"] == str(doc1.id) for d in c1_detail["documents"])
    assert not any(d["id"] == str(doc2.id) for d in c1_detail["documents"])

    # Verify Chat 2 state
    c2_detail = auth_client.get(f"/api/chat/conversations/{c2_id}").json()
    assert c2_detail["selected_document_id"] == str(doc2.id)
    assert any(d["id"] == str(doc2.id) for d in c2_detail["documents"])
    assert not any(d["id"] == str(doc1.id) for d in c2_detail["documents"])


def test_groq_model_configuration_default_and_env_override(monkeypatch):
    """Test that Groq model defaults to openai/gpt-oss-20b and honors env override."""
    from backend.config import Settings

    # Default without custom GROQ_MODEL in env
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    default_settings = Settings()
    assert default_settings.GROQ_MODEL == "openai/gpt-oss-20b"

    # Environment override
    monkeypatch.setenv("GROQ_MODEL", "custom/test-model-32b")
    custom_settings = Settings()
    assert custom_settings.GROQ_MODEL == "custom/test-model-32b"


def test_chat_client_sends_configured_groq_model(
    auth_client: TestClient,
    user_a_document: Document,
    monkeypatch,
):
    """Test that chat service sends the configured GROQ_MODEL in the payload."""
    import backend.services.chat as chat_module
    from backend.config import settings

    captured_requests = []

    def _mock_requests_post(url, headers=None, json=None, **kwargs):
        captured_requests.append({"url": url, "json": json})
        class _MockResponse:
            status_code = 200
            def json(self):
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "Mocked answer using configured model."}}
                    ]
                }
            @property
            def text(self):
                return "Mocked text"
        return _MockResponse()

    monkeypatch.setattr(chat_module.requests, "post", _mock_requests_post)
    monkeypatch.setattr(chat_module.settings, "GROQ_API_KEY", "gsk_test_mock_key")

    conv_res = auth_client.post("/api/chat/conversations", json={"title": "Model Test"})
    conv_id = conv_res.json()["id"]

    res = auth_client.post(
        f"/api/chat/conversations/{conv_id}/messages",
        json={
            "document_id": str(user_a_document.id),
            "content": "What model are you using?",
        },
    )
    assert res.status_code == 200
    assert len(captured_requests) == 1
    assert captured_requests[0]["json"]["model"] == settings.GROQ_MODEL
    assert captured_requests[0]["json"]["model"] == "openai/gpt-oss-20b"


