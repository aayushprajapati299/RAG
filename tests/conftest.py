"""
Pytest configuration and minimum required fixtures for testing the FastAPI application.
Configures a dedicated test database, overrides dependencies, and mocks external APIs.
"""
import os
import uuid
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from backend.config import settings
from backend.database import Base, get_db
import backend.database as db_module
import backend.routers.documents as docs_router_module
import backend.services.chat as chat_service_module
from backend.main import app
from backend.models import User
from backend.security import hash_password, create_session_token, create_csrf_token, SESSION_COOKIE_NAME

# Dedicated test database URL
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/rag_test_db",
)

test_engine = create_engine(
    TEST_DB_URL,
    pool_pre_ping=True,
)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Create all tables in the test database once per test session."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(autouse=True)
def clean_database_tables():
    """Ensure every test runs against clean, empty tables."""
    with test_engine.connect() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE messages, conversation_documents, conversations, chunks, ingestion_jobs, documents, users CASCADE"
            )
        )
        conn.commit()
    yield
    with test_engine.connect() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE messages, conversation_documents, conversations, chunks, ingestion_jobs, documents, users CASCADE"
            )
        )
        conn.commit()


@pytest.fixture(autouse=True)
def override_db_session(monkeypatch):
    """
    Ensure all application components (dependency injection, background tasks,
    and services) use the test database session.
    """
    # Monkeypatch SessionLocal references across modules
    monkeypatch.setattr(db_module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(docs_router_module, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(chat_service_module, "SessionLocal", TestingSessionLocal)

    def _override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a clean test database session for individual tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def mock_embedding_models(monkeypatch):
    """
    Mock sentence-transformers embedding functions so tests never call
    external or paid APIs and run instantaneously.
    """
    def _mock_generate_embedding(text: str):
        # Deterministic 384-dimensional unit-like vector
        val = 0.05 if text else 0.0
        return [val] * 384

    def _mock_generate_embeddings_batch(texts):
        return [[0.05] * 384 for _ in texts]

    import backend.services.embeddings as emb_module
    import backend.services.retrieval as ret_module
    import backend.services.document_ingestion as ing_module

    monkeypatch.setattr(emb_module, "generate_embedding", _mock_generate_embedding)
    monkeypatch.setattr(emb_module, "generate_embeddings_batch", _mock_generate_embeddings_batch)
    monkeypatch.setattr(ret_module, "generate_embedding", _mock_generate_embedding)
    monkeypatch.setattr(ing_module, "generate_embeddings_batch", _mock_generate_embeddings_batch)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Unauthenticated TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Create and persist a primary test user."""
    user = User(
        id=uuid.uuid4(),
        email="primary_user@example.com",
        hashed_password=hash_password("Password123!"),
        full_name="Primary Tester",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def second_user(db_session: Session) -> User:
    """Create and persist a second test user for cross-user isolation tests."""
    user = User(
        id=uuid.uuid4(),
        email="second_user@example.com",
        hashed_password=hash_password("Password456!"),
        full_name="Second Tester",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_client(client: TestClient, test_user: User) -> TestClient:
    """TestClient authenticated as test_user with session cookie and CSRF token."""
    token = create_session_token(test_user.id)
    client.cookies.set(SESSION_COOKIE_NAME, token)
    csrf_token = create_csrf_token(test_user.id)
    client.headers.update({"X-CSRF-Token": csrf_token})
    return client


@pytest.fixture
def second_auth_client(test_user: User, second_user: User) -> TestClient:
    """TestClient authenticated as second_user with session cookie and CSRF token."""
    with TestClient(app) as c:
        token = create_session_token(second_user.id)
        c.cookies.set(SESSION_COOKIE_NAME, token)
        csrf_token = create_csrf_token(second_user.id)
        c.headers.update({"X-CSRF-Token": csrf_token})
        yield c
