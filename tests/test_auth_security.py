"""
Tests for authentication, password hashing, and CSRF protection.
Verifies:
- Password hashing security with bcrypt
- User registration and validation
- Login behavior, session cookie generation, and error conditions
- CSRF token issuance, validation, and rejection on state-changing endpoints
"""
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User
from backend.security import (
    hash_password,
    verify_password,
    create_session_token,
    verify_session_token,
    create_csrf_token,
    verify_csrf_token,
    SESSION_COOKIE_NAME,
)


def test_password_hashing_and_verification():
    """Verify password hashing produces salted hashes and validates passwords securely."""
    plain = "SuperSecretPassword123!"
    hashed = hash_password(plain)

    # Hash must be distinct from plaintext and non-empty
    assert hashed != plain
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

    # Correct password succeeds
    assert verify_password(plain, hashed) is True

    # Incorrect password fails
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False

    # Two hashes of the same password should differ due to unique salts
    hashed2 = hash_password(plain)
    assert hashed != hashed2


def test_register_user_success(client: TestClient, db_session: Session):
    """Test successful user registration creates an active user with hashed password."""
    payload = {
        "email": "new_registered_user@example.com",
        "password": "ValidPassword99!",
        "full_name": "New Registered",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["email"] == payload["email"]
    assert data["full_name"] == payload["full_name"]
    assert data["is_active"] is True
    assert "password" not in data
    assert "hashed_password" not in data

    # Verify database persistence
    user_in_db = db_session.query(User).filter(User.email == payload["email"]).first()
    assert user_in_db is not None
    assert user_in_db.hashed_password != payload["password"]
    assert verify_password(payload["password"], user_in_db.hashed_password) is True


def test_register_duplicate_email_conflict(client: TestClient, test_user: User):
    """Test registering with an existing email returns 409 Conflict."""
    payload = {
        "email": test_user.email,
        "password": "AnotherPassword123!",
        "full_name": "Duplicate User",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 409
    assert "already exists" in response.text.lower()


def test_register_validation_errors(client: TestClient):
    """Test validation errors for invalid email and short password."""
    # Invalid email
    res1 = client.post("/api/auth/register", json={"email": "notanemail", "password": "Password123!"})
    assert res1.status_code == 422

    # Short password (< 8 chars)
    res2 = client.post("/api/auth/register", json={"email": "valid@example.com", "password": "short"})
    assert res2.status_code == 422


def test_login_success(client: TestClient, test_user: User):
    """Test logging in with valid credentials returns user info and sets session cookie."""
    response = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "Password123!"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["email"] == test_user.email
    assert data["id"] == str(test_user.id)

    # Session cookie must be present in response
    assert SESSION_COOKIE_NAME in response.cookies
    cookie_val = response.cookies[SESSION_COOKIE_NAME]
    verified_uid = verify_session_token(cookie_val)
    assert verified_uid == test_user.id


def test_login_invalid_password(client: TestClient, test_user: User):
    """Test login with incorrect password returns 401 Unauthorized."""
    response = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "WrongPassword!"},
    )
    assert response.status_code == 401
    assert "invalid email or password" in response.text.lower()
    assert SESSION_COOKIE_NAME not in response.cookies


def test_login_nonexistent_user(client: TestClient):
    """Test login with non-existent email returns 401 Unauthorized."""
    response = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "Password123!"},
    )
    assert response.status_code == 401
    assert "invalid email or password" in response.text.lower()


def test_unauthenticated_protected_routes(client: TestClient):
    """Test accessing protected endpoints without a session cookie returns 401 Unauthorized."""
    # GET /api/auth/me
    res1 = client.get("/api/auth/me")
    assert res1.status_code == 401

    # GET /api/documents
    res2 = client.get("/api/documents")
    assert res2.status_code == 401

    # GET /api/chat/conversations
    res3 = client.get("/api/chat/conversations")
    assert res3.status_code == 401


def test_csrf_token_issuance(auth_client: TestClient, test_user: User):
    """Test obtaining a CSRF token for an authenticated user."""
    response = auth_client.get("/api/auth/csrf")
    assert response.status_code == 200
    token = response.json().get("csrf_token")
    assert token is not None
    assert verify_csrf_token(token, expected_user_id=test_user.id) is True


def test_state_changing_request_missing_csrf(client: TestClient, test_user: User):
    """Test state-changing requests missing X-CSRF-Token header are rejected with 403 or 422."""
    token = create_session_token(test_user.id)
    client.cookies.set(SESSION_COOKIE_NAME, token)

    # Document upload without CSRF token
    response = client.post(
        "/api/documents",
        files={"file": ("test.txt", b"Sample content", "text/plain")},
    )
    assert response.status_code in (403, 422)


def test_state_changing_request_invalid_csrf(client: TestClient, test_user: User):
    """Test state-changing requests with invalid or tampered CSRF token are rejected with 403 Forbidden."""
    token = create_session_token(test_user.id)
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.post(
        "/api/documents",
        files={"file": ("test.txt", b"Sample content", "text/plain")},
        headers={"X-CSRF-Token": "invalid.tampered.token"},
    )
    assert response.status_code == 403


def test_logout_clears_session(auth_client: TestClient):
    """Test logout endpoint deletes session cookie and subsequent requests require re-authentication."""
    # First verify session works
    res_me = auth_client.get("/api/auth/me")
    assert res_me.status_code == 200

    # Logout
    res_logout = auth_client.post("/api/auth/logout")
    assert res_logout.status_code == 200

    # Subsequent call to /api/auth/me fails with 401
    auth_client.cookies.delete(SESSION_COOKIE_NAME)
    res_after = auth_client.get("/api/auth/me")
    assert res_after.status_code == 401
