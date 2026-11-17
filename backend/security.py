"""
Security utilities for password hashing, signed session cookies, and CSRF protection.

Session Mechanism:
- Sessions are stored in an HttpOnly, SameSite=Lax signed cookie.
- The cookie payload contains the user's UUID and an expiration timestamp.
- The payload is signed with an HMAC-SHA256 signature generated using the application's SECRET_KEY.
- In production (ENVIRONMENT=production), cookies have the 'secure=True' flag set.

CSRF Protection Mechanism:
- An endpoint (/api/auth/csrf) provides signed, time-limited CSRF tokens.
- State-changing authenticated requests must supply the token in the 'X-CSRF-Token' header.
- The token is verified against the authenticated user and application SECRET_KEY.
"""
import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Optional
from uuid import UUID

import bcrypt
from fastapi import Depends, Header, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models import User

# Cookie and token lifetime constants
SESSION_COOKIE_NAME = "session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days
CSRF_MAX_AGE_SECONDS = 24 * 3600  # 24 hours


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash a plaintext password with bcrypt using a generated salt.
    Never stores or logs plaintext passwords.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.
    Safely returns False on malformed hashes or comparison errors.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Signed Session Token & Cookie Helpers
# ---------------------------------------------------------------------------

def create_session_token(user_id: UUID, expires_in_seconds: int = SESSION_MAX_AGE_SECONDS) -> str:
    """
    Create a signed session token containing the user ID and expiration time.
    Format: <base64url_encoded_json_payload>.<hmac_sha256_hex_signature>
    """
    exp = int(time.time()) + expires_in_seconds
    payload = {
        "sub": str(user_id),
        "exp": exp,
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    sig = hmac.new(settings.SECRET_KEY.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def verify_session_token(token: str) -> Optional[UUID]:
    """
    Validate a signed session token. Returns the user UUID if valid and unexpired;
    otherwise returns None.
    """
    try:
        if not token or "." not in token:
            return None

        raw, sig = token.split(".", 1)
        expected_sig = hmac.new(
            settings.SECRET_KEY.encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, expected_sig):
            return None

        payload_bytes = base64.urlsafe_b64decode(raw.encode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))

        if payload.get("exp", 0) < time.time():
            return None

        return UUID(payload["sub"])
    except Exception:
        return None


def set_session_cookie(response: Response, token: str) -> None:
    """
    Attach an HttpOnly, SameSite=Lax signed session cookie to the response.
    Production environment enforces the Secure attribute.
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Delete the session cookie by expiring it."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
        path="/",
    )


# ---------------------------------------------------------------------------
# CSRF Protection Helpers
# ---------------------------------------------------------------------------

def create_csrf_token(user_id: Optional[UUID] = None) -> str:
    """
    Generate a signed CSRF token optionally bound to an authenticated user ID.
    Format: <base64url_encoded_payload>.<hmac_sha256_hex_signature>
    """
    payload = {
        "uid": str(user_id) if user_id else None,
        "nonce": secrets.token_hex(8),
        "exp": int(time.time()) + CSRF_MAX_AGE_SECONDS,
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    sig = hmac.new(settings.SECRET_KEY.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def verify_csrf_token(token: str, expected_user_id: Optional[UUID] = None) -> bool:
    """
    Verify the signature, expiration, and optional user binding of a CSRF token.
    """
    try:
        if not token or "." not in token:
            return False

        raw, sig = token.split(".", 1)
        expected_sig = hmac.new(
            settings.SECRET_KEY.encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, expected_sig):
            return False

        payload_bytes = base64.urlsafe_b64decode(raw.encode("utf-8"))
        payload = json.loads(payload_bytes.decode("utf-8"))

        if payload.get("exp", 0) < time.time():
            return False

        if expected_user_id is not None:
            token_uid = payload.get("uid")
            if token_uid != str(expected_user_id):
                return False

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FastAPI Route Dependencies
# ---------------------------------------------------------------------------

def get_optional_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Retrieve the authenticated User from the session cookie, or None if unauthenticated."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None

    user_id = verify_session_token(token)
    if not user_id:
        return None

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    return user


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """
    Retrieve the active authenticated User from the session cookie.
    Raises HTTP 401 if unauthenticated or user is inactive.
    """
    user = get_optional_current_user(request=request, db=db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


def require_csrf(
    x_csrf_token: str = Header(
        ...,
        alias="X-CSRF-Token",
        description="CSRF token for state-changing requests",
    ),
    user: User = Depends(get_current_user),
) -> User:
    """
    Dependency for state-changing authenticated routes.
    Requires and verifies the X-CSRF-Token header matching the current user session.
    Returns the authenticated User.
    """
    if not x_csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required CSRF token in X-CSRF-Token header",
        )

    if not verify_csrf_token(x_csrf_token, expected_user_id=user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired CSRF token",
        )

    return user
