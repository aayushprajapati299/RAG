from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas.auth import (
    CsrfTokenResponse,
    MessageResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from backend.security import (
    clear_session_cookie,
    create_csrf_token,
    create_session_token,
    get_current_user,
    get_optional_current_user,
    hash_password,
    require_csrf,
    set_session_cookie,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(
    payload: UserRegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Register a new user account with hashed password storage.
    Establishes an initial session cookie upon successful registration.
    """
    # Check for duplicate email
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists",
        )

    # Hash password with bcrypt before persistence
    hashed = hash_password(payload.password)

    user = User(
        email=payload.email,
        hashed_password=hashed,
        full_name=payload.full_name,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Issue signed session cookie
    session_token = create_session_token(user.id)
    set_session_cookie(response, session_token)

    return user


@router.post(
    "/login",
    response_model=UserResponse,
    summary="Authenticate user and establish session",
)
def login(
    payload: UserLoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Authenticate a user by email and password.
    Sets an HttpOnly SameSite=Lax signed session cookie upon success.
    """
    user = db.query(User).filter(User.email == payload.email).first()

    # Generic invalid credentials response to prevent user enumeration
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Establish session
    session_token = create_session_token(user.id)
    set_session_cookie(response, session_token)

    return user


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
def get_me(current_user: User = Depends(get_current_user)):
    """Return the profile of the currently logged-in user."""
    return current_user


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout current session",
)
def logout(
    response: Response,
    current_user: User = Depends(require_csrf),
):
    """
    Terminate the current user session by clearing the session cookie.
    Requires a valid CSRF token in the X-CSRF-Token header.
    """
    clear_session_cookie(response)
    return MessageResponse(message="Successfully logged out")


@router.get(
    "/csrf",
    response_model=CsrfTokenResponse,
    summary="Fetch a CSRF token for state-changing requests",
)
def get_csrf(current_user: Optional[User] = Depends(get_optional_current_user)):
    """
    Generate and return a signed CSRF token.
    Token is bound to the current authenticated user if logged in.
    """
    user_id = current_user.id if current_user else None
    token = create_csrf_token(user_id=user_id)
    return CsrfTokenResponse(csrf_token=token)
