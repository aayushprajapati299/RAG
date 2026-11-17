import re
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class UserRegisterRequest(BaseModel):
    """Payload for user registration."""
    email: str = Field(..., max_length=255, description="Unique email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Password between 8 and 72 characters",
    )
    full_name: Optional[str] = Field(default=None, max_length=255)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email address format")
        return clean


class UserLoginRequest(BaseModel):
    """Payload for user login."""
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=72)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    """Public user representation (plaintext password is never returned)."""
    id: UUID
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CsrfTokenResponse(BaseModel):
    """CSRF token response."""
    csrf_token: str


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
