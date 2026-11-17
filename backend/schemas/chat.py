"""
Chat and conversation schemas.

Provides request/response Pydantic models for:
- Creating and listing conversations
- Fetching conversation details and message history
- Sending messages to a conversation against a selected document
- Provenance citations containing chunk ID and page number
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.schemas.documents import DocumentResponse


class Citation(BaseModel):
    """Citation metadata referencing the source chunk and page."""
    chunk_id: UUID = Field(..., description="UUID of the retrieved chunk")
    page_number: int = Field(..., description="1-based page number within the source document")
    text_snippet: Optional[str] = Field(None, description="Short snippet of the matched text")
    score: Optional[float] = Field(None, description="Retrieval relevance score")

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Single message response representation."""
    id: UUID
    conversation_id: UUID
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str
    sources: Optional[List[Citation]] = Field(
        default=None,
        description="List of citation sources (for assistant messages)",
    )
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return None
        return v


class ConversationCreate(BaseModel):
    """Request payload to create a new conversation."""
    title: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional title for the conversation",
    )
    selected_document_id: Optional[UUID] = Field(
        default=None,
        description="Optional document ID to select for this conversation",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Optional list of document IDs to attach to this conversation",
    )


class ConversationUpdate(BaseModel):
    """Request payload to update conversation title or selected document."""
    title: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Updated conversation title",
    )
    selected_document_id: Optional[UUID] = Field(
        default=None,
        description="Updated selected document ID (or null to clear)",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Updated list of associated document IDs",
    )


class ConversationResponse(BaseModel):
    """Conversation summary response."""
    id: UUID
    user_id: UUID
    title: str
    selected_document_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = Field(
        default=None,
        description="Total number of messages in the conversation",
    )

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(BaseModel):
    """Detailed conversation response including message history."""
    id: UUID
    user_id: UUID
    title: str
    selected_document_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    documents: List[DocumentResponse] = []
    messages: List[MessageResponse] = []

    model_config = ConfigDict(from_attributes=True)


class ChatMessageRequest(BaseModel):
    """Request payload to post a message to a conversation."""
    document_id: Optional[UUID] = Field(
        default=None,
        description="UUID of the selected user-owned document. If omitted, uses conversation's selected document.",
    )
    content: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=10000,
        description="Message content or question text",
    )
    question: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=10000,
        description="Alias for message content / question",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response as server-sent events",
    )

    @model_validator(mode="after")
    def resolve_content(self):
        if not self.content and not self.question:
            raise ValueError("Either 'content' or 'question' must be provided.")
        if not self.content:
            self.content = self.question
        return self


class StreamTextData(BaseModel):
    """Payload for SSE 'text' event containing streaming text token."""
    delta: str


class StreamCompleteData(BaseModel):
    """Payload for SSE 'complete' event containing final message provenance."""
    message_id: UUID
    conversation_id: UUID
    answer: str
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source citations with chunk ID and page number",
    )
    finish_reason: str = "stop"


class StreamErrorData(BaseModel):
    """Payload for SSE 'error' event containing safe error details."""
    error: str
    message: str


class ChatAnswerResponse(BaseModel):
    """Structured response returned upon answering a chat message."""
    conversation_id: UUID
    answer: str
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source citations with chunk ID and page number",
    )
    user_message: MessageResponse
    assistant_message: MessageResponse

    model_config = ConfigDict(from_attributes=True)
