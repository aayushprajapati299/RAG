"""
Chat and conversation router.

Endpoints:
- POST /api/chat/conversations: Create a new user conversation
- GET /api/chat/conversations: List all conversations belonging to the authenticated user
- GET /api/chat/conversations/{conversation_id}: Retrieve detailed conversation with message history
- POST /api/chat/conversations/{conversation_id}/messages: Post a message for a selected document
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas.chat import (
    ChatAnswerResponse,
    ChatMessageRequest,
    Citation,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)
from backend.security import get_current_user, verify_csrf_token
from backend.services.chat import (
    ConversationNotFoundError,
    DocumentNotFoundError,
    ProviderAPIError,
    attach_document_to_conversation,
    create_conversation,
    delete_conversation,
    detach_document_from_conversation,
    get_user_conversation,
    get_user_document,
    list_user_conversations,
    post_conversation_message,
    select_document_for_conversation,
    stream_conversation_message,
    update_conversation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])
conversations_router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _validate_optional_csrf(x_csrf_token: Optional[str], user_id: UUID) -> None:
    """Validate CSRF token if one is supplied in the request header."""
    if x_csrf_token and not verify_csrf_token(x_csrf_token, expected_user_id=user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired CSRF token",
        )


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
@conversations_router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
def handle_create_conversation(
    payload: Optional[ConversationCreate] = None,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new conversation session owned by the authenticated user.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)
    title = payload.title if payload else None
    selected_doc_id = payload.selected_document_id if payload else None
    doc_ids = payload.document_ids if payload else None

    try:
        conv = create_conversation(
            db=db,
            user_id=current_user.id,
            title=title,
            selected_document_id=selected_doc_id,
            document_ids=doc_ids,
        )
    except DocumentNotFoundError as de:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(de),
        )

    return conv


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Update conversation metadata",
)
@conversations_router.patch(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Update conversation metadata",
)
def handle_update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdate,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update conversation title, associated documents, or selected document.
    Enforces user ownership.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)
    try:
        conv = update_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
            title=payload.title,
            selected_document_id=payload.selected_document_id,
            document_ids=payload.document_ids,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return conv


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete a conversation",
)
@conversations_router.delete(
    "/{conversation_id}",
    summary="Delete a conversation",
)
def handle_delete_conversation(
    conversation_id: UUID,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a conversation owned by the authenticated user.
    Cascades to associated messages and document links.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)
    try:
        delete_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return {"status": "ok", "message": "Conversation deleted"}


@router.post(
    "/conversations/{conversation_id}/documents/{document_id}",
    response_model=ConversationDetailResponse,
    summary="Attach document to conversation",
)
@conversations_router.post(
    "/{conversation_id}/documents/{document_id}",
    response_model=ConversationDetailResponse,
    summary="Attach document to conversation",
)
def handle_attach_document(
    conversation_id: UUID,
    document_id: UUID,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Attach an existing document to a conversation.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)
    try:
        conv = attach_document_to_conversation(
            db=db,
            conversation_id=conversation_id,
            document_id=document_id,
            user_id=current_user.id,
        )
    except (ConversationNotFoundError, DocumentNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation or document not found",
        )
    return conv


@router.delete(
    "/conversations/{conversation_id}/documents/{document_id}",
    response_model=ConversationDetailResponse,
    summary="Detach document from conversation",
)
@conversations_router.delete(
    "/{conversation_id}/documents/{document_id}",
    response_model=ConversationDetailResponse,
    summary="Detach document from conversation",
)
def handle_detach_document(
    conversation_id: UUID,
    document_id: UUID,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Detach a document from a conversation.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)
    try:
        conv = detach_document_from_conversation(
            db=db,
            conversation_id=conversation_id,
            document_id=document_id,
            user_id=current_user.id,
        )
    except (ConversationNotFoundError, DocumentNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation or document not found",
        )
    return conv


@router.put(
    "/conversations/{conversation_id}/select-document/{document_id}",
    response_model=ConversationDetailResponse,
    summary="Select active document for conversation",
)
@conversations_router.put(
    "/{conversation_id}/select-document/{document_id}",
    response_model=ConversationDetailResponse,
    summary="Select active document for conversation",
)
def handle_select_document(
    conversation_id: UUID,
    document_id: UUID,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Select active document for conversation.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)
    try:
        conv = select_document_for_conversation(
            db=db,
            conversation_id=conversation_id,
            document_id=document_id,
            user_id=current_user.id,
        )
    except (ConversationNotFoundError, DocumentNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation or document not found",
        )
    return conv


@router.get(
    "/conversations",
    response_model=List[ConversationResponse],
    summary="List current user's conversations",
)
@conversations_router.get(
    "",
    response_model=List[ConversationResponse],
    summary="List current user's conversations",
)
def handle_list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all conversations belonging to the authenticated user.
    Never returns conversations owned by another user.
    """
    conversations = list_user_conversations(db=db, user_id=current_user.id)
    # Populate message counts
    results = []
    for c in conversations:
        resp = ConversationResponse.model_validate(c)
        resp.message_count = len(c.messages) if c.messages else 0
        results.append(resp)
    return results


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Retrieve conversation details and messages",
)
@conversations_router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Retrieve conversation details and messages",
)
def handle_get_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve an individual conversation and its message history.
    Enforces user ownership.
    """
    try:
        conversation = get_user_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=current_user.id,
        )
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=ChatAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Post a message to a conversation for a selected document",
)
@router.post(
    "/conversations/{conversation_id}",
    response_model=ChatAnswerResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
@conversations_router.post(
    "/{conversation_id}/messages",
    response_model=ChatAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Post a message to a conversation for a selected document",
)
@conversations_router.post(
    "/{conversation_id}",
    response_model=ChatAnswerResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def handle_post_message(
    conversation_id: UUID,
    payload: ChatMessageRequest,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Post a user question to an existing conversation against a selected document:
    1. Verifies ownership of conversation and document.
    2. Persists user question and assistant answer in Message rows.
    3. Retrieves relevant chunks and builds defensive prompt with bounded context.
    4. Queries Groq chat completions API with request timeout.
    5. Returns answer with citations containing chunk ID and page number.
    If payload.stream is True, streams the response via Server-Sent Events.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)

    message_content = payload.content or payload.question
    if not message_content or not message_content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message content cannot be empty",
        )

    # Check ownership upfront
    try:
        conv = get_user_conversation(db, conversation_id, current_user.id)
        target_doc_id = payload.document_id or conv.selected_document_id
        if not target_doc_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No document selected for this conversation",
            )
        get_user_document(db, target_doc_id, current_user.id)
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if payload.stream:
        return StreamingResponse(
            stream_conversation_message(
                user_id=current_user.id,
                conversation_id=conversation_id,
                document_id=target_doc_id,
                content=message_content,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        user_msg, assistant_msg, citations = post_conversation_message(
            db=db,
            user_id=current_user.id,
            conversation_id=conversation_id,
            document_id=target_doc_id,
            content=message_content,
        )
    except ProviderAPIError as pe:
        logger.error("LLM Provider failure in conversation %s: %s", conversation_id, pe.message)
        raise HTTPException(
            status_code=pe.status_code,
            detail={
                "error": "ProviderError",
                "message": pe.message,
                "details": pe.details,
            },
        )

    return ChatAnswerResponse(
        conversation_id=conversation_id,
        answer=assistant_msg.content,
        citations=citations,
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
    )


@router.post(
    "/conversations/{conversation_id}/messages/stream",
    summary="Stream assistant response for a selected document (SSE)",
)
@router.post(
    "/conversations/{conversation_id}/stream",
    include_in_schema=False,
)
@conversations_router.post(
    "/{conversation_id}/messages/stream",
    summary="Stream assistant response for a selected document (SSE)",
)
@conversations_router.post(
    "/{conversation_id}/stream",
    include_in_schema=False,
)
def handle_post_message_stream(
    conversation_id: UUID,
    payload: ChatMessageRequest,
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Stream assistant response for a conversation as Server-Sent Events (SSE).

    Documented SSE Event Sequence:
    1. 'text': Emitted per text token delta:
       data: {"delta": "<token>"}
    2. 'complete': Emitted on successful stream completion with provenance citations and message ID:
       data: {"message_id": "<uuid>", "conversation_id": "<uuid>", "answer": "<full_text>", "citations": [{"chunk_id": "<uuid>", "page_number": <int>}], "finish_reason": "stop"}
    3. 'error': Emitted if an error occurs during processing:
       data: {"error": "ProviderError", "message": "<safe_sanitized_message>"}

    Guarantees:
    - Ownership verified prior to stream initiation (HTTP 404 returned on failure).
    - Assembled assistant message persisted in database ONLY after successful stream completion.
    - Closes upstream HTTP connection reliably on completion, error, or client disconnect.
    """
    _validate_optional_csrf(x_csrf_token, current_user.id)

    message_content = payload.content or payload.question
    if not message_content or not message_content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message content cannot be empty",
        )

    # Verify ownership before opening stream
    try:
        conv = get_user_conversation(db, conversation_id, current_user.id)
        target_doc_id = payload.document_id or conv.selected_document_id
        if not target_doc_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No document selected for this conversation",
            )
        get_user_document(db, target_doc_id, current_user.id)
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return StreamingResponse(
        stream_conversation_message(
            user_id=current_user.id,
            conversation_id=conversation_id,
            document_id=target_doc_id,
            content=message_content,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
