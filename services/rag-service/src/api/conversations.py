"""Conversation and memory management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.memory.manager import get_conversation_manager, get_user_memory_manager
from shared.models.conversation import MessageRole
from shared.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


# ===== Request/Response Models =====


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""

    user_id: str = Field(..., description="User identifier")
    title: str | None = Field(None, description="Optional conversation title")


class CreateConversationResponse(BaseModel):
    """Response for creating a conversation."""

    conversation_id: str
    user_id: str
    title: str
    created_at: str


class GetConversationsRequest(BaseModel):
    """Request to get user's conversations."""

    user_id: str = Field(..., description="User identifier")
    limit: int = Field(20, description="Maximum conversations to retrieve", ge=1, le=100)
    include_archived: bool = Field(False, description="Include archived conversations")


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    conversation_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str
    is_active: bool


class GetConversationsResponse(BaseModel):
    """Response with list of conversations."""

    conversations: list[ConversationSummary]
    user_id: str
    total: int


class GetConversationHistoryRequest(BaseModel):
    """Request to get conversation history."""

    conversation_id: str = Field(..., description="Conversation identifier")
    limit: int = Field(10, description="Maximum messages to retrieve", ge=1, le=100)


class ConversationMessage(BaseModel):
    """A message in a conversation."""

    role: str
    content: str
    created_at: str
    sequence_number: int
    metadata: dict | None = None


class GetConversationHistoryResponse(BaseModel):
    """Response with conversation history."""

    conversation_id: str
    messages: list[ConversationMessage]
    total_messages: int


class GetUserMemoriesRequest(BaseModel):
    """Request to get user's long-term memories."""

    user_id: str = Field(..., description="User identifier")
    memory_type: str | None = Field(
        None, description="Filter by memory type (fact, preference, interaction, context)"
    )
    limit: int = Field(50, description="Maximum memories to retrieve", ge=1, le=100)


class UserMemory(BaseModel):
    """A user's long-term memory."""

    type: str
    key: str
    value: str
    confidence: int
    importance: int
    created_at: str
    updated_at: str


class GetUserMemoriesResponse(BaseModel):
    """Response with user memories."""

    user_id: str
    memories: list[UserMemory]
    total: int


# ===== Endpoints =====


@router.post("/create", response_model=CreateConversationResponse)
async def create_conversation(request: CreateConversationRequest):
    """
    Create a new conversation thread for a user.

    Args:
        request: Conversation creation request

    Returns:
        Created conversation details

    Example:
        POST /api/v1/conversations/create
        {
            "user_id": "user-123",
            "title": "EU AI Act Compliance Questions"
        }
    """
    try:
        conv_manager = get_conversation_manager()

        logger.info(
            "Creating new conversation",
            extra={"user_id": request.user_id, "title": request.title},
        )

        # Create conversation
        conversation_id = conv_manager.create_conversation(
            user_id=request.user_id, title=request.title
        )

        # Get conversation details to return
        from shared.memory.manager import DatabaseManager

        db = DatabaseManager()
        with db.get_session() as session:
            from shared.models.conversation import Conversation

            conversation = (
                session.query(Conversation)
                .filter(Conversation.conversation_id == conversation_id)
                .first()
            )

            if not conversation:
                raise HTTPException(status_code=500, detail="Failed to create conversation")

            return CreateConversationResponse(
                conversation_id=conversation.conversation_id,
                user_id=request.user_id,
                title=conversation.title,
                created_at=conversation.created_at.isoformat(),
            )

    except Exception as e:
        logger.error(f"Failed to create conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@router.post("/list", response_model=GetConversationsResponse)
async def list_conversations(request: GetConversationsRequest):
    """
    List all conversations for a user.

    Args:
        request: Request with user_id and filters

    Returns:
        List of user's conversations

    Example:
        POST /api/v1/conversations/list
        {
            "user_id": "user-123",
            "limit": 20,
            "include_archived": false
        }
    """
    try:
        conv_manager = get_conversation_manager()

        logger.info(
            "Listing conversations",
            extra={
                "user_id": request.user_id,
                "limit": request.limit,
                "include_archived": request.include_archived,
            },
        )

        conversations = conv_manager.get_user_conversations(
            user_id=request.user_id,
            limit=request.limit,
            include_archived=request.include_archived,
        )

        return GetConversationsResponse(
            conversations=[
                ConversationSummary(
                    conversation_id=conv["conversation_id"],
                    title=conv["title"],
                    message_count=conv["message_count"],
                    created_at=conv["created_at"],
                    updated_at=conv["updated_at"],
                    is_active=conv["is_active"],
                )
                for conv in conversations
            ],
            user_id=request.user_id,
            total=len(conversations),
        )

    except Exception as e:
        logger.error(f"Failed to list conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@router.post("/history", response_model=GetConversationHistoryResponse)
async def get_conversation_history(request: GetConversationHistoryRequest):
    """
    Get message history for a conversation.

    Args:
        request: Request with conversation_id and limit

    Returns:
        Conversation message history

    Example:
        POST /api/v1/conversations/history
        {
            "conversation_id": "conv-456",
            "limit": 10
        }
    """
    try:
        conv_manager = get_conversation_manager()

        logger.info(
            "Getting conversation history",
            extra={"conversation_id": request.conversation_id, "limit": request.limit},
        )

        messages = conv_manager.get_conversation_history(
            conversation_id=request.conversation_id, limit=request.limit
        )

        return GetConversationHistoryResponse(
            conversation_id=request.conversation_id,
            messages=[
                ConversationMessage(
                    role=msg["role"],
                    content=msg["content"],
                    created_at=msg["created_at"],
                    sequence_number=msg["sequence_number"],
                    metadata=msg.get("metadata"),
                )
                for msg in messages
            ],
            total_messages=len(messages),
        )

    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get conversation history: {str(e)}"
        )


@router.post("/memories", response_model=GetUserMemoriesResponse)
async def get_user_memories(request: GetUserMemoriesRequest):
    """
    Get user's long-term memories.

    Args:
        request: Request with user_id and filters

    Returns:
        User's long-term memories

    Example:
        POST /api/v1/conversations/memories
        {
            "user_id": "user-123",
            "memory_type": "preference",
            "limit": 50
        }
    """
    try:
        memory_manager = get_user_memory_manager()

        logger.info(
            "Getting user memories",
            extra={
                "user_id": request.user_id,
                "memory_type": request.memory_type,
                "limit": request.limit,
            },
        )

        memories = memory_manager.get_user_memories(
            user_id=request.user_id, memory_type=request.memory_type, limit=request.limit
        )

        return GetUserMemoriesResponse(
            user_id=request.user_id,
            memories=[
                UserMemory(
                    type=mem["type"],
                    key=mem["key"],
                    value=mem["value"],
                    confidence=mem["confidence"],
                    importance=mem["importance"],
                    created_at=mem["created_at"],
                    updated_at=mem["updated_at"],
                )
                for mem in memories
            ],
            total=len(memories),
        )

    except Exception as e:
        logger.error(f"Failed to get user memories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get user memories: {str(e)}")


@router.post("/{conversation_id}/archive")
async def archive_conversation(conversation_id: str):
    """
    Archive a conversation.

    Args:
        conversation_id: Conversation to archive

    Returns:
        Success confirmation

    Example:
        POST /api/v1/conversations/conv-456/archive
    """
    try:
        conv_manager = get_conversation_manager()

        logger.info("Archiving conversation", extra={"conversation_id": conversation_id})

        conv_manager.archive_conversation(conversation_id=conversation_id)

        return {"success": True, "conversation_id": conversation_id, "status": "archived"}

    except Exception as e:
        logger.error(f"Failed to archive conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to archive conversation: {str(e)}")
