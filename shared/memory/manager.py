"""
Memory Management

Handles conversation history and long-term user memory storage/retrieval.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session, sessionmaker

from shared.config import get_settings
from shared.models.conversation import (
    Base,
    Conversation,
    ConversationSummary,
    Message,
    MessageRole,
    User,
    UserMemory,
)
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DatabaseManager:
    """Base database manager."""

    def __init__(self):
        """Initialize database connection."""
        self.engine = create_engine(settings.database_url, echo=False)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Created memory tables")

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()


class ConversationMemoryManager:
    """Manages conversation threads and message history."""

    def __init__(self):
        """Initialize conversation memory manager."""
        self.db = DatabaseManager()
        self.db.create_tables()

    def get_or_create_user(self, user_id: str, **kwargs) -> User:
        """
        Get existing user or create new one.

        Args:
            user_id: Unique user identifier
            **kwargs: Additional user fields (email, full_name)

        Returns:
            User object
        """
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                user = User(user_id=user_id, **kwargs)
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"Created new user: {user_id}")
            else:
                # Update last_active
                user.last_active = datetime.utcnow()
                session.commit()

            return user

    def create_conversation(
        self, user_id: str, conversation_id: str | None = None, title: str | None = None
    ) -> str:
        """
        Create a new conversation thread.

        Args:
            user_id: User identifier
            conversation_id: Optional conversation ID (auto-generated if not provided)
            title: Optional conversation title

        Returns:
            Conversation ID
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        with self.db.get_session() as session:
            # Get or create user
            user = self.get_or_create_user(user_id)

            # Create conversation
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user.id,
                title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            session.add(conversation)
            session.commit()

            logger.info(
                f"Created conversation: {conversation_id} for user: {user_id}"
            )

        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Add a message to conversation.

        Args:
            conversation_id: Conversation identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata (RAG context, citations, etc.)

        Returns:
            Message ID
        """
        with self.db.get_session() as session:
            # Get conversation
            conversation = (
                session.query(Conversation)
                .filter(Conversation.conversation_id == conversation_id)
                .first()
            )

            if not conversation:
                raise ValueError(f"Conversation not found: {conversation_id}")

            # Get next sequence number
            sequence_number = conversation.message_count + 1

            # Create message
            message = Message(
                conversation_id=conversation.id,
                role=role.value,
                content=content,
                metadata=metadata,
                sequence_number=sequence_number,
                created_at=datetime.utcnow(),
            )

            session.add(message)

            # Update conversation
            conversation.message_count = sequence_number
            conversation.last_message_at = datetime.utcnow()
            conversation.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(message)

            logger.debug(
                f"Added {role.value} message to conversation {conversation_id}"
            )

            return message.id

    def get_conversation_history(
        self, conversation_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get recent messages from conversation.

        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages (oldest first)
        """
        with self.db.get_session() as session:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.conversation_id == conversation_id)
                .first()
            )

            if not conversation:
                return []

            messages = (
                session.query(Message)
                .filter(Message.conversation_id == conversation.id)
                .order_by(desc(Message.sequence_number))
                .limit(limit)
                .all()
            )

            # Reverse to get chronological order
            messages = list(reversed(messages))

            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.metadata,
                    "created_at": msg.created_at.isoformat(),
                    "sequence_number": msg.sequence_number,
                }
                for msg in messages
            ]

    def get_user_conversations(
        self, user_id: str, limit: int = 20, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get user's conversation threads.

        Args:
            user_id: User identifier
            limit: Maximum number of conversations
            include_archived: Include archived conversations

        Returns:
            List of conversations with metadata
        """
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return []

            query = session.query(Conversation).filter(
                Conversation.user_id == user.id
            )

            if not include_archived:
                query = query.filter(Conversation.is_active == 1)

            conversations = (
                query.order_by(desc(Conversation.updated_at)).limit(limit).all()
            )

            return [
                {
                    "conversation_id": conv.conversation_id,
                    "title": conv.title,
                    "message_count": conv.message_count,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "is_active": bool(conv.is_active),
                }
                for conv in conversations
            ]

    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title."""
        with self.db.get_session() as session:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.conversation_id == conversation_id)
                .first()
            )

            if conversation:
                conversation.title = title
                conversation.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Updated conversation title: {conversation_id}")

    def archive_conversation(self, conversation_id: str):
        """Archive a conversation."""
        with self.db.get_session() as session:
            conversation = (
                session.query(Conversation)
                .filter(Conversation.conversation_id == conversation_id)
                .first()
            )

            if conversation:
                conversation.is_active = 0
                conversation.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Archived conversation: {conversation_id}")


class UserMemoryManager:
    """Manages long-term user memory (facts, preferences, context)."""

    def __init__(self):
        """Initialize user memory manager."""
        self.db = DatabaseManager()
        self.db.create_tables()

    def store_memory(
        self,
        user_id: str,
        memory_type: str,
        key: str,
        value: str,
        source_conversation_id: str | None = None,
        confidence: int = 5,
        importance: int = 5,
    ) -> int:
        """
        Store a long-term memory for user.

        Args:
            user_id: User identifier
            memory_type: Type of memory (fact, preference, interaction, context)
            key: Memory key
            value: Memory value
            source_conversation_id: Where this memory originated
            confidence: Confidence score (1-10)
            importance: Importance score (1-10)

        Returns:
            Memory ID
        """
        with self.db.get_session() as session:
            # Get or create user
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                # Create user if doesn't exist
                user = User(user_id=user_id, created_at=datetime.utcnow())
                session.add(user)
                session.commit()
                session.refresh(user)

            # Check if memory exists
            existing = (
                session.query(UserMemory)
                .filter(
                    UserMemory.user_id == user.id,
                    UserMemory.memory_type == memory_type,
                    UserMemory.key == key,
                )
                .first()
            )

            if existing:
                # Update existing memory
                existing.value = value
                existing.updated_at = datetime.utcnow()
                existing.confidence = confidence
                existing.importance = importance
                session.commit()
                memory_id = existing.id
                logger.info(f"Updated memory: {key} for user {user_id}")
            else:
                # Create new memory
                memory = UserMemory(
                    user_id=user.id,
                    memory_type=memory_type,
                    key=key,
                    value=value,
                    source_conversation_id=source_conversation_id,
                    confidence=confidence,
                    importance=importance,
                    created_at=datetime.utcnow(),
                )

                session.add(memory)
                session.commit()
                session.refresh(memory)
                memory_id = memory.id
                logger.info(f"Stored new memory: {key} for user {user_id}")

            return memory_id

    def get_user_memories(
        self, user_id: str, memory_type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Retrieve user's long-term memories.

        Args:
            user_id: User identifier
            memory_type: Optional filter by memory type
            limit: Maximum memories to retrieve

        Returns:
            List of memories
        """
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return []

            query = session.query(UserMemory).filter(UserMemory.user_id == user.id)

            if memory_type:
                query = query.filter(UserMemory.memory_type == memory_type)

            memories = (
                query.order_by(desc(UserMemory.importance), desc(UserMemory.updated_at))
                .limit(limit)
                .all()
            )

            # Update access tracking
            for memory in memories:
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1

            session.commit()

            return [
                {
                    "type": mem.memory_type,
                    "key": mem.key,
                    "value": mem.value,
                    "confidence": mem.confidence,
                    "importance": mem.importance,
                    "created_at": mem.created_at.isoformat(),
                    "updated_at": mem.updated_at.isoformat(),
                }
                for mem in memories
            ]

    def get_memory(self, user_id: str, key: str) -> dict[str, Any] | None:
        """
        Get a specific memory by key.

        Args:
            user_id: User identifier
            key: Memory key

        Returns:
            Memory dict or None
        """
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return None

            memory = (
                session.query(UserMemory)
                .filter(UserMemory.user_id == user.id, UserMemory.key == key)
                .first()
            )

            if memory:
                # Update access tracking
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
                session.commit()

                return {
                    "type": memory.memory_type,
                    "key": memory.key,
                    "value": memory.value,
                    "confidence": memory.confidence,
                    "importance": memory.importance,
                }

            return None

    def delete_memory(self, user_id: str, key: str) -> bool:
        """
        Delete a user memory.

        Args:
            user_id: User identifier
            key: Memory key

        Returns:
            True if deleted, False if not found
        """
        with self.db.get_session() as session:
            user = session.query(User).filter(User.user_id == user_id).first()

            if not user:
                return False

            memory = (
                session.query(UserMemory)
                .filter(UserMemory.user_id == user.id, UserMemory.key == key)
                .first()
            )

            if memory:
                session.delete(memory)
                session.commit()
                logger.info(f"Deleted memory: {key} for user {user_id}")
                return True

            return False


# Convenience functions
def get_conversation_manager() -> ConversationMemoryManager:
    """Get conversation memory manager instance."""
    return ConversationMemoryManager()


def get_user_memory_manager() -> UserMemoryManager:
    """Get user memory manager instance."""
    return UserMemoryManager()
