"""
Conversation and Memory Models

Database models for managing user conversations, message history,
and long-term user memory in the RAG system.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MessageRole(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryType(str, Enum):
    """Type of long-term memory."""

    FACT = "fact"  # Factual information about user
    PREFERENCE = "preference"  # User preferences
    INTERACTION = "interaction"  # Past interaction summary
    CONTEXT = "context"  # Domain-specific context


class User(Base):
    """User model for authentication and memory association."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    full_name = Column(String(255), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    memories = relationship(
        "UserMemory", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}')>"


class Conversation(Base):
    """Conversation thread containing multiple messages."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Conversation metadata
    title = Column(String(500), nullable=True)  # Auto-generated or user-provided
    summary = Column(Text, nullable=True)  # Summary of conversation
    topic = Column(String(255), nullable=True)  # Main topic (e.g., "EU AI Act")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_message_at = Column(DateTime, nullable=True)

    # State
    is_active = Column(Integer, default=1)  # 1=active, 0=archived
    message_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Conversation(id='{self.conversation_id}', user_id={self.user_id}, messages={self.message_count})>"


class Message(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False, index=True
    )

    # Message content
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)

    # Message metadata
    metadata = Column(JSON, nullable=True)  # Store RAG metadata, citations, etc.
    tokens_used = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    sequence_number = Column(Integer, nullable=False)  # Order in conversation

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(role='{self.role}', content='{preview}')>"


class UserMemory(Base):
    """Long-term memory associated with a user."""

    __tablename__ = "user_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Memory content
    memory_type = Column(String(50), nullable=False)  # fact, preference, interaction, context
    key = Column(String(255), nullable=False)  # Memory key (e.g., "job_role", "company_industry")
    value = Column(Text, nullable=False)  # Memory value

    # Metadata
    source_conversation_id = Column(
        String(255), nullable=True
    )  # Where this memory came from
    confidence = Column(Integer, default=5)  # 1-10 confidence score
    importance = Column(Integer, default=5)  # 1-10 importance score

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_accessed = Column(DateTime, nullable=True)
    access_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="memories")

    def __repr__(self):
        return f"<UserMemory(type='{self.memory_type}', key='{self.key}', user_id={self.user_id})>"


class ConversationSummary(Base):
    """Periodic summaries of conversations for efficient retrieval."""

    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(255), ForeignKey("conversations.conversation_id"), nullable=False)

    # Summary content
    summary = Column(Text, nullable=False)
    key_topics = Column(JSON, nullable=True)  # List of main topics discussed
    message_range_start = Column(Integer, nullable=False)  # First message in range
    message_range_end = Column(Integer, nullable=False)  # Last message in range

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<ConversationSummary(conversation_id='{self.conversation_id}', messages={self.message_range_start}-{self.message_range_end})>"
