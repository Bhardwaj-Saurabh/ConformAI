#!/usr/bin/env python3
"""
Initialize Memory Database

This script creates all necessary tables for conversation and user memory persistence.
Run this script before starting the RAG service for the first time.

Usage:
    python scripts/init_memory_database.py

Tables created:
- users: User accounts with relationships to conversations and memories
- conversations: Conversation threads with metadata
- messages: Individual messages in conversations
- user_memories: Long-term user memories (facts, preferences, context)
- conversation_summaries: Periodic conversation summaries for efficient retrieval
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import get_settings
from shared.models.conversation import Base
from shared.utils import get_logger
from sqlalchemy import create_engine, inspect

logger = get_logger(__name__)
settings = get_settings()


def check_existing_tables(engine):
    """Check which tables already exist."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    return existing_tables


def initialize_memory_database():
    """Initialize all memory database tables."""
    logger.info("=" * 70)
    logger.info("MEMORY DATABASE INITIALIZATION")
    logger.info("=" * 70)

    # Create engine
    logger.info(f"Connecting to database: {settings.database_url.split('@')[1] if '@' in settings.database_url else settings.database_url}")
    engine = create_engine(settings.database_url, echo=True)

    # Check existing tables
    logger.info("\nChecking existing tables...")
    existing_tables = check_existing_tables(engine)

    if existing_tables:
        logger.info(f"Found {len(existing_tables)} existing tables: {', '.join(existing_tables)}")
    else:
        logger.info("No existing tables found.")

    # Tables to create
    tables_to_create = [
        "users",
        "conversations",
        "messages",
        "user_memories",
        "conversation_summaries",
    ]

    # Check which tables need to be created
    tables_needed = [t for t in tables_to_create if t not in existing_tables]

    if not tables_needed:
        logger.info("\n✓ All memory tables already exist!")
        logger.info("Existing tables:")
        for table in tables_to_create:
            logger.info(f"  ✓ {table}")
        return

    logger.info(f"\nCreating {len(tables_needed)} tables: {', '.join(tables_needed)}")

    try:
        # Create all tables
        logger.info("\nExecuting table creation...")
        Base.metadata.create_all(bind=engine)

        # Verify creation
        new_tables = check_existing_tables(engine)
        created_count = len([t for t in tables_to_create if t in new_tables])

        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\n✓ Successfully created {created_count}/{len(tables_to_create)} tables")

        logger.info("\nTable status:")
        for table in tables_to_create:
            status = "✓" if table in new_tables else "✗"
            logger.info(f"  {status} {table}")

        logger.info("\n" + "=" * 70)
        logger.info("NEXT STEPS")
        logger.info("=" * 70)
        logger.info("\n1. Start the RAG service:")
        logger.info("   cd services/rag-service")
        logger.info("   python -m src.api.main")
        logger.info("\n2. Test conversation memory:")
        logger.info('   POST /api/v1/conversations/create {"user_id": "user-123", "title": "Test"}')
        logger.info("\n3. Test query with memory:")
        logger.info('   POST /api/v1/query {"query": "...", "user_id": "user-123", "conversation_id": "conv-..."}')
        logger.info("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"\n✗ Failed to initialize database: {str(e)}", exc_info=True)
        logger.error("\nPlease check:")
        logger.error("1. DATABASE_URL is correctly set in .env")
        logger.error("2. PostgreSQL is running and accessible")
        logger.error("3. Database exists and user has CREATE TABLE permissions")
        raise


if __name__ == "__main__":
    try:
        initialize_memory_database()
    except KeyboardInterrupt:
        logger.info("\n\nInitialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nInitialization failed: {str(e)}")
        sys.exit(1)
