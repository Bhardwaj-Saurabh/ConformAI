#!/usr/bin/env python3
"""
Initialize ConformAI project.

This script:
1. Validates environment variables
2. Creates necessary directories
3. Initializes database schemas
4. Creates Qdrant collections
5. Runs basic health checks
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)


def validate_environment():
    """Validate required environment variables."""
    logger.info("Validating environment variables...")

    settings = get_settings()
    required_keys = []

    if settings.llm_provider == "anthropic":
        required_keys.append("ANTHROPIC_API_KEY")
    elif settings.llm_provider == "openai":
        required_keys.append("OPENAI_API_KEY")

    missing_keys = [key for key in required_keys if not getattr(settings, key.lower(), None)]

    if missing_keys:
        logger.error(f"Missing required environment variables: {', '.join(missing_keys)}")
        return False

    logger.info("Environment validation passed")
    return True


def create_directories():
    """Create necessary data directories."""
    logger.info("Creating data directories...")

    directories = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "logs",
    ]

    for directory in directories:
        path = project_root / directory
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")

    logger.info("Directory creation completed")


def init_qdrant():
    """Initialize Qdrant collections."""
    logger.info("Initializing Qdrant vector database...")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        settings = get_settings()
        client = QdrantClient(url=settings.qdrant_url)

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        collection_name = settings.qdrant_collection_name

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
        else:
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {collection_name}")

        # Test connection
        info = client.get_collection(collection_name)
        logger.info(f"Qdrant collection info: {info}")
        logger.info("Qdrant initialization completed")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")
        logger.warning("Make sure Qdrant is running (docker-compose up -d qdrant)")
        return False


def init_postgres():
    """Initialize PostgreSQL database."""
    logger.info("Initializing PostgreSQL database...")

    try:
        from sqlalchemy import create_engine, text

        settings = get_settings()
        engine = create_engine(settings.database_url)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"PostgreSQL version: {version}")

        logger.info("PostgreSQL initialization completed")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {str(e)}")
        logger.warning("Make sure PostgreSQL is running (docker-compose up -d postgres)")
        return False


def main():
    """Main initialization function."""
    logger.info("=" * 60)
    logger.info("ConformAI Project Initialization")
    logger.info("=" * 60)

    # Step 1: Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)

    # Step 2: Create directories
    create_directories()

    # Step 3: Initialize Qdrant (optional - may not be running yet)
    init_qdrant()

    # Step 4: Initialize PostgreSQL (optional - may not be running yet)
    init_postgres()

    logger.info("=" * 60)
    logger.info("Initialization completed!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Start services: docker-compose up -d")
    logger.info("2. Run Airflow migrations: docker-compose exec airflow-webserver airflow db migrate")
    logger.info("3. Create Airflow admin user: docker-compose exec airflow-webserver airflow users create ...")
    logger.info("4. Access Airflow UI: http://localhost:8080")
    logger.info("5. Access API Gateway: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
