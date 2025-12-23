"""
LangGraph Checkpointer Integration

Provides persistent state management for LangGraph workflows using PostgreSQL.
"""

from langgraph.checkpoint.postgres import PostgresSaver

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ConformAICheckpointer:
    """Wrapper for LangGraph PostgreSQL checkpointer."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for checkpointer."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize checkpointer."""
        if self._initialized:
            return

        try:
            # Create PostgreSQL checkpointer
            self.checkpointer = PostgresSaver.from_conn_string(
                settings.database_url
            )

            # Create checkpoint tables if they don't exist
            self.checkpointer.setup()

            logger.info("✓ LangGraph checkpointer initialized")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {str(e)}")
            self.checkpointer = None
            self._initialized = False

    def get_checkpointer(self) -> PostgresSaver | None:
        """
        Get checkpointer instance.

        Returns:
            PostgresSaver instance or None if initialization failed
        """
        return self.checkpointer

    @classmethod
    def is_available(cls) -> bool:
        """Check if checkpointer is available."""
        instance = cls()
        return instance._initialized and instance.checkpointer is not None


def get_checkpointer() -> PostgresSaver | None:
    """
    Get checkpointer instance.

    Returns:
        PostgresSaver instance or None
    """
    wrapper = ConformAICheckpointer()
    return wrapper.get_checkpointer()


def is_checkpointer_enabled() -> bool:
    """Check if checkpointer is enabled and available."""
    return ConformAICheckpointer.is_available()
