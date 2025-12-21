"""Production-grade logging configuration with structured logging and context support."""

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from ..config import get_settings

# Context variables for request tracking
request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
conversation_id_context: ContextVar[str | None] = ContextVar("conversation_id", default=None)
user_id_context: ContextVar[str | None] = ContextVar("user_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context information
        request_id = request_id_context.get()
        if request_id:
            log_data["request_id"] = request_id

        conversation_id = conversation_id_context.get()
        if conversation_id:
            log_data["conversation_id"] = conversation_id

        user_id = user_id_context.get()
        if user_id:
            log_data["user_id"] = user_id

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add stack info if present
        if record.stack_info:
            log_data["stack_info"] = record.stack_info

        return json.dumps(log_data)


class ProductionLogger(logging.LoggerAdapter):
    """
    Enhanced logger with production features:
    - Structured logging
    - Context tracking (request_id, conversation_id)
    - Performance logging
    - Audit logging
    - Error enrichment
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Add contextual information to log messages."""
        # Extract extra fields if provided
        extra = kwargs.get("extra", {})

        # Add context from ContextVars
        request_id = request_id_context.get()
        if request_id and "request_id" not in extra:
            extra["request_id"] = request_id

        conversation_id = conversation_id_context.get()
        if conversation_id and "conversation_id" not in extra:
            extra["conversation_id"] = conversation_id

        user_id = user_id_context.get()
        if user_id and "user_id" not in extra:
            extra["user_id"] = user_id

        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs

    def with_context(self, **context: Any) -> "ProductionLogger":
        """Create a new logger with additional context."""
        new_extra = {**self.extra, **context}
        return ProductionLogger(self.logger, new_extra)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        **metadata: Any
    ) -> None:
        """
        Log performance metrics.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        self.info(
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "metric_type": "performance",
                **metadata,
            },
        )

    def log_audit(
        self,
        action: str,
        resource: str,
        result: str,
        **metadata: Any
    ) -> None:
        """
        Log audit events for compliance and security.

        Args:
            action: Action performed (e.g., "query", "retrieve", "synthesize")
            resource: Resource affected (e.g., "legal_documents", "rag_pipeline")
            result: Result of the action (e.g., "success", "failure", "denied")
            **metadata: Additional metadata
        """
        self.info(
            f"Audit: {action} on {resource} - {result}",
            extra={
                "action": action,
                "resource": resource,
                "result": result,
                "audit_event": True,
                **metadata,
            },
        )

    def log_error_with_context(
        self,
        message: str,
        error: Exception,
        **context: Any
    ) -> None:
        """
        Log errors with full context and stack trace.

        Args:
            message: Error message
            error: Exception instance
            **context: Additional context
        """
        self.error(
            message,
            exc_info=True,
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                **context,
            },
        )

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        **metadata: Any
    ) -> None:
        """
        Log API requests.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            **metadata: Additional metadata
        """
        level = logging.INFO if status_code < 400 else logging.ERROR

        self.log(
            level,
            f"API {method} {endpoint} - {status_code} ({duration_ms:.2f}ms)",
            extra={
                "http_method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "metric_type": "api_request",
                **metadata,
            },
        )

    def log_llm_call(
        self,
        model: str,
        provider: str,
        duration_ms: float,
        tokens_used: int = 0,
        **metadata: Any
    ) -> None:
        """
        Log LLM calls for monitoring and cost tracking.

        Args:
            model: Model name
            provider: LLM provider
            duration_ms: Call duration in milliseconds
            tokens_used: Number of tokens used
            **metadata: Additional metadata
        """
        self.info(
            f"LLM call: {provider}/{model} - {tokens_used} tokens in {duration_ms:.2f}ms",
            extra={
                "llm_provider": provider,
                "llm_model": model,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
                "metric_type": "llm_call",
                **metadata,
            },
        )


def get_logger(name: str) -> ProductionLogger:
    """
    Get a production-grade configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured ProductionLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing request", extra={"user_id": "123"})
        logger.log_performance("db_query", 45.2, query_type="select")
        logger.log_audit("query", "legal_documents", "success", count=10)
    """
    settings = get_settings()

    # Get base logger
    base_logger = logging.getLogger(name)

    if not base_logger.handlers:
        # Set level
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        base_logger.setLevel(log_level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create formatter based on environment
        if settings.environment == "production" or settings.log_format == "json":
            # Structured JSON logging for production
            formatter = StructuredFormatter()
        else:
            # Human-readable format for development
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        handler.setFormatter(formatter)
        base_logger.addHandler(handler)

        # Prevent propagation to root logger
        base_logger.propagate = False

    # Wrap in ProductionLogger for enhanced features
    return ProductionLogger(base_logger, {})


def set_request_context(
    request_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None
) -> None:
    """
    Set request context for logging.

    Args:
        request_id: Unique request identifier
        conversation_id: Conversation identifier
        user_id: User identifier
    """
    if request_id:
        request_id_context.set(request_id)
    if conversation_id:
        conversation_id_context.set(conversation_id)
    if user_id:
        user_id_context.set(user_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_context.set(None)
    conversation_id_context.set(None)
    user_id_context.set(None)
