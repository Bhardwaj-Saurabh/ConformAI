"""Shared utilities."""

from .logger import get_logger
from .opik_tracer import (
    get_opik_client,
    log_event,
    log_metric,
    track_embedding_call,
    track_llm_call,
    track_operation,
)

__all__ = [
    "get_logger",
    "get_opik_client",
    "track_operation",
    "track_llm_call",
    "track_embedding_call",
    "log_metric",
    "log_event",
]
