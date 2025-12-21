"""
Logging utilities for data pipeline components.

Provides consistent, extensive logging across all data pipeline operations with proper context.
"""

import time

from shared.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineStageLogger:
    """Helper class for logging data pipeline stage execution."""

    def __init__(self, stage_name: str, stage_type: str = "processing"):
        """
        Initialize pipeline stage logger.

        Args:
            stage_name: Name of the pipeline stage
            stage_type: Type of stage (download, parse, chunk, embed, index)
        """
        self.stage_name = stage_name
        self.stage_type = stage_type
        self.start_time = None

    def log_start(self, **extra):
        """
        Log stage start.

        Args:
            **extra: Additional context to log
        """
        self.start_time = time.time()

        logger.info(
            f"┌─ [{self.stage_type.upper()}] {self.stage_name} - STARTED",
        )

        logger.debug(
            f"Stage '{self.stage_name}' execution started",
            extra={
                "stage_name": self.stage_name,
                "stage_type": self.stage_type,
                **extra,
            },
        )

    def log_complete(self, **extra):
        """
        Log stage completion with performance metrics.

        Args:
            **extra: Additional context to log
        """
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
        else:
            duration_ms = 0

        logger.info(
            f"└─ [{self.stage_type.upper()}] {self.stage_name} - COMPLETED ({duration_ms:.0f}ms)",
        )

        logger.log_performance(
            operation=f"pipeline_stage_{self.stage_name}",
            duration_ms=duration_ms,
            **extra,
        )

    def log_error(self, error: Exception, **extra):
        """
        Log stage error.

        Args:
            error: Exception that occurred
            **extra: Additional context
        """
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        logger.error(
            f"✗ [{self.stage_type.upper()}] {self.stage_name} - FAILED ({duration_ms:.0f}ms)",
        )

        logger.log_error_with_context(
            message=f"Error in pipeline stage '{self.stage_name}'",
            error=error,
            stage_name=self.stage_name,
            stage_type=self.stage_type,
            **extra,
        )

    def log_info(self, message: str, **extra):
        """Log informational message."""
        logger.info(f"  [{self.stage_name}] {message}", extra=extra)

    def log_debug(self, message: str, **extra):
        """Log debug message."""
        logger.debug(f"  [{self.stage_name}] {message}", extra=extra)

    def log_warning(self, message: str, **extra):
        """Log warning message."""
        logger.warning(f"  [{self.stage_name}] {message}", extra=extra)


def log_download(
    celex: str,
    format: str,
    file_size: int,
    duration_ms: float,
    stage_name: str = "download"
):
    """
    Log document download operation.

    Args:
        celex: CELEX identifier
        format: Document format
        file_size: Downloaded file size in bytes
        duration_ms: Download duration
        stage_name: Stage name for logging
    """
    logger.info(
        f"  [{stage_name}] Downloaded {celex} ({format}, {file_size:,} bytes, {duration_ms:.0f}ms)",
    )

    logger.debug(
        "Document download completed",
        extra={
            "stage_name": stage_name,
            "celex": celex,
            "format": format,
            "file_size_bytes": file_size,
            "duration_ms": duration_ms,
            "throughput_kbps": (file_size / 1024) / (duration_ms / 1000) if duration_ms > 0 else 0,
        },
    )


def log_parsing_result(
    celex: str,
    chapters_count: int,
    articles_count: int,
    duration_ms: float,
    stage_name: str = "parse"
):
    """
    Log document parsing result.

    Args:
        celex: CELEX identifier
        chapters_count: Number of chapters parsed
        articles_count: Number of articles parsed
        duration_ms: Parsing duration
        stage_name: Stage name for logging
    """
    logger.info(
        f"  [{stage_name}] Parsed {celex}: {chapters_count} chapters, {articles_count} articles ({duration_ms:.0f}ms)",
    )

    logger.debug(
        "Document parsing completed",
        extra={
            "stage_name": stage_name,
            "celex": celex,
            "chapters_count": chapters_count,
            "articles_count": articles_count,
            "duration_ms": duration_ms,
        },
    )


def log_chunking_result(
    celex: str,
    chunks_created: int,
    chunks_filtered: int,
    avg_chunk_length: float,
    duration_ms: float,
    stage_name: str = "chunk"
):
    """
    Log document chunking result.

    Args:
        celex: CELEX identifier
        chunks_created: Number of valid chunks created
        chunks_filtered: Number of chunks filtered out
        avg_chunk_length: Average chunk length in characters
        duration_ms: Chunking duration
        stage_name: Stage name for logging
    """
    logger.info(
        f"  [{stage_name}] Chunked {celex}: {chunks_created} chunks (filtered: {chunks_filtered}, avg length: {avg_chunk_length:.0f} chars, {duration_ms:.0f}ms)",
    )

    logger.debug(
        "Document chunking completed",
        extra={
            "stage_name": stage_name,
            "celex": celex,
            "chunks_created": chunks_created,
            "chunks_filtered": chunks_filtered,
            "avg_chunk_length": avg_chunk_length,
            "duration_ms": duration_ms,
        },
    )


def log_embedding_batch(
    batch_index: int,
    total_batches: int,
    batch_size: int,
    duration_ms: float,
    stage_name: str = "embed"
):
    """
    Log embedding generation batch.

    Args:
        batch_index: Current batch index
        total_batches: Total number of batches
        batch_size: Number of chunks in batch
        duration_ms: Batch processing duration
        stage_name: Stage name for logging
    """
    logger.debug(
        f"  [{stage_name}] Batch {batch_index + 1}/{total_batches}: {batch_size} chunks ({duration_ms:.0f}ms)",
    )

    logger.log_performance(
        operation=f"{stage_name}_batch",
        duration_ms=duration_ms,
        batch_index=batch_index,
        batch_size=batch_size,
        chunks_per_second=batch_size / (duration_ms / 1000) if duration_ms > 0 else 0,
    )


def log_embedding_result(
    celex: str,
    embeddings_count: int,
    total_duration_ms: float,
    avg_batch_duration_ms: float,
    stage_name: str = "embed"
):
    """
    Log embedding generation result.

    Args:
        celex: CELEX identifier
        embeddings_count: Number of embeddings generated
        total_duration_ms: Total duration
        avg_batch_duration_ms: Average batch duration
        stage_name: Stage name for logging
    """
    throughput = embeddings_count / (total_duration_ms / 1000) if total_duration_ms > 0 else 0

    logger.info(
        f"  [{stage_name}] Generated embeddings for {celex}: {embeddings_count} embeddings ({total_duration_ms:.0f}ms, {throughput:.1f} emb/sec)",
    )

    logger.debug(
        "Embedding generation completed",
        extra={
            "stage_name": stage_name,
            "celex": celex,
            "embeddings_count": embeddings_count,
            "total_duration_ms": total_duration_ms,
            "avg_batch_duration_ms": avg_batch_duration_ms,
            "throughput_per_second": throughput,
        },
    )


def log_indexing_batch(
    batch_index: int,
    batch_size: int,
    duration_ms: float,
    stage_name: str = "index"
):
    """
    Log indexing batch operation.

    Args:
        batch_index: Batch index
        batch_size: Number of points in batch
        duration_ms: Batch indexing duration
        stage_name: Stage name for logging
    """
    logger.debug(
        f"  [{stage_name}] Indexed batch {batch_index}: {batch_size} points ({duration_ms:.0f}ms)",
    )


def log_indexing_result(
    celex: str,
    points_indexed: int,
    collection_name: str,
    duration_ms: float,
    stage_name: str = "index"
):
    """
    Log indexing result.

    Args:
        celex: CELEX identifier
        points_indexed: Number of points indexed
        collection_name: Qdrant collection name
        duration_ms: Total indexing duration
        stage_name: Stage name for logging
    """
    throughput = points_indexed / (duration_ms / 1000) if duration_ms > 0 else 0

    logger.info(
        f"  [{stage_name}] Indexed {celex}: {points_indexed} points to '{collection_name}' ({duration_ms:.0f}ms, {throughput:.1f} pts/sec)",
    )

    logger.debug(
        "Indexing completed",
        extra={
            "stage_name": stage_name,
            "celex": celex,
            "points_indexed": points_indexed,
            "collection_name": collection_name,
            "duration_ms": duration_ms,
            "throughput_per_second": throughput,
        },
    )


def log_api_call(
    api_name: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    retry_count: int = 0,
    **extra
):
    """
    Log API call.

    Args:
        api_name: API name (e.g., "EUR-Lex", "OpenAI")
        endpoint: Endpoint or operation name
        status_code: HTTP status code or success indicator
        duration_ms: Request duration
        retry_count: Number of retries
        **extra: Additional context
    """
    status = "success" if status_code < 400 else "error"

    logger.debug(
        f"API call: {api_name} {endpoint} ({status}, {duration_ms:.0f}ms, retries: {retry_count})",
        extra={
            "api_name": api_name,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "retry_count": retry_count,
            **extra,
        },
    )


def log_validation_result(
    validation_name: str,
    passed: bool,
    items_validated: int,
    items_failed: int = 0,
    stage_name: str = "validation"
):
    """
    Log validation result.

    Args:
        validation_name: Name of validation
        passed: Whether validation passed
        items_validated: Number of items validated
        items_failed: Number of items that failed
        stage_name: Stage name for logging
    """
    status = "PASSED ✓" if passed else "FAILED ✗"

    logger.info(
        f"  [{stage_name}] {validation_name}: {status} ({items_validated} validated, {items_failed} failed)",
    )

    logger.debug(
        "Validation result",
        extra={
            "stage_name": stage_name,
            "validation_name": validation_name,
            "passed": passed,
            "items_validated": items_validated,
            "items_failed": items_failed,
        },
    )


def log_data_quality(
    metric_name: str,
    value: float,
    threshold: float = None,
    passed: bool = None,
    stage_name: str = "quality_check"
):
    """
    Log data quality metric.

    Args:
        metric_name: Name of quality metric
        value: Metric value
        threshold: Threshold value (if applicable)
        passed: Whether quality check passed
        stage_name: Stage name for logging
    """
    if passed is not None:
        status = "✓" if passed else "✗"
        logger.info(
            f"  [{stage_name}] {metric_name}: {value:.2f} {status}",
        )
    else:
        logger.info(
            f"  [{stage_name}] {metric_name}: {value:.2f}",
        )

    logger.debug(
        "Data quality metric",
        extra={
            "stage_name": stage_name,
            "metric_name": metric_name,
            "value": value,
            "threshold": threshold,
            "passed": passed,
        },
    )
