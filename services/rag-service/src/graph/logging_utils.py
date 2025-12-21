"""
Logging utilities for RAG pipeline graph nodes.

Provides consistent, extensive logging across all graph nodes with proper context.
"""

import time
from typing import Any

from shared.utils.logger import get_logger

logger = get_logger(__name__)


class NodeLogger:
    """Helper class for logging graph node execution."""

    def __init__(self, node_name: str, node_type: str = "processing"):
        """
        Initialize node logger.

        Args:
            node_name: Name of the graph node
            node_type: Type of node (analysis, agent, synthesis, validation)
        """
        self.node_name = node_name
        self.node_type = node_type
        self.start_time = None

    def log_entry(self, state: dict[str, Any], **extra):
        """
        Log node entry.

        Args:
            state: Current RAG state
            **extra: Additional context to log
        """
        self.start_time = time.time()

        logger.info(
            f"┌─ [{self.node_type.upper()}] {self.node_name} - STARTED",
        )

        logger.debug(
            f"Node '{self.node_name}' execution started",
            extra={
                "node_name": self.node_name,
                "node_type": self.node_type,
                "iteration": state.get("iteration_count", 0),
                "query_preview": state.get("query", "")[:100],
                **extra,
            },
        )

    def log_exit(self, state: dict[str, Any], **extra):
        """
        Log node exit with performance metrics.

        Args:
            state: Updated RAG state
            **extra: Additional context to log
        """
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
        else:
            duration_ms = 0

        logger.info(
            f"└─ [{self.node_type.upper()}] {self.node_name} - COMPLETED ({duration_ms:.0f}ms)",
        )

        logger.log_performance(
            operation=f"node_{self.node_name}",
            duration_ms=duration_ms,
            **extra,
        )

    def log_error(self, error: Exception, state: dict[str, Any], **extra):
        """
        Log node error.

        Args:
            error: Exception that occurred
            state: Current RAG state
            **extra: Additional context
        """
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        logger.error(
            f"✗ [{self.node_type.upper()}] {self.node_name} - FAILED ({duration_ms:.0f}ms)",
        )

        logger.log_error_with_context(
            message=f"Error in node '{self.node_name}'",
            error=error,
            node_name=self.node_name,
            node_type=self.node_type,
            iteration=state.get("iteration_count", 0),
            **extra,
        )

    def log_info(self, message: str, **extra):
        """Log informational message."""
        logger.info(f"  [{self.node_name}] {message}", extra=extra)

    def log_debug(self, message: str, **extra):
        """Log debug message."""
        logger.debug(f"  [{self.node_name}] {message}", extra=extra)

    def log_warning(self, message: str, **extra):
        """Log warning message."""
        logger.warning(f"  [{self.node_name}] {message}", extra=extra)


def log_state_transition(from_node: str, to_node: str, decision: str = None, state: dict[str, Any] = None):
    """
    Log state transition between nodes.

    Args:
        from_node: Source node name
        to_node: Destination node name
        decision: Decision logic (for conditional edges)
        state: Current state (optional)
    """
    transition_arrow = "→"

    if decision:
        logger.info(
            f"  {from_node} {transition_arrow} {to_node} (decision: {decision})",
        )
        logger.debug(
            "Graph state transition",
            extra={
                "from_node": from_node,
                "to_node": to_node,
                "decision": decision,
                "iteration": state.get("iteration_count", 0) if state else None,
            },
        )
    else:
        logger.debug(
            f"  {from_node} {transition_arrow} {to_node}",
        )


def log_llm_call(node_name: str, prompt_preview: str, response_preview: str, duration_ms: float, **extra):
    """
    Log LLM API call from a node.

    Args:
        node_name: Node making the LLM call
        prompt_preview: First 100 chars of prompt
        response_preview: First 100 chars of response
        duration_ms: Call duration in milliseconds
        **extra: Additional context (model, tokens, etc.)
    """
    logger.debug(
        f"  [{node_name}] LLM call completed ({duration_ms:.0f}ms)",
        extra={
            "node_name": node_name,
            "prompt_preview": prompt_preview[:100],
            "response_preview": response_preview[:100],
            "duration_ms": duration_ms,
            **extra,
        },
    )


def log_retrieval(node_name: str, query: str, chunks_retrieved: int, top_score: float = None, **extra):
    """
    Log retrieval operation from a node.

    Args:
        node_name: Node performing retrieval
        query: Retrieval query
        chunks_retrieved: Number of chunks retrieved
        top_score: Score of best match
        **extra: Additional context
    """
    logger.info(
        f"  [{node_name}] Retrieved {chunks_retrieved} chunks (top_score: {top_score:.3f if top_score else 'N/A'})",
    )

    logger.debug(
        "Retrieval operation",
        extra={
            "node_name": node_name,
            "retrieval_query": query[:200],
            "chunks_retrieved": chunks_retrieved,
            "top_score": top_score,
            **extra,
        },
    )


def log_sub_queries(sub_queries: list, node_name: str = "decompose_query"):
    """
    Log generated sub-queries.

    Args:
        sub_queries: List of SubQuery objects
        node_name: Node that generated sub-queries
    """
    logger.info(f"  [{node_name}] Generated {len(sub_queries)} sub-queries")

    for i, sq in enumerate(sub_queries):
        priority_icon = "🔴" if sq.priority == 1 else "🟡" if sq.priority == 2 else "🟢"
        logger.info(
            f"    {i+1}. {priority_icon} [P{sq.priority}] {sq.aspect}: {sq.question[:100]}",
        )

    logger.debug(
        "Sub-queries generated",
        extra={
            "node_name": node_name,
            "count": len(sub_queries),
            "priorities": {
                "critical": sum(1 for sq in sub_queries if sq.priority == 1),
                "important": sum(1 for sq in sub_queries if sq.priority == 2),
                "supplementary": sum(1 for sq in sub_queries if sq.priority == 3),
            },
            "aspects": [sq.aspect for sq in sub_queries],
        },
    )


def log_agent_action(action: Any, node_name: str = "react"):
    """
    Log ReAct agent action.

    Args:
        action: AgentAction object
        node_name: Agent node name
    """
    logger.info(
        f"  [{node_name}] Step {action.step}: {action.action}",
    )

    logger.debug(
        "Agent action",
        extra={
            "node_name": node_name,
            "step": action.step,
            "action": action.action,
            "thought_preview": action.thought[:200] if action.thought else None,
            "observation_preview": action.observation[:200] if action.observation else None,
        },
    )


def log_validation_result(passed: bool, reason: str = None, score: float = None, node_name: str = "validation"):
    """
    Log validation result.

    Args:
        passed: Whether validation passed
        reason: Reason for validation result
        score: Validation score (if applicable)
        node_name: Validation node name
    """
    status = "PASSED ✓" if passed else "FAILED ✗"
    logger.info(
        f"  [{node_name}] Validation {status}" + (f" (score: {score:.3f})" if score else ""),
    )

    if not passed and reason:
        logger.warning(
            f"    Reason: {reason}",
        )

    logger.debug(
        "Validation result",
        extra={
            "node_name": node_name,
            "passed": passed,
            "reason": reason,
            "score": score,
        },
    )


def log_synthesis_metrics(answer_length: int, citations_count: int, confidence: float = None, node_name: str = "synthesis"):
    """
    Log answer synthesis metrics.

    Args:
        answer_length: Length of generated answer
        citations_count: Number of citations
        confidence: Confidence score
        node_name: Synthesis node name
    """
    logger.info(
        f"  [{node_name}] Answer synthesized: {answer_length} chars, {citations_count} citations",
    )

    logger.debug(
        "Synthesis metrics",
        extra={
            "node_name": node_name,
            "answer_length": answer_length,
            "citations_count": citations_count,
            "confidence_score": confidence,
        },
    )
