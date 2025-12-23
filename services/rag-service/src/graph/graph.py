"""LangGraph compilation for agentic RAG pipeline."""

from typing import Literal

from graph.nodes.analysis import (
    analyze_query,
    decompose_query,
    safety_check,
)
from graph.nodes.memory import (
    extract_user_memories,
    retrieve_conversation_context,
    retrieve_user_memory,
    store_conversation_message,
)
from graph.nodes.react_agent import (
    react_act,
    react_observe,
    react_plan,
)
from graph.nodes.synthesis import (
    format_response,
    synthesize_answer,
)
from graph.nodes.validation import validate_grounding
from graph.state import RAGState
from langgraph.graph import END, StateGraph

from shared.memory.checkpointer import get_checkpointer
from shared.utils.logger import get_logger
from shared.utils.opik_tracer import track_rag_pipeline

logger = get_logger(__name__)


def should_continue_react(
    state: RAGState,
) -> Literal["continue_planning", "synthesize"]:
    """
    Decide whether to continue ReAct loop or move to synthesis.

    Args:
        state: Current RAG state

    Returns:
        "continue_planning" to loop back to planning
        "synthesize" to move to synthesis
    """
    agent_state = state.get("agent_state", "done")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    logger.debug(
        "ReAct loop decision point",
        extra={
            "agent_state": agent_state,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "has_error": bool(state.get("error")),
            "retrieved_chunks_count": len(state.get("all_retrieved_chunks", [])),
        },
    )

    # Check if agent marked as done
    if agent_state == "done":
        logger.info(
            "ReAct loop complete - Agent marked as done",
            extra={
                "iterations_used": iteration_count,
                "total_retrievals": len(state.get("retrieval_history", [])),
                "total_actions": len(state.get("agent_actions", [])),
            },
        )
        return "synthesize"

    # Check iteration limit
    if iteration_count >= max_iterations:
        logger.warning(
            "ReAct loop terminated - Max iterations reached",
            extra={
                "max_iterations": max_iterations,
                "total_retrievals": len(state.get("retrieval_history", [])),
                "last_action": state.get("agent_actions", [])[-1].action if state.get("agent_actions") else None,
            },
        )
        return "synthesize"

    # Check if error occurred
    if state.get("error"):
        logger.error(
            "ReAct loop terminated - Error occurred",
            extra={"error": state["error"], "iteration_count": iteration_count},
        )
        return "synthesize"

    # Continue planning
    logger.debug(
        "ReAct loop continuing",
        extra={"next_iteration": iteration_count + 1, "max_iterations": max_iterations},
    )
    return "continue_planning"


def should_refuse(state: RAGState) -> Literal["continue", "refuse"]:
    """
    Check if query should be refused due to safety concerns.

    Args:
        state: Current RAG state

    Returns:
        "continue" to proceed with processing
        "refuse" to skip to response formatting with refusal
    """
    is_safe = state.get("is_safe", True)
    refusal_reason = state.get("refusal_reason")

    logger.debug(
        "Safety check decision point",
        extra={
            "is_safe": is_safe,
            "has_refusal_reason": bool(refusal_reason),
            "query_length": len(state.get("query", "")),
        },
    )

    if not is_safe:
        logger.warning(
            "Query refused - Safety concerns",
            extra={
                "refusal_reason": refusal_reason,
                "query_preview": state.get("query", "")[:100],
                "intent": state.get("intent"),
            },
        )
        return "refuse"

    logger.debug("Safety check passed - Proceeding to ReAct agent")
    return "continue"


def should_regenerate(state: RAGState) -> Literal["success", "regenerate"]:
    """
    Check if answer passed grounding validation.

    Args:
        state: Current RAG state

    Returns:
        "success" if validation passed
        "regenerate" if answer needs regeneration (handled within validation node)
    """
    if state.get("grounding_validated", False):
        return "success"

    # Check if we've exceeded retries
    if state.get("retry_count", 0) >= 2:
        logger.warning("Max retries reached for grounding validation")
        return "success"  # Give up and format as-is

    # Validation node handles regeneration internally
    return "success"


def build_rag_graph() -> StateGraph:
    """
    Build the complete agentic RAG graph.

    Graph structure:
    1. Memory Retrieval → Conversation Context → User Memory
    2. Query Analysis → Decomposition → Safety Check
    3. ReAct Loop: Plan → Act → Observe → (loop or continue)
    4. Synthesis → Validation → Memory Storage → Format Response

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building agentic RAG graph")

    workflow = StateGraph(RAGState)

    # ===== Add Nodes =====

    # Memory retrieval phase
    workflow.add_node("retrieve_conversation_context", retrieve_conversation_context)
    workflow.add_node("retrieve_user_memory", retrieve_user_memory)

    # Analysis phase
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("safety_check", safety_check)

    # ReAct agent loop
    workflow.add_node("react_plan", react_plan)
    workflow.add_node("react_act", react_act)
    workflow.add_node("react_observe", react_observe)

    # Synthesis and validation
    workflow.add_node("synthesize_answer", synthesize_answer)
    workflow.add_node("validate_grounding", validate_grounding)

    # Memory storage phase
    workflow.add_node("store_conversation_message", store_conversation_message)
    workflow.add_node("extract_user_memories", extract_user_memories)

    # Final formatting
    workflow.add_node("format_response", format_response)

    # ===== Define Edges =====

    # Entry point - Start with memory retrieval
    workflow.set_entry_point("retrieve_conversation_context")

    # Memory retrieval flow
    workflow.add_edge("retrieve_conversation_context", "retrieve_user_memory")
    workflow.add_edge("retrieve_user_memory", "analyze_query")

    # Analysis flow
    workflow.add_edge("analyze_query", "decompose_query")
    workflow.add_edge("decompose_query", "safety_check")

    # Safety check conditional
    workflow.add_conditional_edges(
        "safety_check",
        should_refuse,
        {
            "continue": "react_plan",
            "refuse": "format_response",  # Skip to formatting with refusal
        },
    )

    # ReAct loop
    workflow.add_edge("react_plan", "react_act")
    workflow.add_edge("react_act", "react_observe")

    # Observe conditional: loop or synthesize
    workflow.add_conditional_edges(
        "react_observe",
        should_continue_react,
        {
            "continue_planning": "react_plan",  # Loop back
            "synthesize": "synthesize_answer",  # Move forward
        },
    )

    # Synthesis and validation flow
    workflow.add_edge("synthesize_answer", "validate_grounding")

    # Validation conditional (validation handles regeneration internally)
    workflow.add_conditional_edges(
        "validate_grounding",
        should_regenerate,
        {
            "success": "store_conversation_message",
            "regenerate": "validate_grounding",  # Shouldn't reach here
        },
    )

    # Memory storage flow
    workflow.add_edge("store_conversation_message", "extract_user_memories")
    workflow.add_edge("extract_user_memories", "format_response")

    # Format and end
    workflow.add_edge("format_response", END)

    logger.info("RAG graph built successfully")

    return workflow


def compile_rag_graph() -> StateGraph:
    """
    Compile the RAG graph for execution with persistent checkpointing.

    Returns:
        Compiled LangGraph workflow ready for invocation with memory persistence
    """
    workflow = build_rag_graph()

    # Get PostgreSQL checkpointer for persistent state
    checkpointer = get_checkpointer()

    if checkpointer:
        logger.info("Compiling RAG graph with PostgreSQL checkpointer for persistent memory")
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        logger.warning("PostgreSQL checkpointer not available - compiling without persistence")
        compiled = workflow.compile()

    logger.info("RAG graph compiled and ready")

    return compiled


# ===== Convenience function for running the graph =====


@track_rag_pipeline("eu_compliance_rag")
async def run_rag_pipeline(
    query: str,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> RAGState:
    """
    Run the complete RAG pipeline on a query with memory persistence.

    Args:
        query: User's compliance question
        conversation_id: Optional conversation tracking ID
        user_id: Optional user identifier for memory persistence

    Returns:
        Final RAG state with answer

    Example:
        result = await run_rag_pipeline(
            query="What are the obligations for high-risk AI systems in recruitment?",
            conversation_id="conv-123",
            user_id="user-456"
        )
        print(result["final_answer"])
    """
    import time

    from graph.state import create_initial_state

    start_time = time.time()

    logger.info(
        "╔═══════════════════════════════════════════════════════════════════╗",
    )
    logger.info(
        "║                 STARTING RAG PIPELINE EXECUTION                   ║",
    )
    logger.info(
        "╚═══════════════════════════════════════════════════════════════════╝",
    )

    logger.info(
        "Pipeline initialization",
        extra={
            "query": query,
            "query_length": len(query),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "has_memory_context": bool(conversation_id and user_id),
            "timestamp": time.time(),
        },
    )

    # Create initial state with user context
    logger.debug("Creating initial state with memory context")
    initial_state = create_initial_state(query, conversation_id, user_id)

    logger.debug(
        "Initial state created",
        extra={
            "max_iterations": initial_state.get("max_iterations"),
            "state_keys": list(initial_state.keys()),
        },
    )

    # Compile graph
    logger.debug("Compiling RAG graph")
    graph = compile_rag_graph()

    # Execute
    try:
        logger.info("▶ Executing RAG graph workflow")

        final_state = await graph.ainvoke(initial_state)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        final_state["processing_time_ms"] = processing_time

        # Comprehensive completion logging
        logger.info(
            "╔═══════════════════════════════════════════════════════════════════╗",
        )
        logger.info(
            "║              RAG PIPELINE COMPLETED SUCCESSFULLY                  ║",
        )
        logger.info(
            "╚═══════════════════════════════════════════════════════════════════╝",
        )

        logger.log_performance(
            operation="rag_pipeline_execution",
            duration_ms=processing_time,
            confidence_score=final_state.get("confidence_score", 0),
            iterations_used=final_state.get("iteration_count", 0),
            chunks_retrieved=len(final_state.get("all_retrieved_chunks", [])),
            citations_count=len(final_state.get("citations", [])),
            answer_length=len(final_state.get("final_answer", "")),
            was_refused=bool(final_state.get("refusal_reason")),
            grounding_validated=final_state.get("grounding_validated", False),
        )

        logger.info(
            "Pipeline results summary",
            extra={
                "processing_time_ms": processing_time,
                "confidence_score": final_state.get("confidence_score", 0),
                "iterations_used": final_state.get("iteration_count", 0),
                "max_iterations": final_state.get("max_iterations", 5),
                "total_retrievals": len(final_state.get("retrieval_history", [])),
                "total_chunks_retrieved": len(final_state.get("all_retrieved_chunks", [])),
                "unique_regulations": len(
                    set(chunk.get("regulation", "") for chunk in final_state.get("all_retrieved_chunks", []))
                ),
                "citations_count": len(final_state.get("citations", [])),
                "answer_length": len(final_state.get("final_answer", "")),
                "was_refused": bool(final_state.get("refusal_reason")),
                "grounding_validated": final_state.get("grounding_validated", False),
                "sub_queries_count": len(final_state.get("sub_queries", [])),
                "agent_actions_count": len(final_state.get("agent_actions", [])),
                "query_complexity": final_state.get("query_complexity"),
                "intent": final_state.get("intent"),
                "ai_domain": final_state.get("ai_domain"),
            },
        )

        # Log audit trail
        logger.log_audit(
            action="rag_pipeline_completed",
            resource="compliance_query",
            result="success",
            processing_time_ms=processing_time,
            confidence_score=final_state.get("confidence_score", 0),
            was_refused=bool(final_state.get("refusal_reason")),
        )

        return final_state

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000

        logger.error(
            "╔═══════════════════════════════════════════════════════════════════╗",
        )
        logger.error(
            "║                RAG PIPELINE EXECUTION FAILED                      ║",
        )
        logger.error(
            "╚═══════════════════════════════════════════════════════════════════╝",
        )

        logger.log_error_with_context(
            message="RAG pipeline execution failed",
            error=e,
            query=query,
            processing_time_ms=processing_time,
            conversation_id=conversation_id,
        )

        logger.log_audit(
            action="rag_pipeline_failed",
            resource="compliance_query",
            result="error",
            error_type=type(e).__name__,
            error_message=str(e),
            processing_time_ms=processing_time,
        )

        # Return state with error
        initial_state["error"] = str(e)
        initial_state["final_answer"] = ""
        initial_state["refusal_reason"] = f"An error occurred during processing: {str(e)}"
        initial_state["processing_time_ms"] = processing_time

        return initial_state
