"""LangGraph compilation for agentic RAG pipeline."""

from typing import Literal

from langgraph.graph import END, StateGraph

from graph.nodes.analysis import (
    analyze_query,
    decompose_query,
    safety_check,
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

    # Check if agent marked as done
    if agent_state == "done":
        return "synthesize"

    # Check iteration limit
    if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
        logger.warning("Max iterations reached, moving to synthesis")
        return "synthesize"

    # Check if error occurred
    if state.get("error"):
        logger.error(f"Error occurred: {state['error']}, stopping loop")
        return "synthesize"

    # Continue planning
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
    if not state.get("is_safe", True):
        logger.warning(f"Query refused: {state.get('refusal_reason')}")
        return "refuse"

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
    1. Query Analysis → Decomposition → Safety Check
    2. ReAct Loop: Plan → Act → Observe → (loop or continue)
    3. Synthesis → Validation → Format Response

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building agentic RAG graph")

    workflow = StateGraph(RAGState)

    # ===== Add Nodes =====

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
    workflow.add_node("format_response", format_response)

    # ===== Define Edges =====

    # Entry point
    workflow.set_entry_point("analyze_query")

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
            "success": "format_response",
            "regenerate": "validate_grounding",  # Shouldn't reach here
        },
    )

    # Format and end
    workflow.add_edge("format_response", END)

    logger.info("RAG graph built successfully")

    return workflow


def compile_rag_graph() -> StateGraph:
    """
    Compile the RAG graph for execution.

    Returns:
        Compiled LangGraph workflow ready for invocation
    """
    workflow = build_rag_graph()
    compiled = workflow.compile()

    logger.info("RAG graph compiled and ready")

    return compiled


# ===== Convenience function for running the graph =====


@track_rag_pipeline("eu_compliance_rag")
async def run_rag_pipeline(query: str, conversation_id: str | None = None) -> RAGState:
    """
    Run the complete RAG pipeline on a query.

    Args:
        query: User's compliance question
        conversation_id: Optional conversation tracking ID

    Returns:
        Final RAG state with answer

    Example:
        result = await run_rag_pipeline(
            "What are the obligations for high-risk AI systems in recruitment?"
        )
        print(result["final_answer"])
    """
    from graph.state import create_initial_state
    import time

    start_time = time.time()

    logger.info(f"Running RAG pipeline for query: {query[:100]}...")

    # Create initial state
    initial_state = create_initial_state(query, conversation_id)

    # Compile graph
    graph = compile_rag_graph()

    # Execute
    try:
        final_state = await graph.ainvoke(initial_state)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        final_state["processing_time_ms"] = processing_time

        logger.info(
            f"RAG pipeline completed in {processing_time:.0f}ms. "
            f"Confidence: {final_state.get('confidence_score', 0):.2f}"
        )

        return final_state

    except Exception as e:
        logger.error(f"Error running RAG pipeline: {e}", exc_info=True)
        # Return state with error
        initial_state["error"] = str(e)
        initial_state["final_answer"] = ""
        initial_state["refusal_reason"] = f"An error occurred during processing: {str(e)}"
        return initial_state
