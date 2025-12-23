"""
Memory Context Nodes

Retrieve and inject conversation history and long-term user memory
into the RAG pipeline.
"""

from graph.state import RAGState

from shared.memory.manager import (
    get_conversation_manager,
    get_user_memory_manager,
)
from shared.utils.logger import get_logger
from shared.utils.opik_tracer import track_langgraph_node

logger = get_logger(__name__)


@track_langgraph_node("retrieve_conversation_context")
def retrieve_conversation_context(state: RAGState) -> RAGState:
    """
    Retrieve conversation history for context.

    Args:
        state: Current RAG state

    Returns:
        Updated state with conversation history
    """
    conversation_id = state.get("conversation_id")
    user_id = state.get("user_id")

    logger.info(
        "Retrieving conversation context",
        extra={
            "conversation_id": conversation_id,
            "user_id": user_id,
            "has_conversation": bool(conversation_id),
        },
    )

    # Initialize conversation context in state
    state["conversation_history"] = []
    state["user_context_summary"] = ""

    if not conversation_id:
        logger.debug("No conversation_id provided - starting fresh conversation")
        return state

    try:
        # Get conversation history
        conv_manager = get_conversation_manager()
        messages = conv_manager.get_conversation_history(
            conversation_id, limit=10  # Last 10 messages
        )

        if messages:
            state["conversation_history"] = messages
            logger.info(
                f"Retrieved {len(messages)} messages from conversation history",
                extra={
                    "conversation_id": conversation_id,
                    "message_count": len(messages),
                },
            )

            # Create summary of conversation context
            summary_parts = []
            for msg in messages[-5:]:  # Last 5 messages for immediate context
                role = msg["role"]
                content = msg["content"][:200]  # Truncate for summary
                summary_parts.append(f"{role.upper()}: {content}")

            state["conversation_context_summary"] = "\n".join(summary_parts)
        else:
            logger.debug("No previous messages in conversation")

    except Exception as e:
        logger.error(
            f"Error retrieving conversation context: {str(e)}",
            extra={"conversation_id": conversation_id},
        )

    return state


@track_langgraph_node("retrieve_user_memory")
def retrieve_user_memory(state: RAGState) -> RAGState:
    """
    Retrieve long-term user memory.

    Args:
        state: Current RAG state

    Returns:
        Updated state with user memory context
    """
    user_id = state.get("user_id")

    logger.info(
        "Retrieving user memory",
        extra={"user_id": user_id, "has_user_id": bool(user_id)},
    )

    # Initialize user memory in state
    state["user_memories"] = []
    state["user_profile"] = {}

    if not user_id:
        logger.debug("No user_id provided - no long-term memory to retrieve")
        return state

    try:
        memory_manager = get_user_memory_manager()

        # Retrieve all memory types
        all_memories = memory_manager.get_user_memories(user_id, limit=50)

        if all_memories:
            state["user_memories"] = all_memories

            # Group memories by type for easy access
            user_profile = {
                "facts": [],
                "preferences": [],
                "interactions": [],
                "context": [],
            }

            for mem in all_memories:
                mem_type = mem["type"]
                if mem_type in user_profile:
                    user_profile[mem_type].append(
                        {"key": mem["key"], "value": mem["value"]}
                    )

            state["user_profile"] = user_profile

            logger.info(
                f"Retrieved {len(all_memories)} user memories",
                extra={
                    "user_id": user_id,
                    "memory_count": len(all_memories),
                    "facts": len(user_profile.get("facts", [])),
                    "preferences": len(user_profile.get("preferences", [])),
                },
            )

            # Create user context summary
            summary_parts = []

            if user_profile.get("facts"):
                facts_str = ", ".join(
                    [f"{f['key']}: {f['value']}" for f in user_profile["facts"][:5]]
                )
                summary_parts.append(f"User Facts: {facts_str}")

            if user_profile.get("preferences"):
                prefs_str = ", ".join(
                    [
                        f"{p['key']}: {p['value']}"
                        for p in user_profile["preferences"][:5]
                    ]
                )
                summary_parts.append(f"User Preferences: {prefs_str}")

            state["user_context_summary"] = " | ".join(summary_parts)

        else:
            logger.debug("No user memories found")

    except Exception as e:
        logger.error(
            f"Error retrieving user memory: {str(e)}", extra={"user_id": user_id}
        )

    return state


@track_langgraph_node("store_conversation_message")
def store_conversation_message(state: RAGState) -> RAGState:
    """
    Store conversation messages after processing.

    This node should be called AFTER the final answer is generated.

    Args:
        state: Current RAG state

    Returns:
        Updated state
    """
    conversation_id = state.get("conversation_id")
    user_id = state.get("user_id")
    query = state.get("query")
    final_answer = state.get("final_answer")

    if not conversation_id or not user_id:
        logger.debug("No conversation_id or user_id - skipping message storage")
        return state

    try:
        from shared.models.conversation import MessageRole

        conv_manager = get_conversation_manager()

        # Store user message
        conv_manager.add_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=query,
            metadata={
                "intent": state.get("intent"),
                "ai_domain": str(state.get("ai_domain")),
                "risk_category": str(state.get("risk_category")),
            },
        )

        # Store assistant response
        if final_answer:
            conv_manager.add_message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=final_answer,
                metadata={
                    "citations": [
                        {
                            "regulation": c.regulation,
                            "article": c.article,
                            "excerpt": c.excerpt[:100],
                        }
                        for c in state.get("citations", [])
                    ],
                    "confidence_score": state.get("confidence_score"),
                    "iteration_count": state.get("iteration_count"),
                    "processing_time_ms": state.get("processing_time_ms"),
                },
            )

        logger.info(
            "Stored conversation messages",
            extra={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "has_answer": bool(final_answer),
            },
        )

    except Exception as e:
        logger.error(
            f"Error storing conversation message: {str(e)}",
            extra={"conversation_id": conversation_id},
        )

    return state


@track_langgraph_node("extract_user_memories")
def extract_user_memories(state: RAGState) -> RAGState:
    """
    Extract and store long-term memories from conversation.

    Analyzes the conversation to extract facts, preferences, and context
    about the user for future personalization.

    Args:
        state: Current RAG state

    Returns:
        Updated state
    """
    user_id = state.get("user_id")
    conversation_id = state.get("conversation_id")

    if not user_id:
        logger.debug("No user_id - skipping memory extraction")
        return state

    try:
        query = state.get("query", "")
        intent = state.get("intent", "")
        ai_domain = state.get("ai_domain")

        memory_manager = get_user_memory_manager()

        # Extract domain context
        if ai_domain:
            memory_manager.store_memory(
                user_id=user_id,
                memory_type="context",
                key="primary_ai_domain",
                value=str(ai_domain),
                source_conversation_id=conversation_id,
                confidence=7,
                importance=6,
            )

        # Extract intent patterns
        if intent:
            memory_manager.store_memory(
                user_id=user_id,
                memory_type="interaction",
                key="common_query_types",
                value=intent,
                source_conversation_id=conversation_id,
                confidence=6,
                importance=5,
            )

        # TODO: Add LLM-based memory extraction
        # This could use an LLM to analyze the conversation and extract:
        # - Job role mentions (e.g., "As a developer...")
        # - Company/industry context
        # - Specific compliance concerns
        # - Preferences for answer style

        logger.debug(
            f"Extracted memories for user {user_id}",
            extra={"user_id": user_id, "conversation_id": conversation_id},
        )

    except Exception as e:
        logger.error(
            f"Error extracting user memories: {str(e)}",
            extra={"user_id": user_id},
        )

    return state
