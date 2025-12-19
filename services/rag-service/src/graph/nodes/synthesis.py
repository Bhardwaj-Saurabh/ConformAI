"""Answer synthesis and formatting nodes."""

import re
from typing import Any

from langchain_core.messages import HumanMessage

from services.rag_service.src.graph.state import Citation, RAGState
from services.rag_service.src.llm.client import get_generation_llm
from shared.utils.logger import get_logger

logger = get_logger(__name__)


async def synthesize_answer(state: RAGState) -> dict[str, Any]:
    """
    Synthesize final answer from intermediate answers or direct generation.

    Args:
        state: Current RAG state

    Returns:
        Updated state with final answer
    """
    logger.info("Synthesizing final answer")

    # Check if we have intermediate answers to synthesize
    if state["intermediate_answers"] and len(state["intermediate_answers"]) > 1:
        # Multi-step synthesis from sub-answers
        return await synthesize_from_sub_answers(state)
    elif state["all_retrieved_chunks"]:
        # Direct generation from retrieved chunks
        return await generate_direct_answer(state)
    else:
        # No sources available
        logger.error("No sources available for answer generation")
        state["final_answer"] = ""
        state["refusal_reason"] = "Unable to find relevant legal sources for this query"
        return state


async def synthesize_from_sub_answers(state: RAGState) -> dict[str, Any]:
    """Synthesize answer from multiple sub-question answers."""

    logger.info(
        f"Synthesizing from {len(state['intermediate_answers'])} sub-answers"
    )

    try:
        llm = get_generation_llm()

        # Format sub-answers
        formatted_answers = "\n\n".join(
            [
                f"**Sub-question {i+1}**: {question}\n**Answer**: {answer}"
                for i, (question, answer) in enumerate(state["intermediate_answers"].items())
            ]
        )

        synthesis_prompt = f"""You are an EU AI regulation expert. Synthesize these sub-answers into a comprehensive response.

ORIGINAL QUERY:
{state["query"]}

SUB-QUESTION ANSWERS:
{formatted_answers}

TASK:
1. Create a coherent, well-structured answer that fully addresses the original query
2. Maintain ALL citations from sub-answers in [Source X: Article Y] format
3. Resolve any contradictions or overlaps between sub-answers
4. Ensure logical flow and readability
5. Use clear markdown headings (##) if the answer has multiple distinct sections
6. Do NOT add new information beyond what's in the sub-answers

SYNTHESIZED ANSWER:"""

        response = await llm.ainvoke([HumanMessage(content=synthesis_prompt)])

        state["final_answer"] = response.content
        state["reasoning_trace"].append("Synthesized final answer from sub-answers")

        logger.info("Synthesis complete")

        return state

    except Exception as e:
        logger.error(f"Error in synthesize_from_sub_answers: {e}")
        # Fallback: concatenate sub-answers
        state["final_answer"] = "\n\n".join(state["intermediate_answers"].values())
        return state


async def generate_direct_answer(state: RAGState) -> dict[str, Any]:
    """Generate answer directly from retrieved chunks (single-pass)."""

    logger.info(
        f"Generating direct answer from {len(state['all_retrieved_chunks'])} chunks"
    )

    try:
        llm = get_generation_llm()

        # Format retrieved sources
        formatted_sources = "\n\n".join(
            [
                f"[Source {i+1}] {chunk.get('regulation', 'Unknown')}, "
                f"{chunk.get('article', '')}\n{chunk.get('content', '')}"
                for i, chunk in enumerate(state["all_retrieved_chunks"][:15])  # Limit to top 15
            ]
        )

        generation_prompt = f"""You are an EU AI regulation compliance assistant. Answer this question using ONLY the provided legal sources.

LEGAL SOURCES:
{formatted_sources}

USER QUERY:
{state["query"]}

INSTRUCTIONS:
1. Answer using ONLY the provided sources - do not add external knowledge
2. Cite every claim with [Source X: Article Y]
3. If sources don't fully cover the question, explicitly state "The provided sources do not contain complete information about [specific aspect]"
4. DO NOT speculate or make claims not supported by sources
5. Use clear, precise legal language
6. Structure answer with markdown headings (##) if complex
7. Be comprehensive but concise

ANSWER:"""

        response = await llm.ainvoke([HumanMessage(content=generation_prompt)])

        state["final_answer"] = response.content
        state["reasoning_trace"].append("Generated answer directly from retrieved sources")

        logger.info("Direct generation complete")

        return state

    except Exception as e:
        logger.error(f"Error in generate_direct_answer: {e}")
        state["final_answer"] = ""
        state["error"] = f"Answer generation failed: {str(e)}"
        return state


async def format_response(state: RAGState) -> dict[str, Any]:
    """
    Format final response with citations, disclaimer, and metadata.

    Args:
        state: Current RAG state

    Returns:
        Updated state with formatted response
    """
    logger.info("Formatting final response")

    # Extract citations from final answer
    state["citations"] = extract_citations(state["final_answer"], state["all_retrieved_chunks"])

    # Add disclaimer if not a refusal
    if not state.get("refusal_reason"):
        disclaimer = (
            "\n\n---\n\n"
            "⚖️ **Disclaimer**: This information is for educational and informational purposes only. "
            "It does not constitute legal advice. For compliance decisions affecting your organization, "
            "consult a qualified legal professional."
        )

        state["final_answer"] = state["final_answer"] + disclaimer

    # Calculate confidence score
    state["confidence_score"] = calculate_confidence(state)

    # Add reasoning trace summary
    if state["reasoning_trace"]:
        state["reasoning_trace"].append(
            f"Final confidence score: {state['confidence_score']:.2f}"
        )

    logger.info(f"Response formatted. Confidence: {state['confidence_score']:.2f}")

    return state


# ===== Helper Functions =====


def extract_citations(answer: str, chunks: list[dict[str, Any]]) -> list[Citation]:
    """
    Extract citation references from answer text.

    Args:
        answer: Generated answer with citations
        chunks: Retrieved legal chunks

    Returns:
        List of Citation objects
    """
    citations = []

    # Pattern: [Source X: Article Y, ...] or [Source X]
    citation_pattern = r"\[Source (\d+):?\s*([^\]]*)\]"

    matches = re.finditer(citation_pattern, answer)

    for match in matches:
        source_num = int(match.group(1))
        article_ref = match.group(2).strip() if match.group(2) else None

        # Get chunk if index is valid
        if 0 < source_num <= len(chunks):
            chunk = chunks[source_num - 1]

            citation = Citation(
                source_id=source_num,
                regulation=chunk.get("regulation", "Unknown"),
                article=chunk.get("article") or article_ref,
                paragraph=chunk.get("paragraph"),
                celex=chunk.get("celex"),
                excerpt=chunk.get("content", "")[:200],  # First 200 chars
                chunk_id=chunk.get("id", f"chunk_{source_num}"),
            )

            citations.append(citation)

    logger.info(f"Extracted {len(citations)} citations from answer")

    return citations


def calculate_confidence(state: RAGState) -> float:
    """
    Calculate confidence score for the answer.

    Based on:
    - Retrieval quality (scores)
    - Number of sources
    - Citation density
    - Grounding validation

    Args:
        state: Current RAG state

    Returns:
        Confidence score (0.0 - 1.0)
    """
    confidence = 0.0

    # Retrieval quality (0-0.4)
    if state["retrieval_scores"]:
        avg_score = sum(state["retrieval_scores"]) / len(state["retrieval_scores"])
        confidence += min(avg_score, 1.0) * 0.4

    # Number of sources (0-0.2)
    chunk_count = len(state["all_retrieved_chunks"])
    if chunk_count >= 10:
        confidence += 0.2
    elif chunk_count >= 5:
        confidence += 0.15
    elif chunk_count >= 3:
        confidence += 0.1

    # Citation density (0-0.2)
    if state["citations"] and state["final_answer"]:
        citation_density = len(state["citations"]) / max(len(state["final_answer"].split()), 1)
        confidence += min(citation_density * 100, 1.0) * 0.2  # Normalize

    # Grounding validation (0-0.2)
    if state.get("grounding_validated", False):
        confidence += 0.2

    return min(confidence, 1.0)  # Cap at 1.0
