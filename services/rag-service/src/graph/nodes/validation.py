"""Grounding validation and hallucination detection."""

import re
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from graph.state import RAGState
from llm.client import get_planning_llm, invoke_llm
from shared.utils.logger import get_logger
from shared.utils.opik_tracer import track_langgraph_node

logger = get_logger(__name__)


class HallucinationCheckOutput(BaseModel):
    """Structured output for hallucination detection."""

    has_hallucination: bool
    unsupported_claims: list[str]
    explanation: str


@track_langgraph_node("validate_grounding", "validation")
async def validate_grounding(state: RAGState) -> dict[str, Any]:
    """
    Validate that answer is fully grounded in retrieved sources.

    Performs:
    1. Citation completeness check
    2. Citation validity check
    3. LLM-based hallucination detection

    Args:
        state: Current RAG state with final answer

    Returns:
        Updated state with grounding validation results
    """
    logger.info("Validating answer grounding")

    answer = state.get("final_answer", "")

    # Skip validation if no answer or it's a refusal
    if not answer or state.get("refusal_reason"):
        state["grounding_validated"] = True  # Refusals don't need validation
        return state

    chunks = state["all_retrieved_chunks"]

    if not chunks:
        logger.warning("No retrieved chunks to validate against")
        state["grounding_validated"] = False
        state["hallucination_detected"] = True
        return state

    try:
        # 1. Check citation completeness
        has_citations = check_citation_completeness(answer)

        if not has_citations:
            logger.warning("Answer lacks proper citations")
            state["grounding_validated"] = False
            state["hallucination_detected"] = False  # Not hallucination, just missing citations

            # Increment retry
            if state.get("retry_count", 0) < 2:
                state["retry_count"] += 1
                logger.info(
                    f"Regenerating with stricter prompt (retry {state['retry_count']}/2)"
                )
                return await regenerate_with_citations(state)
            else:
                state["refusal_reason"] = (
                    "Unable to generate properly cited answer after multiple attempts"
                )
                return state

        # 2. Validate citations reference valid chunks
        invalid_citations = validate_citations(answer, chunks)

        if invalid_citations:
            logger.warning(f"Found {len(invalid_citations)} invalid citation references")
            state["grounding_validated"] = False
            state["hallucination_detected"] = True

            if state.get("retry_count", 0) < 2:
                state["retry_count"] += 1
                return await regenerate_with_citations(state)
            else:
                state["refusal_reason"] = "Unable to generate answer with valid citations"
                return state

        # 3. LLM-based hallucination detection
        hallucination_result = await detect_hallucinations(answer, chunks)

        if hallucination_result.has_hallucination:
            logger.warning(
                f"Detected potential hallucinations: {hallucination_result.unsupported_claims}"
            )
            state["grounding_validated"] = False
            state["hallucination_detected"] = True

            if state.get("retry_count", 0) < 2:
                state["retry_count"] += 1
                return await regenerate_with_citations(state)
            else:
                state["refusal_reason"] = (
                    "Unable to generate fully grounded answer. "
                    "The available sources may not contain sufficient information."
                )
                return state

        # All checks passed
        state["grounding_validated"] = True
        state["hallucination_detected"] = False

        logger.info("Answer passed grounding validation")

        return state

    except Exception as e:
        logger.error(f"Error in validate_grounding: {e}")
        # On error, pass through (fail open)
        state["grounding_validated"] = True
        return state


# ===== Helper Functions =====


def check_citation_completeness(answer: str) -> bool:
    """
    Check if answer contains citations.

    Args:
        answer: Generated answer

    Returns:
        True if answer has citations, False otherwise
    """
    # Pattern: [Source X] or [Source X: ...]
    citation_pattern = r"\[Source \d+[:\]]"

    matches = re.findall(citation_pattern, answer)

    # Require at least one citation for substantial answers
    if len(answer.split()) > 50:  # Answers > 50 words need citations
        return len(matches) > 0
    else:
        return True  # Short answers may not need citations


def validate_citations(answer: str, chunks: list[dict[str, Any]]) -> list[str]:
    """
    Validate that citation references point to valid chunks.

    Args:
        answer: Generated answer with citations
        chunks: Retrieved chunks

    Returns:
        List of invalid citation references
    """
    invalid = []

    # Extract all citation numbers
    citation_pattern = r"\[Source (\d+)"
    matches = re.findall(citation_pattern, answer)

    for source_num_str in matches:
        source_num = int(source_num_str)

        # Check if source number is within valid range
        if source_num < 1 or source_num > len(chunks):
            invalid.append(f"Source {source_num}")

    return invalid


async def detect_hallucinations(
    answer: str, chunks: list[dict[str, Any]]
) -> HallucinationCheckOutput:
    """
    Use LLM to detect unsupported claims (hallucinations).

    Args:
        answer: Generated answer
        chunks: Retrieved legal chunks

    Returns:
        Hallucination detection results
    """
    try:
        llm = get_planning_llm()

        # Format sources
        formatted_sources = "\n\n".join(
            [
                f"[Source {i+1}] {chunk.get('regulation', '')}, "
                f"{chunk.get('article', '')}\n{chunk.get('content', '')}"
                for i, chunk in enumerate(chunks[:15])  # Limit to top 15
            ]
        )

        hallucination_prompt = f"""You are a fact-checker. Verify if this answer contains any claims not supported by the legal sources.

ANSWER TO CHECK:
{answer}

LEGAL SOURCES:
{formatted_sources}

TASK:
1. Identify any claims or statements in the answer that are NOT supported by the sources
2. Check if citations are accurate (do the cited sources actually support the claims?)
3. Look for speculative language or added information beyond the sources

Return JSON format:
{{
    "has_hallucination": <true if unsupported claims found, false otherwise>,
    "unsupported_claims": [
        "<list of specific claims that are not supported>"
    ],
    "explanation": "<brief explanation of findings>"
}}

Be strict: if a claim is not explicitly or clearly implied by the sources, mark it as unsupported."""

        response = await invoke_llm(llm, [HumanMessage(content=hallucination_prompt)])

        # Parse JSON response (simplified)
        import json

        try:
            result_json = json.loads(response.content)
            result = HallucinationCheckOutput(**result_json)
        except (json.JSONDecodeError, Exception):
            # Fallback: check for "yes" or "true" in response
            has_hallucination = any(
                word in response.content.lower()[:100]
                for word in ["true", "yes", "hallucination"]
            )

            result = HallucinationCheckOutput(
                has_hallucination=has_hallucination,
                unsupported_claims=[],
                explanation="Could not parse structured output",
            )

        return result

    except Exception as e:
        logger.error(f"Error in detect_hallucinations: {e}")
        # On error, pass (fail open)
        return HallucinationCheckOutput(
            has_hallucination=False,
            unsupported_claims=[],
            explanation="Error during hallucination check",
        )


async def regenerate_with_citations(state: RAGState) -> dict[str, Any]:
    """
    Regenerate answer with stricter citation requirements.

    Args:
        state: Current RAG state

    Returns:
        Updated state with regenerated answer
    """
    logger.info("Regenerating answer with stricter grounding requirements")

    from llm.client import get_generation_llm
    from langchain_core.messages import HumanMessage

    try:
        llm = get_generation_llm()

        # Format sources
        formatted_sources = "\n\n".join(
            [
                f"[Source {i+1}] {chunk.get('regulation', '')}, "
                f"{chunk.get('article', '')}\n{chunk.get('content', '')}"
                for i, chunk in enumerate(state["all_retrieved_chunks"][:15])
            ]
        )

        strict_prompt = f"""You are an EU AI regulation compliance assistant. Answer this question using ONLY the provided legal sources.

LEGAL SOURCES:
{formatted_sources}

USER QUERY:
{state["query"]}

CRITICAL INSTRUCTIONS:
1. **You MUST cite EVERY claim** with [Source X: Article Y]
2. **ONLY use information directly from the sources** - no external knowledge
3. **Do NOT make ANY claim without a citation**
4. If sources don't cover something, explicitly state "The provided sources do not address [topic]"
5. **Do NOT speculate, infer, or add information not in sources**
6. Use precise legal language from the sources
7. Structure with markdown headings if needed

EXAMPLE OF PROPER CITATION:
"High-risk AI systems must maintain technical documentation [Source 1: Article 11, EU AI Act]."

ANSWER (with citations for EVERY claim):"""

        response = await invoke_llm(llm, [HumanMessage(content=strict_prompt)])

        state["final_answer"] = response.content

        logger.info("Regenerated answer with stricter requirements")

        return state

    except Exception as e:
        logger.error(f"Error in regenerate_with_citations: {e}")
        return state
