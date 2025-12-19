"""Agent tools for ReAct loop - retrieval, generation, validation, etc."""

import asyncio
import httpx
import time
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shared.config.settings import get_settings
from shared.models.legal_document import Chunk
from shared.utils.logger import get_logger
from shared.utils.opik_tracer import trace_context, log_metric

settings = get_settings()
logger = get_logger(__name__)


# ===== Tool Input Schemas =====


class RetrievalInput(BaseModel):
    """Input schema for retrieve_legal_chunks tool."""

    query: str = Field(description="Search query for retrieving legal documents")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata filters (domain, regulation, risk_category)",
    )
    top_k: int = Field(default=10, description="Number of chunks to retrieve")


class AnswerInput(BaseModel):
    """Input schema for answer_sub_question tool."""

    sub_question: str = Field(description="The specific question to answer")
    legal_sources: list[dict[str, Any]] = Field(
        description="Retrieved legal chunks to use as evidence"
    )


class SynthesisInput(BaseModel):
    """Input schema for synthesize_information tool."""

    answers: dict[str, str] = Field(description="Sub-question to answer mapping")
    original_query: str = Field(description="The original user query")


class ValidationInput(BaseModel):
    """Input schema for validate_claim tool."""

    claim: str = Field(description="The claim to validate")
    legal_sources: list[dict[str, Any]] = Field(
        description="Legal chunks to check against"
    )


class ArticleSearchInput(BaseModel):
    """Input schema for search_specific_article tool."""

    regulation: str = Field(description="Regulation name (e.g., 'EU AI Act', 'GDPR')")
    article_number: str = Field(description="Article number (e.g., 'Article 9')")


# ===== Agent Tools =====


@tool(args_schema=RetrievalInput)
async def retrieve_legal_chunks(
    query: str, filters: dict[str, Any] | None = None, top_k: int = 10
) -> list[dict[str, Any]]:
    """
    Retrieve relevant legal document chunks from vector database.

    This tool searches the Qdrant vector database for legal chunks that are
    semantically similar to the query and match the provided metadata filters.

    Args:
        query: Search query describing what legal information to find
        filters: Optional metadata filters:
            - domain: AI application domain (e.g., "recruitment", "biometrics")
            - regulation: Regulation name (e.g., "EU AI Act", "GDPR")
            - risk_category: Risk level (e.g., "high", "prohibited")
        top_k: Number of most relevant chunks to return (default: 10)

    Returns:
        List of legal chunk dictionaries with content, metadata, and scores

    Example:
        retrieve_legal_chunks(
            query="documentation requirements for high-risk AI",
            filters={"regulation": "EU AI Act", "risk_category": "high"},
            top_k=10
        )
    """
    start_time = time.time()

    with trace_context(
        name="retrieve_legal_chunks",
        tags=["retrieval", "agent_tool", "conformai"],
        metadata={
            "query": query[:200],
            "filters": filters or {},
            "top_k": top_k,
        }
    ) as trace:
        try:
            # Log input
            if trace:
                trace.log_input({
                    "query": query,
                    "filters": filters,
                    "top_k": top_k,
                })

            # Call Retrieval Service
            retrieval_service_url = (
                f"http://{settings.retrieval_service_host}:"
                f"{settings.retrieval_service_port}/api/v1/retrieve"
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    retrieval_service_url,
                    json={"query": query, "filters": filters or {}, "top_k": top_k},
                )

                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()
                    chunks = result.get("chunks", [])
                    logger.info(
                        f"Retrieved {len(chunks)} chunks for query: {query[:50]}..."
                    )

                    # Log output
                    if trace:
                        trace.log_output({
                            "chunks_count": len(chunks),
                            "status": "success",
                        })
                        trace.update(output={
                            "status": "success",
                            "chunks_retrieved": len(chunks),
                            "duration_ms": duration_ms,
                        })

                    # Log metrics
                    log_metric("retrieval_duration_ms", duration_ms, {"top_k": str(top_k)})
                    log_metric("chunks_retrieved", len(chunks), {"top_k": str(top_k)})

                    return chunks
                else:
                    logger.error(
                        f"Retrieval service error: {response.status_code} - {response.text}"
                    )

                    if trace:
                        trace.update(output={
                            "status": "error",
                            "error": f"HTTP {response.status_code}",
                            "duration_ms": duration_ms,
                        }, error=True)

                    return []

        except httpx.RequestError as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to connect to retrieval service: {e}")

            if trace:
                trace.update(output={
                    "status": "error",
                    "error": f"Connection failed: {str(e)}",
                    "duration_ms": duration_ms,
                }, error=True)

            return []
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error in retrieve_legal_chunks: {e}")

            if trace:
                trace.update(output={
                    "status": "error",
                    "error": str(e),
                    "duration_ms": duration_ms,
                }, error=True)

            return []


@tool(args_schema=AnswerInput)
async def answer_sub_question(
    sub_question: str, legal_sources: list[dict[str, Any]]
) -> str:
    """
    Generate grounded answer for a sub-question using legal sources.

    This tool uses an LLM to answer a specific sub-question based solely on
    the provided legal source chunks. All claims must be cited and grounded.

    Args:
        sub_question: The specific question to answer
        legal_sources: List of legal chunk dictionaries to use as evidence

    Returns:
        Grounded answer with citations to legal sources

    Example:
        answer_sub_question(
            sub_question="What documentation must be provided for high-risk AI?",
            legal_sources=[{chunk1}, {chunk2}, ...]
        )
    """
    from llm.client import get_llm_client, invoke_llm

    try:
        # Format sources for prompt
        formatted_sources = "\n\n".join(
            [
                f"[Source {i+1}] {source.get('regulation', 'Unknown')}, "
                f"{source.get('article', '')}\n{source.get('content', '')}"
                for i, source in enumerate(legal_sources[:10])  # Limit to top 10
            ]
        )

        # Generation prompt
        prompt = f"""You are an EU AI regulation compliance assistant. Answer this question using ONLY the provided legal sources.

LEGAL SOURCES:
{formatted_sources}

QUESTION:
{sub_question}

INSTRUCTIONS:
1. Answer using ONLY the provided sources
2. Cite every claim with [Source X: Article Y]
3. If sources don't cover the question fully, state "The provided sources do not contain complete information about..."
4. DO NOT speculate or add information not in sources
5. Be precise and use exact legal language from sources

ANSWER:"""

        llm = get_llm_client()
        response = await invoke_llm(llm, prompt)

        answer = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Generated answer for sub-question: {sub_question[:50]}...")
        return answer

    except Exception as e:
        logger.error(f"Error in answer_sub_question: {e}")
        return f"Error generating answer: {str(e)}"


@tool(args_schema=SynthesisInput)
async def synthesize_information(answers: dict[str, str], original_query: str) -> str:
    """
    Synthesize multiple sub-answers into a coherent final answer.

    This tool combines answers to multiple sub-questions into a single,
    well-structured response that addresses the original complex query.

    Args:
        answers: Dictionary mapping sub-questions to their answers
        original_query: The original user query to address

    Returns:
        Coherent synthesized answer maintaining all citations

    Example:
        synthesize_information(
            answers={
                "What are obligations?": "...",
                "What are prohibitions?": "..."
            },
            original_query="What are the obligations and prohibitions for recruitment AI?"
        )
    """
    from llm.client import get_llm_client, invoke_llm

    try:
        # Format sub-answers
        formatted_answers = "\n\n".join(
            [f"**Sub-question**: {q}\n**Answer**: {a}" for q, a in answers.items()]
        )

        synthesis_prompt = f"""You are an EU AI regulation expert. Synthesize these sub-answers into a comprehensive response.

ORIGINAL QUERY:
{original_query}

SUB-QUESTION ANSWERS:
{formatted_answers}

TASK:
1. Create a coherent, well-structured answer that fully addresses the original query
2. Maintain ALL citations from sub-answers [Source X: Article Y]
3. Resolve any contradictions or overlaps between sub-answers
4. Ensure logical flow and readability
5. Use clear headings (##) if the answer has multiple distinct sections
6. Do NOT add new information beyond what's in the sub-answers

SYNTHESIZED ANSWER:"""

        llm = get_llm_client()
        response = await invoke_llm(llm, synthesis_prompt)

        synthesized = response.content if hasattr(response, "content") else str(response)

        logger.info("Synthesized final answer from sub-answers")
        return synthesized

    except Exception as e:
        logger.error(f"Error in synthesize_information: {e}")
        return f"Error synthesizing answer: {str(e)}"


@tool(args_schema=ValidationInput)
async def validate_claim(claim: str, legal_sources: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate if a claim is supported by legal sources.

    This tool checks whether a specific claim made in an answer is actually
    grounded in the provided legal source chunks.

    Args:
        claim: The claim to validate
        legal_sources: Legal chunks to check against

    Returns:
        Dictionary with:
            - is_valid: Whether claim is supported
            - supporting_source: The source that supports it (if any)
            - confidence: Confidence score (0.0-1.0)

    Example:
        validate_claim(
            claim="High-risk AI systems must maintain documentation",
            legal_sources=[{chunk1}, {chunk2}, ...]
        )
    """
    from llm.client import get_llm_client, invoke_llm

    try:
        # Format sources
        formatted_sources = "\n\n".join(
            [
                f"[Source {i+1}] {source.get('content', '')}"
                for i, source in enumerate(legal_sources[:10])
            ]
        )

        validation_prompt = f"""Validate if this claim is supported by the legal sources.

CLAIM:
{claim}

LEGAL SOURCES:
{formatted_sources}

Is the claim supported by the sources? Answer in JSON format:
{{
    "is_valid": true/false,
    "supporting_source_index": <index if valid, else null>,
    "confidence": <float between 0.0 and 1.0>,
    "explanation": "<brief explanation>"
}}"""

        llm = get_llm_client()
        response = await invoke_llm(llm, validation_prompt)

        # Parse JSON response (simplified - would need proper parsing)
        result = {
            "is_valid": "true" in response.content.lower(),
            "supporting_source": None,
            "confidence": 0.8 if "true" in response.content.lower() else 0.2,
        }

        return result

    except Exception as e:
        logger.error(f"Error in validate_claim: {e}")
        return {"is_valid": False, "supporting_source": None, "confidence": 0.0}


@tool(args_schema=ArticleSearchInput)
async def search_specific_article(
    regulation: str, article_number: str
) -> dict[str, Any] | None:
    """
    Retrieve a specific article from a regulation by exact reference.

    This tool performs a precise lookup for a specific article in a named
    regulation, useful when the agent needs exact legal text.

    Args:
        regulation: Regulation name (e.g., "EU AI Act", "GDPR")
        article_number: Article number (e.g., "Article 9", "Article 22")

    Returns:
        Dictionary with article content and metadata, or None if not found

    Example:
        search_specific_article(
            regulation="EU AI Act",
            article_number="Article 9"
        )
    """
    try:
        # Use retrieval tool with exact filters
        result = await retrieve_legal_chunks(
            query=f"{regulation} {article_number}",
            filters={"regulation": regulation, "article": article_number},
            top_k=1,
        )

        if result and len(result) > 0:
            logger.info(f"Found {article_number} in {regulation}")
            return result[0]
        else:
            logger.warning(f"Article not found: {article_number} in {regulation}")
            return None

    except Exception as e:
        logger.error(f"Error in search_specific_article: {e}")
        return None


# ===== Tool Registry =====

AGENT_TOOLS = {
    "retrieve_legal_chunks": retrieve_legal_chunks,
    "answer_sub_question": answer_sub_question,
    "synthesize_information": synthesize_information,
    "validate_claim": validate_claim,
    "search_specific_article": search_specific_article,
}


def get_agent_tools() -> list:
    """
    Get list of all available agent tools.

    Returns:
        List of LangChain tool objects for agent use
    """
    return list(AGENT_TOOLS.values())
