"""Query analysis and decomposition nodes."""

import json
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from services.rag_service.src.graph.state import RAGState, SubQuery
from services.rag_service.src.llm.client import get_planning_llm
from shared.models.legal_document import AIDomain, RiskCategory
from shared.utils.logger import get_logger

logger = get_logger(__name__)


class QueryAnalysisOutput(BaseModel):
    """Structured output for query analysis."""

    intent: str  # compliance_question, risk_assessment, obligation_lookup, etc.
    ai_domain: str | None  # biometrics, recruitment, healthcare, etc.
    risk_category: str | None  # prohibited, high, limited, minimal
    entities: list[str]  # Extracted regulations, articles, systems
    complexity_score: int  # 0-3: 0=simple, 1-2=medium, 3+=complex


class QueryDecompositionOutput(BaseModel):
    """Structured output for query decomposition."""

    class SubQuestionData(BaseModel):
        question: str
        aspect: str  # obligations, prohibitions, definitions, etc.
        priority: int  # 1=critical, 2=important, 3=supplementary

    sub_questions: list[SubQuestionData]


async def analyze_query(state: RAGState) -> dict[str, Any]:
    """
    Analyze query to understand intent, domain, and complexity.

    Args:
        state: Current RAG state

    Returns:
        Updated state with query analysis results
    """
    query = state["query"]

    logger.info(f"Analyzing query: {query[:100]}...")

    try:
        llm = get_planning_llm()

        analysis_prompt = f"""Analyze this EU AI regulation compliance query:

QUERY: {query}

Extract the following information in JSON format:
{{
    "intent": "<compliance_question|risk_assessment|obligation_lookup|prohibition_check|definition_query|documentation_requirements>",
    "ai_domain": "<biometrics|recruitment|healthcare|law_enforcement|education|critical_infrastructure|general|null>",
    "risk_category": "<prohibited|high|limited|minimal|unclassified|null>",
    "entities": ["<list of regulations, articles, or AI systems mentioned>"],
    "complexity_score": <integer 0-5, based on: multiple aspects, multiple regulations, comparative analysis, conditional reasoning>
}}

Examples:
- "What is a high-risk AI system?" → simple (score: 0)
- "What are the obligations for recruitment AI?" → medium (score: 1-2)
- "Compare documentation requirements for recruitment vs healthcare AI and explain differences" → complex (score: 3+)

Return ONLY the JSON, no additional text."""

        response = await llm.ainvoke([HumanMessage(content=analysis_prompt)])

        # Parse JSON response
        try:
            analysis_json = json.loads(response.content)
            analysis = QueryAnalysisOutput(**analysis_json)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse structured output, using fallback: {e}")
            # Fallback to simple analysis
            analysis = QueryAnalysisOutput(
                intent="compliance_question",
                ai_domain=None,
                risk_category=None,
                entities=[],
                complexity_score=1,
            )

        # Map to state
        state["intent"] = analysis.intent
        state["entities"] = analysis.entities

        # Map AI domain
        try:
            state["ai_domain"] = (
                AIDomain(analysis.ai_domain) if analysis.ai_domain else None
            )
        except ValueError:
            state["ai_domain"] = None

        # Map risk category
        try:
            state["risk_category"] = (
                RiskCategory(analysis.risk_category) if analysis.risk_category else None
            )
        except ValueError:
            state["risk_category"] = None

        # Determine complexity
        if analysis.complexity_score >= 3:
            state["query_complexity"] = "complex"
        elif analysis.complexity_score >= 1:
            state["query_complexity"] = "medium"
        else:
            state["query_complexity"] = "simple"

        logger.info(
            f"Query analysis complete: intent={state['intent']}, "
            f"complexity={state['query_complexity']}, "
            f"domain={state['ai_domain']}"
        )

        return state

    except Exception as e:
        logger.error(f"Error in analyze_query: {e}")
        # Fallback values
        state["intent"] = "compliance_question"
        state["query_complexity"] = "medium"
        state["entities"] = []
        return state


async def decompose_query(state: RAGState) -> dict[str, Any]:
    """
    Decompose complex queries into sub-questions.

    Args:
        state: Current RAG state with query analysis

    Returns:
        Updated state with sub-queries
    """
    query = state["query"]
    complexity = state["query_complexity"]

    logger.info(f"Decomposing query (complexity: {complexity})")

    # Skip decomposition for simple queries
    if complexity == "simple":
        state["decomposition_needed"] = False
        state["sub_queries"] = [
            SubQuery(question=query, aspect="main", priority=1, status="pending")
        ]
        logger.info("Simple query - no decomposition needed")
        return state

    # Decompose complex queries
    try:
        llm = get_planning_llm()

        decomposition_prompt = f"""Break down this complex EU AI regulation query into specific sub-questions.

ORIGINAL QUERY: {query}

CONTEXT:
- Intent: {state["intent"]}
- AI Domain: {state.get("ai_domain", "general")}
- Entities mentioned: {", ".join(state["entities"]) if state["entities"] else "none"}

For each sub-question, identify:
1. The specific aspect being asked (e.g., "obligations", "prohibitions", "definitions", "risk_classification", "documentation", "procedures")
2. Priority: 1=critical (must answer), 2=important (should answer), 3=supplementary (nice to have)

Generate 2-5 sub-questions that together comprehensively answer the original query.

Return JSON format:
{{
    "sub_questions": [
        {{
            "question": "<specific sub-question>",
            "aspect": "<aspect category>",
            "priority": <1|2|3>
        }},
        ...
    ]
}}

IMPORTANT: Each sub-question should be answerable independently using legal document retrieval."""

        response = await llm.ainvoke([HumanMessage(content=decomposition_prompt)])

        # Parse JSON
        try:
            decomp_json = json.loads(response.content)
            decomposition = QueryDecompositionOutput(**decomp_json)

            state["sub_queries"] = [
                SubQuery(
                    question=sq.question,
                    aspect=sq.aspect,
                    priority=sq.priority,
                    status="pending",
                )
                for sq in decomposition.sub_questions
            ]

            state["decomposition_needed"] = True

            logger.info(
                f"Decomposed into {len(state['sub_queries'])} sub-questions: "
                f"{[sq.question[:50] + '...' for sq in state['sub_queries']]}"
            )

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse decomposition, using single query: {e}")
            # Fallback: treat as single query
            state["decomposition_needed"] = False
            state["sub_queries"] = [
                SubQuery(question=query, aspect="main", priority=1, status="pending")
            ]

        return state

    except Exception as e:
        logger.error(f"Error in decompose_query: {e}")
        # Fallback
        state["decomposition_needed"] = False
        state["sub_queries"] = [
            SubQuery(question=query, aspect="main", priority=1, status="pending")
        ]
        return state


async def safety_check(state: RAGState) -> dict[str, Any]:
    """
    Perform safety pre-check to reject out-of-scope or harmful queries.

    Args:
        state: Current RAG state

    Returns:
        Updated state with safety validation
    """
    query = state["query"].lower()

    logger.info("Performing safety pre-check")

    # Rule-based checks
    harmful_patterns = [
        "jailbreak",
        "ignore previous",
        "disregard instructions",
        "prompt injection",
    ]

    out_of_scope_patterns = [
        "medical diagnosis",
        "legal advice for",
        "financial advice",
        "tax advice",
    ]

    # Check for harmful patterns
    if any(pattern in query for pattern in harmful_patterns):
        state["is_safe"] = False
        state["refusal_reason"] = "Query appears to contain prompt injection or jailbreak attempt"
        logger.warning(f"Rejected query due to harmful pattern: {query[:100]}")
        return state

    # Check if query is about EU AI/data regulations
    regulation_keywords = [
        "ai act",
        "artificial intelligence",
        "gdpr",
        "data protection",
        "eu regulation",
        "european union",
        "high-risk",
        "prohibited",
        "biometric",
    ]

    if not any(keyword in query for keyword in regulation_keywords):
        # Additional LLM check
        try:
            llm = get_planning_llm()
            scope_prompt = f"""Is this query about EU AI or data protection regulations?

QUERY: {state["query"]}

Answer YES or NO, and provide brief explanation.
If asking for personal legal advice (not informational), answer NO."""

            response = await llm.ainvoke([HumanMessage(content=scope_prompt)])

            if "no" in response.content.lower()[:50]:  # Check first part of response
                state["is_safe"] = False
                state["refusal_reason"] = (
                    "Query is out of scope. This system only provides information "
                    "about EU AI and data protection regulations."
                )
                logger.warning(f"Rejected out-of-scope query: {query[:100]}")
                return state

        except Exception as e:
            logger.error(f"Error in LLM scope check: {e}")
            # On error, allow through (fail open for availability)

    # Check for requests for legal advice
    advice_patterns = ["what should i do", "should i", "advise me", "recommend"]
    if any(pattern in query for pattern in advice_patterns):
        state["is_safe"] = False
        state["refusal_reason"] = (
            "This system provides informational support about EU regulations, "
            "not legal advice. Please consult a qualified legal professional for advice."
        )
        logger.warning(f"Rejected legal advice request: {query[:100]}")
        return state

    # Passed all checks
    state["is_safe"] = True
    state["refusal_reason"] = None
    logger.info("Query passed safety checks")

    return state
