# ConformAI Agentic RAG Service - Production Implementation Plan

## Overview
Advanced agentic RAG system using LangGraph with:
- **Query Decomposition**: Break complex queries into sub-questions
- **ReAct Agent**: Planning and reasoning with tool use
- **Multi-step Reasoning**: Iterative answer construction
- **Legal-Grade Accuracy**: Citation enforcement and grounding validation

---

## Architecture Design

### Enhanced LangGraph State Machine with ReAct Agent

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Query Analysis     │  ← Classify intent, extract entities, detect complexity
│  - Intent detection │
│  - Domain classify  │
│  - Complexity score │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Query Decomposition │  ← Break complex queries into sub-questions
│  - Identify aspects │
│  - Generate         │
│    sub-queries      │
└──────┬──────────────┘
       │
       ├─── [Simple Query] ──► Skip decomposition
       │
       ▼
┌─────────────────────┐
│  Safety Pre-Check   │  ← Reject off-topic or harmful queries
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│         ReAct Agent Loop (Max 5 iterations) │
│ ┌─────────────────────────────────────┐     │
│ │  1. PLAN                            │     │
│ │  - Analyze current state            │     │
│ │  - Decide next action               │     │
│ │  - Select tool to use               │     │
│ └──────────┬──────────────────────────┘     │
│            │                                  │
│            ▼                                  │
│ ┌─────────────────────────────────────┐     │
│ │  2. ACT                             │     │
│ │  - Execute selected tool:           │     │
│ │    • retrieve_legal_chunks()        │     │
│ │    • answer_sub_question()          │     │
│ │    • synthesize_information()       │     │
│ │    • validate_claim()               │     │
│ └──────────┬──────────────────────────┘     │
│            │                                  │
│            ▼                                  │
│ ┌─────────────────────────────────────┐     │
│ │  3. OBSERVE                         │     │
│ │  - Record tool results              │     │
│ │  - Update working memory            │     │
│ │  - Check completion criteria        │     │
│ └──────────┬──────────────────────────┘     │
│            │                                  │
│            ├─── [Not Done] ──► Loop back     │
│            │                     to PLAN     │
│            │                                  │
│            └─── [Done] ──────────────────────┤
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Synthesize Answer  │  ← Combine sub-answers into coherent response
│  - Merge findings   │
│  - Resolve conflicts│
│  - Ensure coherence │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Grounding Check    │  ← Validate all claims are cited
│  - Citation match   │
│  - Hallucination    │
│  - Source verify    │
└──────┬──────────────┘
       │
       ├─── [Failed] ──► REGENERATE or REFUSE
       │
       ▼
┌─────────────────────┐
│  Format Response    │  ← Structure final output with metadata
│  - Add disclaimer   │
│  - Format citations │
│  - Add reasoning    │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│     END     │
└─────────────┘
```

---

## Enhanced State Schema

```python
from typing import TypedDict, Literal
from dataclasses import dataclass

@dataclass
class SubQuery:
    """Decomposed sub-question."""
    question: str
    aspect: str  # "obligations", "prohibitions", "risk_classification", etc.
    priority: int
    answer: str | None = None
    sources: list[Chunk] = None
    status: Literal["pending", "in_progress", "completed"] = "pending"

@dataclass
class AgentAction:
    """ReAct agent action."""
    step: int
    thought: str  # Agent's reasoning
    action: str  # Tool name
    action_input: dict  # Tool parameters
    observation: str | None = None  # Tool result
    timestamp: float

class RAGState(TypedDict):
    """LangGraph state for agentic RAG pipeline."""

    # ===== Input =====
    query: str
    conversation_id: str | None
    user_context: dict | None  # Additional context from user

    # ===== Query Analysis =====
    intent: str  # "compliance_question", "risk_assessment", "obligation_lookup"
    ai_domain: AIDomain | None
    risk_category: RiskCategory | None
    entities: list[str]
    query_complexity: Literal["simple", "medium", "complex"]

    # ===== Query Decomposition =====
    sub_queries: list[SubQuery]
    decomposition_needed: bool

    # ===== ReAct Agent Loop =====
    agent_actions: list[AgentAction]  # History of agent steps
    agent_state: Literal["planning", "acting", "observing", "done"]
    iteration_count: int
    max_iterations: int  # Default: 5
    working_memory: dict  # Agent's scratchpad

    # ===== Retrieval =====
    all_retrieved_chunks: list[Chunk]  # All chunks across all sub-queries
    retrieval_history: list[dict]  # Track all retrieval calls

    # ===== Generation =====
    intermediate_answers: dict[str, str]  # Sub-query -> answer mapping
    final_answer: str
    reasoning_trace: list[str]  # Explanation of reasoning steps

    # ===== Citations & Grounding =====
    citations: list[Citation]
    grounding_validated: bool
    hallucination_detected: bool

    # ===== Safety & Validation =====
    is_safe: bool
    refusal_reason: str | None
    confidence_score: float

    # ===== Metadata =====
    processing_time_ms: float
    model_used: str
    total_llm_calls: int
    total_tokens_used: int

    # ===== Error Handling =====
    error: str | None
    retry_count: int
```

---

## Node Implementations

### 1. Query Analysis Node
**Enhanced to detect complexity**

```python
async def analyze_query(state: RAGState) -> RAGState:
    """Analyze query and determine complexity."""

    # LLM call with structured output
    analysis = await llm_analyze_query(state["query"])

    state["intent"] = analysis.intent
    state["ai_domain"] = analysis.ai_domain
    state["entities"] = analysis.entities

    # Complexity scoring
    complexity_signals = {
        "multi_aspect": "and" in state["query"].lower(),
        "multiple_regulations": len(analysis.entities) > 2,
        "comparative": any(word in state["query"].lower()
                          for word in ["compare", "difference", "versus"]),
        "conditional": any(word in state["query"].lower()
                          for word in ["if", "when", "unless"]),
    }

    if sum(complexity_signals.values()) >= 2:
        state["query_complexity"] = "complex"
    elif sum(complexity_signals.values()) == 1:
        state["query_complexity"] = "medium"
    else:
        state["query_complexity"] = "simple"

    return state
```

---

### 2. Query Decomposition Node
**Break complex queries into sub-questions**

```python
async def decompose_query(state: RAGState) -> RAGState:
    """Decompose complex queries into sub-questions."""

    if state["query_complexity"] == "simple":
        state["decomposition_needed"] = False
        state["sub_queries"] = [
            SubQuery(
                question=state["query"],
                aspect="main",
                priority=1
            )
        ]
        return state

    # LLM-based decomposition
    decomposition_prompt = f"""
You are an EU AI regulation expert. Break down this complex query into specific sub-questions:

Query: {state["query"]}

For each sub-question, identify:
1. The specific aspect being asked (obligations, prohibitions, definitions, procedures, etc.)
2. Priority (1=critical, 2=important, 3=supplementary)

Return 2-5 sub-questions that together answer the original query comprehensively.
    """

    decomposition = await llm_decompose_query(decomposition_prompt)

    state["sub_queries"] = [
        SubQuery(
            question=sq.question,
            aspect=sq.aspect,
            priority=sq.priority
        )
        for sq in decomposition.sub_questions
    ]

    state["decomposition_needed"] = True

    return state
```

**Example Decomposition**:
```
Original Query: "What are the obligations for AI systems used in recruitment,
                 and how do they differ from healthcare AI systems?"

Sub-queries:
1. [Priority 1, Aspect: obligations] "What are the specific obligations for AI systems used in recruitment under the EU AI Act?"
2. [Priority 1, Aspect: obligations] "What are the specific obligations for AI systems used in healthcare under the EU AI Act?"
3. [Priority 2, Aspect: comparison] "What are the key differences between recruitment and healthcare AI obligations?"
4. [Priority 3, Aspect: risk_classification] "What risk categories apply to recruitment vs healthcare AI systems?"
```

---

### 3. ReAct Agent Loop
**Iterative planning, acting, and observing**

#### Agent Tools

```python
class AgentTools:
    """Tools available to the ReAct agent."""

    @tool
    async def retrieve_legal_chunks(
        query: str,
        filters: dict | None = None,
        top_k: int = 10
    ) -> list[Chunk]:
        """Retrieve relevant legal document chunks from vector database.

        Args:
            query: Search query
            filters: Metadata filters (domain, regulation, risk_category)
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant legal chunks with scores
        """
        # Call Retrieval Service
        response = await retrieval_client.search(query, filters, top_k)
        return response.chunks

    @tool
    async def answer_sub_question(
        sub_question: str,
        legal_sources: list[Chunk]
    ) -> str:
        """Generate grounded answer for a sub-question using legal sources.

        Args:
            sub_question: The specific question to answer
            legal_sources: Retrieved legal chunks to use as evidence

        Returns:
            Grounded answer with citations
        """
        # Generate answer with strict grounding
        answer = await llm_generate_answer(sub_question, legal_sources)
        return answer

    @tool
    async def synthesize_information(
        answers: dict[str, str],
        original_query: str
    ) -> str:
        """Synthesize multiple sub-answers into coherent final answer.

        Args:
            answers: Sub-question -> answer mapping
            original_query: The original user query

        Returns:
            Coherent synthesized answer
        """
        synthesis_prompt = f"""
Synthesize these sub-answers into a coherent response to the original query:

Original Query: {original_query}

Sub-answers:
{format_sub_answers(answers)}

Create a structured, coherent answer that:
1. Addresses all aspects of the original query
2. Maintains all citations from sub-answers
3. Resolves any contradictions
4. Provides clear, actionable information
        """

        return await llm_synthesize(synthesis_prompt)

    @tool
    async def validate_claim(
        claim: str,
        legal_sources: list[Chunk]
    ) -> dict:
        """Validate if a claim is supported by legal sources.

        Args:
            claim: The claim to validate
            legal_sources: Legal chunks to check against

        Returns:
            {"is_valid": bool, "supporting_source": Chunk | None, "confidence": float}
        """
        # LLM-based claim validation
        validation = await llm_validate_claim(claim, legal_sources)
        return validation.dict()

    @tool
    async def search_specific_article(
        regulation: str,
        article_number: str
    ) -> Chunk | None:
        """Retrieve a specific article from a regulation.

        Args:
            regulation: Regulation name (e.g., "EU AI Act", "GDPR")
            article_number: Article number (e.g., "Article 9")

        Returns:
            The article chunk if found
        """
        # Direct lookup in vector DB with exact metadata match
        return await retrieval_client.get_article(regulation, article_number)
```

#### ReAct Planning Node

```python
async def react_plan(state: RAGState) -> RAGState:
    """Agent decides next action based on current state."""

    iteration = state["iteration_count"]

    # Check termination conditions
    if iteration >= state["max_iterations"]:
        state["agent_state"] = "done"
        return state

    # Build context for agent
    context = build_agent_context(state)

    planning_prompt = f"""
You are a legal research agent analyzing EU AI regulations.

CURRENT SITUATION:
{context}

WORKING MEMORY:
{state["working_memory"]}

PREVIOUS ACTIONS:
{format_action_history(state["agent_actions"])}

SUB-QUERIES STATUS:
{format_subquery_status(state["sub_queries"])}

AVAILABLE TOOLS:
1. retrieve_legal_chunks(query, filters, top_k) - Search vector database
2. answer_sub_question(sub_question, legal_sources) - Generate grounded answer
3. synthesize_information(answers, original_query) - Combine sub-answers
4. validate_claim(claim, legal_sources) - Check if claim is supported
5. search_specific_article(regulation, article_number) - Get specific article

TASK: Decide the next action to progress toward answering the user's query.

Think step-by-step:
1. What have I accomplished so far?
2. What still needs to be done?
3. Which tool should I use next and why?
4. What are the tool parameters?

Respond in this format:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: <JSON parameters>
    """

    # LLM decides next action
    agent_decision = await llm_plan_action(planning_prompt)

    # Record action
    action = AgentAction(
        step=iteration + 1,
        thought=agent_decision.thought,
        action=agent_decision.action,
        action_input=agent_decision.action_input,
        timestamp=time.time()
    )

    state["agent_actions"].append(action)
    state["agent_state"] = "acting"

    return state
```

#### ReAct Action Node

```python
async def react_act(state: RAGState) -> RAGState:
    """Execute the planned action."""

    current_action = state["agent_actions"][-1]

    # Get tool
    tool = AGENT_TOOLS[current_action.action]

    # Execute tool
    try:
        observation = await tool.ainvoke(current_action.action_input)

        # Store result
        current_action.observation = str(observation)

        # Update working memory based on action
        if current_action.action == "retrieve_legal_chunks":
            state["all_retrieved_chunks"].extend(observation)
            state["retrieval_history"].append({
                "query": current_action.action_input["query"],
                "count": len(observation),
                "step": current_action.step
            })

        elif current_action.action == "answer_sub_question":
            sub_q = current_action.action_input["sub_question"]
            state["intermediate_answers"][sub_q] = observation

            # Mark sub-query as completed
            for sq in state["sub_queries"]:
                if sq.question == sub_q:
                    sq.status = "completed"
                    sq.answer = observation

        # Update working memory
        state["working_memory"]["last_action"] = current_action.action
        state["working_memory"]["last_result"] = observation[:200]  # Truncate

    except Exception as e:
        current_action.observation = f"ERROR: {str(e)}"
        state["error"] = str(e)

    state["agent_state"] = "observing"

    return state
```

#### ReAct Observe Node

```python
async def react_observe(state: RAGState) -> RAGState:
    """Observe results and decide whether to continue."""

    # Check completion criteria
    all_subqueries_answered = all(
        sq.status == "completed" for sq in state["sub_queries"]
    )

    sufficient_sources = len(state["all_retrieved_chunks"]) >= 5

    has_intermediate_answers = len(state["intermediate_answers"]) > 0

    # Decide if done
    if all_subqueries_answered and sufficient_sources:
        state["agent_state"] = "done"
    else:
        # Continue looping
        state["iteration_count"] += 1
        state["agent_state"] = "planning"

    return state
```

#### ReAct Loop Conditional Edge

```python
def should_continue_react(state: RAGState) -> Literal["continue", "synthesize"]:
    """Decide whether to continue ReAct loop or move to synthesis."""

    if state["agent_state"] == "done":
        return "synthesize"

    if state["iteration_count"] >= state["max_iterations"]:
        return "synthesize"

    return "continue"
```

---

### 4. Synthesize Answer Node
**Combine all findings into coherent response**

```python
async def synthesize_answer(state: RAGState) -> RAGState:
    """Synthesize final answer from all sub-answers and sources."""

    if not state["intermediate_answers"]:
        # Fallback: single-pass generation
        state["final_answer"] = await generate_simple_answer(
            state["query"],
            state["all_retrieved_chunks"]
        )
    else:
        # Multi-step synthesis
        synthesis_prompt = f"""
You are an EU AI regulation expert. Synthesize a comprehensive answer to this query:

ORIGINAL QUERY:
{state["query"]}

SUB-QUESTION ANSWERS:
{format_intermediate_answers(state["intermediate_answers"])}

ALL LEGAL SOURCES:
{format_all_sources(state["all_retrieved_chunks"])}

TASK:
1. Create a coherent, well-structured answer that addresses all aspects of the original query
2. Maintain ALL citations from sub-answers
3. Resolve any contradictions or overlaps
4. Ensure logical flow and readability
5. Use clear headings if the answer is complex

ANSWER:
        """

        state["final_answer"] = await llm_synthesize(synthesis_prompt)

    # Extract citations
    state["citations"] = extract_citations(state["final_answer"])

    return state
```

---

### 5. Grounding Validation Node
**Validate answer is fully grounded**

```python
async def validate_grounding(state: RAGState) -> RAGState:
    """Validate all claims are grounded in retrieved sources."""

    # 1. Check all citations are valid
    for citation in state["citations"]:
        if not citation_exists_in_chunks(citation, state["all_retrieved_chunks"]):
            state["grounding_validated"] = False
            state["hallucination_detected"] = True
            return state

    # 2. LLM-based hallucination detection
    hallucination_prompt = f"""
You are a fact-checker. Verify if this answer contains any claims not supported by the sources.

ANSWER:
{state["final_answer"]}

LEGAL SOURCES:
{format_all_sources(state["all_retrieved_chunks"])}

Does the answer contain any unsupported claims or hallucinations? YES or NO
If YES, list the unsupported claims.
    """

    hallucination_check = await llm_check_hallucination(hallucination_prompt)

    if hallucination_check.has_hallucination:
        state["hallucination_detected"] = True
        state["grounding_validated"] = False

        # Increment retry
        state["retry_count"] += 1

        if state["retry_count"] < 2:
            # Regenerate with stricter prompt
            return await regenerate_with_stricter_grounding(state)
        else:
            # Give up, refuse
            state["refusal_reason"] = "Unable to generate fully grounded answer"
            state["final_answer"] = ""
    else:
        state["grounding_validated"] = True

    return state
```

---

## LangGraph Compilation

```python
from langgraph.graph import StateGraph, END

def build_rag_graph() -> StateGraph:
    """Build the complete agentic RAG graph."""

    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("safety_check", safety_check)
    workflow.add_node("react_plan", react_plan)
    workflow.add_node("react_act", react_act)
    workflow.add_node("react_observe", react_observe)
    workflow.add_node("synthesize_answer", synthesize_answer)
    workflow.add_node("validate_grounding", validate_grounding)
    workflow.add_node("format_response", format_response)

    # Define edges
    workflow.set_entry_point("analyze_query")

    workflow.add_edge("analyze_query", "decompose_query")
    workflow.add_edge("decompose_query", "safety_check")

    # Safety check conditional
    workflow.add_conditional_edges(
        "safety_check",
        lambda state: "continue" if state["is_safe"] else "refuse",
        {
            "continue": "react_plan",
            "refuse": "format_response"
        }
    )

    # ReAct loop
    workflow.add_edge("react_plan", "react_act")
    workflow.add_edge("react_act", "react_observe")

    workflow.add_conditional_edges(
        "react_observe",
        should_continue_react,
        {
            "continue": "react_plan",
            "synthesize": "synthesize_answer"
        }
    )

    # Synthesis and validation
    workflow.add_edge("synthesize_answer", "validate_grounding")

    workflow.add_conditional_edges(
        "validate_grounding",
        lambda state: "success" if state["grounding_validated"] else "refuse",
        {
            "success": "format_response",
            "refuse": "format_response"
        }
    )

    workflow.add_edge("format_response", END)

    return workflow.compile()
```

---

## Example Execution Trace

**User Query**: "What are the documentation requirements for high-risk AI systems in recruitment, and how should they handle transparency obligations?"

### Execution Steps:

1. **Query Analysis**
   - Intent: compliance_question
   - Domain: recruitment
   - Complexity: complex (multiple aspects: documentation + transparency)

2. **Query Decomposition**
   - Sub-query 1: "What documentation must be provided for high-risk AI systems under the EU AI Act?"
   - Sub-query 2: "What specific documentation requirements apply to recruitment AI systems?"
   - Sub-query 3: "What are the transparency obligations for high-risk AI systems?"
   - Sub-query 4: "How do transparency obligations apply specifically to recruitment AI?"

3. **ReAct Agent Loop**

   **Iteration 1**:
   - THOUGHT: "I need to first understand general documentation requirements for high-risk AI"
   - ACTION: retrieve_legal_chunks
   - ACTION_INPUT: {"query": "documentation requirements high-risk AI systems", "filters": {"risk_category": "high"}}
   - OBSERVATION: Retrieved 8 chunks from EU AI Act Articles 9, 11, 13

   **Iteration 2**:
   - THOUGHT: "Now I can answer the first sub-question about general documentation"
   - ACTION: answer_sub_question
   - ACTION_INPUT: {"sub_question": "What documentation must be provided...", "legal_sources": [chunks from iteration 1]}
   - OBSERVATION: Generated answer with citations

   **Iteration 3**:
   - THOUGHT: "I need recruitment-specific requirements"
   - ACTION: retrieve_legal_chunks
   - ACTION_INPUT: {"query": "recruitment AI documentation transparency", "filters": {"domain": "recruitment"}}
   - OBSERVATION: Retrieved 6 chunks including Annex III references

   **Iteration 4**:
   - THOUGHT: "Answer recruitment-specific sub-question"
   - ACTION: answer_sub_question
   - OBSERVATION: Generated answer

   **Iteration 5**:
   - THOUGHT: "All sub-queries answered, ready to synthesize"
   - ACTION: synthesize_information
   - OBSERVATION: Coherent final answer generated

4. **Synthesis**: Combine all sub-answers into structured response

5. **Grounding Validation**: All citations verified ✓

6. **Final Response**:
```json
{
  "answer": "High-risk AI systems in recruitment must comply with comprehensive documentation and transparency obligations under the EU AI Act:\n\n## Documentation Requirements\n\n1. **Technical Documentation** [Source 1: Article 11, EU AI Act]:\n   - Detailed description of system design and development\n   - Data governance and management practices\n   - Computational resources used\n   - Testing and validation procedures\n\n2. **Recruitment-Specific Documentation** [Source 2: Annex III, EU AI Act]:\n   - Clear description of evaluation criteria\n   - Information about data sets used for training\n   - Explanations of decision-making logic\n\n## Transparency Obligations\n\n3. **User Information** [Source 3: Article 13, EU AI Act]:\n   - Candidates must be informed they are interacting with an AI system\n   - Clear explanation of how the system works\n   - Information about rights to human review\n\n4. **Accountability** [Source 4: Article 9, EU AI Act]:\n   - Maintain logs of system operations\n   - Enable traceability of decisions\n   - Ensure human oversight capabilities\n\n...",

  "reasoning_trace": [
    "Retrieved general documentation requirements for high-risk AI",
    "Identified recruitment-specific obligations in Annex III",
    "Retrieved transparency provisions from Article 13",
    "Synthesized requirements into structured response"
  ],

  "agent_steps": 5,
  "sub_queries_answered": 4
}
```

---

## Configuration

```python
# Max iterations for ReAct loop
RAG_MAX_ITERATIONS: int = 5

# Query complexity thresholds
QUERY_COMPLEXITY_THRESHOLDS = {
    "simple": 0,      # Single aspect, single regulation
    "medium": 1,      # 2 aspects or 2 regulations
    "complex": 2      # 3+ aspects or complex reasoning
}

# Agent planning model (more capable model for reasoning)
AGENT_PLANNING_MODEL = "claude-3-5-sonnet-20241022"

# Generation model (can be cheaper)
GENERATION_MODEL = "gpt-4o-mini"
```

---

## Updated File Structure

```
services/rag-service/
├── src/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py              # RAGState with agent fields
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py       # Query analysis + decomposition
│   │   │   ├── react_agent.py    # ReAct planning/acting/observing
│   │   │   ├── synthesis.py      # Answer synthesis
│   │   │   └── validation.py     # Grounding validation
│   │   └── graph.py              # LangGraph compilation
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py              # Agent tool implementations
│   │   └── prompts.py            # Agent planning prompts
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py             # LLM client factory
│   │   └── structured.py         # Structured output parsing
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI app
│       └── routes.py             # API endpoints
```

---

## Next Steps

1. ✅ Implement enhanced RAGState with agent fields
2. ✅ Build query decomposition node
3. ✅ Implement ReAct agent tools
4. ✅ Create ReAct planning/acting/observing nodes
5. ✅ Build synthesis node
6. ✅ Integrate all nodes into LangGraph
7. ✅ Test with complex queries
8. ⏳ Optimize iteration limits
9. ⏳ Add comprehensive error handling

---

**Status**: Ready to implement agentic RAG! 🚀
