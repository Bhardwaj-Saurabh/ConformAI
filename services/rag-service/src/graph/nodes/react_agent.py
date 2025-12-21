"""ReAct agent nodes for planning, acting, and observing."""

import json
import time
from typing import Any

from agent.tools import AGENT_TOOLS
from graph.state import AgentAction, RAGState
from langchain_core.messages import HumanMessage
from llm.client import get_planning_llm, invoke_llm
from pydantic import BaseModel

from shared.utils.logger import get_logger
from shared.utils.opik_tracer import track_langgraph_node

logger = get_logger(__name__)


class AgentDecision(BaseModel):
    """Structured output for agent planning."""

    thought: str
    action: str
    action_input: dict[str, Any]
    is_final: bool = False  # Whether agent is done


@track_langgraph_node("react_plan", "agent")
async def react_plan(state: RAGState) -> dict[str, Any]:
    """
    Agent decides next action based on current state (PLAN step).

    Args:
        state: Current RAG state

    Returns:
        Updated state with planned action
    """
    iteration = state["iteration_count"]
    max_iter = state["max_iterations"]

    logger.info(f"ReAct PLAN - Iteration {iteration + 1}/{max_iter}")

    # Check termination conditions
    if iteration >= max_iter:
        logger.warning("Max iterations reached, forcing completion")
        state["agent_state"] = "done"
        return state

    # Check if all sub-queries are answered
    all_answered = all(sq.status == "completed" for sq in state["sub_queries"])
    if all_answered and state["intermediate_answers"]:
        logger.info("All sub-queries answered, moving to synthesis")
        state["agent_state"] = "done"
        return state

    try:
        llm = get_planning_llm()

        # Build context
        context = build_planning_context(state)

        planning_prompt = f"""You are a legal research agent for EU AI regulations. Plan the next action.

{context}

AVAILABLE TOOLS:
1. retrieve_legal_chunks(query, filters, top_k) - Search vector database for legal documents
2. answer_sub_question(sub_question, legal_sources) - Generate answer using retrieved sources
3. synthesize_information(answers, original_query) - Combine multiple answers into final response
4. validate_claim(claim, legal_sources) - Verify if claim is supported
5. search_specific_article(regulation, article_number) - Get exact article text

TASK: Decide the next action to progress toward answering the user's query.

Think step-by-step:
1. What have I accomplished? (Check sub-queries status and retrieved chunks)
2. What is the next priority sub-question to work on?
3. Do I need to retrieve more sources or can I answer with what I have?
4. Which tool should I use and with what parameters?

Respond in JSON format:
{{
    "thought": "<your step-by-step reasoning>",
    "action": "<tool_name>",
    "action_input": {{<JSON parameters for the tool>}},
    "is_final": <true if this is the last action needed, false otherwise>
}}

Example:
{{
    "thought": "I need to retrieve legal sources about high-risk AI obligations before I can answer sub-question 1",
    "action": "retrieve_legal_chunks",
    "action_input": {{"query": "high-risk AI system obligations", "filters": {{"risk_category": "high"}}, "top_k": 10}},
    "is_final": false
}}"""

        response = await invoke_llm(llm, [HumanMessage(content=planning_prompt)])

        # Parse decision
        try:
            decision_json = json.loads(response.content)
            decision = AgentDecision(**decision_json)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse agent decision: {e}")
            # Fallback decision: retrieve for first pending sub-query
            decision = get_fallback_decision(state)

        # Validate action exists
        if decision.action not in AGENT_TOOLS:
            logger.error(f"Invalid action: {decision.action}")
            decision = get_fallback_decision(state)

        # Record action
        action = AgentAction(
            step=iteration + 1,
            thought=decision.thought,
            action=decision.action,
            action_input=decision.action_input,
            timestamp=time.time(),
        )

        state["agent_actions"].append(action)
        state["agent_state"] = "acting"

        logger.info(
            f"Planned action: {decision.action} - {decision.thought[:100]}..."
        )

        # Check if final
        if decision.is_final:
            state["agent_state"] = "done"

        return state

    except Exception as e:
        logger.error(f"Error in react_plan: {e}")
        state["error"] = f"Planning error: {str(e)}"
        state["agent_state"] = "done"
        return state


@track_langgraph_node("react_act", "agent")
async def react_act(state: RAGState) -> dict[str, Any]:
    """
    Execute the planned action (ACT step).

    Args:
        state: Current RAG state with planned action

    Returns:
        Updated state with action results
    """
    if not state["agent_actions"]:
        logger.error("No actions to execute")
        state["agent_state"] = "done"
        return state

    current_action = state["agent_actions"][-1]

    logger.info(f"ReAct ACT - Executing: {current_action.action}")

    try:
        # Get tool
        tool = AGENT_TOOLS[current_action.action]

        # Execute tool
        observation = await tool.ainvoke(current_action.action_input)

        # Store result
        current_action.observation = str(observation)[:500]  # Truncate for logging

        # Update state based on action type
        if current_action.action == "retrieve_legal_chunks":
            # Store retrieved chunks
            if isinstance(observation, list):
                state["all_retrieved_chunks"].extend(observation)
                state["retrieval_history"].append({
                    "query": current_action.action_input.get("query", ""),
                    "count": len(observation),
                    "step": current_action.step,
                })

                # Update working memory
                state["working_memory"]["last_retrieval_count"] = len(observation)

                logger.info(f"Retrieved {len(observation)} chunks")

        elif current_action.action == "answer_sub_question":
            # Store answer for sub-question
            sub_q = current_action.action_input.get("sub_question", "")
            if isinstance(observation, str):
                state["intermediate_answers"][sub_q] = observation

                # Mark sub-query as completed
                for sq in state["sub_queries"]:
                    if sq.question == sub_q:
                        sq.status = "completed"
                        sq.answer = observation
                        logger.info(f"Completed sub-question: {sub_q[:50]}...")
                        break

        elif current_action.action == "synthesize_information":
            # Store final synthesized answer
            if isinstance(observation, str):
                state["final_answer"] = observation
                logger.info("Synthesized final answer")

        # Update working memory
        state["working_memory"]["last_action"] = current_action.action
        state["working_memory"]["last_observation"] = str(observation)[:200]

        state["agent_state"] = "observing"

        return state

    except Exception as e:
        logger.error(f"Error executing action {current_action.action}: {e}")
        current_action.observation = f"ERROR: {str(e)}"
        current_action.error = str(e)
        state["retry_count"] += 1

        # Move to observing to decide next step
        state["agent_state"] = "observing"

        return state


@track_langgraph_node("react_observe", "agent")
async def react_observe(state: RAGState) -> dict[str, Any]:
    """
    Observe results and decide whether to continue (OBSERVE step).

    Args:
        state: Current RAG state with action results

    Returns:
        Updated state with continuation decision
    """
    logger.info("ReAct OBSERVE - Checking completion criteria")

    # Check completion criteria
    all_answered = all(sq.status == "completed" for sq in state["sub_queries"])
    has_chunks = len(state["all_retrieved_chunks"]) >= 3
    has_answers = len(state["intermediate_answers"]) > 0

    # Check if we have a final answer already
    if state.get("final_answer", ""):
        logger.info("Final answer generated, marking as done")
        state["agent_state"] = "done"
        return state

    # Check if all sub-queries answered
    if all_answered and has_chunks and has_answers:
        logger.info(
            "All sub-queries answered with sufficient sources, ready to synthesize"
        )
        state["agent_state"] = "done"
        return state

    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"] - 1:
        logger.warning(
            "Approaching max iterations, will synthesize with available answers"
        )
        state["agent_state"] = "done"
        return state

    # Continue looping
    state["iteration_count"] += 1
    state["agent_state"] = "planning"

    logger.info(
        f"Continuing ReAct loop - Iteration {state['iteration_count']}/{state['max_iterations']}"
    )

    return state


# ===== Helper Functions =====


def build_planning_context(state: RAGState) -> str:
    """Build context summary for agent planning."""

    sub_queries_status = "\n".join(
        [
            f"  [{sq.priority}] {sq.question[:80]}... - Status: {sq.status}"
            for sq in state["sub_queries"]
        ]
    )

    previous_actions = "\n".join(
        [
            f"  Step {action.step}: {action.action} - {action.thought[:60]}..."
            for action in state["agent_actions"][-3:]  # Last 3 actions
        ]
    )

    retrieval_summary = (
        f"{len(state['all_retrieved_chunks'])} total chunks retrieved"
        if state["all_retrieved_chunks"]
        else "No chunks retrieved yet"
    )

    answers_summary = (
        f"{len(state['intermediate_answers'])} sub-questions answered"
        if state["intermediate_answers"]
        else "No answers generated yet"
    )

    context = f"""CURRENT SITUATION:
- Original Query: {state["query"]}
- Intent: {state["intent"]}
- Complexity: {state["query_complexity"]}
- Iteration: {state["iteration_count"] + 1}/{state["max_iterations"]}

SUB-QUERIES:
{sub_queries_status}

PROGRESS:
- {retrieval_summary}
- {answers_summary}

PREVIOUS ACTIONS:
{previous_actions if previous_actions else "  (None yet)"}

WORKING MEMORY:
{json.dumps(state.get("working_memory", {}), indent=2)}
"""

    return context


def get_fallback_decision(state: RAGState) -> AgentDecision:
    """Generate fallback decision when parsing fails."""

    # Find first pending sub-query
    pending = [sq for sq in state["sub_queries"] if sq.status == "pending"]

    if pending:
        # Retrieve for first pending query
        return AgentDecision(
            thought="Fallback: retrieving sources for next pending sub-question",
            action="retrieve_legal_chunks",
            action_input={
                "query": pending[0].question,
                "filters": {},
                "top_k": 10,
            },
            is_final=False,
        )
    else:
        # All pending, synthesize
        return AgentDecision(
            thought="Fallback: all sub-queries processed, synthesizing answer",
            action="synthesize_information",
            action_input={
                "answers": state["intermediate_answers"],
                "original_query": state["query"],
            },
            is_final=True,
        )
