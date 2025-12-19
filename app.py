"""
ConformAI Agent Test App

Interactive Streamlit interface for testing the agentic RAG system.
"""

import asyncio
import json
import time
from datetime import datetime

import streamlit as st

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.rag_service.src.graph.graph import run_rag_pipeline
    from services.rag_service.src.graph.state import create_initial_state
    from services.rag_service.src.graph.graph import compile_rag_graph
except ImportError:
    # Handle hyphenated directory names
    import importlib.util
    import os

    # Add services/rag-service/src to path
    rag_service_path = Path(__file__).parent / "services" / "rag-service" / "src"
    if rag_service_path.exists():
        sys.path.insert(0, str(rag_service_path))

    from graph.graph import run_rag_pipeline, compile_rag_graph
    from graph.state import create_initial_state

from shared.config.settings import get_settings

# Page config
st.set_page_config(
    page_title="ConformAI Agent Tester",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .agent-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .citation {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .sub-query {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-radius: 0.4rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<div class="main-header">🤖 ConformAI Agent Tester</div>', unsafe_allow_html=True)
st.markdown("Test the agentic RAG system with query decomposition and ReAct agent")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Agent settings
    st.subheader("Agent Parameters")
    max_iterations = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of ReAct loop iterations"
    )

    # Display settings
    st.subheader("Display Options")
    show_reasoning = st.checkbox("Show Agent Reasoning", value=True)
    show_subqueries = st.checkbox("Show Sub-Queries", value=True)
    show_retrievals = st.checkbox("Show Retrieval Details", value=True)
    show_raw_state = st.checkbox("Show Raw State (Debug)", value=False)

    # LLM info
    st.subheader("System Info")
    try:
        settings = get_settings()
        st.info(f"""
**LLM Provider:** {settings.llm_provider}
**Model:** {settings.llm_model}
**Environment:** {settings.environment}
        """)
    except Exception as e:
        st.error(f"Settings error: {e}")

    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Query Input")

    # Sample queries
    sample_queries = {
        "Simple": "What is a high-risk AI system?",
        "Medium": "What are the documentation requirements for high-risk AI systems?",
        "Complex": "What are the documentation requirements for recruitment AI vs healthcare AI systems, and how do they differ?",
        "Multi-Aspect": "What are the obligations, prohibitions, and transparency requirements for biometric identification systems?",
        "Comparative": "Compare the risk classifications and obligations for AI in recruitment versus AI in law enforcement.",
    }

    selected_sample = st.selectbox(
        "Sample Queries (optional)",
        ["Custom Query"] + list(sample_queries.keys())
    )

    if selected_sample != "Custom Query":
        query = st.text_area(
            "Enter your compliance query:",
            value=sample_queries[selected_sample],
            height=100,
            placeholder="e.g., What are the obligations for high-risk AI systems in recruitment?"
        )
    else:
        query = st.text_area(
            "Enter your compliance query:",
            height=100,
            placeholder="e.g., What are the obligations for high-risk AI systems in recruitment?"
        )

    col_btn1, col_btn2 = st.columns([1, 4])

    with col_btn1:
        submit_button = st.button("🚀 Run Query", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Quick Stats")
    if st.session_state.history:
        latest = st.session_state.history[-1]

        st.metric("Last Query Time", f"{latest.get('processing_time_ms', 0):.0f}ms")
        st.metric("Confidence", f"{latest.get('confidence_score', 0):.2f}")
        st.metric("Iterations", latest.get('iteration_count', 0))
        st.metric("Chunks Retrieved", len(latest.get('all_retrieved_chunks', [])))
    else:
        st.info("Run a query to see stats")

# Process query
if submit_button and query:
    with st.spinner("🤖 Agent is thinking..."):
        try:
            # Run the RAG pipeline
            start_time = time.time()

            # Create initial state with custom max_iterations
            initial_state = create_initial_state(query)
            initial_state["max_iterations"] = max_iterations

            # Compile and run graph
            graph = compile_rag_graph()

            # Run asynchronously
            result = asyncio.run(graph.ainvoke(initial_state))

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time

            # Save to history
            st.session_state.history.append(result)

            # Display results
            st.success("✅ Query processed successfully!")

            # Query Analysis
            st.markdown('<div class="sub-header">🔍 Query Analysis</div>', unsafe_allow_html=True)

            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                st.metric("Intent", result.get("intent", "unknown"))
            with col_a2:
                st.metric("Complexity", result.get("query_complexity", "unknown"))
            with col_a3:
                domain = result.get("ai_domain")
                st.metric("AI Domain", str(domain) if domain else "general")
            with col_a4:
                risk = result.get("risk_category")
                st.metric("Risk Category", str(risk) if risk else "N/A")

            # Sub-queries
            if show_subqueries and result.get("sub_queries"):
                st.markdown('<div class="sub-header">📋 Query Decomposition</div>', unsafe_allow_html=True)

                if result.get("decomposition_needed"):
                    st.info(f"Complex query decomposed into {len(result['sub_queries'])} sub-questions")

                for i, sq in enumerate(result["sub_queries"]):
                    priority_emoji = "🔴" if sq.priority == 1 else "🟡" if sq.priority == 2 else "🟢"
                    status_emoji = "✅" if sq.status == "completed" else "⏳" if sq.status == "in_progress" else "⏸️"

                    st.markdown(f"""
                    <div class="sub-query">
                        <strong>{priority_emoji} {status_emoji} Sub-Query {i+1}</strong> (Priority: {sq.priority}, Aspect: {sq.aspect})<br/>
                        <em>{sq.question}</em>
                    </div>
                    """, unsafe_allow_html=True)

            # Agent Reasoning
            if show_reasoning and result.get("agent_actions"):
                st.markdown('<div class="sub-header">🧠 Agent Reasoning (ReAct Loop)</div>', unsafe_allow_html=True)

                st.info(f"Agent completed {result.get('iteration_count', 0)} iterations")

                for action in result["agent_actions"]:
                    with st.expander(f"Step {action.step}: {action.action}"):
                        st.markdown(f"""
                        <div class="agent-step">
                            <strong>💭 Thought:</strong><br/>
                            {action.thought}<br/><br/>
                            <strong>🔧 Action:</strong> <code>{action.action}</code><br/><br/>
                            <strong>👀 Observation:</strong><br/>
                            {action.observation or "No observation recorded"}
                        </div>
                        """, unsafe_allow_html=True)

            # Retrieval Details
            if show_retrievals and result.get("retrieval_history"):
                st.markdown('<div class="sub-header">🔎 Retrieval History</div>', unsafe_allow_html=True)

                for i, retrieval in enumerate(result["retrieval_history"]):
                    st.markdown(f"""
**Retrieval {i+1}:** Retrieved {retrieval.get('count', 0)} chunks for query: *"{retrieval.get('query', '')[:100]}..."*
                    """)

            # Final Answer
            st.markdown('<div class="sub-header">💡 Final Answer</div>', unsafe_allow_html=True)

            if result.get("refusal_reason"):
                st.error(f"❌ Query Refused: {result['refusal_reason']}")
            elif result.get("final_answer"):
                st.markdown(result["final_answer"])
            else:
                st.warning("No answer generated")

            # Citations
            if result.get("citations"):
                st.markdown('<div class="sub-header">📚 Citations</div>', unsafe_allow_html=True)

                for i, citation in enumerate(result["citations"]):
                    st.markdown(f"""
                    <div class="citation">
                        <strong>Source {citation.source_id}:</strong> {citation.regulation}
                        {f", {citation.article}" if citation.article else ""}
                        {f" (CELEX: {citation.celex})" if citation.celex else ""}<br/>
                        <em>{citation.excerpt[:150]}...</em>
                    </div>
                    """, unsafe_allow_html=True)

            # Metadata
            st.markdown('<div class="sub-header">📊 Performance Metrics</div>', unsafe_allow_html=True)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.0f}ms")
            with col_m2:
                st.metric("Confidence Score", f"{result.get('confidence_score', 0):.2f}")
            with col_m3:
                st.metric("Iterations", result.get('iteration_count', 0))
            with col_m4:
                st.metric("Total Chunks", len(result.get('all_retrieved_chunks', [])))

            col_m5, col_m6, col_m7, col_m8 = st.columns(4)
            with col_m5:
                st.metric("Sub-Questions", len(result.get('sub_queries', [])))
            with col_m6:
                st.metric("Citations", len(result.get('citations', [])))
            with col_m7:
                scores = result.get('retrieval_scores', [])
                avg_score = sum(scores) / len(scores) if scores else 0
                st.metric("Avg Retrieval Score", f"{avg_score:.2f}")
            with col_m8:
                st.metric("Grounding Validated", "✅" if result.get('grounding_validated') else "❌")

            # Reasoning Trace
            if result.get("reasoning_trace"):
                with st.expander("🗺️ View Reasoning Trace"):
                    for i, step in enumerate(result["reasoning_trace"]):
                        st.markdown(f"{i+1}. {step}")

            # Raw State (Debug)
            if show_raw_state:
                with st.expander("🔧 Raw State (Debug)"):
                    # Convert to JSON-serializable format
                    debug_state = {
                        k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                        for k, v in result.items()
                    }
                    st.json(debug_state)

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)

# Query History
if st.session_state.history:
    st.markdown('<div class="sub-header">📜 Query History</div>', unsafe_allow_html=True)

    for i, hist in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
        with st.expander(f"Query {len(st.session_state.history) - i}: {hist.get('query', '')[:80]}..."):
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.metric("Complexity", hist.get("query_complexity", "unknown"))
            with col_h2:
                st.metric("Time", f"{hist.get('processing_time_ms', 0):.0f}ms")
            with col_h3:
                st.metric("Confidence", f"{hist.get('confidence_score', 0):.2f}")

            if hist.get("final_answer"):
                st.markdown("**Answer:**")
                st.markdown(hist["final_answer"][:300] + "...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    ConformAI - EU AI Compliance RAG System | Built with ❤️ using LangGraph & Streamlit
</div>
""", unsafe_allow_html=True)
