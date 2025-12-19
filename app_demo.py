"""
ConformAI Agent Test App - Demo Version with Mock Data

This demo version shows the UI working with sample data.
Full version requires fixing module imports (rag-service → rag_service).
"""

import time
from datetime import datetime
from dataclasses import dataclass

import streamlit as st

# Page config
st.set_page_config(
    page_title="ConformAI Agent Tester (Demo)",
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
    .sub-query {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-radius: 0.4rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    .citation {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Mock data structures
@dataclass
class SubQuery:
    question: str
    aspect: str
    priority: int
    status: str = "completed"

@dataclass
class AgentAction:
    step: int
    thought: str
    action: str
    observation: str

@dataclass
class Citation:
    source_id: int
    regulation: str
    article: str | None
    celex: str | None
    excerpt: str

def generate_mock_result(query: str, complexity: str, max_iterations: int):
    """Generate mock RAG result for demo purposes."""

    if complexity == "simple":
        sub_queries = [
            SubQuery(
                question=query,
                aspect="definition",
                priority=1,
                status="completed"
            )
        ]

        agent_actions = [
            AgentAction(
                step=1,
                thought="I need to retrieve the definition of high-risk AI systems from the EU AI Act",
                action="retrieve_legal_chunks",
                observation="Retrieved 8 relevant chunks from EU AI Act, Annex III"
            ),
            AgentAction(
                step=2,
                thought="Now I can answer the question using the retrieved sources",
                action="answer_sub_question",
                observation="Generated answer with 3 citations"
            ),
        ]

        answer = """A high-risk AI system is an AI system that is either listed in Annex III of the EU AI Act or is an AI system that is a safety component of a product covered by certain EU harmonization legislation [Source 1: Article 6, EU AI Act].

According to Annex III, high-risk AI systems include:
- AI systems used for biometric identification and categorization of natural persons [Source 2: Annex III, Point 1]
- AI systems used in employment, workers management and access to self-employment [Source 2: Annex III, Point 4]
- AI systems used in education and vocational training [Source 2: Annex III, Point 3]
- AI systems used in the administration of justice and democratic processes [Source 2: Annex III, Point 8]

⚖️ **Disclaimer**: This information is for educational and informational purposes only. It does not constitute legal advice. For compliance decisions affecting your organization, consult a qualified legal professional."""

        citations = [
            Citation(1, "EU AI Act", "Article 6", "32024R1689", "High-risk AI systems as referred to in Article 6(1) are the AI systems listed in Annex III..."),
            Citation(2, "EU AI Act", "Annex III", "32024R1689", "AI systems intended to be used for the biometric identification and categorisation of natural persons..."),
            Citation(3, "EU AI Act", "Annex III", "32024R1689", "AI systems intended to be used for recruitment or selection of natural persons..."),
        ]

        iterations = 2

    elif complexity == "medium":
        sub_queries = [
            SubQuery(
                question="What documentation must be provided for high-risk AI systems?",
                aspect="obligations",
                priority=1,
                status="completed"
            ),
            SubQuery(
                question="What are the specific documentation requirements from Article 11?",
                aspect="details",
                priority=2,
                status="completed"
            ),
        ]

        agent_actions = [
            AgentAction(
                step=1,
                thought="I need to retrieve information about documentation obligations for high-risk AI",
                action="retrieve_legal_chunks",
                observation="Retrieved 10 chunks covering Article 11 and related obligations"
            ),
            AgentAction(
                step=2,
                thought="Answer the first sub-question about general documentation",
                action="answer_sub_question",
                observation="Generated answer for sub-question 1"
            ),
            AgentAction(
                step=3,
                thought="Retrieve more specific details about Article 11 requirements",
                action="retrieve_legal_chunks",
                observation="Retrieved 6 additional chunks with detailed requirements"
            ),
            AgentAction(
                step=4,
                thought="Answer the detailed requirements sub-question",
                action="answer_sub_question",
                observation="Generated answer for sub-question 2"
            ),
            AgentAction(
                step=5,
                thought="All sub-questions answered, synthesize final response",
                action="synthesize_information",
                observation="Created comprehensive answer combining both sub-answers"
            ),
        ]

        answer = """High-risk AI systems must maintain comprehensive technical documentation as specified in Article 11 of the EU AI Act [Source 1: Article 11].

## Required Documentation

1. **General Description** [Source 1: Article 11]:
   - Detailed description of the AI system and its intended purpose
   - Methods and steps performed for development
   - Main design choices and assumptions made

2. **Technical Specifications** [Source 2: Annex IV]:
   - Computational resources used
   - Data governance and management practices
   - Testing and validation procedures
   - Risk management system description

3. **Performance Metrics** [Source 3: Article 13]:
   - Accuracy, robustness, and cybersecurity measures
   - Expected level of accuracy for different demographic groups
   - Known or foreseeable circumstances that may impact performance

4. **Compliance Documentation** [Source 4: Article 11]:
   - EU declaration of conformity
   - Quality management system documentation
   - Change logs and modifications

This documentation must be kept up-to-date and made available to competent authorities upon request [Source 5: Article 11].

⚖️ **Disclaimer**: This information is for educational and informational purposes only. It does not constitute legal advice. For compliance decisions affecting your organization, consult a qualified legal professional."""

        citations = [
            Citation(1, "EU AI Act", "Article 11", "32024R1689", "Providers shall draw up the technical documentation in such a way as to demonstrate that the high-risk AI system complies..."),
            Citation(2, "EU AI Act", "Annex IV", "32024R1689", "Technical documentation referred to in Article 11 shall contain at a minimum the following information..."),
            Citation(3, "EU AI Act", "Article 13", "32024R1689", "High-risk AI systems shall be designed to achieve an appropriate level of accuracy, robustness..."),
            Citation(4, "EU AI Act", "Article 11", "32024R1689", "The technical documentation shall be kept up-to-date..."),
            Citation(5, "EU AI Act", "Article 11", "32024R1689", "The technical documentation shall be made available to the relevant competent authorities..."),
        ]

        iterations = 5

    else:  # complex
        sub_queries = [
            SubQuery("What documentation is required for recruitment AI?", "obligations_recruitment", 1, "completed"),
            SubQuery("What documentation is required for healthcare AI?", "obligations_healthcare", 1, "completed"),
            SubQuery("What are the key differences between them?", "comparison", 2, "completed"),
            SubQuery("What risk categories apply to each?", "risk_classification", 3, "completed"),
        ]

        agent_actions = [
            AgentAction(1, "Start by retrieving recruitment AI documentation requirements", "retrieve_legal_chunks", "Retrieved 8 chunks about recruitment AI obligations"),
            AgentAction(2, "Answer recruitment-specific sub-question", "answer_sub_question", "Generated answer with 4 citations"),
            AgentAction(3, "Retrieve healthcare AI documentation requirements", "retrieve_legal_chunks", "Retrieved 7 chunks about healthcare AI obligations"),
            AgentAction(4, "Answer healthcare-specific sub-question", "answer_sub_question", "Generated answer with 3 citations"),
            AgentAction(5, "Now I need to identify the key differences", "retrieve_legal_chunks", "Retrieved 5 chunks comparing different domains"),
            AgentAction(6, "Generate comparative analysis", "answer_sub_question", "Generated comparison highlighting 3 key differences"),
            AgentAction(7, "Synthesize all findings into final answer", "synthesize_information", "Created comprehensive comparative answer"),
        ]

        answer = """# Documentation Requirements: Recruitment AI vs Healthcare AI

## Recruitment AI Systems

Recruitment AI systems are classified as high-risk under Annex III, Point 4 [Source 1]. They must comply with the following documentation requirements:

1. **Technical Documentation** [Source 2: Article 11]:
   - Description of selection criteria and decision logic
   - Data sets used for training (job descriptions, candidate profiles)
   - Bias testing results across protected characteristics

2. **Transparency Requirements** [Source 3: Article 13]:
   - Candidates must be informed of AI system use
   - Explanation of decision-making process
   - Information on how to request human review

## Healthcare AI Systems

Healthcare AI systems fall under Annex III, Point 5(b) as high-risk systems [Source 4]. Documentation requirements include:

1. **Clinical Documentation** [Source 5: Article 11 + MDR]:
   - Clinical evaluation data
   - Intended medical purpose and use
   - Clinical performance metrics and validation

2. **Patient Safety Documentation** [Source 6: Annex IV]:
   - Risk-benefit analysis
   - Post-market surveillance plan
   - Adverse event reporting procedures

3. **Medical Device Compliance** [Source 7: Article 6]:
   - CE marking documentation
   - Notified body certificates
   - Compliance with Medical Device Regulation (MDR)

## Key Differences

1. **Regulatory Framework**: Healthcare AI has additional MDR compliance [Source 7], recruitment AI does not
2. **Clinical Requirements**: Healthcare requires clinical evaluation and validation [Source 5], recruitment focuses on bias testing [Source 2]
3. **Oversight**: Healthcare AI requires notified body involvement [Source 7], recruitment AI can use self-assessment in some cases [Source 8]

Both systems must maintain comprehensive technical documentation under Article 11, but healthcare AI has significantly more stringent clinical and safety requirements due to the potential impact on human health.

⚖️ **Disclaimer**: This information is for educational and informational purposes only. It does not constitute legal advice. For compliance decisions affecting your organization, consult a qualified legal professional."""

        citations = [
            Citation(1, "EU AI Act", "Annex III, Point 4", "32024R1689", "AI systems intended to be used for recruitment or selection of natural persons..."),
            Citation(2, "EU AI Act", "Article 11", "32024R1689", "Providers of high-risk AI systems shall draw up technical documentation..."),
            Citation(3, "EU AI Act", "Article 13", "32024R1689", "High-risk AI systems shall be designed to provide information to deployers..."),
            Citation(4, "EU AI Act", "Annex III, Point 5(b)", "32024R1689", "AI systems intended to be used for making or assisting in making decisions related to healthcare..."),
            Citation(5, "EU AI Act", "Article 11 + MDR", "32024R1689", "Clinical evaluation shall follow the requirements laid down in the MDR..."),
            Citation(6, "EU AI Act", "Annex IV", "32024R1689", "The technical documentation shall contain detailed information about the risk management system..."),
            Citation(7, "EU AI Act", "Article 6", "32024R1689", "AI systems that are safety components of products or systems covered by EU harmonisation legislation..."),
            Citation(8, "EU AI Act", "Article 43", "32024R1689", "Conformity assessment procedures for high-risk AI systems..."),
        ]

        iterations = 7

    return {
        "query": query,
        "intent": "compliance_question",
        "ai_domain": "recruitment" if "recruitment" in query.lower() else "general",
        "risk_category": "high",
        "query_complexity": complexity,
        "sub_queries": sub_queries,
        "decomposition_needed": complexity in ["medium", "complex"],
        "agent_actions": agent_actions,
        "iteration_count": iterations,
        "max_iterations": max_iterations,
        "retrieval_history": [{"query": action.action, "count": 8} for action in agent_actions if "retrieve" in action.action],
        "all_retrieved_chunks": ["dummy"] * (8 * len([a for a in agent_actions if "retrieve" in a.action])),
        "final_answer": answer,
        "citations": citations,
        "grounding_validated": True,
        "confidence_score": 0.89 if complexity != "simple" else 0.92,
        "processing_time_ms": 1500 if complexity == "simple" else 2800 if complexity == "medium" else 4200,
        "retrieval_scores": [0.85, 0.82, 0.79, 0.88, 0.76, 0.83, 0.81, 0.80],
        "reasoning_trace": [
            f"Analyzed query and detected {complexity} complexity",
            f"Decomposed into {len(sub_queries)} sub-questions" if complexity != "simple" else "No decomposition needed",
            f"Executed {iterations} agent iterations",
            f"Retrieved {len([a for a in agent_actions if 'retrieve' in a.action]) * 8} total chunks",
            "Synthesized final answer from sub-answers" if complexity != "simple" else "Generated direct answer",
            "Validated grounding and citations",
            f"Final confidence score: {0.89 if complexity != 'simple' else 0.92}"
        ]
    }

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<div class="main-header">🤖 ConformAI Agent Tester (Demo)</div>', unsafe_allow_html=True)
st.markdown("**Demo version with mock data** - Shows UI functionality")
st.info("💡 This is a demo version using sample data to show the interface. The full version requires fixing module imports.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    max_iterations = st.slider("Max Iterations", 1, 10, 5)

    st.subheader("Display Options")
    show_reasoning = st.checkbox("Show Agent Reasoning", value=True)
    show_subqueries = st.checkbox("Show Sub-Queries", value=True)
    show_retrievals = st.checkbox("Show Retrieval Details", value=True)

    st.subheader("System Info")
    st.info("""
**Mode:** Demo (Mock Data)
**LLM:** GPT-4o-mini
**Environment:** Development
    """)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Query Input")

    sample_queries = {
        "Simple": "What is a high-risk AI system?",
        "Medium": "What are the documentation requirements for high-risk AI systems?",
        "Complex": "What are the documentation requirements for recruitment AI vs healthcare AI systems, and how do they differ?",
    }

    selected_sample = st.selectbox("Sample Queries", list(sample_queries.keys()))
    query = st.text_area("Query:", value=sample_queries[selected_sample], height=100)

    submit_button = st.button("🚀 Run Query (Demo)", type="primary")

with col2:
    st.subheader("📊 Quick Stats")
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.metric("Last Query Time", f"{latest.get('processing_time_ms', 0):.0f}ms")
        st.metric("Confidence", f"{latest.get('confidence_score', 0):.2f}")
        st.metric("Iterations", latest.get('iteration_count', 0))
    else:
        st.info("Run a query to see stats")

# Process query
if submit_button and query:
    with st.spinner("🤖 Agent is thinking..."):
        time.sleep(1)  # Simulate processing

        # Generate mock result
        result = generate_mock_result(query, selected_sample.lower(), max_iterations)
        st.session_state.history.append(result)

        st.success("✅ Query processed successfully!")

        # Display results (same as full app)
        st.markdown('<div class="sub-header">🔍 Query Analysis</div>', unsafe_allow_html=True)

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        col_a1.metric("Intent", result["intent"])
        col_a2.metric("Complexity", result["query_complexity"])
        col_a3.metric("AI Domain", result["ai_domain"])
        col_a4.metric("Risk Category", result["risk_category"])

        # Sub-queries
        if show_subqueries and result["sub_queries"]:
            st.markdown('<div class="sub-header">📋 Query Decomposition</div>', unsafe_allow_html=True)

            if result["decomposition_needed"]:
                st.info(f"Complex query decomposed into {len(result['sub_queries'])} sub-questions")

            for i, sq in enumerate(result["sub_queries"]):
                priority_emoji = "🔴" if sq.priority == 1 else "🟡" if sq.priority == 2 else "🟢"
                st.markdown(f"""
                <div class="sub-query">
                    <strong>{priority_emoji} ✅ Sub-Query {i+1}</strong> (Priority: {sq.priority}, Aspect: {sq.aspect})<br/>
                    <em>{sq.question}</em>
                </div>
                """, unsafe_allow_html=True)

        # Agent reasoning
        if show_reasoning and result["agent_actions"]:
            st.markdown('<div class="sub-header">🧠 Agent Reasoning (ReAct Loop)</div>', unsafe_allow_html=True)
            st.info(f"Agent completed {result['iteration_count']} iterations")

            for action in result["agent_actions"]:
                with st.expander(f"Step {action.step}: {action.action}"):
                    st.markdown(f"""
                    <div class="agent-step">
                        <strong>💭 Thought:</strong><br/>{action.thought}<br/><br/>
                        <strong>🔧 Action:</strong> <code>{action.action}</code><br/><br/>
                        <strong>👀 Observation:</strong><br/>{action.observation}
                    </div>
                    """, unsafe_allow_html=True)

        # Final Answer
        st.markdown('<div class="sub-header">💡 Final Answer</div>', unsafe_allow_html=True)
        st.markdown(result["final_answer"])

        # Citations
        if result["citations"]:
            st.markdown('<div class="sub-header">📚 Citations</div>', unsafe_allow_html=True)
            for cit in result["citations"]:
                st.markdown(f"""
                <div class="citation">
                    <strong>Source {cit.source_id}:</strong> {cit.regulation}, {cit.article or 'N/A'}<br/>
                    <em>{cit.excerpt[:100]}...</em>
                </div>
                """, unsafe_allow_html=True)

        # Metrics
        st.markdown('<div class="sub-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Processing Time", f"{result['processing_time_ms']}ms")
        col_m2.metric("Confidence Score", f"{result['confidence_score']:.2f}")
        col_m3.metric("Iterations", result['iteration_count'])
        col_m4.metric("Grounding", "✅" if result['grounding_validated'] else "❌")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'>ConformAI Demo | Built with Streamlit</div>", unsafe_allow_html=True)
