"""
Simple test script for RAG service.

Test the agentic RAG pipeline from the command line.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.rag_service.src.graph.graph import run_rag_pipeline
from services.rag_service.src.graph.state import create_initial_state


async def test_query(query: str, max_iterations: int = 5):
    """
    Test a single query through the RAG pipeline.

    Args:
        query: User's compliance question
        max_iterations: Max ReAct iterations
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")

    # Create initial state
    initial_state = create_initial_state(query)
    initial_state["max_iterations"] = max_iterations

    # Run pipeline
    from services.rag_service.src.graph.graph import compile_rag_graph

    graph = compile_rag_graph()
    result = await graph.ainvoke(initial_state)

    # Display results
    print(f"📊 ANALYSIS")
    print(f"  Intent: {result.get('intent')}")
    print(f"  Complexity: {result.get('query_complexity')}")
    print(f"  AI Domain: {result.get('ai_domain')}")
    print(f"  Risk Category: {result.get('risk_category')}")

    # Sub-queries
    if result.get("sub_queries"):
        print(f"\n📋 SUB-QUERIES ({len(result['sub_queries'])})")
        for i, sq in enumerate(result["sub_queries"]):
            status = "✅" if sq.status == "completed" else "⏳"
            print(f"  {i+1}. [{status}] {sq.question}")

    # Agent actions
    if result.get("agent_actions"):
        print(f"\n🧠 AGENT REASONING ({result.get('iteration_count')} iterations)")
        for action in result["agent_actions"]:
            print(f"\n  Step {action.step}: {action.action}")
            print(f"    💭 {action.thought[:100]}...")
            if action.observation:
                print(f"    👀 {str(action.observation)[:100]}...")

    # Retrieval
    if result.get("retrieval_history"):
        print(f"\n🔎 RETRIEVAL HISTORY")
        for i, ret in enumerate(result["retrieval_history"]):
            print(f"  {i+1}. Retrieved {ret.get('count', 0)} chunks")

    # Answer
    print(f"\n💡 ANSWER")
    if result.get("refusal_reason"):
        print(f"  ❌ Refused: {result['refusal_reason']}")
    elif result.get("final_answer"):
        # Print first 500 chars
        answer = result["final_answer"]
        print(f"  {answer[:500]}")
        if len(answer) > 500:
            print(f"  ... (truncated, total {len(answer)} chars)")
    else:
        print("  ⚠️ No answer generated")

    # Citations
    if result.get("citations"):
        print(f"\n📚 CITATIONS ({len(result['citations'])})")
        for cit in result["citations"][:5]:  # Show first 5
            print(f"  - Source {cit.source_id}: {cit.regulation}, {cit.article or 'N/A'}")

    # Metrics
    print(f"\n📊 METRICS")
    print(f"  Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
    print(f"  Confidence Score: {result.get('confidence_score', 0):.2f}")
    print(f"  Iterations: {result.get('iteration_count', 0)}")
    print(f"  Total Chunks: {len(result.get('all_retrieved_chunks', []))}")
    print(f"  Grounding Validated: {'✅' if result.get('grounding_validated') else '❌'}")

    print(f"\n{'='*80}\n")


async def main():
    """Run test queries."""
    test_queries = [
        # Simple
        "What is a high-risk AI system?",

        # Medium complexity
        "What are the documentation requirements for high-risk AI systems?",

        # Complex
        "What are the documentation requirements for recruitment AI vs healthcare AI systems, and how do they differ?",
    ]

    for query in test_queries:
        try:
            await test_query(query, max_iterations=5)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "─" * 80 + "\n")
        input("Press Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
