"""
Evaluation Dashboard for ConformAI RAG System

Interactive Streamlit dashboard for viewing and analyzing evaluation results.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tests.evaluation.base import MetricType
    from tests.evaluation.pipeline_evaluator import PipelineEvaluator, PipelineOutput, TestCase
    from tests.evaluation.report_generator import EvaluationReportGenerator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

try:
    from services.rag_service.src.graph.graph import run_rag_pipeline
except ImportError:
    # Handle hyphenated directory
    rag_service_path = Path(__file__).parent.parent / "services" / "rag-service" / "src"
    if rag_service_path.exists():
        sys.path.insert(0, str(rag_service_path))
    from graph.graph import run_rag_pipeline

# Page config
st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .pass-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .fail-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "report_generator" not in st.session_state:
    st.session_state.report_generator = EvaluationReportGenerator()
if "selected_report" not in st.session_state:
    st.session_state.selected_report = None

# Header
st.markdown('<div class="main-header">📊 Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown("Comprehensive evaluation and analysis of the ConformAI RAG system")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Evaluation mode
    eval_mode = st.radio(
        "Evaluation Mode",
        ["Run New Evaluation", "View Saved Reports", "Compare Reports"],
        help="Choose evaluation mode",
    )

    st.divider()

    if eval_mode == "Run New Evaluation":
        st.subheader("Evaluation Settings")

        # Thresholds
        retrieval_threshold = st.slider(
            "Retrieval Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum score for retrieval metrics",
        )

        answer_threshold = st.slider(
            "Answer Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum score for answer quality metrics",
        )

        # Test dataset selection
        test_dataset = st.selectbox(
            "Test Dataset",
            ["Golden EU AI Act Q&A", "Custom Dataset"],
            help="Select test dataset to use",
        )

        # Number of test cases
        num_test_cases = st.slider(
            "Number of Test Cases",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of test cases to evaluate (limited for demo)",
        )

        st.divider()

        # Run evaluation button
        run_evaluation = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)

    elif eval_mode == "View Saved Reports":
        st.subheader("Saved Reports")

        # List available reports
        reports_dir = Path(__file__).parent.parent / "tests" / "evaluation_reports"
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.json"))
            if report_files:
                selected_file = st.selectbox(
                    "Select Report",
                    [f.stem for f in report_files],
                    help="Choose a saved evaluation report",
                )
                load_report = st.button("📂 Load Report", use_container_width=True)

                if load_report:
                    report_path = reports_dir / f"{selected_file}.json"
                    with open(report_path) as f:
                        st.session_state.selected_report = json.load(f)
                    st.success(f"Loaded report: {selected_file}")
            else:
                st.info("No saved reports found. Run an evaluation first.")
        else:
            st.info("No reports directory found.")

    else:  # Compare Reports
        st.subheader("Compare Reports")
        st.info("Comparison feature coming soon!")


# Main content
if eval_mode == "Run New Evaluation" and run_evaluation:
    st.markdown("### 🔄 Running Evaluation...")

    # Load test dataset
    dataset_path = (
        Path(__file__).parent.parent / "tests" / "test_datasets" / "golden_qa_eu_ai_act.json"
    )

    if not dataset_path.exists():
        st.error(f"Test dataset not found: {dataset_path}")
        st.stop()

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Limit test cases
    test_subset = dataset[:num_test_cases]

    # Create test cases
    test_cases = [
        TestCase(
            id=item["id"],
            query=item["query"],
            ground_truth_answer=item.get("ground_truth_answer"),
            relevant_chunk_ids=item.get("relevant_chunk_ids"),
            expected_aspects=item.get("expected_aspects"),
            metadata={"difficulty": item.get("difficulty"), "category": item.get("category")},
        )
        for item in test_subset
    ]

    # Create evaluator
    evaluator = PipelineEvaluator(
        retrieval_threshold=retrieval_threshold,
        answer_threshold=answer_threshold,
    )

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run pipeline for each test case
    pipeline_outputs = []
    for i, test_case in enumerate(test_cases):
        status_text.text(f"Processing test case {i+1}/{len(test_cases)}: {test_case.query[:50]}...")
        progress_bar.progress((i + 1) / len(test_cases))

        try:
            # Run RAG pipeline
            result = asyncio.run(run_rag_pipeline(test_case.query))

            # Create pipeline output
            pipeline_output = PipelineOutput(
                query=test_case.query,
                answer=result.get("final_answer", ""),
                retrieved_chunk_ids=[],
                retrieved_chunks=[
                    chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])
                ],
                citations=result.get("citations", []),
                metadata={
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "confidence_score": result.get("confidence_score", 0.0),
                },
            )
            pipeline_outputs.append(pipeline_output)

        except Exception as e:
            st.error(f"Error processing test case {test_case.id}: {e}")
            # Create empty output
            pipeline_output = PipelineOutput(
                query=test_case.query,
                answer="",
                retrieved_chunk_ids=[],
                retrieved_chunks=[],
                citations=[],
                metadata={"error": str(e)},
            )
            pipeline_outputs.append(pipeline_output)

    status_text.text("Evaluating results...")

    # Batch evaluation
    try:
        evaluation_report = asyncio.run(evaluator.evaluate_batch(test_cases, pipeline_outputs))
        st.session_state.evaluation_results = evaluation_report

        # Generate and save report
        report_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.report_generator.generate_report(
            [
                asyncio.run(evaluator.evaluate_pipeline(tc, po))
                for tc, po in zip(test_cases, pipeline_outputs)
            ],
            report_name=report_name,
        )

        progress_bar.progress(1.0)
        status_text.text("✅ Evaluation complete!")

    except Exception as e:
        st.error(f"Evaluation error: {e}")
        st.exception(e)


# Display results
if eval_mode == "Run New Evaluation" and st.session_state.evaluation_results:
    report = st.session_state.evaluation_results
    summary = report["summary"]

    st.markdown("---")
    st.markdown("### 📈 Evaluation Results")

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{summary['total_count']}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #2ecc71;">{summary['passed_count']}</div>
            <div class="metric-label">Passed</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #e74c3c;">{summary['failed_count']}</div>
            <div class="metric-label">Failed</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        pass_rate_color = "#2ecc71" if summary["pass_rate"] >= 0.7 else "#e74c3c"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {pass_rate_color};">{summary['pass_rate']:.1%}</div>
            <div class="metric-label">Pass Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        score_color = "#2ecc71" if summary["average_score"] >= 0.7 else "#e74c3c"
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {score_color};">{summary['average_score']:.3f}</div>
            <div class="metric-label">Average Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Metrics by type
    st.markdown("### 📊 Metrics by Type")

    metric_df = pd.DataFrame(
        [
            {"Metric Type": metric_type, "Average Score": score}
            for metric_type, score in report.get("scores_by_metric_type", {}).items()
        ]
    )

    if not metric_df.empty:
        st.bar_chart(metric_df.set_index("Metric Type"))

        # Detailed metrics table
        with st.expander("📋 Detailed Metrics Table"):
            st.dataframe(metric_df, use_container_width=True)

    # Failed tests
    if report.get("failed_tests"):
        st.markdown("### ❌ Failed Tests")

        for failed_test in report["failed_tests"]:
            with st.expander(f"Test: {failed_test['test_case_id']} - Score: {failed_test['overall_score']:.3f}"):
                st.markdown(f"**Query:** {failed_test['query']}")
                st.markdown(f"**Failed Metrics:** {failed_test['failed_metric_count']}")

                # Failed metrics table
                failed_metrics_df = pd.DataFrame(failed_test["failed_metrics"])
                st.dataframe(failed_metrics_df, use_container_width=True)

    # All results
    with st.expander("📄 View All Results (JSON)"):
        st.json(report)


# Display saved report
elif eval_mode == "View Saved Reports" and st.session_state.selected_report:
    report = st.session_state.selected_report
    summary = report.get("summary", {})

    st.markdown("---")
    st.markdown(f"### 📊 Report: {report.get('report_name', 'Unknown')}")
    st.markdown(f"**Generated:** {report.get('generated_at', 'Unknown')}")

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Tests", summary.get("total_count", 0))
    with col2:
        st.metric("Passed", summary.get("passed_count", 0))
    with col3:
        st.metric("Failed", summary.get("failed_count", 0))
    with col4:
        st.metric("Pass Rate", f"{summary.get('pass_rate', 0):.1%}")
    with col5:
        st.metric("Average Score", f"{summary.get('average_score', 0):.3f}")

    # Metric analysis
    if "metric_analysis" in report:
        st.markdown("### 📊 Metrics Analysis")

        metric_data = []
        for metric_type, data in report["metric_analysis"].items():
            metric_data.append(
                {
                    "Metric Type": metric_type,
                    "Average Score": data.get("average_score", 0),
                    "Pass Rate": data.get("pass_rate", 0),
                    "Total Evaluations": data.get("total_evaluations", 0),
                }
            )

        if metric_data:
            df = pd.DataFrame(metric_data)
            st.dataframe(df, use_container_width=True)

    # Chart data
    if "chart_data" in report and report["chart_data"]:
        st.markdown("### 📈 Visualizations")

        # Score histogram
        if "score_histogram" in report["chart_data"]:
            st.markdown("#### Score Distribution")
            scores = report["chart_data"]["score_histogram"]["scores"]
            st.bar_chart(pd.DataFrame({"Scores": scores}))

    # Failed tests
    if "failed_tests" in report and report["failed_tests"]:
        st.markdown("### ❌ Failed Tests")
        st.write(f"Total failed: {len(report['failed_tests'])}")

        for failed_test in report["failed_tests"]:
            with st.expander(f"{failed_test['test_case_id']} - Score: {failed_test['overall_score']:.3f}"):
                st.markdown(f"**Query:** {failed_test.get('query', 'N/A')}")
                if failed_test.get("failed_metrics"):
                    st.dataframe(pd.DataFrame(failed_test["failed_metrics"]))

    # Download report
    st.markdown("### 💾 Export Report")
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        # JSON download
        json_str = json.dumps(report, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{report.get('report_name', 'report')}.json",
            mime="application/json",
        )

    with col_export2:
        # CSV download (if results available)
        if "all_results" in report:
            # Flatten results for CSV
            csv_rows = []
            for result in report["all_results"]:
                for metric in result.get("results", []):
                    csv_rows.append(
                        {
                            "test_case_id": result.get("test_case_id"),
                            "query": result.get("query"),
                            "metric_name": metric.get("metric_name"),
                            "metric_type": metric.get("metric_type"),
                            "score": metric.get("score"),
                            "passed": metric.get("passed"),
                        }
                    )

            if csv_rows:
                csv_df = pd.DataFrame(csv_rows)
                csv_str = csv_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"{report.get('report_name', 'report')}.csv",
                    mime="text/csv",
                )


# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    ConformAI Evaluation Dashboard | Built with ❤️ using Streamlit
</div>
""",
    unsafe_allow_html=True,
)
