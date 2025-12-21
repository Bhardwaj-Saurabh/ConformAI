"""Report generation utilities for evaluation results with charts and visualizations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from shared.utils.logger import get_logger
from tests.evaluation.base import EvaluationMetrics, MetricType

logger = get_logger(__name__)


class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports with visualizations."""

    def __init__(self, output_dir: Path = None):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports (default: tests/evaluation_reports)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "evaluation_reports"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        all_metrics: list[EvaluationMetrics],
        report_name: str = None,
        include_charts: bool = True,
    ) -> dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            all_metrics: List of evaluation metrics
            report_name: Name for the report
            include_charts: Whether to generate chart data

        Returns:
            Report dictionary with summary, metrics, and chart data
        """
        if report_name is None:
            report_name = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating evaluation report: {report_name}")

        # Generate report sections
        summary = self._generate_summary(all_metrics)
        metric_analysis = self._analyze_metrics(all_metrics)
        failed_tests = self._extract_failed_tests(all_metrics)
        chart_data = self._generate_chart_data(all_metrics) if include_charts else {}

        report = {
            "report_name": report_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary": summary,
            "metric_analysis": metric_analysis,
            "failed_tests": failed_tests,
            "chart_data": chart_data,
            "all_results": [m.to_dict() for m in all_metrics],
        }

        # Save to file
        report_path = self.output_dir / f"{report_name}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {report_path}")

        return report

    def _generate_summary(self, all_metrics: list[EvaluationMetrics]) -> dict[str, Any]:
        """Generate summary statistics."""
        total_count = len(all_metrics)
        passed_count = sum(1 for m in all_metrics if m.passed)
        failed_count = total_count - passed_count

        # Calculate average scores
        average_score = sum(m.overall_score for m in all_metrics) / total_count if total_count > 0 else 0.0

        # Calculate score distribution
        score_ranges = {
            "excellent": sum(1 for m in all_metrics if m.overall_score >= 0.9),
            "good": sum(1 for m in all_metrics if 0.75 <= m.overall_score < 0.9),
            "fair": sum(1 for m in all_metrics if 0.5 <= m.overall_score < 0.75),
            "poor": sum(1 for m in all_metrics if m.overall_score < 0.5),
        }

        return {
            "total_count": total_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0.0,
            "average_score": average_score,
            "score_distribution": score_ranges,
            "median_score": sorted([m.overall_score for m in all_metrics])[len(all_metrics) // 2]
            if all_metrics
            else 0.0,
            "min_score": min(m.overall_score for m in all_metrics) if all_metrics else 0.0,
            "max_score": max(m.overall_score for m in all_metrics) if all_metrics else 0.0,
        }

    def _analyze_metrics(self, all_metrics: list[EvaluationMetrics]) -> dict[str, Any]:
        """Analyze metrics by type."""
        metric_analysis = {}

        # Group results by metric type
        for metric_type in MetricType:
            scores = []
            passed_count = 0
            total_count = 0

            for metrics in all_metrics:
                for result in metrics.results:
                    if result.metric_type == metric_type:
                        scores.append(result.score)
                        total_count += 1
                        if result.passed:
                            passed_count += 1

            if scores:
                metric_analysis[metric_type.value] = {
                    "average_score": sum(scores) / len(scores),
                    "median_score": sorted(scores)[len(scores) // 2],
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "pass_rate": passed_count / total_count if total_count > 0 else 0.0,
                    "total_evaluations": total_count,
                }

        return metric_analysis

    def _extract_failed_tests(self, all_metrics: list[EvaluationMetrics]) -> list[dict[str, Any]]:
        """Extract details of failed tests."""
        failed_tests = []

        for metrics in all_metrics:
            if not metrics.passed:
                failed_metrics = metrics.get_failed_metrics()

                failed_tests.append(
                    {
                        "test_case_id": metrics.test_case_id,
                        "query": metrics.query,
                        "overall_score": metrics.overall_score,
                        "failed_metric_count": len(failed_metrics),
                        "failed_metrics": [
                            {
                                "name": r.metric_name,
                                "type": r.metric_type.value,
                                "score": r.score,
                                "threshold": r.threshold,
                                "explanation": r.explanation,
                            }
                            for r in failed_metrics
                        ],
                    }
                )

        return failed_tests

    def _generate_chart_data(self, all_metrics: list[EvaluationMetrics]) -> dict[str, Any]:
        """Generate data for charts and visualizations."""
        chart_data = {}

        # 1. Score distribution histogram
        scores = [m.overall_score for m in all_metrics]
        chart_data["score_histogram"] = {
            "scores": scores,
            "bins": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
            "labels": ["Poor", "Fair", "Good", "Very Good", "Excellent"],
        }

        # 2. Metrics by type (radar chart data)
        metric_type_scores = {}
        for metric_type in MetricType:
            scores = []
            for metrics in all_metrics:
                for result in metrics.results:
                    if result.metric_type == metric_type:
                        scores.append(result.score)
            if scores:
                metric_type_scores[metric_type.value] = sum(scores) / len(scores)

        chart_data["radar_chart"] = {
            "labels": list(metric_type_scores.keys()),
            "values": list(metric_type_scores.values()),
        }

        # 3. Pass/Fail by test case (bar chart)
        chart_data["pass_fail_by_test"] = [
            {
                "test_case_id": m.test_case_id,
                "passed": m.passed,
                "score": m.overall_score,
            }
            for m in all_metrics
        ]

        # 4. Time series (if timestamps available)
        chart_data["time_series"] = [
            {
                "timestamp": m.metadata.get("timestamp", datetime.utcnow().isoformat()),
                "score": m.overall_score,
                "test_case_id": m.test_case_id,
            }
            for m in all_metrics
        ]

        # 5. Metric type breakdown (stacked bar)
        metric_breakdown = []
        for metrics in all_metrics:
            breakdown = {"test_case_id": metrics.test_case_id}
            for result in metrics.results:
                breakdown[result.metric_type.value] = result.score
            metric_breakdown.append(breakdown)

        chart_data["metric_breakdown"] = metric_breakdown

        return chart_data

    def export_to_csv(self, all_metrics: list[EvaluationMetrics], filename: str = None) -> Path:
        """
        Export evaluation results to CSV.

        Args:
            all_metrics: List of evaluation metrics
            filename: Output filename (optional)

        Returns:
            Path to CSV file
        """
        if filename is None:
            filename = f"evaluation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Flatten metrics to rows
        rows = []
        for metrics in all_metrics:
            for result in metrics.results:
                rows.append(
                    {
                        "test_case_id": metrics.test_case_id,
                        "query": metrics.query,
                        "metric_name": result.metric_name,
                        "metric_type": result.metric_type.value,
                        "score": result.score,
                        "passed": result.passed,
                        "threshold": result.threshold,
                        "explanation": result.explanation,
                        "timestamp": result.timestamp.isoformat(),
                    }
                )

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / filename

        df.to_csv(csv_path, index=False)
        logger.info(f"CSV export saved to: {csv_path}")

        return csv_path

    def generate_comparison_report(
        self,
        baseline_metrics: list[EvaluationMetrics],
        current_metrics: list[EvaluationMetrics],
        report_name: str = None,
    ) -> dict[str, Any]:
        """
        Generate comparison report between baseline and current results.

        Args:
            baseline_metrics: Baseline evaluation metrics
            current_metrics: Current evaluation metrics
            report_name: Name for comparison report

        Returns:
            Comparison report dictionary
        """
        if report_name is None:
            report_name = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Generating comparison report: {report_name}")

        baseline_summary = self._generate_summary(baseline_metrics)
        current_summary = self._generate_summary(current_metrics)

        # Calculate changes
        changes = {
            "pass_rate_change": current_summary["pass_rate"] - baseline_summary["pass_rate"],
            "average_score_change": current_summary["average_score"] - baseline_summary["average_score"],
            "passed_count_change": current_summary["passed_count"] - baseline_summary["passed_count"],
            "failed_count_change": current_summary["failed_count"] - baseline_summary["failed_count"],
        }

        # Metric-by-metric comparison
        baseline_analysis = self._analyze_metrics(baseline_metrics)
        current_analysis = self._analyze_metrics(current_metrics)

        metric_changes = {}
        for metric_type in set(list(baseline_analysis.keys()) + list(current_analysis.keys())):
            baseline_score = baseline_analysis.get(metric_type, {}).get("average_score", 0.0)
            current_score = current_analysis.get(metric_type, {}).get("average_score", 0.0)

            metric_changes[metric_type] = {
                "baseline_score": baseline_score,
                "current_score": current_score,
                "change": current_score - baseline_score,
                "percent_change": ((current_score - baseline_score) / baseline_score * 100)
                if baseline_score > 0
                else 0.0,
            }

        comparison_report = {
            "report_name": report_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "baseline_summary": baseline_summary,
            "current_summary": current_summary,
            "overall_changes": changes,
            "metric_changes": metric_changes,
            "improvement_detected": changes["average_score_change"] > 0,
            "regression_detected": changes["average_score_change"] < -0.05,  # 5% threshold
        }

        # Save to file
        report_path = self.output_dir / f"{report_name}.json"
        with open(report_path, "w") as f:
            json.dump(comparison_report, f, indent=2)

        logger.info(f"Comparison report saved to: {report_path}")

        return comparison_report

    def generate_html_report(self, report: dict[str, Any], output_path: Path = None) -> Path:
        """
        Generate HTML report from JSON report.

        Args:
            report: Report dictionary
            output_path: Output path for HTML file

        Returns:
            Path to HTML file
        """
        if output_path is None:
            output_path = self.output_dir / f"{report['report_name']}.html"

        html_content = self._generate_html_template(report)

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {output_path}")
        return output_path

    def _generate_html_template(self, report: dict[str, Any]) -> str:
        """Generate HTML template for report."""
        summary = report["summary"]
        metric_analysis = report.get("metric_analysis", {})

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report['report_name']} - Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .header {{ background-color: #1f77b4; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-card {{ display: inline-block; background-color: white; padding: 15px; margin: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #1f77b4; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .pass {{ color: #2ecc71; }}
        .fail {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; background-color: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #1f77b4; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Evaluation Report: {report['report_name']}</h1>
        <p>Generated: {report['generated_at']}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric-card">
            <div class="metric-label">Total Tests</div>
            <div class="metric-value">{summary['total_count']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Passed</div>
            <div class="metric-value pass">{summary['passed_count']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Failed</div>
            <div class="metric-value fail">{summary['failed_count']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Pass Rate</div>
            <div class="metric-value">{summary['pass_rate']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value">{summary['average_score']:.3f}</div>
        </div>
    </div>

    <div class="summary">
        <h2>Metrics by Type</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric Type</th>
                    <th>Average Score</th>
                    <th>Pass Rate</th>
                    <th>Total Evaluations</th>
                </tr>
            </thead>
            <tbody>
"""

        for metric_type, data in metric_analysis.items():
            html += f"""
                <tr>
                    <td>{metric_type}</td>
                    <td>{data['average_score']:.3f}</td>
                    <td>{data['pass_rate']:.1%}</td>
                    <td>{data['total_evaluations']}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html
