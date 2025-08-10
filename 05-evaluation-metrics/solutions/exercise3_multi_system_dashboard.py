#!/usr/bin/env python3
"""
Exercise 3 Solution: Multi-System Evaluation Dashboard

This solution demonstrates how to create an evaluation dashboard that can
compare multiple systems across multiple metrics with ranking, visualization,
and export functionality.
"""

import json
import statistics
from collections import defaultdict
from typing import Any

import dspy


class MultiSystemEvaluator:
    """Evaluation dashboard for comparing multiple systems across multiple metrics."""

    def __init__(self):
        self.systems = {}
        self.metrics = {}
        self.evaluation_results = {}
        self.rankings = {}

    def add_system(self, name: str, system: Any) -> None:
        """Add a system to evaluate."""
        self.systems[name] = {
            "system": system,
            "name": name,
            "description": getattr(system, "description", f"System: {name}"),
        }
        print(f"Added system: {name}")

    def add_metric(self, name: str, metric: Any, weight: float = 1.0) -> None:
        """Add a metric with optional weight."""
        self.metrics[name] = {
            "metric": metric,
            "name": name,
            "weight": weight,
            "description": getattr(metric, "description", f"Metric: {name}"),
        }
        print(f"Added metric: {name} (weight: {weight})")

    def evaluate_all_systems(self, test_examples: list[dspy.Example]) -> dict[str, Any]:
        """Evaluate all systems with all metrics."""
        print(
            f"Evaluating {len(self.systems)} systems with {len(self.metrics)} metrics..."
        )

        self.evaluation_results = {}

        # For each system, generate predictions and evaluate
        for system_name, system_info in self.systems.items():
            print(f"  Evaluating {system_name}...")
            system = system_info["system"]

            # Generate predictions
            predictions = []
            for example in test_examples:
                if hasattr(system, "forward"):
                    inputs = example.inputs()
                    pred = system(**inputs)
                else:
                    pred = system(example)
                predictions.append(pred)

            # Evaluate with each metric
            system_results = {}
            for metric_name, metric_info in self.metrics.items():
                metric = metric_info["metric"]

                # Calculate scores
                scores = []
                detailed_results = []

                for example, prediction in zip(
                    test_examples, predictions, strict=False
                ):
                    if hasattr(metric, "evaluate"):
                        result = metric.evaluate(example, prediction)
                        score = (
                            result.score if hasattr(result, "score") else float(result)
                        )
                        detailed_results.append(result)
                    else:
                        score = float(metric(example, prediction))
                        detailed_results.append({"score": score})
                    scores.append(score)

                # Aggregate results
                system_results[metric_name] = {
                    "scores": scores,
                    "mean": statistics.mean(scores) if scores else 0.0,
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min": min(scores) if scores else 0.0,
                    "max": max(scores) if scores else 0.0,
                    "count": len(scores),
                    "detailed_results": detailed_results,
                }

            self.evaluation_results[system_name] = system_results

        # Calculate rankings
        self.rankings = self.calculate_rankings()

        return {
            "systems_evaluated": len(self.systems),
            "metrics_used": len(self.metrics),
            "examples_processed": len(test_examples),
            "results": self.evaluation_results,
            "rankings": self.rankings,
        }

    def calculate_rankings(self) -> dict[str, list[tuple[str, float]]]:
        """Calculate rankings for each metric and overall."""
        rankings = {}

        # Rank systems for each individual metric
        for metric_name in self.metrics.keys():
            metric_scores = []
            for system_name, system_results in self.evaluation_results.items():
                if metric_name in system_results:
                    score = system_results[metric_name]["mean"]
                    metric_scores.append((system_name, score))

            # Sort by score (descending)
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric_name] = metric_scores

        # Calculate weighted overall ranking
        overall_scores = defaultdict(float)
        total_weight = sum(
            metric_info["weight"] for metric_info in self.metrics.values()
        )

        for system_name, system_results in self.evaluation_results.items():
            weighted_score = 0.0
            for metric_name, metric_info in self.metrics.items():
                if metric_name in system_results:
                    score = system_results[metric_name]["mean"]
                    weight = metric_info["weight"]
                    weighted_score += score * weight

            # Normalize by total weight
            if total_weight > 0:
                overall_scores[system_name] = weighted_score / total_weight

        # Sort overall scores
        overall_ranking = sorted(
            overall_scores.items(), key=lambda x: x[1], reverse=True
        )
        rankings["overall"] = overall_ranking

        return rankings

    def generate_comparison_matrix(self) -> dict[str, dict[str, float]]:
        """Generate pairwise comparison matrix."""
        comparison_matrix = {}
        system_names = list(self.systems.keys())

        for system_a in system_names:
            comparison_matrix[system_a] = {}

            for system_b in system_names:
                if system_a == system_b:
                    comparison_matrix[system_a][system_b] = 0.0  # Tie
                else:
                    # Calculate how much better system_a is than system_b
                    wins = 0
                    total_comparisons = 0

                    for metric_name in self.metrics.keys():
                        if (
                            metric_name in self.evaluation_results[system_a]
                            and metric_name in self.evaluation_results[system_b]
                        ):

                            score_a = self.evaluation_results[system_a][metric_name][
                                "mean"
                            ]
                            score_b = self.evaluation_results[system_b][metric_name][
                                "mean"
                            ]

                            if score_a > score_b:
                                wins += 1
                            total_comparisons += 1

                    # Win rate
                    win_rate = (
                        wins / total_comparisons if total_comparisons > 0 else 0.0
                    )
                    comparison_matrix[system_a][system_b] = win_rate

        return comparison_matrix

    def create_visualization_data(self) -> dict[str, Any]:
        """Create data structure for visualization."""
        viz_data = {
            "bar_chart": self._create_bar_chart_data(),
            "radar_chart": self._create_radar_chart_data(),
            "heatmap": self._create_heatmap_data(),
            "scatter_plot": self._create_scatter_plot_data(),
        }
        return viz_data

    def _create_bar_chart_data(self) -> dict[str, Any]:
        """Create data for bar chart visualization."""
        data = {"systems": list(self.systems.keys()), "metrics": {}}

        for metric_name in self.metrics.keys():
            scores = []
            for system_name in self.systems.keys():
                if (
                    system_name in self.evaluation_results
                    and metric_name in self.evaluation_results[system_name]
                ):
                    score = self.evaluation_results[system_name][metric_name]["mean"]
                else:
                    score = 0.0
                scores.append(score)
            data["metrics"][metric_name] = scores

        return data

    def _create_radar_chart_data(self) -> dict[str, Any]:
        """Create data for radar chart visualization."""
        data = {"metrics": list(self.metrics.keys()), "systems": {}}

        for system_name in self.systems.keys():
            scores = []
            for metric_name in self.metrics.keys():
                if (
                    system_name in self.evaluation_results
                    and metric_name in self.evaluation_results[system_name]
                ):
                    score = self.evaluation_results[system_name][metric_name]["mean"]
                else:
                    score = 0.0
                scores.append(score)
            data["systems"][system_name] = scores

        return data

    def _create_heatmap_data(self) -> dict[str, Any]:
        """Create data for heatmap visualization."""
        systems = list(self.systems.keys())
        metrics = list(self.metrics.keys())

        # Create matrix of scores
        matrix = []
        for system_name in systems:
            row = []
            for metric_name in metrics:
                if (
                    system_name in self.evaluation_results
                    and metric_name in self.evaluation_results[system_name]
                ):
                    score = self.evaluation_results[system_name][metric_name]["mean"]
                else:
                    score = 0.0
                row.append(score)
            matrix.append(row)

        return {"systems": systems, "metrics": metrics, "matrix": matrix}

    def _create_scatter_plot_data(self) -> dict[str, Any]:
        """Create data for scatter plot visualization (metric1 vs metric2)."""
        if len(self.metrics) < 2:
            return {"error": "Need at least 2 metrics for scatter plot"}

        metric_names = list(self.metrics.keys())
        metric1, metric2 = metric_names[0], metric_names[1]

        data = {"metric1": metric1, "metric2": metric2, "points": []}

        for system_name in self.systems.keys():
            x = 0.0
            y = 0.0

            if (
                system_name in self.evaluation_results
                and metric1 in self.evaluation_results[system_name]
            ):
                x = self.evaluation_results[system_name][metric1]["mean"]

            if (
                system_name in self.evaluation_results
                and metric2 in self.evaluation_results[system_name]
            ):
                y = self.evaluation_results[system_name][metric2]["mean"]

            data["points"].append({"system": system_name, "x": x, "y": y})

        return data

    def export_results(self, format_type: str = "json") -> str:
        """Export evaluation results in specified format."""

        if format_type == "json":
            export_data = {
                "systems": {name: info["name"] for name, info in self.systems.items()},
                "metrics": {
                    name: {"name": info["name"], "weight": info["weight"]}
                    for name, info in self.metrics.items()
                },
                "results": self.evaluation_results,
                "rankings": self.rankings,
                "comparison_matrix": self.generate_comparison_matrix(),
                "visualization_data": self.create_visualization_data(),
            }
            return json.dumps(export_data, indent=2)

        elif format_type == "csv":
            # Create CSV format
            lines = ["System,Metric,Score,Std,Min,Max"]

            for system_name, system_results in self.evaluation_results.items():
                for metric_name, metric_data in system_results.items():
                    line = (
                        f"{system_name},{metric_name},{metric_data['mean']:.4f},"
                        + f"{metric_data['std']:.4f},{metric_data['min']:.4f},{metric_data['max']:.4f}"
                    )
                    lines.append(line)

            return "\n".join(lines)

        elif format_type == "text":
            return self._generate_text_report()

        else:
            return f"Unsupported format: {format_type}"

    def _generate_text_report(self) -> str:
        """Generate formatted text report."""
        report = []
        report.append("MULTI-SYSTEM EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")

        # Summary
        report.append("SUMMARY:")
        report.append(f"  Systems Evaluated: {len(self.systems)}")
        report.append(f"  Metrics Used: {len(self.metrics)}")
        report.append("")

        # Overall Rankings
        if "overall" in self.rankings:
            report.append("OVERALL RANKINGS:")
            for i, (system_name, score) in enumerate(self.rankings["overall"], 1):
                report.append(f"  {i}. {system_name}: {score:.4f}")
            report.append("")

        # Detailed Results
        report.append("DETAILED RESULTS:")
        report.append("-" * 30)

        for system_name, system_results in self.evaluation_results.items():
            report.append(f"\n{system_name}:")
            for metric_name, metric_data in system_results.items():
                report.append(
                    f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}"
                )

        # Metric Rankings
        report.append("\nMETRIC-SPECIFIC RANKINGS:")
        report.append("-" * 30)

        for metric_name, ranking in self.rankings.items():
            if metric_name != "overall":
                report.append(f"\n{metric_name}:")
                for i, (system_name, score) in enumerate(ranking, 1):
                    report.append(f"  {i}. {system_name}: {score:.4f}")

        return "\n".join(report)

    def generate_insights(self) -> list[str]:
        """Generate insights from evaluation results."""
        insights = []

        if not self.evaluation_results or not self.rankings:
            insights.append("No evaluation results available for analysis.")
            return insights

        # Overall winner
        if "overall" in self.rankings and self.rankings["overall"]:
            winner = self.rankings["overall"][0]
            insights.append(
                f"Best overall performer: {winner[0]} (score: {winner[1]:.4f})"
            )

        # Consistency analysis
        system_std_devs = {}
        for system_name, system_results in self.evaluation_results.items():
            std_devs = [metric_data["std"] for metric_data in system_results.values()]
            avg_std = statistics.mean(std_devs) if std_devs else 0.0
            system_std_devs[system_name] = avg_std

        most_consistent = min(system_std_devs.items(), key=lambda x: x[1])
        insights.append(
            f"Most consistent performer: {most_consistent[0]} (avg std: {most_consistent[1]:.4f})"
        )

        # Metric-specific dominance
        metric_winners = {}
        for metric_name, ranking in self.rankings.items():
            if metric_name != "overall" and ranking:
                winner = ranking[0][0]
                metric_winners[winner] = metric_winners.get(winner, 0) + 1

        if metric_winners:
            dominant_system = max(metric_winners.items(), key=lambda x: x[1])
            insights.append(
                f"Most metric wins: {dominant_system[0]} ({dominant_system[1]} out of {len(self.metrics)} metrics)"
            )

        # Performance spread analysis
        overall_scores = (
            [score for _, score in self.rankings["overall"]]
            if "overall" in self.rankings
            else []
        )
        if len(overall_scores) > 1:
            score_range = max(overall_scores) - min(overall_scores)
            if score_range > 0.2:
                insights.append(
                    "Large performance differences detected between systems"
                )
            elif score_range < 0.05:
                insights.append("Systems show similar performance levels")
            else:
                insights.append("Moderate performance differences between systems")

        # Recommendations
        if "overall" in self.rankings and len(self.rankings["overall"]) > 1:
            best_system = self.rankings["overall"][0]
            second_best = self.rankings["overall"][1]

            if best_system[1] - second_best[1] > 0.1:
                insights.append(f"Strong recommendation: Deploy {best_system[0]}")
            else:
                insights.append(
                    f"Consider A/B testing between {best_system[0]} and {second_best[0]}"
                )

        return insights


# Sample systems for testing
class HighPerformanceSystem:
    """High performance system for testing."""

    def __init__(self):
        self.name = "High Performance System"
        self.description = "System optimized for high accuracy"

    def forward(self, question):
        # Simulate high accuracy responses
        answers = {
            "What is 2+2?": "4",
            "What is the capital of France?": "Paris",
            "Who wrote Romeo and Juliet?": "William Shakespeare",
        }
        return dspy.Prediction(answer=answers.get(question, "High quality answer"))


class BalancedSystem:
    """Balanced system for testing."""

    def __init__(self):
        self.name = "Balanced System"
        self.description = "System with balanced performance"

    def forward(self, question):
        # Simulate moderate accuracy responses
        answers = {
            "What is 2+2?": "Four",
            "What is the capital of France?": "Paris, France",
            "Who wrote Romeo and Juliet?": "Shakespeare",
        }
        return dspy.Prediction(answer=answers.get(question, "Balanced answer"))


class FastSystem:
    """Fast but lower accuracy system for testing."""

    def __init__(self):
        self.name = "Fast System"
        self.description = "System optimized for speed"

    def forward(self, question):
        # Simulate fast but less accurate responses
        return dspy.Prediction(answer="Quick response")


# Sample metrics
class ExactMatchMetric:
    """Exact match metric."""

    def __init__(self):
        self.name = "Exact Match"
        self.description = "Exact string matching"

    def evaluate(self, example, prediction):
        expected = getattr(example, "answer", "").lower().strip()
        predicted = getattr(prediction, "answer", str(prediction)).lower().strip()
        return 1.0 if expected == predicted else 0.0


class LengthMetric:
    """Response length metric."""

    def __init__(self):
        self.name = "Response Length"
        self.description = "Normalized response length score"

    def evaluate(self, example, prediction):
        predicted = getattr(prediction, "answer", str(prediction))
        # Normalize length score (optimal around 10-50 characters)
        length = len(predicted)
        if 10 <= length <= 50:
            return 1.0
        elif length < 10:
            return length / 10.0
        else:
            return max(0.0, 1.0 - (length - 50) / 100.0)


def test_multi_system_evaluator():
    """Test the multi-system evaluation dashboard."""
    print("Testing Multi-System Evaluation Dashboard")
    print("=" * 50)

    # Create evaluator
    evaluator = MultiSystemEvaluator()

    # Add systems
    evaluator.add_system("high_performance", HighPerformanceSystem())
    evaluator.add_system("balanced", BalancedSystem())
    evaluator.add_system("fast", FastSystem())

    # Add metrics with weights
    evaluator.add_metric("exact_match", ExactMatchMetric(), weight=0.7)
    evaluator.add_metric("length", LengthMetric(), weight=0.3)

    # Create test examples
    test_examples = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(
            question="What is the capital of France?", answer="Paris"
        ).with_inputs("question"),
        dspy.Example(
            question="Who wrote Romeo and Juliet?", answer="William Shakespeare"
        ).with_inputs("question"),
    ] * 5  # Repeat for more data

    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.evaluate_all_systems(test_examples)

    # Display results
    print("\nEvaluation completed:")
    print(f"  Systems: {results['systems_evaluated']}")
    print(f"  Metrics: {results['metrics_used']}")
    print(f"  Examples: {results['examples_processed']}")

    # Show rankings
    print("\nRankings:")
    for metric_name, ranking in evaluator.rankings.items():
        print(f"\n{metric_name.title()}:")
        for i, (system_name, score) in enumerate(ranking, 1):
            print(f"  {i}. {system_name}: {score:.4f}")

    # Generate insights
    print("\nInsights:")
    insights = evaluator.generate_insights()
    for insight in insights:
        print(f"  • {insight}")

    # Test export functionality
    print("\n" + "=" * 50)
    print("Testing Export Functionality:")
    print("-" * 30)

    # Export as text
    text_report = evaluator.export_results("text")
    print("\nText Report (first 500 chars):")
    print(text_report[:500] + "..." if len(text_report) > 500 else text_report)

    # Export as CSV
    csv_data = evaluator.export_results("csv")
    print(f"\nCSV Export (lines: {len(csv_data.split(chr(10)))})")
    print("First few lines:")
    print("\n".join(csv_data.split("\n")[:5]))

    # Test visualization data
    print("\n" + "=" * 50)
    print("Testing Visualization Data:")
    print("-" * 30)

    viz_data = evaluator.create_visualization_data()
    print(f"Visualization components: {list(viz_data.keys())}")

    # Show bar chart data
    bar_data = viz_data["bar_chart"]
    print("\nBar chart data:")
    print(f"  Systems: {bar_data['systems']}")
    for metric, scores in bar_data["metrics"].items():
        print(f"  {metric}: {[f'{s:.3f}' for s in scores]}")

    # Test comparison matrix
    print("\n" + "=" * 50)
    print("Testing Comparison Matrix:")
    print("-" * 30)

    comparison_matrix = evaluator.generate_comparison_matrix()
    print("Win rates (row vs column):")
    systems = list(comparison_matrix.keys())

    # Print header
    print("System".ljust(20), end="")
    for system in systems:
        print(system[:10].ljust(12), end="")
    print()

    # Print matrix
    for system_a in systems:
        print(system_a[:19].ljust(20), end="")
        for system_b in systems:
            win_rate = comparison_matrix[system_a][system_b]
            print(f"{win_rate:.3f}".ljust(12), end="")
        print()


if __name__ == "__main__":
    test_multi_system_evaluator()
