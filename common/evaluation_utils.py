"""
Evaluation utilities and metrics framework for DSPy applications.

This module provides custom metrics, evaluation tools, A/B testing framework,
and performance benchmarking utilities for DSPy systems.
"""

# Standard Library
import json
import logging
import random
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Third-Party Library
import dspy
import numpy as np
import pandas as pd

# Local Modules
from .config import get_config
from .utils import ProgressTracker, timer

logger = logging.getLogger(__name__)


# =============================================================================
# Base Metric Classes
# =============================================================================


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize the metric.

        Args:
            name: Metric name
            description: Metric description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single prediction against ground truth.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth value
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult object
        """
        pass

    def evaluate_batch(
        self, predictions: List[Any], ground_truths: List[Any], **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of predictions.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truth values
            **kwargs: Additional evaluation parameters

        Returns:
            List of EvaluationResult objects
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")

        results = []
        progress = ProgressTracker(len(predictions), f"Evaluating {self.name}")

        for pred, gt in zip(predictions, ground_truths):
            result = self.evaluate(pred, gt, **kwargs)
            results.append(result)
            progress.update()

        return results

    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Aggregate multiple evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Aggregated metrics
        """
        if not results:
            return {}

        scores = [r.score for r in results]

        aggregated = {
            "metric_name": self.name,
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "total_execution_time": sum(r.execution_time for r in results),
        }

        return aggregated


# =============================================================================
# Common Metrics
# =============================================================================


class ExactMatchMetric(BaseMetric):
    """Exact string match metric."""

    def __init__(self, case_sensitive: bool = True, strip_whitespace: bool = True):
        """
        Initialize exact match metric.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
            strip_whitespace: Whether to strip whitespace before comparison
        """
        super().__init__("exact_match", "Exact string match")
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def evaluate(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> EvaluationResult:
        """Evaluate exact match."""
        start_time = time.time()

        pred_str = str(prediction)
        gt_str = str(ground_truth)

        if self.strip_whitespace:
            pred_str = pred_str.strip()
            gt_str = gt_str.strip()

        if not self.case_sensitive:
            pred_str = pred_str.lower()
            gt_str = gt_str.lower()

        score = 1.0 if pred_str == gt_str else 0.0

        end_time = time.time()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details={
                "prediction": str(prediction),
                "ground_truth": str(ground_truth),
                "case_sensitive": self.case_sensitive,
                "strip_whitespace": self.strip_whitespace,
            },
            timestamp=datetime.now(),
            execution_time=end_time - start_time,
        )


class ContainsMetric(BaseMetric):
    """Check if prediction contains ground truth."""

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize contains metric.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
        """
        super().__init__("contains", "Check if prediction contains ground truth")
        self.case_sensitive = case_sensitive

    def evaluate(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> EvaluationResult:
        """Evaluate contains match."""
        start_time = time.time()

        pred_str = str(prediction)
        gt_str = str(ground_truth)

        if not self.case_sensitive:
            pred_str = pred_str.lower()
            gt_str = gt_str.lower()

        score = 1.0 if gt_str in pred_str else 0.0

        end_time = time.time()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details={
                "prediction": str(prediction),
                "ground_truth": str(ground_truth),
                "case_sensitive": self.case_sensitive,
            },
            timestamp=datetime.now(),
            execution_time=end_time - start_time,
        )


class NumericMetric(BaseMetric):
    """Numeric comparison metric with tolerance."""

    def __init__(self, tolerance: float = 0.01, relative: bool = False):
        """
        Initialize numeric metric.

        Args:
            tolerance: Tolerance for numeric comparison
            relative: Whether tolerance is relative (percentage) or absolute
        """
        super().__init__("numeric", "Numeric comparison with tolerance")
        self.tolerance = tolerance
        self.relative = relative

    def evaluate(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> EvaluationResult:
        """Evaluate numeric match."""
        start_time = time.time()

        try:
            pred_num = float(prediction)
            gt_num = float(ground_truth)

            if self.relative:
                # Relative tolerance
                if gt_num == 0:
                    score = 1.0 if pred_num == 0 else 0.0
                else:
                    relative_error = abs(pred_num - gt_num) / abs(gt_num)
                    score = 1.0 if relative_error <= self.tolerance else 0.0
            else:
                # Absolute tolerance
                absolute_error = abs(pred_num - gt_num)
                score = 1.0 if absolute_error <= self.tolerance else 0.0

            details = {
                "prediction": pred_num,
                "ground_truth": gt_num,
                "tolerance": self.tolerance,
                "relative": self.relative,
                "error": abs(pred_num - gt_num),
            }

        except (ValueError, TypeError):
            score = 0.0
            details = {
                "prediction": str(prediction),
                "ground_truth": str(ground_truth),
                "error": "Could not convert to numeric values",
            }

        end_time = time.time()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details=details,
            timestamp=datetime.now(),
            execution_time=end_time - start_time,
        )


class SemanticSimilarityMetric(BaseMetric):
    """Semantic similarity metric using embeddings."""

    def __init__(
        self,
        threshold: float = 0.8,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize semantic similarity metric.

        Args:
            threshold: Similarity threshold for binary classification
            model_name: Sentence transformer model name
        """
        super().__init__("semantic_similarity", "Semantic similarity using embeddings")
        self.threshold = threshold
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic similarity metric"
                )
        return self._model

    def evaluate(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> EvaluationResult:
        """Evaluate semantic similarity."""
        start_time = time.time()

        try:
            model = self._get_model()

            pred_str = str(prediction)
            gt_str = str(ground_truth)

            # Get embeddings
            embeddings = model.encode([pred_str, gt_str])

            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            # Binary score based on threshold
            binary_score = 1.0 if similarity >= self.threshold else 0.0

            details = {
                "prediction": pred_str,
                "ground_truth": gt_str,
                "similarity": float(similarity),
                "threshold": self.threshold,
                "model": self.model_name,
            }

        except Exception as e:
            logger.error(f"Semantic similarity evaluation failed: {e}")
            binary_score = 0.0
            details = {
                "prediction": str(prediction),
                "ground_truth": str(ground_truth),
                "error": str(e),
            }

        end_time = time.time()

        return EvaluationResult(
            metric_name=self.name,
            score=binary_score,
            details=details,
            timestamp=datetime.now(),
            execution_time=end_time - start_time,
        )


# =============================================================================
# Custom Metric Builder
# =============================================================================


class CustomMetric(BaseMetric):
    """Custom metric defined by a user function."""

    def __init__(
        self,
        name: str,
        evaluation_function: Callable[[Any, Any], float],
        description: str = "",
    ):
        """
        Initialize custom metric.

        Args:
            name: Metric name
            evaluation_function: Function that takes (prediction, ground_truth) and returns a score
            description: Metric description
        """
        super().__init__(name, description)
        self.evaluation_function = evaluation_function

    def evaluate(
        self, prediction: Any, ground_truth: Any, **kwargs
    ) -> EvaluationResult:
        """Evaluate using custom function."""
        start_time = time.time()

        try:
            score = self.evaluation_function(prediction, ground_truth)
            if not isinstance(score, (int, float)):
                raise ValueError("Evaluation function must return a numeric score")

            details = {
                "prediction": str(prediction),
                "ground_truth": str(ground_truth),
                "custom_function": self.evaluation_function.__name__,
            }

        except Exception as e:
            logger.error(f"Custom metric evaluation failed: {e}")
            score = 0.0
            details = {
                "prediction": str(prediction),
                "ground_truth": str(ground_truth),
                "error": str(e),
            }

        end_time = time.time()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details=details,
            timestamp=datetime.now(),
            execution_time=end_time - start_time,
        )


# =============================================================================
# Evaluation Framework
# =============================================================================


class EvaluationSuite:
    """
    Comprehensive evaluation suite for DSPy systems.
    """

    def __init__(self, name: str = "Evaluation Suite"):
        """
        Initialize evaluation suite.

        Args:
            name: Suite name
        """
        self.name = name
        self.metrics = {}
        self.evaluation_history = []

    def add_metric(self, metric: BaseMetric):
        """
        Add a metric to the suite.

        Args:
            metric: Metric to add
        """
        self.metrics[metric.name] = metric

    def remove_metric(self, metric_name: str):
        """
        Remove a metric from the suite.

        Args:
            metric_name: Name of metric to remove
        """
        if metric_name in self.metrics:
            del self.metrics[metric_name]

    def evaluate_single(
        self,
        prediction: Any,
        ground_truth: Any,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single prediction with all or specified metrics.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth value
            metric_names: Specific metrics to use (None for all)

        Returns:
            Dictionary mapping metric names to results
        """
        if metric_names is None:
            metric_names = list(self.metrics.keys())

        results = {}
        for metric_name in metric_names:
            if metric_name in self.metrics:
                result = self.metrics[metric_name].evaluate(prediction, ground_truth)
                results[metric_name] = result

        return results

    def evaluate_dataset(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate a dataset with all or specified metrics.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truth values
            metric_names: Specific metrics to use (None for all)

        Returns:
            Dictionary mapping metric names to result lists
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")

        if metric_names is None:
            metric_names = list(self.metrics.keys())

        results = {}
        for metric_name in metric_names:
            if metric_name in self.metrics:
                metric_results = self.metrics[metric_name].evaluate_batch(
                    predictions, ground_truths
                )
                results[metric_name] = metric_results

        # Store evaluation in history
        evaluation_record = {
            "timestamp": datetime.now(),
            "dataset_size": len(predictions),
            "metrics_used": metric_names,
            "results": results,
        }
        self.evaluation_history.append(evaluation_record)

        return results

    def get_aggregated_results(
        self, results: Dict[str, List[EvaluationResult]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated results for all metrics.

        Args:
            results: Results from evaluate_dataset

        Returns:
            Aggregated results for each metric
        """
        aggregated = {}
        for metric_name, metric_results in results.items():
            if metric_name in self.metrics:
                aggregated[metric_name] = self.metrics[metric_name].aggregate_results(
                    metric_results
                )

        return aggregated

    def compare_systems(
        self, system_results: Dict[str, Dict[str, List[EvaluationResult]]]
    ) -> pd.DataFrame:
        """
        Compare multiple systems using evaluation results.

        Args:
            system_results: Dictionary mapping system names to their evaluation results

        Returns:
            DataFrame with comparison results
        """
        comparison_data = []

        for system_name, results in system_results.items():
            aggregated = self.get_aggregated_results(results)

            row = {"System": system_name}
            for metric_name, agg_results in aggregated.items():
                row[f"{metric_name}_mean"] = agg_results["mean"]
                row[f"{metric_name}_std"] = agg_results["std"]
                row[f"{metric_name}_count"] = agg_results["count"]

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)


# =============================================================================
# A/B Testing Framework
# =============================================================================


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    name: str
    description: str
    traffic_split: float = 0.5  # Fraction of traffic to system A
    min_sample_size: int = 100
    significance_level: float = 0.05
    power: float = 0.8


class ABTester:
    """
    A/B testing framework for comparing DSPy systems.
    """

    def __init__(self, config: ABTestConfig):
        """
        Initialize A/B tester.

        Args:
            config: A/B test configuration
        """
        self.config = config
        self.system_a_results = []
        self.system_b_results = []
        self.assignments = {}  # Track which system each sample was assigned to

    def assign_system(self, sample_id: str) -> str:
        """
        Assign a sample to system A or B.

        Args:
            sample_id: Unique identifier for the sample

        Returns:
            System assignment ('A' or 'B')
        """
        # Use deterministic assignment based on sample_id hash
        random.seed(hash(sample_id))
        assignment = "A" if random.random() < self.config.traffic_split else "B"
        self.assignments[sample_id] = assignment
        return assignment

    def add_result(self, sample_id: str, system: str, score: float):
        """
        Add a result for a specific system.

        Args:
            sample_id: Sample identifier
            system: System name ('A' or 'B')
            score: Evaluation score
        """
        if system == "A":
            self.system_a_results.append({"sample_id": sample_id, "score": score})
        elif system == "B":
            self.system_b_results.append({"sample_id": sample_id, "score": score})
        else:
            raise ValueError("System must be 'A' or 'B'")

    def get_sample_sizes(self) -> Tuple[int, int]:
        """Get sample sizes for both systems."""
        return len(self.system_a_results), len(self.system_b_results)

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate test statistics."""
        if not self.system_a_results or not self.system_b_results:
            return {"error": "Insufficient data for both systems"}

        scores_a = [r["score"] for r in self.system_a_results]
        scores_b = [r["score"] for r in self.system_b_results]

        stats = {
            "system_a": {
                "count": len(scores_a),
                "mean": statistics.mean(scores_a),
                "std": statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0,
                "min": min(scores_a),
                "max": max(scores_a),
            },
            "system_b": {
                "count": len(scores_b),
                "mean": statistics.mean(scores_b),
                "std": statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0,
                "min": min(scores_b),
                "max": max(scores_b),
            },
        }

        # Calculate effect size and confidence interval
        mean_diff = stats["system_a"]["mean"] - stats["system_b"]["mean"]
        stats["effect_size"] = mean_diff

        return stats

    def run_significance_test(self) -> Dict[str, Any]:
        """
        Run statistical significance test.

        Returns:
            Test results including p-value and significance
        """
        try:
            from scipy import stats as scipy_stats

            scores_a = [r["score"] for r in self.system_a_results]
            scores_b = [r["score"] for r in self.system_b_results]

            # Perform t-test
            t_stat, p_value = scipy_stats.ttest_ind(scores_a, scores_b)

            is_significant = p_value < self.config.significance_level

            result = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significance_level": self.config.significance_level,
                "is_significant": is_significant,
                "winner": (
                    "A"
                    if statistics.mean(scores_a) > statistics.mean(scores_b)
                    else "B"
                ),
                "confidence": 1 - self.config.significance_level,
            }

            return result

        except ImportError:
            return {"error": "scipy is required for significance testing"}
        except Exception as e:
            return {"error": f"Statistical test failed: {str(e)}"}

    def get_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        summary = {
            "config": asdict(self.config),
            "sample_sizes": self.get_sample_sizes(),
            "statistics": self.calculate_statistics(),
            "significance_test": self.run_significance_test(),
        }

        return summary


# =============================================================================
# Performance Benchmarking
# =============================================================================


class PerformanceBenchmark:
    """
    Performance benchmarking utility for DSPy systems.
    """

    def __init__(self, name: str = "Performance Benchmark"):
        """
        Initialize performance benchmark.

        Args:
            name: Benchmark name
        """
        self.name = name
        self.benchmark_results = []

    def benchmark_system(
        self,
        system: Callable,
        test_inputs: List[Any],
        system_name: str = "System",
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark a system's performance.

        Args:
            system: System to benchmark (callable)
            test_inputs: List of test inputs
            system_name: Name for the system
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {system_name} with {len(test_inputs)} inputs")

        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            for test_input in test_inputs[
                : min(5, len(test_inputs))
            ]:  # Use first 5 inputs for warmup
                try:
                    system(test_input)
                except Exception:
                    pass  # Ignore errors during warmup

        # Benchmark runs
        execution_times = []
        successful_runs = 0
        failed_runs = 0

        logger.info(f"Running {benchmark_runs} benchmark runs...")
        progress = ProgressTracker(
            benchmark_runs * len(test_inputs), f"Benchmarking {system_name}"
        )

        for run in range(benchmark_runs):
            for test_input in test_inputs:
                start_time = time.time()
                try:
                    result = system(test_input)
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                    successful_runs += 1
                except Exception as e:
                    end_time = time.time()
                    failed_runs += 1
                    logger.debug(f"Benchmark run failed: {e}")

                progress.update()

        # Calculate statistics
        if execution_times:
            benchmark_result = {
                "system_name": system_name,
                "timestamp": datetime.now(),
                "total_runs": len(execution_times) + failed_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / (successful_runs + failed_runs),
                "execution_times": {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "std": (
                        statistics.stdev(execution_times)
                        if len(execution_times) > 1
                        else 0.0
                    ),
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "p95": (
                        np.percentile(execution_times, 95) if execution_times else 0.0
                    ),
                    "p99": (
                        np.percentile(execution_times, 99) if execution_times else 0.0
                    ),
                },
                "throughput": {
                    "requests_per_second": (
                        len(execution_times) / sum(execution_times)
                        if execution_times
                        else 0.0
                    ),
                    "total_time": sum(execution_times),
                },
            }
        else:
            benchmark_result = {
                "system_name": system_name,
                "timestamp": datetime.now(),
                "error": "No successful runs completed",
            }

        self.benchmark_results.append(benchmark_result)
        return benchmark_result

    def compare_systems(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare benchmark results from multiple systems.

        Args:
            results: List of benchmark results

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for result in results:
            if "error" not in result:
                row = {
                    "System": result["system_name"],
                    "Success Rate": result["success_rate"],
                    "Mean Time (s)": result["execution_times"]["mean"],
                    "Median Time (s)": result["execution_times"]["median"],
                    "P95 Time (s)": result["execution_times"]["p95"],
                    "P99 Time (s)": result["execution_times"]["p99"],
                    "Requests/sec": result["throughput"]["requests_per_second"],
                    "Total Runs": result["total_runs"],
                }
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)


# =============================================================================
# Utility Functions
# =============================================================================


def create_evaluation_suite() -> EvaluationSuite:
    """Create a standard evaluation suite with common metrics."""
    suite = EvaluationSuite("Standard Evaluation Suite")

    # Add common metrics
    suite.add_metric(ExactMatchMetric())
    suite.add_metric(ContainsMetric())
    suite.add_metric(NumericMetric())

    # Add semantic similarity if available
    try:
        suite.add_metric(SemanticSimilarityMetric())
    except ImportError:
        logger.warning(
            "Semantic similarity metric not available (sentence-transformers not installed)"
        )

    return suite


def quick_evaluate(
    predictions: List[Any],
    ground_truths: List[Any],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation with standard metrics.

    Args:
        predictions: List of predictions
        ground_truths: List of ground truth values
        metrics: Specific metrics to use

    Returns:
        Aggregated evaluation results
    """
    suite = create_evaluation_suite()
    results = suite.evaluate_dataset(predictions, ground_truths, metrics)
    return suite.get_aggregated_results(results)
