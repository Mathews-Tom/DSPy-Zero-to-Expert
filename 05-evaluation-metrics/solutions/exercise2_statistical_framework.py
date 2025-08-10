#!/usr/bin/env python3
"""
Exercise 2 Solution: Statistical Evaluation Framework

This solution demonstrates how to build a statistical evaluation framework
with bootstrap confidence intervals, significance testing, and comprehensive
statistical reporting.
"""

import math
import random
import statistics
from typing import Any, Dict, List, Tuple

import dspy


class StatisticalEvaluator:
    """Statistical evaluation framework with confidence intervals and significance testing."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def bootstrap_confidence_interval(
        self, scores: List[float], n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean score."""
        if not scores:
            return (0.0, 0.0)

        bootstrap_means = []

        # Generate bootstrap samples
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = [random.choice(scores) for _ in range(len(scores))]
            bootstrap_means.append(statistics.mean(bootstrap_sample))

        # Calculate confidence interval
        bootstrap_means.sort()
        lower_idx = int((self.alpha / 2) * n_bootstrap)
        upper_idx = int((1 - self.alpha / 2) * n_bootstrap)

        lower_bound = (
            bootstrap_means[lower_idx]
            if lower_idx < len(bootstrap_means)
            else bootstrap_means[0]
        )
        upper_bound = (
            bootstrap_means[upper_idx]
            if upper_idx < len(bootstrap_means)
            else bootstrap_means[-1]
        )

        return (lower_bound, upper_bound)

    def calculate_effect_size(
        self, scores_a: List[float], scores_b: List[float]
    ) -> float:
        """Calculate Cohen's d effect size."""
        if not scores_a or not scores_b:
            return 0.0

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        # Calculate pooled standard deviation
        if len(scores_a) == 1 and len(scores_b) == 1:
            return 0.0

        var_a = statistics.variance(scores_a) if len(scores_a) > 1 else 0.0
        var_b = statistics.variance(scores_b) if len(scores_b) > 1 else 0.0

        n_a, n_b = len(scores_a), len(scores_b)
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = math.sqrt(pooled_var)

        if pooled_std == 0:
            return 0.0

        # Cohen's d = (mean_b - mean_a) / pooled_standard_deviation
        cohens_d = (mean_b - mean_a) / pooled_std
        return cohens_d

    def permutation_test(
        self, scores_a: List[float], scores_b: List[float], n_permutations: int = 1000
    ) -> float:
        """Perform permutation test for significance."""
        if not scores_a or not scores_b:
            return 1.0

        # Calculate observed difference
        observed_diff = statistics.mean(scores_b) - statistics.mean(scores_a)

        # Combine all scores
        all_scores = scores_a + scores_b
        n_a = len(scores_a)

        # Generate permuted samples and calculate differences
        permuted_diffs = []
        for _ in range(n_permutations):
            # Randomly shuffle and split
            shuffled = all_scores.copy()
            random.shuffle(shuffled)

            perm_a = shuffled[:n_a]
            perm_b = shuffled[n_a:]

            perm_diff = statistics.mean(perm_b) - statistics.mean(perm_a)
            permuted_diffs.append(perm_diff)

        # Calculate p-value (two-tailed)
        extreme_count = sum(
            1 for diff in permuted_diffs if abs(diff) >= abs(observed_diff)
        )
        p_value = extreme_count / n_permutations

        return p_value

    def welch_t_test(
        self, scores_a: List[float], scores_b: List[float]
    ) -> Tuple[float, float, int]:
        """Perform Welch's t-test (unequal variances)."""
        if len(scores_a) < 2 or len(scores_b) < 2:
            return 0.0, 1.0, 0

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        var_a = statistics.variance(scores_a)
        var_b = statistics.variance(scores_b)
        n_a = len(scores_a)
        n_b = len(scores_b)

        # Standard error of difference
        se_diff = math.sqrt(var_a / n_a + var_b / n_b)

        if se_diff == 0:
            return 0.0, 1.0, 0

        # t-statistic
        t_stat = (mean_b - mean_a) / se_diff

        # Degrees of freedom (Welch-Satterthwaite equation)
        if var_a > 0 and var_b > 0:
            df = (var_a / n_a + var_b / n_b) ** 2 / (
                (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
            )
        else:
            df = n_a + n_b - 2

        # Approximate p-value (simplified)
        abs_t = abs(t_stat)
        if abs_t > 3.5:
            p_value = 0.001
        elif abs_t > 2.8:
            p_value = 0.01
        elif abs_t > 2.0:
            p_value = 0.05
        elif abs_t > 1.5:
            p_value = 0.1
        else:
            p_value = 0.2

        return t_stat, p_value, int(df)

    def evaluate_with_statistics(
        self,
        system_a: Any,
        system_b: Any,
        test_examples: List[dspy.Example],
        metric: Any,
    ) -> Dict[str, Any]:
        """Comprehensive statistical evaluation of two systems."""

        # Generate predictions for both systems
        predictions_a = []
        predictions_b = []

        for example in test_examples:
            # System A predictions
            if hasattr(system_a, "forward"):
                inputs = example.inputs()
                pred_a = system_a(**inputs)
            else:
                pred_a = system_a(example)
            predictions_a.append(pred_a)

            # System B predictions
            if hasattr(system_b, "forward"):
                inputs = example.inputs()
                pred_b = system_b(**inputs)
            else:
                pred_b = system_b(example)
            predictions_b.append(pred_b)

        # Calculate scores with the metric
        scores_a = []
        scores_b = []

        for example, pred_a, pred_b in zip(test_examples, predictions_a, predictions_b):
            # Score for system A
            if hasattr(metric, "evaluate"):
                result_a = metric.evaluate(example, pred_a)
                score_a = (
                    result_a.score if hasattr(result_a, "score") else float(result_a)
                )
            else:
                score_a = float(metric(example, pred_a))
            scores_a.append(score_a)

            # Score for system B
            if hasattr(metric, "evaluate"):
                result_b = metric.evaluate(example, pred_b)
                score_b = (
                    result_b.score if hasattr(result_b, "score") else float(result_b)
                )
            else:
                score_b = float(metric(example, pred_b))
            scores_b.append(score_b)

        # Calculate statistical measures
        mean_a = statistics.mean(scores_a) if scores_a else 0.0
        mean_b = statistics.mean(scores_b) if scores_b else 0.0
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0

        # Bootstrap confidence intervals
        ci_a = self.bootstrap_confidence_interval(scores_a)
        ci_b = self.bootstrap_confidence_interval(scores_b)

        # Effect size
        effect_size = self.calculate_effect_size(scores_a, scores_b)

        # Statistical tests
        t_stat, t_p_value, df = self.welch_t_test(scores_a, scores_b)
        perm_p_value = self.permutation_test(scores_a, scores_b)

        # Determine winner
        if abs(mean_b - mean_a) < 0.001:
            winner = "Tie"
            improvement = 0.0
        elif mean_b > mean_a:
            winner = "System B"
            improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0.0
        else:
            winner = "System A"
            improvement = ((mean_a - mean_b) / mean_b * 100) if mean_b > 0 else 0.0

        return {
            "system_a": {
                "name": getattr(system_a, "name", "System A"),
                "mean": mean_a,
                "std": std_a,
                "confidence_interval": ci_a,
                "sample_size": len(scores_a),
                "scores": scores_a,
            },
            "system_b": {
                "name": getattr(system_b, "name", "System B"),
                "mean": mean_b,
                "std": std_b,
                "confidence_interval": ci_b,
                "sample_size": len(scores_b),
                "scores": scores_b,
            },
            "comparison": {
                "mean_difference": mean_b - mean_a,
                "effect_size": effect_size,
                "winner": winner,
                "improvement_percent": improvement,
            },
            "statistical_tests": {
                "t_test": {
                    "t_statistic": t_stat,
                    "p_value": t_p_value,
                    "degrees_of_freedom": df,
                    "significant": t_p_value < self.alpha,
                },
                "permutation_test": {
                    "p_value": perm_p_value,
                    "significant": perm_p_value < self.alpha,
                },
            },
            "metadata": {
                "confidence_level": self.confidence_level,
                "alpha": self.alpha,
                "total_examples": len(test_examples),
            },
        }

    def generate_statistical_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive statistical report."""

        system_a = results["system_a"]
        system_b = results["system_b"]
        comparison = results["comparison"]
        t_test = results["statistical_tests"]["t_test"]
        perm_test = results["statistical_tests"]["permutation_test"]

        # Effect size interpretation
        abs_effect = abs(comparison["effect_size"])
        if abs_effect < 0.2:
            effect_magnitude = "negligible"
        elif abs_effect < 0.5:
            effect_magnitude = "small"
        elif abs_effect < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        # Statistical significance
        is_significant = t_test["significant"] and perm_test["significant"]
        significance_text = (
            "statistically significant"
            if is_significant
            else "not statistically significant"
        )

        report = f"""
STATISTICAL EVALUATION REPORT
{'=' * 50}

SYSTEM PERFORMANCE:
{'-' * 20}
{system_a['name']}:
  Mean Score: {system_a['mean']:.4f} ± {system_a['std']:.4f}
  95% CI: [{system_a['confidence_interval'][0]:.4f}, {system_a['confidence_interval'][1]:.4f}]
  Sample Size: {system_a['sample_size']}

{system_b['name']}:
  Mean Score: {system_b['mean']:.4f} ± {system_b['std']:.4f}
  95% CI: [{system_b['confidence_interval'][0]:.4f}, {system_b['confidence_interval'][1]:.4f}]
  Sample Size: {system_b['sample_size']}

COMPARISON ANALYSIS:
{'-' * 20}
Winner: {comparison['winner']}
Mean Difference: {comparison['mean_difference']:.4f}
Improvement: {comparison['improvement_percent']:.2f}%
Effect Size (Cohen's d): {comparison['effect_size']:.4f} ({effect_magnitude})

STATISTICAL TESTS:
{'-' * 20}
Welch's t-test:
  t-statistic: {t_test['t_statistic']:.4f}
  p-value: {t_test['p_value']:.4f}
  df: {t_test['degrees_of_freedom']}
  Result: {significance_text}

Permutation test:
  p-value: {perm_test['p_value']:.4f}
  Result: {significance_text}

INTERPRETATION:
{'-' * 20}
The difference between systems is {significance_text} at α = {results['metadata']['alpha']:.3f}.
The effect size is {effect_magnitude}, indicating a {'practically meaningful' if abs_effect >= 0.5 else 'small practical'} difference.

RECOMMENDATION:
{'-' * 20}
"""

        if is_significant and abs_effect >= 0.5:
            report += f"Strong evidence favors {comparison['winner']}. Recommend implementation."
        elif is_significant:
            report += f"Weak evidence favors {comparison['winner']}. Consider additional testing."
        else:
            report += "No significant difference detected. Consider collecting more data or testing larger changes."

        return report


# Sample systems for testing
class HighAccuracySystem:
    """High accuracy system for testing."""

    def __init__(self):
        self.name = "High Accuracy System"
        self.answers = {
            "What is 2+2?": "4",
            "What is the capital of France?": "Paris",
            "Who wrote Romeo and Juliet?": "William Shakespeare",
        }

    def forward(self, question):
        answer = self.answers.get(question, "I don't know")
        return dspy.Prediction(answer=answer)


class ModerateAccuracySystem:
    """Moderate accuracy system for testing."""

    def __init__(self):
        self.name = "Moderate Accuracy System"
        self.answers = {
            "What is 2+2?": "Four",  # Different format
            "What is the capital of France?": "Paris, France",  # Verbose
            "Who wrote Romeo and Juliet?": "Shakespeare",  # Abbreviated
        }

    def forward(self, question):
        answer = self.answers.get(question, "Not sure")
        return dspy.Prediction(answer=answer)


class SimpleMetric:
    """Simple exact match metric."""

    def evaluate(self, example, prediction):
        expected = getattr(example, "answer", "").lower().strip()
        predicted = getattr(prediction, "answer", str(prediction)).lower().strip()
        return 1.0 if expected == predicted else 0.0


def test_statistical_evaluator():
    """Test the statistical evaluation framework."""
    print("Testing Statistical Evaluation Framework")
    print("=" * 50)

    # Create test systems
    system_a = HighAccuracySystem()
    system_b = ModerateAccuracySystem()

    # Create test examples
    test_examples = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(
            question="What is the capital of France?", answer="Paris"
        ).with_inputs("question"),
        dspy.Example(
            question="Who wrote Romeo and Juliet?", answer="William Shakespeare"
        ).with_inputs("question"),
    ] * 10  # Repeat for larger sample size

    # Create metric
    metric = SimpleMetric()

    # Create statistical evaluator
    evaluator = StatisticalEvaluator(confidence_level=0.95)

    # Run statistical evaluation
    print("Running statistical evaluation...")
    results = evaluator.evaluate_with_statistics(
        system_a, system_b, test_examples, metric
    )

    # Generate and display report
    report = evaluator.generate_statistical_report(results)
    print(report)

    # Test individual statistical methods
    print("\n" + "=" * 50)
    print("Testing Individual Statistical Methods:")
    print("-" * 30)

    # Sample scores for testing
    scores_a = [0.8, 0.9, 0.7, 0.85, 0.92, 0.88, 0.76, 0.91, 0.83, 0.87]
    scores_b = [0.6, 0.7, 0.5, 0.65, 0.72, 0.68, 0.56, 0.71, 0.63, 0.67]

    # Bootstrap confidence intervals
    ci_a = evaluator.bootstrap_confidence_interval(scores_a)
    ci_b = evaluator.bootstrap_confidence_interval(scores_b)
    print(f"Bootstrap CI A: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"Bootstrap CI B: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

    # Effect size
    effect_size = evaluator.calculate_effect_size(scores_a, scores_b)
    print(f"Effect Size (Cohen's d): {effect_size:.4f}")

    # Permutation test
    p_value = evaluator.permutation_test(scores_a, scores_b)
    print(f"Permutation test p-value: {p_value:.4f}")

    # Welch's t-test
    t_stat, t_p_value, df = evaluator.welch_t_test(scores_a, scores_b)
    print(f"Welch's t-test: t={t_stat:.4f}, p={t_p_value:.4f}, df={df}")


if __name__ == "__main__":
    test_statistical_evaluator()
