#!/usr/bin/env python3
"""
Exercise 1 Solution: Domain-Specific Evaluation Metrics

This solution demonstrates how to create domain-specific evaluation metrics
for a code review assistant system, including helpfulness, accuracy, and
actionability metrics, as well as a composite metric.
"""

import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import dspy


@dataclass
class MetricResult:
    """Result of a metric evaluation with detailed metadata."""

    metric_name: str
    score: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
        """Evaluate a single example and return the result."""
        pass


class CodeReviewHelpfulnessMetric(EvaluationMetric):
    """Evaluates how helpful a code review comment is."""

    def __init__(self):
        super().__init__(
            "code_review_helpfulness", "Measures helpfulness of code review comments"
        )

    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
        """Evaluate helpfulness based on constructive feedback indicators."""
        try:
            review_comment = getattr(prediction, "review_comment", str(prediction))
            expected_issues = getattr(example, "expected_issues", [])

            score = 0.0
            metadata = {}
            explanations = []

            # Check for constructive language (30% of score)
            constructive_indicators = [
                "consider",
                "suggest",
                "recommend",
                "might",
                "could",
                "perhaps",
                "alternative",
                "improvement",
                "better",
            ]
            constructive_count = sum(
                1
                for indicator in constructive_indicators
                if indicator in review_comment.lower()
            )
            constructive_score = min(0.3, constructive_count * 0.1)
            score += constructive_score
            metadata["constructive_indicators"] = constructive_count
            explanations.append(f"Constructive language: {constructive_score:.2f}")

            # Check for specific suggestions (40% of score)
            specific_patterns = [
                r"use \w+",
                r"change \w+",
                r"add \w+",
                r"remove \w+",
                r"replace \w+ with \w+",
                r"extract \w+",
                r"refactor \w+",
            ]
            specific_suggestions = sum(
                1
                for pattern in specific_patterns
                if re.search(pattern, review_comment.lower())
            )
            specific_score = min(0.4, specific_suggestions * 0.15)
            score += specific_score
            metadata["specific_suggestions"] = specific_suggestions
            explanations.append(f"Specific suggestions: {specific_score:.2f}")

            # Check for explanation of reasoning (30% of score)
            reasoning_indicators = [
                "because",
                "since",
                "as",
                "due to",
                "this will",
                "this helps",
                "this improves",
                "this reduces",
            ]
            reasoning_count = sum(
                1
                for indicator in reasoning_indicators
                if indicator in review_comment.lower()
            )
            reasoning_score = min(0.3, reasoning_count * 0.15)
            score += reasoning_score
            metadata["reasoning_indicators"] = reasoning_count
            explanations.append(f"Reasoning provided: {reasoning_score:.2f}")

            return MetricResult(
                metric_name=self.name,
                score=min(1.0, score),
                confidence=0.8,
                metadata=metadata,
                explanation="; ".join(explanations),
            )

        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                explanation=f"Error: {str(e)}",
            )


class CodeReviewAccuracyMetric(EvaluationMetric):
    """Evaluates accuracy of identified issues."""

    def __init__(self):
        super().__init__(
            "code_review_accuracy", "Measures accuracy of issue identification"
        )

    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
        """Evaluate accuracy based on correct issue identification."""
        try:
            review_comment = getattr(prediction, "review_comment", str(prediction))
            expected_issues = getattr(example, "expected_issues", [])
            code_snippet = getattr(example, "code_snippet", "")

            if not expected_issues:
                # If no expected issues, check for false positives
                issue_indicators = [
                    "bug",
                    "error",
                    "issue",
                    "problem",
                    "wrong",
                    "incorrect",
                    "fix",
                    "broken",
                    "fail",
                ]
                false_positives = sum(
                    1
                    for indicator in issue_indicators
                    if indicator in review_comment.lower()
                )
                score = (
                    1.0
                    if false_positives == 0
                    else max(0.0, 1.0 - false_positives * 0.2)
                )

                return MetricResult(
                    metric_name=self.name,
                    score=score,
                    confidence=0.7,
                    metadata={"false_positives": false_positives},
                    explanation=f"No issues expected, {false_positives} false positives detected",
                )

            # Check for correct issue identification
            identified_issues = []
            for issue in expected_issues:
                issue_keywords = issue.lower().split()
                if any(keyword in review_comment.lower() for keyword in issue_keywords):
                    identified_issues.append(issue)

            # Calculate precision and recall
            precision = len(identified_issues) / max(1, len(expected_issues))

            # Check for false positives (issues mentioned but not expected)
            all_issue_keywords = set()
            for issue in expected_issues:
                all_issue_keywords.update(issue.lower().split())

            review_words = set(review_comment.lower().split())
            potential_false_positives = len(review_words - all_issue_keywords)
            false_positive_penalty = min(0.3, potential_false_positives * 0.05)

            score = max(0.0, precision - false_positive_penalty)

            return MetricResult(
                metric_name=self.name,
                score=score,
                confidence=0.9,
                metadata={
                    "expected_issues": expected_issues,
                    "identified_issues": identified_issues,
                    "precision": precision,
                    "false_positive_penalty": false_positive_penalty,
                },
                explanation=f"Identified {len(identified_issues)}/{len(expected_issues)} issues correctly",
            )

        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                explanation=f"Error: {str(e)}",
            )


class CodeReviewActionabilityMetric(EvaluationMetric):
    """Evaluates how actionable the review comments are."""

    def __init__(self):
        super().__init__(
            "code_review_actionability", "Measures actionability of review comments"
        )

    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
        """Evaluate actionability based on clear instructions and implementability."""
        try:
            review_comment = getattr(prediction, "review_comment", str(prediction))

            score = 0.0
            metadata = {}
            explanations = []

            # Check for clear action verbs (40% of score)
            action_verbs = [
                "change",
                "modify",
                "add",
                "remove",
                "delete",
                "replace",
                "extract",
                "refactor",
                "rename",
                "move",
                "split",
                "combine",
            ]
            action_count = sum(
                1 for verb in action_verbs if verb in review_comment.lower()
            )
            action_score = min(0.4, action_count * 0.2)
            score += action_score
            metadata["action_verbs"] = action_count
            explanations.append(f"Clear actions: {action_score:.2f}")

            # Check for specific locations/references (30% of score)
            location_patterns = [
                r"line \d+",
                r"function \w+",
                r"class \w+",
                r"variable \w+",
                r"method \w+",
                r"on line",
                r"in function",
                r"this \w+",
            ]
            location_count = sum(
                1
                for pattern in location_patterns
                if re.search(pattern, review_comment.lower())
            )
            location_score = min(0.3, location_count * 0.15)
            score += location_score
            metadata["location_references"] = location_count
            explanations.append(f"Specific locations: {location_score:.2f}")

            # Check for implementation details (30% of score)
            implementation_indicators = [
                "to",
                "with",
                "using",
                "by",
                "like",
                "such as",
                "for example",
                "instead of",
                "rather than",
            ]
            implementation_count = sum(
                1
                for indicator in implementation_indicators
                if indicator in review_comment.lower()
            )
            implementation_score = min(0.3, implementation_count * 0.1)
            score += implementation_score
            metadata["implementation_details"] = implementation_count
            explanations.append(f"Implementation details: {implementation_score:.2f}")

            return MetricResult(
                metric_name=self.name,
                score=min(1.0, score),
                confidence=0.8,
                metadata=metadata,
                explanation="; ".join(explanations),
            )

        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                explanation=f"Error: {str(e)}",
            )


class CompositeCodeReviewMetric(EvaluationMetric):
    """Combines multiple code review metrics."""

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__("composite_code_review", "Combined code review evaluation")
        self.weights = weights or {
            "helpfulness": 0.4,
            "accuracy": 0.4,
            "actionability": 0.2,
        }

        # Initialize sub-metrics
        self.helpfulness_metric = CodeReviewHelpfulnessMetric()
        self.accuracy_metric = CodeReviewAccuracyMetric()
        self.actionability_metric = CodeReviewActionabilityMetric()

    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
        """Combine results from all sub-metrics with weights."""
        try:
            # Evaluate with each sub-metric
            helpfulness_result = self.helpfulness_metric.evaluate(example, prediction)
            accuracy_result = self.accuracy_metric.evaluate(example, prediction)
            actionability_result = self.actionability_metric.evaluate(
                example, prediction
            )

            # Calculate weighted score
            total_score = (
                helpfulness_result.score * self.weights["helpfulness"]
                + accuracy_result.score * self.weights["accuracy"]
                + actionability_result.score * self.weights["actionability"]
            )

            # Calculate weighted confidence
            total_confidence = (
                helpfulness_result.confidence * self.weights["helpfulness"]
                + accuracy_result.confidence * self.weights["accuracy"]
                + actionability_result.confidence * self.weights["actionability"]
            )

            # Combine metadata
            combined_metadata = {
                "weights": self.weights,
                "sub_scores": {
                    "helpfulness": helpfulness_result.score,
                    "accuracy": accuracy_result.score,
                    "actionability": actionability_result.score,
                },
                "sub_metadata": {
                    "helpfulness": helpfulness_result.metadata,
                    "accuracy": accuracy_result.metadata,
                    "actionability": actionability_result.metadata,
                },
            }

            # Combine explanations
            combined_explanation = (
                f"Composite: {total_score:.3f} "
                + f"(H:{helpfulness_result.score:.2f}, "
                + f"A:{accuracy_result.score:.2f}, "
                + f"Act:{actionability_result.score:.2f})"
            )

            return MetricResult(
                metric_name=self.name,
                score=total_score,
                confidence=total_confidence,
                metadata=combined_metadata,
                explanation=combined_explanation,
            )

        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                explanation=f"Error: {str(e)}",
            )


def test_code_review_metrics():
    """Test the code review metrics with sample data."""
    print("Testing Code Review Evaluation Metrics")
    print("=" * 50)

    # Create sample code review examples
    examples = [
        dspy.Example(
            code_snippet="def calculate_total(items):\n    return sum(items)",
            expected_issues=["missing type hints", "no input validation"],
            review_comment="Consider adding type hints for better code clarity and add input validation to handle edge cases",
        ),
        dspy.Example(
            code_snippet="x = [1, 2, 3]\nfor i in range(len(x)):\n    print(x[i])",
            expected_issues=["non-pythonic iteration"],
            review_comment="This code works but you should use 'for item in x:' instead of range(len(x)) for more pythonic iteration",
        ),
        dspy.Example(
            code_snippet="def process_data(data):\n    # TODO: implement\n    pass",
            expected_issues=[],
            review_comment="This function looks good as a placeholder. Remember to implement the actual logic when ready.",
        ),
    ]

    # Create predictions (simulating system outputs)
    predictions = [
        dspy.Prediction(
            review_comment="Consider adding type hints for better code clarity and add input validation to handle edge cases"
        ),
        dspy.Prediction(
            review_comment="This code works but you should use 'for item in x:' instead of range(len(x)) for more pythonic iteration"
        ),
        dspy.Prediction(
            review_comment="This function looks good as a placeholder. Remember to implement the actual logic when ready."
        ),
    ]

    # Test individual metrics
    metrics = [
        CodeReviewHelpfulnessMetric(),
        CodeReviewAccuracyMetric(),
        CodeReviewActionabilityMetric(),
    ]

    print("Individual Metric Results:")
    print("-" * 30)

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        print(f"\nExample {i+1}:")
        print(f"Code: {example.code_snippet.replace(chr(10), ' ')}")
        print(f"Review: {prediction.review_comment}")

        for metric in metrics:
            result = metric.evaluate(example, prediction)
            print(f"  {metric.name}: {result.score:.3f} - {result.explanation}")

    # Test composite metric
    print("\n" + "=" * 50)
    print("Composite Metric Results:")
    print("-" * 30)

    composite_metric = CompositeCodeReviewMetric()

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        result = composite_metric.evaluate(example, prediction)
        print(f"\nExample {i+1}: {result.score:.3f}")
        print(f"  {result.explanation}")
        print(f"  Sub-scores: {result.metadata['sub_scores']}")

    # Calculate overall statistics
    composite_scores = [
        composite_metric.evaluate(example, prediction).score
        for example, prediction in zip(examples, predictions)
    ]

    print(f"\nOverall Statistics:")
    print(f"  Mean Score: {statistics.mean(composite_scores):.3f}")
    print(f"  Score Range: {min(composite_scores):.3f} - {max(composite_scores):.3f}")
    print(f"  Standard Deviation: {statistics.stdev(composite_scores):.3f}")


if __name__ == "__main__":
    test_code_review_metrics()
