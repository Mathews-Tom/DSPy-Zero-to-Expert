"""
Data Quality Metrics System for DSPy Examples

This module provides comprehensive data quality assessment tools for DSPy datasets,
including completeness, consistency, bias detection, and quality scoring mechanisms.

Key Features:
- Comprehensive quality metrics calculation
- Bias detection and analysis
- Data completeness and consistency checks
- Quality scoring and recommendations
- Automated quality reporting
"""

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from dspy import Example

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a dataset"""

    # Basic metrics
    total_examples: int = 0
    valid_examples: int = 0

    # Completeness metrics
    completeness_score: float = 0.0
    field_completeness: dict[str, float] = field(default_factory=dict)
    missing_value_rate: float = 0.0

    # Consistency metrics
    consistency_score: float = 0.0
    type_consistency: dict[str, float] = field(default_factory=dict)
    format_consistency: dict[str, float] = field(default_factory=dict)

    # Uniqueness metrics
    uniqueness_score: float = 0.0
    duplicate_rate: float = 0.0
    field_diversity: dict[str, float] = field(default_factory=dict)

    # Text quality metrics
    text_quality_score: float = 0.0
    avg_text_length: dict[str, float] = field(default_factory=dict)
    readability_scores: dict[str, float] = field(default_factory=dict)

    # Bias metrics
    bias_score: float = 0.0
    length_bias: dict[str, float] = field(default_factory=dict)
    vocabulary_bias: dict[str, float] = field(default_factory=dict)

    # Overall quality
    overall_quality_score: float = 0.0
    quality_grade: str = "Unknown"

    # Issues and recommendations
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class DataQualityAnalyzer:
    """
    Comprehensive data quality analyzer for DSPy Examples

    Provides detailed analysis of data quality across multiple dimensions
    including completeness, consistency, uniqueness, and bias detection.
    """

    def __init__(self):
        """Initialize the data quality analyzer"""
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "fair": 0.6,
            "poor": 0.4,
        }

    def analyze_completeness(
        self, examples: list[Example], required_fields: list[str]
    ) -> dict[str, Any]:
        """
        Analyze data completeness

        Args:
            examples: list of DSPy Examples
            required_fields: list of required field names

        Returns:
            Dictionary with completeness metrics
        """
        if not examples:
            return {
                "completeness_score": 0.0,
                "field_completeness": {},
                "missing_rate": 1.0,
            }

        field_completeness = {}
        total_missing = 0
        total_possible = len(examples) * len(required_fields)

        for field in required_fields:
            complete_count = 0
            for example in examples:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None and str(value).strip():
                        complete_count += 1
                    else:
                        total_missing += 1
                else:
                    total_missing += 1

            field_completeness[field] = (
                complete_count / len(examples) if examples else 0.0
            )

        overall_completeness = (
            sum(field_completeness.values()) / len(required_fields)
            if required_fields
            else 0.0
        )
        missing_rate = total_missing / total_possible if total_possible > 0 else 0.0

        return {
            "completeness_score": overall_completeness,
            "field_completeness": field_completeness,
            "missing_rate": missing_rate,
        }

    def analyze_consistency(
        self, examples: list[Example], text_fields: list[str]
    ) -> dict[str, Any]:
        """
        Analyze data consistency

        Args:
            examples: list of DSPy Examples
            text_fields: list of text field names

        Returns:
            Dictionary with consistency metrics
        """
        if not examples:
            return {
                "consistency_score": 0.0,
                "type_consistency": {},
                "format_consistency": {},
            }

        type_consistency = {}
        format_consistency = {}

        for field in text_fields:
            field_values = []
            field_types = []

            for example in examples:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None:
                        field_values.append(str(value))
                        field_types.append(type(value).__name__)

            if field_values:
                # Type consistency
                type_counter = Counter(field_types)
                most_common_type = type_counter.most_common(1)[0][1]
                type_consistency[field] = most_common_type / len(field_values)

                # Format consistency (for text fields)
                format_score = self._analyze_text_format_consistency(field_values)
                format_consistency[field] = format_score
            else:
                type_consistency[field] = 0.0
                format_consistency[field] = 0.0

        overall_consistency = (
            (sum(type_consistency.values()) + sum(format_consistency.values()))
            / (2 * len(text_fields))
            if text_fields
            else 0.0
        )

        return {
            "consistency_score": overall_consistency,
            "type_consistency": type_consistency,
            "format_consistency": format_consistency,
        }

    def _analyze_text_format_consistency(self, texts: list[str]) -> float:
        """Analyze format consistency in text fields"""
        if not texts:
            return 0.0

        # Check various format patterns
        patterns = {
            "has_punctuation": lambda t: bool(re.search(r"[.!?]", t)),
            "starts_uppercase": lambda t: t[0].isupper() if t else False,
            "has_numbers": lambda t: bool(re.search(r"\d", t)),
            "has_special_chars": lambda t: bool(re.search(r"[^a-zA-Z0-9\s]", t)),
        }

        consistency_scores = []
        for pattern_name, pattern_func in patterns.items():
            matches = sum(1 for text in texts if pattern_func(text))
            # High consistency if most texts follow the same pattern (either all have it or all don't)
            consistency = max(matches, len(texts) - matches) / len(texts)
            consistency_scores.append(consistency)

        return sum(consistency_scores) / len(consistency_scores)

    def analyze_uniqueness(
        self, examples: list[Example], text_fields: list[str]
    ) -> dict[str, Any]:
        """
        Analyze data uniqueness and diversity

        Args:
            examples: list of DSPy Examples
            text_fields: list of text field names

        Returns:
            Dictionary with uniqueness metrics
        """
        if not examples:
            return {
                "uniqueness_score": 0.0,
                "duplicate_rate": 0.0,
                "field_diversity": {},
            }

        field_diversity = {}
        duplicate_signatures = set()
        total_examples = len(examples)

        # Check for exact duplicates
        example_signatures = []
        for example in examples:
            signature_parts = []
            for field in text_fields:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None:
                        signature_parts.append(str(value).strip().lower())
            signature = "|".join(signature_parts)
            example_signatures.append(signature)

        unique_signatures = set(example_signatures)
        duplicate_rate = 1 - (len(unique_signatures) / total_examples)

        # Analyze field diversity
        for field in text_fields:
            field_values = []
            for example in examples:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None:
                        field_values.append(str(value).strip().lower())

            if field_values:
                unique_values = set(field_values)
                diversity = len(unique_values) / len(field_values)
                field_diversity[field] = diversity
            else:
                field_diversity[field] = 0.0

        overall_uniqueness = (
            sum(field_diversity.values()) / len(text_fields) if text_fields else 0.0
        )

        return {
            "uniqueness_score": overall_uniqueness,
            "duplicate_rate": duplicate_rate,
            "field_diversity": field_diversity,
        }

    def analyze_text_quality(
        self, examples: list[Example], text_fields: list[str]
    ) -> dict[str, Any]:
        """
        Analyze text quality metrics

        Args:
            examples: list of DSPy Examples
            text_fields: list of text field names

        Returns:
            Dictionary with text quality metrics
        """
        if not examples:
            return {
                "text_quality_score": 0.0,
                "avg_text_length": {},
                "readability_scores": {},
            }

        avg_text_length = {}
        readability_scores = {}

        for field in text_fields:
            field_texts = []
            for example in examples:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None and isinstance(value, str):
                        field_texts.append(value)

            if field_texts:
                # Average text length
                lengths = [len(text) for text in field_texts]
                avg_text_length[field] = sum(lengths) / len(lengths)

                # Simple readability score based on sentence structure
                readability_scores[field] = self._calculate_readability_score(
                    field_texts
                )
            else:
                avg_text_length[field] = 0.0
                readability_scores[field] = 0.0

        # Overall text quality score
        length_scores = []
        for field, avg_len in avg_text_length.items():
            # Optimal length range: 50-500 characters
            if 50 <= avg_len <= 500:
                length_scores.append(1.0)
            elif avg_len < 50:
                length_scores.append(avg_len / 50)
            else:
                length_scores.append(500 / avg_len)

        length_quality = (
            sum(length_scores) / len(length_scores) if length_scores else 0.0
        )
        readability_quality = (
            sum(readability_scores.values()) / len(readability_scores)
            if readability_scores
            else 0.0
        )

        overall_text_quality = (length_quality + readability_quality) / 2

        return {
            "text_quality_score": overall_text_quality,
            "avg_text_length": avg_text_length,
            "readability_scores": readability_scores,
        }

    def _calculate_readability_score(self, texts: list[str]) -> float:
        """Calculate a simple readability score for texts"""
        if not texts:
            return 0.0

        scores = []
        for text in texts:
            # Simple metrics for readability
            sentences = len(re.split(r"[.!?]+", text))
            words = len(text.split())

            if sentences == 0 or words == 0:
                scores.append(0.0)
                continue

            # Average words per sentence (optimal: 15-20)
            avg_words_per_sentence = words / sentences
            sentence_score = 1.0 if 10 <= avg_words_per_sentence <= 25 else 0.5

            # Punctuation presence
            has_punctuation = bool(re.search(r"[.!?]", text))
            punctuation_score = 1.0 if has_punctuation else 0.5

            # Capitalization
            has_proper_caps = text[0].isupper() if text else False
            caps_score = 1.0 if has_proper_caps else 0.7

            text_score = (sentence_score + punctuation_score + caps_score) / 3
            scores.append(text_score)

        return sum(scores) / len(scores)

    def analyze_bias(
        self, examples: list[Example], text_fields: list[str]
    ) -> dict[str, Any]:
        """
        Analyze potential bias in the dataset

        Args:
            examples: list of DSPy Examples
            text_fields: list of text field names

        Returns:
            Dictionary with bias metrics
        """
        if not examples:
            return {"bias_score": 0.0, "length_bias": {}, "vocabulary_bias": {}}

        length_bias = {}
        vocabulary_bias = {}

        for field in text_fields:
            field_texts = []
            for example in examples:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None and isinstance(value, str):
                        field_texts.append(value)

            if field_texts:
                # Length bias analysis
                lengths = [len(text) for text in field_texts]
                length_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
                length_mean = statistics.mean(lengths)
                length_cv = length_std / length_mean if length_mean > 0 else 0
                # Lower coefficient of variation indicates less length bias
                length_bias[field] = max(0, 1 - length_cv)

                # Vocabulary bias analysis
                all_words = []
                for text in field_texts:
                    words = re.findall(r"\b\w+\b", text.lower())
                    all_words.extend(words)

                if all_words:
                    word_freq = Counter(all_words)
                    total_words = len(all_words)
                    unique_words = len(word_freq)

                    # Calculate vocabulary diversity (higher is better)
                    vocab_diversity = (
                        unique_words / total_words if total_words > 0 else 0
                    )
                    vocabulary_bias[field] = vocab_diversity
                else:
                    vocabulary_bias[field] = 0.0
            else:
                length_bias[field] = 0.0
                vocabulary_bias[field] = 0.0

        # Overall bias score (higher is better - less biased)
        length_bias_avg = (
            sum(length_bias.values()) / len(length_bias) if length_bias else 0.0
        )
        vocab_bias_avg = (
            sum(vocabulary_bias.values()) / len(vocabulary_bias)
            if vocabulary_bias
            else 0.0
        )
        overall_bias_score = (length_bias_avg + vocab_bias_avg) / 2

        return {
            "bias_score": overall_bias_score,
            "length_bias": length_bias,
            "vocabulary_bias": vocabulary_bias,
        }

    def generate_quality_report(
        self,
        examples: list[Example],
        required_fields: list[str],
        text_fields: list[str],
    ) -> QualityMetrics:
        """
        Generate comprehensive quality report

        Args:
            examples: list of DSPy Examples
            required_fields: list of required field names
            text_fields: list of text field names

        Returns:
            QualityMetrics object with comprehensive analysis
        """
        if not examples:
            return QualityMetrics()

        # Analyze all quality dimensions
        completeness = self.analyze_completeness(examples, required_fields)
        consistency = self.analyze_consistency(examples, text_fields)
        uniqueness = self.analyze_uniqueness(examples, text_fields)
        text_quality = self.analyze_text_quality(examples, text_fields)
        bias = self.analyze_bias(examples, text_fields)

        # Calculate overall quality score
        quality_components = [
            completeness["completeness_score"],
            consistency["consistency_score"],
            uniqueness["uniqueness_score"],
            text_quality["text_quality_score"],
            bias["bias_score"],
        ]

        overall_score = sum(quality_components) / len(quality_components)

        # Determine quality grade
        quality_grade = self._get_quality_grade(overall_score)

        # Generate issues and recommendations
        issues, recommendations = self._generate_issues_and_recommendations(
            completeness, consistency, uniqueness, text_quality, bias
        )

        return QualityMetrics(
            total_examples=len(examples),
            valid_examples=len(
                [ex for ex in examples if self._is_valid_example(ex, required_fields)]
            ),
            completeness_score=completeness["completeness_score"],
            field_completeness=completeness["field_completeness"],
            missing_value_rate=completeness["missing_rate"],
            consistency_score=consistency["consistency_score"],
            type_consistency=consistency["type_consistency"],
            format_consistency=consistency["format_consistency"],
            uniqueness_score=uniqueness["uniqueness_score"],
            duplicate_rate=uniqueness["duplicate_rate"],
            field_diversity=uniqueness["field_diversity"],
            text_quality_score=text_quality["text_quality_score"],
            avg_text_length=text_quality["avg_text_length"],
            readability_scores=text_quality["readability_scores"],
            bias_score=bias["bias_score"],
            length_bias=bias["length_bias"],
            vocabulary_bias=bias["vocabulary_bias"],
            overall_quality_score=overall_score,
            quality_grade=quality_grade,
            issues=issues,
            recommendations=recommendations,
        )

    def _is_valid_example(self, example: Example, required_fields: list[str]) -> bool:
        """Check if an example is valid based on required fields"""
        for field in required_fields:
            if not hasattr(example, field):
                return False
            value = getattr(example, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= self.quality_thresholds["excellent"]:
            return "A"
        elif score >= self.quality_thresholds["good"]:
            return "B"
        elif score >= self.quality_thresholds["fair"]:
            return "C"
        elif score >= self.quality_thresholds["poor"]:
            return "D"
        else:
            return "F"

    def _generate_issues_and_recommendations(
        self, completeness, consistency, uniqueness, text_quality, bias
    ) -> tuple[list[str], list[str]]:
        """Generate issues and recommendations based on quality analysis"""
        issues = []
        recommendations = []

        # Completeness issues
        if completeness["completeness_score"] < 0.8:
            issues.append(
                f"Low data completeness: {completeness['completeness_score']:.1%}"
            )
            recommendations.append(
                "Review data collection process to reduce missing values"
            )

        if completeness["missing_rate"] > 0.2:
            issues.append(
                f"High missing value rate: {completeness['missing_rate']:.1%}"
            )
            recommendations.append("Implement data validation at collection time")

        # Consistency issues
        if consistency["consistency_score"] < 0.7:
            issues.append(
                f"Low data consistency: {consistency['consistency_score']:.1%}"
            )
            recommendations.append("Standardize data formats and validation rules")

        # Uniqueness issues
        if uniqueness["duplicate_rate"] > 0.1:
            issues.append(f"High duplicate rate: {uniqueness['duplicate_rate']:.1%}")
            recommendations.append("Implement deduplication process")

        if uniqueness["uniqueness_score"] < 0.5:
            issues.append(f"Low data diversity: {uniqueness['uniqueness_score']:.1%}")
            recommendations.append("Expand data sources to increase variety")

        # Text quality issues
        if text_quality["text_quality_score"] < 0.6:
            issues.append(
                f"Poor text quality: {text_quality['text_quality_score']:.1%}"
            )
            recommendations.append("Implement text preprocessing and cleaning")

        # Bias issues
        if bias["bias_score"] < 0.5:
            issues.append(f"Potential bias detected: {bias['bias_score']:.1%}")
            recommendations.append("Review data sources for bias and balance dataset")

        return issues, recommendations


class QualityReporter:
    """Generate formatted quality reports"""

    def __init__(self):
        """Initialize the quality reporter"""
        pass

    def generate_html_report(
        self, metrics: QualityMetrics, dataset_name: str = "Dataset"
    ) -> str:
        """
        Generate HTML quality report

        Args:
            metrics: QualityMetrics object
            dataset_name: Name of the dataset

        Returns:
            HTML report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .grade-A {{ color: #28a745; }}
                .grade-B {{ color: #17a2b8; }}
                .grade-C {{ color: #ffc107; }}
                .grade-D {{ color: #fd7e14; }}
                .grade-F {{ color: #dc3545; }}
                .issues {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 20px 0; }}
                .recommendations {{ background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 15px; margin: 20px 0; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {dataset_name}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Total Examples:</strong> {metrics.total_examples:,}</p>
                <p><strong>Valid Examples:</strong> {metrics.valid_examples:,}</p>
            </div>
            
            <h2>Overall Quality Score</h2>
            <div class="metric-card">
                <div class="score grade-{metrics.quality_grade}">
                    {metrics.overall_quality_score:.1%} (Grade {metrics.quality_grade})
                </div>
            </div>
            
            <h2>Quality Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Completeness</h3>
                    <div class="score">{metrics.completeness_score:.1%}</div>
                    <p>Missing Rate: {metrics.missing_value_rate:.1%}</p>
                </div>
                <div class="metric-card">
                    <h3>Consistency</h3>
                    <div class="score">{metrics.consistency_score:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Uniqueness</h3>
                    <div class="score">{metrics.uniqueness_score:.1%}</div>
                    <p>Duplicate Rate: {metrics.duplicate_rate:.1%}</p>
                </div>
                <div class="metric-card">
                    <h3>Text Quality</h3>
                    <div class="score">{metrics.text_quality_score:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Bias Score</h3>
                    <div class="score">{metrics.bias_score:.1%}</div>
                </div>
            </div>
        """

        # Add issues section
        if metrics.issues:
            html_report += """
            <div class="issues">
                <h3>⚠️ Issues Identified</h3>
                <ul>
            """
            for issue in metrics.issues:
                html_report += f"<li>{issue}</li>"
            html_report += "</ul></div>"

        # Add recommendations section
        if metrics.recommendations:
            html_report += """
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <ul>
            """
            for rec in metrics.recommendations:
                html_report += f"<li>{rec}</li>"
            html_report += "</ul></div>"

        html_report += """
        </body>
        </html>
        """

        return html_report

    def generate_text_report(
        self, metrics: QualityMetrics, dataset_name: str = "Dataset"
    ) -> str:
        """
        Generate text quality report

        Args:
            metrics: QualityMetrics object
            dataset_name: Name of the dataset

        Returns:
            Text report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
DATA QUALITY REPORT
==================

Dataset: {dataset_name}
Generated: {timestamp}
Total Examples: {metrics.total_examples:,}
Valid Examples: {metrics.valid_examples:,}

OVERALL QUALITY SCORE
====================
Score: {metrics.overall_quality_score:.1%} (Grade {metrics.quality_grade})

DETAILED METRICS
===============
Completeness: {metrics.completeness_score:.1%}
  - Missing Rate: {metrics.missing_value_rate:.1%}

Consistency: {metrics.consistency_score:.1%}

Uniqueness: {metrics.uniqueness_score:.1%}
  - Duplicate Rate: {metrics.duplicate_rate:.1%}

Text Quality: {metrics.text_quality_score:.1%}

Bias Score: {metrics.bias_score:.1%}
"""

        # Add issues
        if metrics.issues:
            report += "\nISSUES IDENTIFIED\n================\n"
            for i, issue in enumerate(metrics.issues, 1):
                report += f"{i}. {issue}\n"

        # Add recommendations
        if metrics.recommendations:
            report += "\nRECOMMENDATIONS\n==============\n"
            for i, rec in enumerate(metrics.recommendations, 1):
                report += f"{i}. {rec}\n"

        return report


# Utility functions


def assess_dataset_quality(
    examples: list[Example],
    required_fields: list[str],
    text_fields: Optional[list[str]] = None,
) -> QualityMetrics:
    """
    Assess the quality of a DSPy dataset

    Args:
        examples: list of DSPy Examples
        required_fields: list of required field names
        text_fields: list of text field names (defaults to required_fields)

    Returns:
        QualityMetrics object
    """
    if text_fields is None:
        text_fields = required_fields

    analyzer = DataQualityAnalyzer()
    return analyzer.generate_quality_report(examples, required_fields, text_fields)


def generate_quality_report(
    metrics: QualityMetrics, dataset_name: str = "Dataset", format: str = "html"
) -> str:
    """
    Generate a quality report in the specified format

    Args:
        metrics: QualityMetrics object
        dataset_name: Name of the dataset
        format: Report format ('html' or 'text')

    Returns:
        Formatted report string
    """
    reporter = QualityReporter()

    if format.lower() == "html":
        return reporter.generate_html_report(metrics, dataset_name)
    else:
        return reporter.generate_text_report(metrics, dataset_name)


if __name__ == "__main__":
    # Example usage
    from dataset_management import DatasetManager

    # Load sample data
    manager = DatasetManager()
    examples = manager.load_from_json("data/sample_qa.json")

    # Assess quality
    quality_metrics = assess_dataset_quality(
        examples,
        required_fields=["question", "answer"],
        text_fields=["question", "answer"],
    )

    # Generate reports
    html_report = generate_quality_report(quality_metrics, "Sample QA Dataset", "html")
    text_report = generate_quality_report(quality_metrics, "Sample QA Dataset", "text")

    # Save reports
    with open("quality_report.html", "w") as f:
        f.write(html_report)

    with open("quality_report.txt", "w") as f:
        f.write(text_report)

    print(
        f"Quality Score: {quality_metrics.overall_quality_score:.1%} (Grade {quality_metrics.quality_grade})"
    )
    print(f"Issues: {len(quality_metrics.issues)}")
    print(f"Recommendations: {len(quality_metrics.recommendations)}")
