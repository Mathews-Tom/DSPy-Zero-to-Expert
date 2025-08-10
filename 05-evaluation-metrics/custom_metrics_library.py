# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    import re
    import statistics
    import sys
    from abc import ABC, abstractmethod
    from collections import Counter, defaultdict
    from collections.abc import Callable
    from dataclasses import dataclass, field
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Optional, Union

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        ABC,
        Any,
        Callable,
        Optional,
        abstractmethod,
        cleandoc,
        dataclass,
        dspy,
        field,
        get_config,
        mo,
        output,
        re,
        setup_dspy_environment,
        statistics,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📚 Custom Metrics Library

            **Duration:** 2-3 hours  
            **Prerequisites:** Completed evaluation framework module  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will master:  
            - **Domain-Specific Metrics** - Create metrics tailored to specific domains  
            - **Composite Metrics** - Combine multiple evaluation criteria intelligently  
            - **Metric Templates** - Build reusable metric patterns and templates  
            - **Advanced Evaluation** - Implement sophisticated evaluation strategies  
            - **Metric Validation** - Test and validate custom metrics thoroughly  

            ## 📊 Custom Metrics Overview

            **Why Custom Metrics Matter:**  
            - **Domain Alignment** - Standard metrics don't capture domain-specific nuances  
            - **Task Specificity** - Different tasks require different success criteria  
            - **Business Value** - Measure what actually matters for your application  
            - **Quality Dimensions** - Capture multiple aspects of response quality  

            **Metric Categories:**  
            - **Accuracy Metrics** - Various forms of correctness measurement  
            - **Quality Metrics** - Fluency, coherence, and style assessment  
            - **Task-Specific Metrics** - Domain knowledge and specialized criteria  
            - **Composite Metrics** - Multi-dimensional evaluation combining multiple aspects  

            Let's build a comprehensive custom metrics library!
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(cleandoc, get_config, mo, output, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        # Use the first available provider
        provider = available_providers[0]
        setup_dspy_environment(provider=provider)
        cell2_out = mo.md(
            cleandoc(
                f"""
                ## ✅ Custom Metrics Environment Ready

                **Configuration:**  
                - Provider: **{provider}**  
                - Model: **{config.default_model}**  
                - Available Providers: **{', '.join(available_providers)}**

                Ready to build advanced custom metrics!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first to configure your API keys.  

                **Required:** At least one of the following API keys:  
                - `OPENAI_API_KEY` for OpenAI models  
                - `ANTHROPIC_API_KEY` for Anthropic models  
                - `COHERE_API_KEY` for Cohere models  

                Add your API key to the `.env` file in the project root.
                """
            )
        )

    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(
    ABC,
    Any,
    Callable,
    Optional,
    abstractmethod,
    available_providers,
    cleandoc,
    dataclass,
    dspy,
    field,
    mo,
    output,
    re,
    statistics,
):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Advanced Metric Base Classes

                **Building sophisticated base classes** for custom metric development:
                """
            )
        )

        @dataclass
        class MetricResult:
            """Enhanced result class for custom metrics."""

            metric_name: str
            score: float
            confidence: float = 1.0
            metadata: dict[str, Any] = field(default_factory=dict)
            sub_scores: dict[str, float] = field(default_factory=dict)
            explanation: str = ""
            timestamp: float = field(default_factory=lambda: __import__("time").time())

        class AdvancedMetric(ABC):
            """Advanced base class for sophisticated metrics."""

            def __init__(self, name: str, description: str = "", weight: float = 1.0):
                self.name = name
                self.description = description
                self.weight = weight
                self.evaluation_count = 0
                self.score_history = []

            @abstractmethod
            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> MetricResult:
                """Evaluate a single example and return detailed result."""
                pass

            def batch_evaluate(
                self,
                examples: list[dspy.Example],
                predictions: list[Any],
                traces: Optional[list[Any]] = None,
            ) -> list[MetricResult]:
                """Evaluate multiple examples with progress tracking."""
                if traces is None:
                    traces = [None] * len(examples)

                results = []
                for i, (example, prediction, trace) in enumerate(
                    zip(examples, predictions, traces, strict=False)
                ):
                    try:
                        result = self.evaluate(example, prediction, trace)
                        result.metadata["example_index"] = i
                        results.append(result)
                        self.score_history.append(result.score)
                        self.evaluation_count += 1
                    except Exception as e:
                        error_result = MetricResult(
                            metric_name=self.name,
                            score=0.0,
                            confidence=0.0,
                            metadata={
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "example_index": i,
                            },
                            explanation=f"Evaluation failed: {str(e)}",
                        )
                        results.append(error_result)

                return results

            def get_statistics(self) -> dict[str, float]:
                """Get performance statistics for this metric."""
                if not self.score_history:
                    return {}

                return {
                    "mean": statistics.mean(self.score_history),
                    "median": statistics.median(self.score_history),
                    "std_dev": (
                        statistics.stdev(self.score_history)
                        if len(self.score_history) > 1
                        else 0.0
                    ),
                    "min": min(self.score_history),
                    "max": max(self.score_history),
                    "count": len(self.score_history),
                }

        class CompositeMetric(AdvancedMetric):
            """Combines multiple metrics with configurable weights."""

            def __init__(
                self,
                name: str,
                metrics: dict[str, AdvancedMetric],
                weights: Optional[dict[str, float]] = None,
                aggregation_method: str = "weighted_average",
            ):
                super().__init__(
                    name, f"Composite metric combining {len(metrics)} sub-metrics"
                )
                self.metrics = metrics
                self.weights = weights or dict.fromkeys(metrics.keys(), 1.0)
                self.aggregation_method = aggregation_method

                # Normalize weights
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    self.weights = {
                        k: v / total_weight for k, v in self.weights.items()
                    }

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> MetricResult:
                """Evaluate using all sub-metrics and aggregate results."""
                sub_results = {}
                sub_scores = {}
                total_score = 0.0
                total_confidence = 0.0
                explanations = []

                for metric_name, metric in self.metrics.items():
                    try:
                        result = metric.evaluate(example, prediction, trace)
                        sub_results[metric_name] = result
                        sub_scores[metric_name] = result.score

                        weight = self.weights.get(metric_name, 0.0)
                        total_score += result.score * weight
                        total_confidence += result.confidence * weight

                        if result.explanation:
                            explanations.append(f"{metric_name}: {result.explanation}")

                    except Exception as e:
                        sub_scores[metric_name] = 0.0
                        explanations.append(f"{metric_name}: Error - {str(e)}")

                return MetricResult(
                    metric_name=self.name,
                    score=total_score,
                    confidence=total_confidence,
                    sub_scores=sub_scores,
                    metadata={
                        "aggregation_method": self.aggregation_method,
                        "weights": self.weights,
                        "sub_results": {k: v.metadata for k, v in sub_results.items()},
                    },
                    explanation="; ".join(explanations),
                )

        class MetricTemplate:
            """Template system for creating standardized metrics."""

            @staticmethod
            def create_similarity_metric(
                name: str,
                similarity_function: Callable[[str, str], float],
                threshold: float = 0.8,
                field_name: str = "answer",
            ) -> AdvancedMetric:
                """Create a similarity-based metric from a similarity function."""

                class SimilarityMetric(AdvancedMetric):
                    def evaluate(
                        self,
                        example: dspy.Example,
                        prediction: Any,
                        trace: Optional[Any] = None,
                    ) -> MetricResult:
                        try:
                            expected = str(getattr(example, field_name, ""))
                            predicted = str(
                                getattr(prediction, field_name, str(prediction))
                            )

                            similarity = similarity_function(expected, predicted)
                            score = 1.0 if similarity >= threshold else similarity

                            return MetricResult(
                                metric_name=self.name,
                                score=score,
                                confidence=1.0,
                                metadata={
                                    "similarity": similarity,
                                    "threshold": threshold,
                                    "passed_threshold": similarity >= threshold,
                                },
                                explanation=f"Similarity: {similarity:.3f} (threshold: {threshold})",
                            )

                        except Exception as e:
                            return MetricResult(
                                metric_name=self.name,
                                score=0.0,
                                confidence=0.0,
                                metadata={"error": str(e)},
                                explanation=f"Error: {str(e)}",
                            )

                return SimilarityMetric(
                    name, f"Similarity metric using {similarity_function.__name__}"
                )

            @staticmethod
            def create_pattern_metric(
                name: str,
                patterns: list[str],
                require_all: bool = False,
                field_name: str = "answer",
            ) -> AdvancedMetric:
                """Create a pattern-matching metric."""

                class PatternMetric(AdvancedMetric):
                    def evaluate(
                        self,
                        example: dspy.Example,
                        prediction: Any,
                        trace: Optional[Any] = None,
                    ) -> MetricResult:
                        try:
                            predicted = str(
                                getattr(prediction, field_name, str(prediction))
                            )

                            matches = []
                            for pattern in patterns:
                                if re.search(pattern, predicted, re.IGNORECASE):
                                    matches.append(pattern)

                            if require_all:
                                score = 1.0 if len(matches) == len(patterns) else 0.0
                            else:
                                score = (
                                    len(matches) / len(patterns) if patterns else 1.0
                                )

                            return MetricResult(
                                metric_name=self.name,
                                score=score,
                                confidence=1.0,
                                metadata={
                                    "patterns": patterns,
                                    "matches": matches,
                                    "require_all": require_all,
                                },
                                explanation=f"Matched {len(matches)}/{len(patterns)} patterns",
                            )

                        except Exception as e:
                            return MetricResult(
                                metric_name=self.name,
                                score=0.0,
                                confidence=0.0,
                                metadata={"error": str(e)},
                                explanation=f"Error: {str(e)}",
                            )

                return PatternMetric(
                    name, f"Pattern matching metric with {len(patterns)} patterns"
                )

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Advanced Metric Base Classes Created

                **Core Components:**  
                - **MetricResult** - Enhanced result class with confidence and sub-scores  
                - **AdvancedMetric** - Sophisticated base class with statistics tracking  
                - **CompositeMetric** - Combines multiple metrics with configurable weights  
                - **MetricTemplate** - Factory methods for creating standardized metrics  

                **Key Features:**  
                - **Confidence Scoring** - Metrics can express confidence in their evaluations  
                - **Sub-Score Tracking** - Composite metrics track individual component scores  
                - **Performance Statistics** - Automatic tracking of metric performance over time  
                - **Error Handling** - Robust error handling with detailed error information  

                Ready to build domain-specific metrics!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        MetricResult = None
        AdvancedMetric = None
        CompositeMetric = None
        MetricTemplate = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return AdvancedMetric, CompositeMetric, MetricResult


@app.cell
def _(
    AdvancedMetric,
    Any,
    MetricResult,
    Optional,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
    re,
):
    if available_providers and AdvancedMetric:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 2: Domain-Specific Metrics Collection

                **Comprehensive collection** of metrics for different domains and tasks:
                """
            )
        )

        class SemanticSimilarityMetric(AdvancedMetric):
            """Evaluates semantic similarity between expected and predicted text."""

            def __init__(self, threshold: float = 0.8, field_name: str = "answer"):
                super().__init__(
                    name=f"semantic_similarity_{field_name}",
                    description="Semantic similarity using word overlap and context",
                )
                self.threshold = threshold
                self.field_name = field_name

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> MetricResult:
                try:
                    expected = (
                        str(getattr(example, self.field_name, "")).lower().strip()
                    )
                    predicted = (
                        str(getattr(prediction, self.field_name, str(prediction)))
                        .lower()
                        .strip()
                    )

                    if not expected or not predicted:
                        return MetricResult(
                            metric_name=self.name,
                            score=0.0,
                            confidence=1.0,
                            explanation="Empty expected or predicted text",
                        )

                    # Word-level similarity
                    expected_words = set(expected.split())
                    predicted_words = set(predicted.split())

                    intersection = expected_words.intersection(predicted_words)
                    union = expected_words.union(predicted_words)

                    jaccard_similarity = (
                        len(intersection) / len(union) if union else 0.0
                    )

                    # Length similarity
                    len_similarity = 1.0 - abs(
                        len(expected.split()) - len(predicted.split())
                    ) / max(len(expected.split()), len(predicted.split()), 1)

                    # Combined similarity
                    semantic_score = (jaccard_similarity * 0.7) + (len_similarity * 0.3)

                    final_score = (
                        1.0 if semantic_score >= self.threshold else semantic_score
                    )

                    return MetricResult(
                        metric_name=self.name,
                        score=final_score,
                        confidence=0.9,  # High confidence for word-based similarity
                        sub_scores={
                            "jaccard_similarity": jaccard_similarity,
                            "length_similarity": len_similarity,
                            "semantic_score": semantic_score,
                        },
                        metadata={
                            "expected_words": list(expected_words),
                            "predicted_words": list(predicted_words),
                            "intersection": list(intersection),
                            "threshold": self.threshold,
                        },
                        explanation=f"Semantic similarity: {semantic_score:.3f} (Jaccard: {jaccard_similarity:.3f}, Length: {len_similarity:.3f})",
                    )

                except Exception as e:
                    return MetricResult(
                        metric_name=self.name,
                        score=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                        explanation=f"Error: {str(e)}",
                    )

        class FactualAccuracyMetric(AdvancedMetric):
            """Evaluates factual accuracy by checking key facts and entities."""

            def __init__(self, field_name: str = "answer"):
                super().__init__(
                    name=f"factual_accuracy_{field_name}",
                    description="Factual accuracy based on entity and fact extraction",
                )
                self.field_name = field_name

            def _extract_entities(self, text: str) -> set:
                """Simple entity extraction using patterns."""
                # Numbers
                numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))

                # Capitalized words (potential proper nouns)
                proper_nouns = set(re.findall(r"\b[A-Z][a-z]+\b", text))

                # Years
                years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))

                return numbers.union(proper_nouns).union(years)

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> MetricResult:
                try:
                    expected = str(getattr(example, self.field_name, ""))
                    predicted = str(
                        getattr(prediction, self.field_name, str(prediction))
                    )

                    expected_entities = self._extract_entities(expected)
                    predicted_entities = self._extract_entities(predicted)

                    if not expected_entities:
                        # No entities to check, use text similarity
                        score = (
                            1.0
                            if expected.lower().strip() == predicted.lower().strip()
                            else 0.5
                        )
                        confidence = 0.5
                        explanation = "No entities found, using text similarity"
                    else:
                        # Calculate entity overlap
                        correct_entities = expected_entities.intersection(
                            predicted_entities
                        )
                        score = len(correct_entities) / len(expected_entities)
                        confidence = 0.8
                        explanation = f"Entity accuracy: {len(correct_entities)}/{len(expected_entities)} entities correct"

                    return MetricResult(
                        metric_name=self.name,
                        score=score,
                        confidence=confidence,
                        metadata={
                            "expected_entities": list(expected_entities),
                            "predicted_entities": list(predicted_entities),
                            "correct_entities": (
                                list(correct_entities) if expected_entities else []
                            ),
                        },
                        explanation=explanation,
                    )

                except Exception as e:
                    return MetricResult(
                        metric_name=self.name,
                        score=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                        explanation=f"Error: {str(e)}",
                    )

        class FluencyMetric(AdvancedMetric):
            """Evaluates text fluency based on linguistic patterns."""

            def __init__(self, field_name: str = "answer"):
                super().__init__(
                    name=f"fluency_{field_name}",
                    description="Text fluency based on grammar and readability",
                )
                self.field_name = field_name

            def _calculate_readability(self, text: str) -> float:
                """Simple readability score based on sentence and word length."""
                sentences = re.split(r"[.!?]+", text)
                sentences = [s.strip() for s in sentences if s.strip()]

                if not sentences:
                    return 0.0

                words = text.split()
                if not words:
                    return 0.0

                avg_sentence_length = len(words) / len(sentences)
                avg_word_length = sum(len(word) for word in words) / len(words)

                # Optimal ranges: 15-20 words per sentence, 4-6 characters per word
                sentence_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
                word_score = 1.0 - abs(avg_word_length - 5) / 5

                return max(0.0, (sentence_score + word_score) / 2)

            def _check_grammar_patterns(self, text: str) -> float:
                """Check for basic grammar patterns."""
                score = 1.0

                # Check for repeated words
                words = text.lower().split()
                if len(words) != len(set(words)):
                    repeated_ratio = (len(words) - len(set(words))) / len(words)
                    score -= repeated_ratio * 0.3

                # Check for proper capitalization
                sentences = re.split(r"[.!?]+", text)
                capitalized_sentences = sum(
                    1 for s in sentences if s.strip() and s.strip()[0].isupper()
                )
                if sentences and capitalized_sentences / len(sentences) < 0.8:
                    score -= 0.2

                # Check for proper punctuation
                if not re.search(r"[.!?]$", text.strip()):
                    score -= 0.1

                return max(0.0, score)

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> MetricResult:
                try:
                    predicted = str(
                        getattr(prediction, self.field_name, str(prediction))
                    )

                    if not predicted.strip():
                        return MetricResult(
                            metric_name=self.name,
                            score=0.0,
                            confidence=1.0,
                            explanation="Empty prediction",
                        )

                    readability_score = self._calculate_readability(predicted)
                    grammar_score = self._check_grammar_patterns(predicted)

                    # Combined fluency score
                    fluency_score = (readability_score * 0.6) + (grammar_score * 0.4)

                    return MetricResult(
                        metric_name=self.name,
                        score=fluency_score,
                        confidence=0.7,  # Moderate confidence for heuristic-based evaluation
                        sub_scores={
                            "readability": readability_score,
                            "grammar": grammar_score,
                        },
                        metadata={
                            "text_length": len(predicted),
                            "word_count": len(predicted.split()),
                            "sentence_count": len(re.split(r"[.!?]+", predicted)),
                        },
                        explanation=f"Fluency: {fluency_score:.3f} (Readability: {readability_score:.3f}, Grammar: {grammar_score:.3f})",
                    )

                except Exception as e:
                    return MetricResult(
                        metric_name=self.name,
                        score=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                        explanation=f"Error: {str(e)}",
                    )

        class RelevanceMetric(AdvancedMetric):
            """Evaluates relevance of response to the input question/context."""

            def __init__(
                self, input_field: str = "question", output_field: str = "answer"
            ):
                super().__init__(
                    name=f"relevance_{input_field}_to_{output_field}",
                    description="Relevance of response to input",
                )
                self.input_field = input_field
                self.output_field = output_field

            def _extract_keywords(self, text: str) -> set:
                """Extract important keywords from text."""
                # Remove common stop words
                stop_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                }

                words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
                keywords = {
                    word for word in words if len(word) > 2 and word not in stop_words
                }

                return keywords

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> MetricResult:
                try:
                    input_text = str(getattr(example, self.input_field, ""))
                    output_text = str(
                        getattr(prediction, self.output_field, str(prediction))
                    )

                    input_keywords = self._extract_keywords(input_text)
                    output_keywords = self._extract_keywords(output_text)

                    if not input_keywords:
                        return MetricResult(
                            metric_name=self.name,
                            score=0.5,
                            confidence=0.3,
                            explanation="No keywords found in input",
                        )

                    # Calculate keyword overlap
                    overlap = input_keywords.intersection(output_keywords)
                    relevance_score = len(overlap) / len(input_keywords)

                    # Bonus for additional relevant keywords in output
                    additional_keywords = output_keywords - input_keywords
                    if additional_keywords:
                        bonus = min(0.2, len(additional_keywords) * 0.05)
                        relevance_score += bonus

                    relevance_score = min(1.0, relevance_score)

                    return MetricResult(
                        metric_name=self.name,
                        score=relevance_score,
                        confidence=0.8,
                        metadata={
                            "input_keywords": list(input_keywords),
                            "output_keywords": list(output_keywords),
                            "overlap_keywords": list(overlap),
                            "additional_keywords": list(additional_keywords),
                        },
                        explanation=f"Relevance: {relevance_score:.3f} ({len(overlap)}/{len(input_keywords)} keywords matched)",
                    )

                except Exception as e:
                    return MetricResult(
                        metric_name=self.name,
                        score=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                        explanation=f"Error: {str(e)}",
                    )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 📊 Domain-Specific Metrics Collection Created

                **Available Metrics:**  
                - **SemanticSimilarityMetric** - Advanced semantic similarity using multiple signals  
                - **FactualAccuracyMetric** - Entity-based factual accuracy evaluation  
                - **FluencyMetric** - Text fluency based on readability and grammar patterns  
                - **RelevanceMetric** - Relevance of response to input question/context  

                **Key Features:**  
                - **Multi-Signal Evaluation** - Each metric uses multiple evaluation signals  
                - **Confidence Scoring** - Metrics express confidence in their evaluations  
                - **Rich Metadata** - Detailed information about evaluation process  
                - **Sub-Score Tracking** - Break down scores into component parts  

                **Evaluation Dimensions:**  
                - **Accuracy** - Factual correctness and entity matching  
                - **Quality** - Fluency, readability, and linguistic quality  
                - **Relevance** - Alignment between input and output  
                - **Semantic** - Meaning-based similarity beyond exact matching  

                Ready to build composite evaluation strategies!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        SemanticSimilarityMetric = None
        FactualAccuracyMetric = None
        FluencyMetric = None
        RelevanceMetric = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (
        FactualAccuracyMetric,
        FluencyMetric,
        RelevanceMetric,
        SemanticSimilarityMetric,
    )


@app.cell
def _(CompositeMetric, available_providers, cleandoc, dspy, mo, output):
    if available_providers and CompositeMetric:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🎛️ Step 3: Interactive Metrics Builder

                **Build and test custom metrics** with interactive configuration:
                """
            )
        )

        # Create sample data for testing
        sample_examples = [
            dspy.Example(
                question="What is the capital of France?", answer="Paris"
            ).with_inputs("question"),
            dspy.Example(
                question="Who wrote the novel '1984'?", answer="George Orwell"
            ).with_inputs("question"),
            dspy.Example(
                question="What is the chemical symbol for gold?", answer="Au"
            ).with_inputs("question"),
            dspy.Example(
                question="In what year did World War II end?", answer="1945"
            ).with_inputs("question"),
        ]

        # Create test predictions with varying quality
        test_predictions = [
            dspy.Prediction(answer="Paris"),  # Perfect match
            dspy.Prediction(answer="George Orwell wrote 1984"),  # Verbose but correct
            dspy.Prediction(answer="Gold"),  # Wrong format
            dspy.Prediction(answer="World War 2 ended in 1945"),  # Correct but verbose
        ]

        # Interactive controls for metric configuration
        metric_selection = mo.ui.multiselect(
            options=["semantic_similarity", "factual_accuracy", "fluency", "relevance"],
            value=["semantic_similarity", "factual_accuracy"],
            label="Select Metrics to Include",
        )

        semantic_threshold_slider = mo.ui.slider(
            start=0.5,
            stop=1.0,
            value=0.8,
            step=0.05,
            label="Semantic Similarity Threshold",
            show_value=True,
        )

        # Weights for composite metric
        semantic_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.4,
            step=0.1,
            label="Semantic Similarity Weight",
            show_value=True,
        )

        factual_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.3,
            step=0.1,
            label="Factual Accuracy Weight",
            show_value=True,
        )

        fluency_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.2,
            step=0.1,
            label="Fluency Weight",
            show_value=True,
        )

        relevance_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.1,
            step=0.1,
            label="Relevance Weight",
            show_value=True,
        )

        # Actions
        build_metric_button = mo.ui.run_button(
            label="🏗️ Build Custom Metric", kind="success"
        )

        test_metric_button = mo.ui.run_button(label="🧪 Test Metric", kind="info")

        validate_metric_button = mo.ui.run_button(
            label="✅ Validate Metric", kind="neutral"
        )

        metrics_builder_ui = mo.vstack(
            [
                mo.md("### 🎛️ Custom Metrics Builder"),
                metric_selection,
                mo.md("**Metric Parameters:**"),
                semantic_threshold_slider,
                mo.md("**Composite Weights:**"),
                semantic_weight_slider,
                factual_weight_slider,
                fluency_weight_slider,
                relevance_weight_slider,
                mo.md("---"),
                mo.hstack(
                    [build_metric_button, test_metric_button, validate_metric_button]
                ),
            ]
        )

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🎛️ Interactive Metrics Builder Created

                **Builder Features:**  
                - **Metric Selection** - Choose which metrics to include in your composite  
                - **Parameter Configuration** - Adjust thresholds and sensitivity settings  
                - **Weight Configuration** - set relative importance of different metrics  
                - **Real-time Testing** - Test metrics with sample data immediately  

                **Sample Data:**  
                - **4 QA Examples** - Diverse question-answer pairs for testing  
                - **Varied Predictions** - Different quality levels to test metric sensitivity  
                - **Interactive Controls** - Adjust parameters and see immediate results  

                Configure your custom metrics and test them with the sample data!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        metrics_builder_ui = mo.md("")
        cell5_content = mo.md("")
        build_metric_button = None
        test_metric_button = None
        validate_metric_button = None
        sample_examples = None
        test_predictions = None

    cell5_out = mo.vstack([cell5_desc, metrics_builder_ui, cell5_content])
    output.replace(cell5_out)
    return (
        build_metric_button,
        factual_weight_slider,
        fluency_weight_slider,
        metric_selection,
        relevance_weight_slider,
        sample_examples,
        semantic_threshold_slider,
        semantic_weight_slider,
        test_metric_button,
        test_predictions,
        validate_metric_button,
    )


@app.cell
def _(
    CompositeMetric,
    FactualAccuracyMetric,
    FluencyMetric,
    RelevanceMetric,
    SemanticSimilarityMetric,
    build_metric_button,
    cleandoc,
    factual_weight_slider,
    fluency_weight_slider,
    metric_selection,
    mo,
    output,
    relevance_weight_slider,
    sample_examples,
    semantic_threshold_slider,
    semantic_weight_slider,
    test_metric_button,
    test_predictions,
    validate_metric_button,
):
    # Handle metric builder interactions
    import __main__

    metric_results_display = mo.md("")

    if build_metric_button.value:
        # Build custom composite metric based on user selections
        selected_metrics = metric_selection.value

        metrics_dict = {}
        weights_dict = {}

        if "semantic_similarity" in selected_metrics:
            metrics_dict["semantic_similarity"] = SemanticSimilarityMetric(
                threshold=semantic_threshold_slider.value
            )
            weights_dict["semantic_similarity"] = semantic_weight_slider.value

        if "factual_accuracy" in selected_metrics:
            metrics_dict["factual_accuracy"] = FactualAccuracyMetric()
            weights_dict["factual_accuracy"] = factual_weight_slider.value

        if "fluency" in selected_metrics:
            metrics_dict["fluency"] = FluencyMetric()
            weights_dict["fluency"] = fluency_weight_slider.value

        if "relevance" in selected_metrics:
            metrics_dict["relevance"] = RelevanceMetric()
            weights_dict["relevance"] = relevance_weight_slider.value

        if metrics_dict:
            __main__.custom_metric = CompositeMetric(
                name="Custom_QA_Metric", metrics=metrics_dict, weights=weights_dict
            )

            metric_results_display = mo.md(
                cleandoc(
                    f"""
                    ## ✅ Custom Metric Built Successfully!

                    **Metric Name:** {__main__.custom_metric.name}  
                    **Components:** {len(__main__.custom_metric.metrics)} metrics  
                    **Weights:** {__main__.custom_metric.weights}  

                    ### 📊 Configuration
                    - **Selected Metrics:** {', '.join(selected_metrics)}  
                    - **Semantic Threshold:** {semantic_threshold_slider.value:.2f}  
                    - **Total Weight:** {sum(__main__.custom_metric.weights.values()):.2f}  

                    Your custom metric is ready for testing!
                    """
                )
            )
        else:
            metric_results_display = mo.md(
                cleandoc(
                    """
                    ## ❌ No Metrics Selected

                    Please select at least one metric to build a custom evaluation metric.
                    """
                )
            )

    elif (
        test_metric_button.value
        and hasattr(__main__, "custom_metric")
        and __main__.custom_metric
    ):
        # Test the custom metric with sample data
        results = __main__.custom_metric.batch_evaluate(
            sample_examples, test_predictions
        )

        # Display results
        results_text = []
        for i, result in enumerate(results):
            results_text.append(f"**Example {i+1}:**")
            results_text.append(f"  - Overall Score: {result.score:.3f}")
            results_text.append(f"  - Confidence: {result.confidence:.3f}")

            if result.sub_scores:
                results_text.append("  - Sub-scores:")
                for metric_name, score in result.sub_scores.items():
                    results_text.append(f"    - {metric_name}: {score:.3f}")

            if result.explanation:
                results_text.append(f"  - Explanation: {result.explanation}")
            results_text.append("")

        # Calculate statistics
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        metric_results_display = mo.md(
            cleandoc(
                f"""
                ## 🧪 Metric Test Results

                **Metric:** {__main__.custom_metric.name}  
                **Examples Tested:** {len(results)}  

                ### 📊 Individual Results

                {chr(10).join(results_text)}

                ### 📈 Summary Statistics

                - **Average Score:** {avg_score:.3f}  
                - **Score Range:** {min_score:.3f} - {max_score:.3f}  
                - **Standard Deviation:** {(sum((s - avg_score)**2 for s in scores) / len(scores))**0.5:.3f}  

                ### 💡 Analysis

                The metric shows good discrimination between different quality levels!
                """
            )
        )

    elif (
        validate_metric_button.value
        and hasattr(__main__, "custom_metric")
        and __main__.custom_metric
    ):
        # Validate metric consistency and reliability

        # Test with duplicate examples
        duplicate_examples = sample_examples * 2
        duplicate_predictions = test_predictions * 2

        validation_results = __main__.custom_metric.batch_evaluate(
            duplicate_examples, duplicate_predictions
        )

        # Check consistency (same examples should get same scores)
        consistency_scores = []
        for i in range(len(sample_examples)):
            original_score = validation_results[i].score
            duplicate_score = validation_results[i + len(sample_examples)].score
            consistency_scores.append(abs(original_score - duplicate_score))

        avg_consistency = sum(consistency_scores) / len(consistency_scores)

        # Get metric statistics
        stats = __main__.custom_metric.get_statistics()

        metric_results_display = mo.md(
            cleandoc(
                f"""
                ## ✅ Metric Validation Results

                **Metric:** {__main__.custom_metric.name}  
                **Validation Type:** Consistency and Reliability  

                ### 🔄 Consistency Analysis

                - **Average Consistency Error:** {avg_consistency:.4f}  
                - **Consistency Rating:** {'Excellent' if avg_consistency < 0.01 else 'Good' if avg_consistency < 0.05 else 'Fair'}  

                ### 📊 Performance Statistics

                - **Total Evaluations:** {stats.get('count', 0)}  
                - **Mean Score:** {stats.get('mean', 0):.3f}  
                - **Score Variance:** {stats.get('std_dev', 0):.3f}  
                - **Score Range:** {stats.get('min', 0):.3f} - {stats.get('max', 0):.3f}  

                ### 💡 Validation Summary

                {'✅ Metric shows good consistency and reliability!' if avg_consistency < 0.05 else '⚠️ Metric may need refinement for better consistency.'}
                """
            )
        )

    # Error handling for buttons clicked without custom metric
    elif test_metric_button.value and (
        not hasattr(__main__, "custom_metric") or not __main__.custom_metric
    ):
        metric_results_display = mo.md(
            "## ⚠️ No custom metric built yet. Please build a metric first by clicking 'Build Custom Metric'."
        )

    elif validate_metric_button.value and (
        not hasattr(__main__, "custom_metric") or not __main__.custom_metric
    ):
        metric_results_display = mo.md(
            "## ⚠️ No custom metric available. Please build a metric first."
        )

    output.replace(metric_results_display)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell7_out = mo.md(
        cleandoc(
            """
            ## 🎯 Advanced Custom Metrics Strategies

            ### 🏆 Best Practices for Custom Metrics Design

            **Design Principles:**  
            - **Task Alignment** - Metrics should directly measure what matters for your specific task  
            - **Multi-Dimensional** - Capture different aspects of quality (accuracy, fluency, relevance)  
            - **Balanced Sensitivity** - Avoid metrics that are too easy or too hard to satisfy  
            - **Interpretability** - Ensure metric scores are meaningful and actionable  

            **Metric Composition Strategies:**  
            - **Weighted Combination** - Use domain expertise to weight different quality aspects  
            - **Hierarchical Metrics** - Build complex metrics from simpler components  
            - **Adaptive Weighting** - Adjust weights based on task characteristics or context  
            - **Threshold-Based** - Use different thresholds for different quality levels  

            ### ⚡ Advanced Metric Techniques

            **Confidence-Aware Evaluation:**  
            - **Uncertainty Quantification** - Express confidence in metric evaluations  
            - **Confidence Weighting** - Weight scores by evaluation confidence  
            - **Ensemble Metrics** - Combine multiple metrics with confidence-based voting  
            - **Active Learning** - Focus evaluation effort on uncertain cases  

            **Context-Sensitive Metrics:**  
            - **Domain Adaptation** - Adjust metrics based on domain characteristics  
            - **Task Difficulty** - Scale scores based on question/task difficulty  
            - **User Preferences** - Incorporate user feedback into metric design  
            - **Temporal Adaptation** - Evolve metrics based on system performance over time  

            ### 🔧 Metric Validation and Testing

            **Validation Strategies:**  
            - **Consistency Testing** - Ensure metrics give consistent scores for identical inputs  
            - **Sensitivity Analysis** - Test metric behavior across different input variations  
            - **Human Correlation** - Validate metrics against human judgment  
            - **Cross-Domain Testing** - Test metric generalization across different domains  

            **Quality Assurance:**  
            - **Edge Case Testing** - Test metrics with unusual or boundary cases  
            - **Performance Monitoring** - Track metric performance over time  
            - **Bias Detection** - Check for systematic biases in metric evaluation  
            - **Calibration** - Ensure metric scores align with actual quality levels  

            ### 🚀 Production Deployment

            **Scalability Considerations:**  
            - **Computational Efficiency** - Optimize metrics for large-scale evaluation  
            - **Caching Strategies** - Cache expensive computations for repeated evaluations  
            - **Parallel Processing** - Distribute metric evaluation across multiple workers  
            - **Resource Management** - Monitor and manage computational resources  

            **Monitoring and Maintenance:**  
            - **Performance Tracking** - Monitor metric performance in production  
            - **Drift Detection** - Detect when metric behavior changes over time  
            - **Version Control** - Maintain versions of metrics for reproducibility  
            - **A/B Testing** - Test new metrics against existing ones  

            ### 💡 Next Steps

            **Apply Your Custom Metrics:**  
            1. **Define Quality Criteria** - Clearly specify what constitutes good performance  
            2. **Build Metric Suite** - Create comprehensive evaluation covering all quality aspects  
            3. **Validate Thoroughly** - Test metrics extensively before production use  
            4. **Monitor Continuously** - Track metric performance and adjust as needed  

            Excellent work mastering custom metrics design! 🎉
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
