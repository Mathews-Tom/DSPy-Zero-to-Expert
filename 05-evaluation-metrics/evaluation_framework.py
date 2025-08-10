# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import statistics
    import sys
    import time
    from abc import ABC, abstractmethod
    from collections import defaultdict
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
        Optional,
        abstractmethod,
        cleandoc,
        dataclass,
        defaultdict,
        dspy,
        field,
        get_config,
        mo,
        output,
        setup_dspy_environment,
        statistics,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 Comprehensive Evaluation Framework

            **Duration:** 2-3 hours  
            **Prerequisites:** Completed Modules 00-04  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will master:  
            - **Evaluation Strategy Design** - Create comprehensive evaluation pipelines  
            - **Multi-Metric Assessment** - Combine multiple evaluation criteria  
            - **Statistical Analysis** - Apply rigorous statistical methods to evaluation  
            - **Automated Evaluation** - Build scalable evaluation systems  
            - **Result Interpretation** - Extract actionable insights from evaluation data  

            ## 📈 Evaluation Framework Overview

            **Why Comprehensive Evaluation Matters:**  
            - **Objective Assessment** - Move beyond subjective quality judgments  
            - **System Comparison** - Compare different approaches systematically  
            - **Performance Tracking** - Monitor improvements over time  
            - **Failure Analysis** - Identify and understand system weaknesses  

            **Framework Components:**  
            - **Evaluation Strategies** - Systematic approaches to assessment  
            - **Metric Aggregation** - Combine multiple evaluation dimensions  
            - **Statistical Testing** - Ensure evaluation reliability and significance  
            - **Report Generation** - Create comprehensive evaluation reports  

            Let's build a world-class evaluation framework!
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
                ## ✅ Evaluation Framework Environment Ready

                **Configuration:**  
                - Provider: **{provider}**  
                - Model: **{config.default_model}**  
                - Available Providers: **{', '.join(available_providers)}**

                Ready to build comprehensive evaluation systems!
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
    Optional,
    abstractmethod,
    available_providers,
    cleandoc,
    dataclass,
    defaultdict,
    dspy,
    field,
    mo,
    output,
    statistics,
    time,
):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Core Evaluation Framework Architecture

                **Building the foundation** for comprehensive evaluation systems:
                """
            )
        )

        @dataclass
        class EvaluationResult:
            """Represents the result of a single evaluation."""

            metric_name: str
            score: float
            metadata: dict[str, Any] = field(default_factory=dict)
            timestamp: float = field(default_factory=time.time)
            example_id: Optional[str] = None
            prediction: Optional[Any] = None
            expected: Optional[Any] = None

        @dataclass
        class EvaluationSummary:
            """Aggregated evaluation results across multiple examples."""

            metric_name: str
            individual_scores: list[float]
            mean_score: float
            median_score: float
            std_deviation: float
            min_score: float
            max_score: float
            total_examples: int
            metadata: dict[str, Any] = field(default_factory=dict)

        class EvaluationMetric(ABC):
            """Abstract base class for evaluation metrics."""

            def __init__(self, name: str, description: str = ""):
                self.name = name
                self.description = description

            @abstractmethod
            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> EvaluationResult:
                """Evaluate a single example and return the result."""
                pass

            def batch_evaluate(
                self,
                examples: list[dspy.Example],
                predictions: list[Any],
                traces: Optional[list[Any]] = None,
            ) -> list[EvaluationResult]:
                """Evaluate multiple examples and return results."""
                if traces is None:
                    traces = [None] * len(examples)

                results = []
                for i, (example, prediction, trace) in enumerate(
                    zip(examples, predictions, traces)
                ):
                    try:
                        result = self.evaluate(example, prediction, trace)
                        result.example_id = f"example_{i}"
                        results.append(result)
                    except Exception as e:
                        # Create error result
                        error_result = EvaluationResult(
                            metric_name=self.name,
                            score=0.0,
                            metadata={"error": str(e), "error_type": type(e).__name__},
                            example_id=f"example_{i}",
                            prediction=prediction,
                            expected=getattr(example, "answer", None),
                        )
                        results.append(error_result)

                return results

        class EvaluationStrategy:
            """Manages multiple metrics and evaluation workflows."""

            def __init__(self, name: str, description: str = ""):
                self.name = name
                self.description = description
                self.metrics: dict[str, EvaluationMetric] = {}
                self.evaluation_history: list[dict[str, Any]] = []

            def add_metric(self, metric: EvaluationMetric) -> None:
                """Add a metric to the evaluation strategy."""
                self.metrics[metric.name] = metric

            def remove_metric(self, metric_name: str) -> None:
                """Remove a metric from the evaluation strategy."""
                if metric_name in self.metrics:
                    del self.metrics[metric_name]

            def evaluate_system(
                self,
                system: Any,
                test_examples: list[dspy.Example],
                capture_traces: bool = False,
            ) -> dict[str, EvaluationSummary]:
                """Evaluate a system using all configured metrics."""
                start_time = time.time()

                # Generate predictions
                predictions = []
                traces = [] if capture_traces else None

                for example in test_examples:
                    try:
                        if hasattr(system, "forward"):
                            # DSPy module
                            inputs = example.inputs()
                            prediction = system(**inputs)
                        else:
                            # Callable function
                            prediction = system(example)

                        predictions.append(prediction)

                        if capture_traces:
                            # In a real implementation, you'd capture the actual trace
                            traces.append(
                                {"prediction": prediction, "timestamp": time.time()}
                            )

                    except Exception as e:
                        # Handle prediction errors
                        error_prediction = {
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                        predictions.append(error_prediction)

                        if capture_traces:
                            traces.append({"error": str(e), "timestamp": time.time()})

                # Evaluate with all metrics
                evaluation_results = {}

                for metric_name, metric in self.metrics.items():
                    try:
                        results = metric.batch_evaluate(
                            test_examples, predictions, traces
                        )
                        scores = [r.score for r in results]

                        if scores:
                            summary = EvaluationSummary(
                                metric_name=metric_name,
                                individual_scores=scores,
                                mean_score=statistics.mean(scores),
                                median_score=statistics.median(scores),
                                std_deviation=(
                                    statistics.stdev(scores) if len(scores) > 1 else 0.0
                                ),
                                min_score=min(scores),
                                max_score=max(scores),
                                total_examples=len(scores),
                                metadata={
                                    "metric_description": metric.description,
                                    "evaluation_time": time.time() - start_time,
                                    "failed_examples": len(
                                        [r for r in results if "error" in r.metadata]
                                    ),
                                },
                            )
                        else:
                            # No valid scores
                            summary = EvaluationSummary(
                                metric_name=metric_name,
                                individual_scores=[],
                                mean_score=0.0,
                                median_score=0.0,
                                std_deviation=0.0,
                                min_score=0.0,
                                max_score=0.0,
                                total_examples=0,
                                metadata={"error": "No valid scores generated"},
                            )

                        evaluation_results[metric_name] = summary

                    except Exception as e:
                        # Handle metric evaluation errors
                        error_summary = EvaluationSummary(
                            metric_name=metric_name,
                            individual_scores=[],
                            mean_score=0.0,
                            median_score=0.0,
                            std_deviation=0.0,
                            min_score=0.0,
                            max_score=0.0,
                            total_examples=0,
                            metadata={"error": str(e), "error_type": type(e).__name__},
                        )
                        evaluation_results[metric_name] = error_summary

                # Store evaluation in history
                evaluation_record = {
                    "timestamp": start_time,
                    "system_name": getattr(system, "__class__", type(system)).__name__,
                    "num_examples": len(test_examples),
                    "metrics_used": list(self.metrics.keys()),
                    "results": evaluation_results,
                    "total_time": time.time() - start_time,
                }
                self.evaluation_history.append(evaluation_record)

                return evaluation_results

            def compare_systems(
                self, systems: dict[str, Any], test_examples: list[dspy.Example]
            ) -> dict[str, Any]:
                """Compare multiple systems using the configured metrics."""
                comparison_results = {}

                for system_name, system in systems.items():
                    results = self.evaluate_system(system, test_examples)
                    comparison_results[system_name] = results

                # Generate comparison analysis
                analysis = self._analyze_comparison(comparison_results)

                return {
                    "individual_results": comparison_results,
                    "analysis": analysis,
                    "timestamp": time.time(),
                    "num_systems": len(systems),
                    "num_examples": len(test_examples),
                }

            def _analyze_comparison(
                self, results: dict[str, dict[str, EvaluationSummary]]
            ) -> dict[str, Any]:
                """Analyze comparison results to identify best performers."""
                analysis = {
                    "best_performers": {},
                    "metric_rankings": {},
                    "statistical_significance": {},
                    "insights": [],
                }

                # Find best performer for each metric
                for metric_name in self.metrics.keys():
                    metric_scores = {}
                    for system_name, system_results in results.items():
                        if metric_name in system_results:
                            metric_scores[system_name] = system_results[
                                metric_name
                            ].mean_score

                    if metric_scores:
                        best_system = max(metric_scores.items(), key=lambda x: x[1])
                        analysis["best_performers"][metric_name] = {
                            "system": best_system[0],
                            "score": best_system[1],
                        }

                        # Rank all systems for this metric
                        ranked_systems = sorted(
                            metric_scores.items(), key=lambda x: x[1], reverse=True
                        )
                        analysis["metric_rankings"][metric_name] = ranked_systems

                # Generate insights
                if analysis["best_performers"]:
                    # Find overall best performer (most wins)
                    system_wins = defaultdict(int)
                    for metric_data in analysis["best_performers"].values():
                        system_wins[metric_data["system"]] += 1

                    if system_wins:
                        overall_best = max(system_wins.items(), key=lambda x: x[1])
                        analysis["insights"].append(
                            f"System '{overall_best[0]}' performs best overall, winning {overall_best[1]} out of {len(self.metrics)} metrics"
                        )

                return analysis

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Core Evaluation Framework Created

                **Framework Components:**  
                - **EvaluationResult** - Individual evaluation result with metadata  
                - **EvaluationSummary** - Aggregated statistics across multiple examples  
                - **EvaluationMetric** - Abstract base class for custom metrics  
                - **EvaluationStrategy** - Orchestrates multiple metrics and workflows  

                **Key Features:**  
                - **Error Handling** - Graceful handling of prediction and evaluation failures  
                - **Batch Processing** - Efficient evaluation of multiple examples  
                - **System Comparison** - Side-by-side comparison of different systems  
                - **Statistical Analysis** - Comprehensive statistical summaries  
                - **History Tracking** - Maintain evaluation history for analysis  

                Ready to build specific evaluation metrics!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        EvaluationResult = None
        EvaluationSummary = None
        EvaluationMetric = None
        EvaluationStrategy = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return EvaluationMetric, EvaluationResult, EvaluationStrategy


@app.cell
def _(
    Any,
    EvaluationMetric,
    EvaluationResult,
    Optional,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
):
    if available_providers and EvaluationMetric:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 2: Built-in Evaluation Metrics Library

                **Comprehensive collection** of ready-to-use evaluation metrics:
                """
            )
        )

        class ExactMatchMetric(EvaluationMetric):
            """Exact string matching evaluation metric."""

            def __init__(
                self, field_name: str = "answer", case_sensitive: bool = False
            ):
                super().__init__(
                    name=f"exact_match_{field_name}",
                    description=f"Exact match evaluation for {field_name} field",
                )
                self.field_name = field_name
                self.case_sensitive = case_sensitive

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> EvaluationResult:
                try:
                    expected = getattr(example, self.field_name, "")
                    predicted = getattr(prediction, self.field_name, str(prediction))

                    if not self.case_sensitive:
                        expected = str(expected).lower().strip()
                        predicted = str(predicted).lower().strip()
                    else:
                        expected = str(expected).strip()
                        predicted = str(predicted).strip()

                    score = 1.0 if expected == predicted else 0.0

                    return EvaluationResult(
                        metric_name=self.name,
                        score=score,
                        metadata={
                            "expected": expected,
                            "predicted": predicted,
                            "case_sensitive": self.case_sensitive,
                        },
                        prediction=prediction,
                        expected=getattr(example, self.field_name, ""),
                    )

                except Exception as e:
                    return EvaluationResult(
                        metric_name=self.name,
                        score=0.0,
                        metadata={"error": str(e)},
                        prediction=prediction,
                        expected=getattr(example, self.field_name, None),
                    )

        class FuzzyMatchMetric(EvaluationMetric):
            """Fuzzy string matching with configurable threshold."""

            def __init__(self, field_name: str = "answer", threshold: float = 0.8):
                super().__init__(
                    name=f"fuzzy_match_{field_name}",
                    description=f"Fuzzy match evaluation for {field_name} field (threshold: {threshold})",
                )
                self.field_name = field_name
                self.threshold = threshold

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> EvaluationResult:
                try:
                    import difflib

                    expected = (
                        str(getattr(example, self.field_name, "")).lower().strip()
                    )
                    predicted = (
                        str(getattr(prediction, self.field_name, str(prediction)))
                        .lower()
                        .strip()
                    )

                    if not expected or not predicted:
                        similarity = 0.0
                    else:
                        similarity = difflib.SequenceMatcher(
                            None, expected, predicted
                        ).ratio()

                    score = 1.0 if similarity >= self.threshold else similarity

                    return EvaluationResult(
                        metric_name=self.name,
                        score=score,
                        metadata={
                            "expected": expected,
                            "predicted": predicted,
                            "similarity": similarity,
                            "threshold": self.threshold,
                            "passed_threshold": similarity >= self.threshold,
                        },
                        prediction=prediction,
                        expected=getattr(example, self.field_name, ""),
                    )

                except Exception as e:
                    return EvaluationResult(
                        metric_name=self.name,
                        score=0.0,
                        metadata={"error": str(e)},
                        prediction=prediction,
                        expected=getattr(example, self.field_name, None),
                    )

        class WordOverlapMetric(EvaluationMetric):
            """Word overlap similarity metric."""

            def __init__(self, field_name: str = "answer", normalize: bool = True):
                super().__init__(
                    name=f"word_overlap_{field_name}",
                    description=f"Word overlap evaluation for {field_name} field",
                )
                self.field_name = field_name
                self.normalize = normalize

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> EvaluationResult:
                try:
                    expected = (
                        str(getattr(example, self.field_name, "")).lower().strip()
                    )
                    predicted = (
                        str(getattr(prediction, self.field_name, str(prediction)))
                        .lower()
                        .strip()
                    )

                    expected_words = set(expected.split())
                    predicted_words = set(predicted.split())

                    if not expected_words:
                        score = 1.0 if not predicted_words else 0.0
                    else:
                        intersection = expected_words.intersection(predicted_words)
                        if self.normalize:
                            score = len(intersection) / len(expected_words)
                        else:
                            score = len(intersection) / max(
                                len(expected_words), len(predicted_words)
                            )

                    return EvaluationResult(
                        metric_name=self.name,
                        score=score,
                        metadata={
                            "expected_words": list(expected_words),
                            "predicted_words": list(predicted_words),
                            "intersection": list(intersection),
                            "normalize": self.normalize,
                        },
                        prediction=prediction,
                        expected=getattr(example, self.field_name, ""),
                    )

                except Exception as e:
                    return EvaluationResult(
                        metric_name=self.name,
                        score=0.0,
                        metadata={"error": str(e)},
                        prediction=prediction,
                        expected=getattr(example, self.field_name, None),
                    )

        class LengthSimilarityMetric(EvaluationMetric):
            """Evaluates similarity based on response length."""

            def __init__(self, field_name: str = "answer", tolerance: float = 0.2):
                super().__init__(
                    name=f"length_similarity_{field_name}",
                    description=f"Length similarity evaluation for {field_name} field",
                )
                self.field_name = field_name
                self.tolerance = tolerance

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> EvaluationResult:
                try:
                    expected = str(getattr(example, self.field_name, ""))
                    predicted = str(
                        getattr(prediction, self.field_name, str(prediction))
                    )

                    expected_len = len(expected.split())
                    predicted_len = len(predicted.split())

                    if expected_len == 0:
                        score = 1.0 if predicted_len == 0 else 0.0
                    else:
                        length_ratio = abs(predicted_len - expected_len) / expected_len
                        score = max(0.0, 1.0 - length_ratio / self.tolerance)

                    return EvaluationResult(
                        metric_name=self.name,
                        score=score,
                        metadata={
                            "expected_length": expected_len,
                            "predicted_length": predicted_len,
                            "length_ratio": length_ratio if expected_len > 0 else 0.0,
                            "tolerance": self.tolerance,
                        },
                        prediction=prediction,
                        expected=getattr(example, self.field_name, ""),
                    )

                except Exception as e:
                    return EvaluationResult(
                        metric_name=self.name,
                        score=0.0,
                        metadata={"error": str(e)},
                        prediction=prediction,
                        expected=getattr(example, self.field_name, None),
                    )

        class ContainsKeywordsMetric(EvaluationMetric):
            """Checks if prediction contains required keywords."""

            def __init__(
                self,
                keywords: list[str],
                field_name: str = "answer",
                require_all: bool = False,
                case_sensitive: bool = False,
            ):
                super().__init__(
                    name=f"contains_keywords_{field_name}",
                    description=f"Keyword presence evaluation for {field_name} field",
                )
                self.keywords = keywords
                self.field_name = field_name
                self.require_all = require_all
                self.case_sensitive = case_sensitive

            def evaluate(
                self,
                example: dspy.Example,
                prediction: Any,
                trace: Optional[Any] = None,
            ) -> EvaluationResult:
                try:
                    predicted = str(
                        getattr(prediction, self.field_name, str(prediction))
                    )

                    if not self.case_sensitive:
                        predicted = predicted.lower()
                        keywords = [k.lower() for k in self.keywords]
                    else:
                        keywords = self.keywords

                    found_keywords = [kw for kw in keywords if kw in predicted]

                    if self.require_all:
                        score = 1.0 if len(found_keywords) == len(keywords) else 0.0
                    else:
                        score = len(found_keywords) / len(keywords) if keywords else 1.0

                    return EvaluationResult(
                        metric_name=self.name,
                        score=score,
                        metadata={
                            "keywords": keywords,
                            "found_keywords": found_keywords,
                            "require_all": self.require_all,
                            "case_sensitive": self.case_sensitive,
                        },
                        prediction=prediction,
                        expected=getattr(example, self.field_name, None),
                    )

                except Exception as e:
                    return EvaluationResult(
                        metric_name=self.name,
                        score=0.0,
                        metadata={"error": str(e)},
                        prediction=prediction,
                        expected=getattr(example, self.field_name, None),
                    )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 📊 Built-in Metrics Library Created

                **Available Metrics:**  
                - **ExactMatchMetric** - Exact string matching with case sensitivity options  
                - **FuzzyMatchMetric** - Similarity-based matching with configurable threshold  
                - **WordOverlapMetric** - Word-level overlap analysis with normalization  
                - **LengthSimilarityMetric** - Response length similarity with tolerance  
                - **ContainsKeywordsMetric** - Keyword presence checking with flexible requirements  

                **Key Features:**  
                - **Configurable Parameters** - Customize each metric for your specific needs  
                - **Robust Error Handling** - Graceful handling of missing fields and errors  
                - **Rich Metadata** - Detailed information about evaluation process  
                - **Field Flexibility** - Evaluate any field in examples and predictions  

                Ready to build composite evaluation strategies!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        ExactMatchMetric = None
        FuzzyMatchMetric = None
        WordOverlapMetric = None
        ContainsKeywordsMetric = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (
        ContainsKeywordsMetric,
        ExactMatchMetric,
        FuzzyMatchMetric,
        WordOverlapMetric,
    )


@app.cell
def _(
    EvaluationStrategy,
    ExactMatchMetric,
    FuzzyMatchMetric,
    WordOverlapMetric,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
):
    if available_providers and EvaluationStrategy:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🎛️ Step 3: Interactive Evaluation Interface

                **Build and test evaluation strategies** with interactive controls:
                """
            )
        )

        # Create sample data for demonstration
        sample_qa_examples = [
            dspy.Example(
                question="What is the capital of France?", answer="Paris"
            ).with_inputs("question"),
            dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
            dspy.Example(
                question="Who wrote Romeo and Juliet?", answer="William Shakespeare"
            ).with_inputs("question"),
            dspy.Example(
                question="What is the largest planet in our solar system?",
                answer="Jupiter",
            ).with_inputs("question"),
        ]

        # Create a simple test system
        class SimpleQASystem:
            """Simple QA system for demonstration."""

            def __init__(self):
                # Predefined answers for demo
                self.answers = {
                    "What is the capital of France?": "Paris",
                    "What is 2 + 2?": "Four",  # Slightly different answer
                    "Who wrote Romeo and Juliet?": "Shakespeare",  # Partial answer
                    "What is the largest planet in our solar system?": "Jupiter is the largest planet",  # Verbose answer
                }

            def forward(self, question):
                answer = self.answers.get(question, "I don't know")
                return dspy.Prediction(answer=answer)

        # Create evaluation strategy
        demo_strategy = EvaluationStrategy(
            name="QA Evaluation Demo",
            description="Demonstration of multi-metric QA evaluation",
        )

        # Add metrics to strategy
        demo_strategy.add_metric(
            ExactMatchMetric(field_name="answer", case_sensitive=False)
        )
        demo_strategy.add_metric(FuzzyMatchMetric(field_name="answer", threshold=0.8))
        demo_strategy.add_metric(WordOverlapMetric(field_name="answer", normalize=True))

        # Interactive controls
        metric_selection = mo.ui.multiselect(
            options=["exact_match", "fuzzy_match", "word_overlap", "contains_keywords"],
            value=["exact_match", "fuzzy_match", "word_overlap"],
            label="Select Metrics to Use",
        )

        fuzzy_threshold_slider = mo.ui.slider(
            start=0.5,
            stop=1.0,
            value=0.8,
            step=0.05,
            label="Fuzzy Match Threshold",
            show_value=True,
        )

        case_sensitive_checkbox = mo.ui.checkbox(
            value=False, label="Case Sensitive Matching"
        )

        keywords_input = mo.ui.text(
            value="Paris, Shakespeare, Jupiter",
            label="Keywords to Check (comma-separated)",
        )

        run_evaluation_button = mo.ui.run_button(
            label="🚀 Run Evaluation", kind="success"
        )

        compare_systems_button = mo.ui.run_button(
            label="📊 Compare Systems", kind="info"
        )

        evaluation_controls = mo.vstack(
            [
                mo.md("### 🎛️ Evaluation Configuration"),
                metric_selection,
                fuzzy_threshold_slider,
                case_sensitive_checkbox,
                keywords_input,
                mo.md("---"),
                mo.hstack([run_evaluation_button, compare_systems_button]),
            ]
        )

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🎛️ Interactive Evaluation Interface Created

                **Interface Features:**  
                - **Metric Selection** - Choose which metrics to apply  
                - **Parameter Configuration** - Adjust thresholds and settings  
                - **Real-time Evaluation** - Run evaluations with current settings  
                - **System Comparison** - Compare multiple systems side-by-side  

                **Sample Data:**  
                - **4 QA Examples** - Diverse question-answer pairs for testing  
                - **Demo System** - Simple QA system with varied response quality  
                - **Multiple Metrics** - Exact match, fuzzy match, and word overlap  

                Configure your evaluation parameters and run the evaluation!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        evaluation_controls = mo.md("")
        cell5_content = mo.md("")
        run_evaluation_button = None
        compare_systems_button = None
        metric_selection = None
        fuzzy_threshold_slider = None
        case_sensitive_checkbox = None
        keywords_input = None
        demo_strategy = None
        sample_qa_examples = None
        SimpleQASystem = None

    cell5_out = mo.vstack([cell5_desc, evaluation_controls, cell5_content])
    output.replace(cell5_out)
    return (
        SimpleQASystem,
        case_sensitive_checkbox,
        compare_systems_button,
        demo_strategy,
        fuzzy_threshold_slider,
        keywords_input,
        metric_selection,
        run_evaluation_button,
        sample_qa_examples,
    )


@app.cell
def _(
    ContainsKeywordsMetric,
    EvaluationStrategy,
    ExactMatchMetric,
    FuzzyMatchMetric,
    SimpleQASystem,
    WordOverlapMetric,
    case_sensitive_checkbox,
    cleandoc,
    compare_systems_button,
    demo_strategy,
    dspy,
    fuzzy_threshold_slider,
    keywords_input,
    metric_selection,
    mo,
    output,
    run_evaluation_button,
    sample_qa_examples,
):
    # Handle evaluation interactions
    evaluation_results_display = mo.md("")

    if run_evaluation_button.value and demo_strategy:
        # Configure strategy based on user inputs
        current_strategy = EvaluationStrategy(
            name="Custom Evaluation", description="User-configured evaluation strategy"
        )

        selected_metrics = metric_selection.value
        fuzzy_threshold = fuzzy_threshold_slider.value
        case_sensitive = case_sensitive_checkbox.value
        keywords = [k.strip() for k in keywords_input.value.split(",") if k.strip()]

        # Add selected metrics
        if "exact_match" in selected_metrics:
            current_strategy.add_metric(ExactMatchMetric(case_sensitive=case_sensitive))

        if "fuzzy_match" in selected_metrics:
            current_strategy.add_metric(FuzzyMatchMetric(threshold=fuzzy_threshold))

        if "word_overlap" in selected_metrics:
            current_strategy.add_metric(WordOverlapMetric())

        if "contains_keywords" in selected_metrics and keywords:
            current_strategy.add_metric(ContainsKeywordsMetric(keywords=keywords))

        # Run evaluation
        test_system = SimpleQASystem()
        results = current_strategy.evaluate_system(test_system, sample_qa_examples)

        # Display results
        results_text = []
        for metric_name, summary in results.items():
            results_text.append(f"**{metric_name}:**")
            results_text.append(f"  - Mean Score: {summary.mean_score:.3f}")
            results_text.append(f"  - Std Dev: {summary.std_deviation:.3f}")
            results_text.append(
                f"  - Range: {summary.min_score:.3f} - {summary.max_score:.3f}"
            )
            results_text.append("")

        evaluation_results_display = mo.md(
            cleandoc(
                f"""
                ## 📊 Evaluation Results

                **Strategy:** {current_strategy.name}  
                **Examples Evaluated:** {len(sample_qa_examples)}  
                **Metrics Used:** {len(current_strategy.metrics)}  

                ### 📈 Metric Scores

                {chr(10).join(results_text)}

                ### 💡 Analysis

                **Best Performing Metric:** {max(results.items(), key=lambda x: x[1].mean_score)[0]}  
                **Most Consistent Metric:** {min(results.items(), key=lambda x: x[1].std_deviation)[0]}  

                The evaluation shows how different metrics capture different aspects of system performance!
                """
            )
        )

    elif compare_systems_button.value and demo_strategy:
        # Create multiple systems for comparison
        class PerfectQASystem:
            """Perfect QA system that always gives correct answers."""

            def forward(self, question):
                perfect_answers = {
                    "What is the capital of France?": "Paris",
                    "What is 2 + 2?": "4",
                    "Who wrote Romeo and Juliet?": "William Shakespeare",
                    "What is the largest planet in our solar system?": "Jupiter",
                }
                answer = perfect_answers.get(question, "Perfect answer")
                return dspy.Prediction(answer=answer)

        class RandomQASystem:
            """Random QA system for comparison."""

            def forward(self, question):
                import random

                random_answers = ["Maybe", "I think so", "Not sure", "Could be"]
                answer = random.choice(random_answers)
                return dspy.Prediction(answer=answer)

        # Compare systems
        systems = {
            "Simple System": SimpleQASystem(),
            "Perfect System": PerfectQASystem(),
            "Random System": RandomQASystem(),
        }

        comparison_results = demo_strategy.compare_systems(systems, sample_qa_examples)

        # Display comparison
        comparison_text = []
        for system_name, system_results in comparison_results[
            "individual_results"
        ].items():
            comparison_text.append(f"**{system_name}:**")
            for metric_name, summary in system_results.items():
                comparison_text.append(f"  - {metric_name}: {summary.mean_score:.3f}")
            comparison_text.append("")

        # Best performers
        best_performers_text = []
        for metric_name, best_data in comparison_results["analysis"][
            "best_performers"
        ].items():
            best_performers_text.append(
                f"- **{metric_name}**: {best_data['system']} ({best_data['score']:.3f})"
            )

        evaluation_results_display = mo.md(
            cleandoc(
                f"""
                ## 🏆 System Comparison Results

                **Systems Compared:** {comparison_results['num_systems']}  
                **Examples Used:** {comparison_results['num_examples']}  

                ### 📊 Individual System Performance

                {chr(10).join(comparison_text)}

                ### 🥇 Best Performers by Metric

                {chr(10).join(best_performers_text)}

                ### 💡 Insights

                {chr(10).join(f"- {insight}" for insight in comparison_results["analysis"]["insights"])}

                System comparison reveals strengths and weaknesses across different evaluation dimensions!
                """
            )
        )

    output.replace(evaluation_results_display)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell7_out = mo.md(
        cleandoc(
            """
            ## 🎯 Advanced Evaluation Strategies

            ### 🏆 Best Practices for Evaluation Framework Design

            **Strategy Design Principles:**  
            - **Multi-Dimensional Assessment** - Use multiple metrics to capture different quality aspects  
            - **Statistical Rigor** - Apply proper statistical methods for reliable results  
            - **Error Analysis** - Understand failure modes and edge cases  
            - **Reproducibility** - Ensure evaluations can be consistently reproduced  

            **Metric Selection Guidelines:**  
            - **Task Alignment** - Choose metrics that align with your specific task requirements  
            - **Complementary Metrics** - Use metrics that capture different aspects of quality  
            - **Balanced Scoring** - Avoid metrics that are too easy or too hard to satisfy  
            - **Domain Expertise** - Incorporate domain knowledge into metric design  

            ### ⚡ Evaluation Workflow Optimization

            **Efficient Evaluation:**  
            - **Batch Processing** - Evaluate multiple examples simultaneously  
            - **Caching** - Cache expensive computations like embeddings  
            - **Parallel Execution** - Run independent evaluations in parallel  
            - **Progressive Evaluation** - Start with fast metrics, then apply expensive ones  

            **Quality Assurance:**  
            - **Validation Sets** - Use separate validation data for evaluation  
            - **Cross-Validation** - Apply k-fold cross-validation for robust results  
            - **Statistical Testing** - Use significance tests for system comparisons  
            - **Confidence Intervals** - Report confidence intervals with scores  

            ### 🔧 Advanced Evaluation Techniques

            **Adaptive Evaluation:**  
            - **Dynamic Thresholds** - Adjust thresholds based on task difficulty  
            - **Context-Aware Metrics** - Consider context in evaluation decisions  
            - **Learning Metrics** - Metrics that improve based on feedback  
            - **Multi-Stage Evaluation** - Progressive evaluation with increasing complexity  

            **Ensemble Evaluation:**  
            - **Metric Ensembles** - Combine multiple metrics with learned weights  
            - **Evaluator Consensus** - Use multiple evaluators and aggregate results  
            - **Human-AI Hybrid** - Combine automated metrics with human judgment  
            - **Uncertainty Quantification** - Measure and report evaluation uncertainty  

            ### 🚀 Production Evaluation Systems

            **Scalability:**  
            - **Distributed Evaluation** - Scale evaluation across multiple machines  
            - **Stream Processing** - Evaluate data streams in real-time  
            - **Resource Management** - Optimize compute and memory usage  
            - **Monitoring** - Track evaluation system health and performance  

            **Integration:**  
            - **CI/CD Integration** - Automated evaluation in deployment pipelines  
            - **A/B Testing** - Continuous evaluation of system variants  
            - **Feedback Loops** - Use evaluation results to improve systems  
            - **Alerting** - Automated alerts for performance degradation  

            ### 💡 Next Steps

            **Apply Your Evaluation Framework:**  
            1. **Define Success Criteria** - Clearly specify what constitutes good performance  
            2. **Select Appropriate Metrics** - Choose metrics that align with your goals  
            3. **Build Evaluation Pipelines** - Create automated evaluation workflows  
            4. **Monitor and Iterate** - Continuously improve your evaluation approach  

            Congratulations on mastering comprehensive evaluation frameworks! 🎉
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
