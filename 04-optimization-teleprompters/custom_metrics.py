# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import difflib
    import json
    import math
    import re
    import sys
    import time
    from collections.abc import Callable
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
        Any,
        Callable,
        Optional,
        Union,
        cleandoc,
        difflib,
        dspy,
        get_config,
        json,
        math,
        mo,
        output,
        re,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 Custom Metrics System

            **Duration:** 90-120 minutes  
            **Prerequisites:** Completed BootstrapFewShot and MIPRO modules  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Master custom metrics creation for domain-specific evaluation  
            - ✅ Implement metric design patterns and templates  
            - ✅ Build metric validation and testing frameworks  
            - ✅ Create metric performance analysis tools  
            - ✅ Understand advanced evaluation strategies  

            ## 📈 Custom Metrics Overview

            **Why Custom Metrics Matter:**  
            - **Domain Specificity** - Standard metrics don't capture domain nuances  
            - **Task Alignment** - Metrics should match your specific success criteria  
            - **Business Value** - Measure what actually matters for your application  
            - **Optimization Quality** - Better metrics lead to better optimization results  

            **Types of Custom Metrics:**  
            - **Accuracy-Based** - Exact match, fuzzy match, semantic similarity  
            - **Quality-Based** - Fluency, coherence, completeness, relevance  
            - **Task-Specific** - Domain knowledge, factual accuracy, style adherence  
            - **Composite** - Multi-dimensional evaluation combining multiple aspects  

            Let's build a comprehensive custom metrics system!
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
        setup_dspy_environment()
        cell2_out = mo.md(
            cleandoc(
                f"""
                ## ✅ Custom Metrics Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**

                Ready to build advanced evaluation systems!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first to configure your API keys.
                """
            )
        )

    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(
    Any,
    Callable,
    Optional,
    Union,
    available_providers,
    cleandoc,
    difflib,
    dspy,
    math,
    mo,
    output,
    re,
):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Core Metric Design Patterns

                **Building foundational patterns** for creating robust custom metrics:
                """
            )
        )

        class MetricTemplate:
            """Base template for creating custom metrics with common patterns."""

            @staticmethod
            def exact_match(
                example, pred, trace=None, field_name: str = "answer"
            ) -> float:
                """Exact string matching metric."""
                try:
                    expected = getattr(example, field_name, "").strip().lower()
                    predicted = getattr(pred, field_name, "").strip().lower()
                    return 1.0 if expected == predicted else 0.0
                except Exception as _:
                    return 0.0

            @staticmethod
            def fuzzy_match(
                example,
                pred,
                trace=None,
                field_name: str = "answer",
                threshold: float = 0.8,
            ) -> float:
                """Fuzzy string matching with configurable threshold."""
                try:
                    expected = getattr(example, field_name, "").strip().lower()
                    predicted = getattr(pred, field_name, "").strip().lower()

                    if not expected or not predicted:
                        return 0.0

                    similarity = difflib.SequenceMatcher(
                        None, expected, predicted
                    ).ratio()
                    return 1.0 if similarity >= threshold else similarity
                except Exception as _:
                    return 0.0

            @staticmethod
            def word_overlap(
                example, pred, trace=None, field_name: str = "answer"
            ) -> float:
                """Word overlap similarity metric."""
                try:
                    expected = set(getattr(example, field_name, "").lower().split())
                    predicted = set(getattr(pred, field_name, "").lower().split())

                    if not expected:
                        return 0.0

                    intersection = expected.intersection(predicted)
                    return len(intersection) / len(expected)
                except Exception as _:
                    return 0.0

            @staticmethod
            def keyword_presence(keywords: list[str], weight: float = 1.0):
                """Create a metric that checks for presence of specific keywords."""

                def metric(
                    example, pred, trace=None, field_name: str = "answer"
                ) -> float:
                    try:
                        predicted_text = getattr(pred, field_name, "").lower()
                        found_keywords = sum(
                            1
                            for keyword in keywords
                            if keyword.lower() in predicted_text
                        )
                        score = (
                            (found_keywords / len(keywords)) * weight
                            if keywords
                            else 0.0
                        )
                        return min(1.0, score)
                    except Exception as _:
                        return 0.0

                return metric

        class CompositeMetric:
            """Combine multiple metrics with weights for comprehensive evaluation."""

            def __init__(
                self, metrics: dict[str, Callable], weights: dict[str, float] = None
            ):
                self.metrics = metrics
                self.weights = weights or dict.fromkeys(metrics.keys(), 1.0)

                # Normalize weights
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    self.weights = {
                        name: weight / total_weight
                        for name, weight in self.weights.items()
                    }

            def __call__(self, example, pred, trace=None) -> float:
                """Evaluate using all metrics and return weighted average."""
                try:
                    total_score = 0.0
                    for name, metric in self.metrics.items():
                        score = metric(example, pred, trace)
                        weighted_score = score * self.weights.get(name, 0.0)
                        total_score += weighted_score

                    return min(1.0, max(0.0, total_score))
                except Exception as _:
                    return 0.0

        class DomainSpecificMetrics:
            """Collection of domain-specific metric creators."""

            @staticmethod
            def qa_accuracy_metric(partial_credit: bool = True):
                """Question answering accuracy with optional partial credit."""

                def metric(example, pred, trace=None) -> float:
                    try:
                        expected = getattr(example, "answer", "").strip().lower()
                        predicted = getattr(pred, "answer", "").strip().lower()

                        # Exact match
                        if expected == predicted:
                            return 1.0

                        if not partial_credit:
                            return 0.0

                        # Partial credit based on word overlap
                        expected_words = set(expected.split())
                        predicted_words = set(predicted.split())

                        if not expected_words:
                            return 0.0

                        overlap = expected_words.intersection(predicted_words)
                        return len(overlap) / len(expected_words)
                    except Exception as _:
                        return 0.0

                return metric

            @staticmethod
            def sentiment_accuracy_metric():
                """Sentiment classification accuracy with fuzzy matching."""

                def metric(example, pred, trace=None) -> float:
                    try:
                        expected = getattr(example, "sentiment", "").strip().lower()
                        predicted = getattr(pred, "sentiment", "").strip().lower()

                        # Direct match
                        if expected == predicted:
                            return 1.0

                        # Fuzzy sentiment matching
                        positive_terms = [
                            "positive",
                            "good",
                            "happy",
                            "great",
                            "excellent",
                        ]
                        negative_terms = ["negative", "bad", "sad", "terrible", "awful"]
                        neutral_terms = ["neutral", "okay", "average", "mixed"]

                        def get_sentiment_category(text):
                            if any(term in text for term in positive_terms):
                                return "positive"
                            elif any(term in text for term in negative_terms):
                                return "negative"
                            elif any(term in text for term in neutral_terms):
                                return "neutral"
                            return text

                        expected_category = get_sentiment_category(expected)
                        predicted_category = get_sentiment_category(predicted)

                        return 1.0 if expected_category == predicted_category else 0.0
                    except Exception as _:
                        return 0.0

                return metric

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Core Metric Design Patterns Created

                **Pattern Categories:**  
                - **MetricTemplate** - Basic patterns (exact match, fuzzy match, word overlap)  
                - **CompositeMetric** - Combine multiple metrics with weights  
                - **DomainSpecificMetrics** - Pre-built metrics for common domains  

                **Key Features:**  
                - **Configurable Thresholds** - Adjust sensitivity for different use cases  
                - **Partial Credit** - Reward partially correct answers  
                - **Error Handling** - Graceful degradation when evaluation fails  

                These patterns provide the foundation for creating robust custom metrics!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        MetricTemplate = None
        CompositeMetric = None
        DomainSpecificMetrics = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return CompositeMetric, DomainSpecificMetrics, MetricTemplate


@app.cell
def _(
    CompositeMetric,
    DomainSpecificMetrics,
    MetricTemplate,
    available_providers,
    cleandoc,
    mo,
    output,
):
    if available_providers and MetricTemplate:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🎛️ Step 2: Interactive Metric Builder

                **Build and test custom metrics** with interactive controls:
                """
            )
        )

        # Metric type selection
        metric_type_dropdown = mo.ui.dropdown(
            options=["qa", "sentiment", "composite"],
            value="qa",
            label="Metric Type",
        )

        # Metric parameters
        fuzzy_threshold_slider = mo.ui.slider(
            start=0.5,
            stop=1.0,
            value=0.8,
            step=0.05,
            label="Fuzzy Match Threshold",
            show_value=True,
        )

        partial_credit_checkbox = mo.ui.checkbox(
            value=True,
            label="Enable Partial Credit",
        )

        # Composite metric weights
        accuracy_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.5,
            step=0.1,
            label="Accuracy Weight",
            show_value=True,
        )

        quality_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.3,
            step=0.1,
            label="Quality Weight",
            show_value=True,
        )

        style_weight_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=0.2,
            step=0.1,
            label="Style Weight",
            show_value=True,
        )

        # Actions
        build_metric_button = mo.ui.button(
            label="🏗️ Build Custom Metric",
            kind="success",
        )

        test_metric_button = mo.ui.button(
            label="🧪 Test Metric",
            kind="primary",
        )

        metric_builder_ui = mo.vstack(
            [
                mo.md("### 🎛️ Custom Metric Builder"),
                metric_type_dropdown,
                mo.md("---"),
                mo.md("**Basic Parameters:**"),
                fuzzy_threshold_slider,
                partial_credit_checkbox,
                mo.md("**Composite Weights:**"),
                accuracy_weight_slider,
                quality_weight_slider,
                style_weight_slider,
                mo.md("---"),
                mo.hstack([build_metric_button, test_metric_button]),
            ]
        )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🎛️ Interactive Metric Builder Created

                **Builder Features:**  
                - **Metric Type Selection** - Choose from QA, sentiment, or composite  
                - **Parameter Configuration** - Adjust thresholds and weights  
                - **Real-time Building** - Create metrics with your specifications  
                - **Testing Interface** - Test metrics with sample data  

                Configure your metric parameters and build custom evaluation functions!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        metric_builder_ui = mo.md("")
        cell4_content = mo.md("")
        build_metric_button = None
        test_metric_button = None
        metric_type_dropdown = None
        fuzzy_threshold_slider = None
        partial_credit_checkbox = None
        accuracy_weight_slider = None
        quality_weight_slider = None
        style_weight_slider = None

    cell4_out = mo.vstack([cell4_desc, metric_builder_ui, cell4_content])
    output.replace(cell4_out)
    return (
        accuracy_weight_slider,
        build_metric_button,
        fuzzy_threshold_slider,
        metric_type_dropdown,
        partial_credit_checkbox,
        quality_weight_slider,
        style_weight_slider,
        test_metric_button,
    )


@app.cell
def _(
    CompositeMetric,
    DomainSpecificMetrics,
    MetricTemplate,
    accuracy_weight_slider,
    build_metric_button,
    cleandoc,
    dspy,
    fuzzy_threshold_slider,
    metric_type_dropdown,
    mo,
    output,
    partial_credit_checkbox,
    quality_weight_slider,
    style_weight_slider,
):
    # Build custom metric when button is clicked
    custom_metric = None
    metric_build_status = mo.md("")

    if build_metric_button and build_metric_button.value:
        metric_type = metric_type_dropdown.value
        fuzzy_threshold = fuzzy_threshold_slider.value
        partial_credit = partial_credit_checkbox.value

        try:
            if metric_type == "qa":
                custom_metric = DomainSpecificMetrics.qa_accuracy_metric(
                    partial_credit=partial_credit
                )
                metric_name = f"QA Accuracy (Partial Credit: {partial_credit})"

            elif metric_type == "sentiment":
                custom_metric = DomainSpecificMetrics.sentiment_accuracy_metric()
                metric_name = "Sentiment Analysis Accuracy"

            elif metric_type == "composite":
                # Build composite metric with user weights
                accuracy_weight = accuracy_weight_slider.value
                quality_weight = quality_weight_slider.value
                style_weight = style_weight_slider.value

                # Normalize weights
                total_weight = accuracy_weight + quality_weight + style_weight
                if total_weight > 0:
                    accuracy_weight /= total_weight
                    quality_weight /= total_weight
                    style_weight /= total_weight

                metrics = {
                    "accuracy": lambda e, p, t: MetricTemplate.exact_match(e, p, t),
                    "quality": lambda e, p, t: MetricTemplate.fuzzy_match(
                        e, p, t, threshold=fuzzy_threshold
                    ),
                    "style": lambda e, p, t: MetricTemplate.word_overlap(e, p, t),
                }

                weights = {
                    "accuracy": accuracy_weight,
                    "quality": quality_weight,
                    "style": style_weight,
                }

                custom_metric = CompositeMetric(metrics, weights)
                metric_name = f"Composite (A:{accuracy_weight:.1f}, Q:{quality_weight:.1f}, S:{style_weight:.1f})"

            metric_build_status = mo.md(
                cleandoc(
                    f"""
                    ## ✅ Custom Metric Built Successfully!

                    **Metric Type:** {metric_type.title()}  
                    **Metric Name:** {metric_name}  

                    ### 📊 Configuration
                    - **Fuzzy Threshold:** {fuzzy_threshold:.2f}  
                    - **Partial Credit:** {partial_credit}  

                    Your custom metric is ready for testing and use!
                    """
                )
            )

        except Exception as e:
            metric_build_status = mo.md(
                cleandoc(
                    f"""
                    ## ❌ Metric Build Failed

                    **Error:** {str(e)}  

                    Please check your configuration and try again.
                    """
                )
            )

    output.replace(metric_build_status)
    return custom_metric, metric_name


@app.cell
def _(
    cleandoc,
    custom_metric,
    dspy,
    metric_type_dropdown,
    mo,
    output,
    test_metric_button,
):
    # Test custom metric when button is clicked
    test_results = mo.md("")

    if test_metric_button and test_metric_button.value and custom_metric:
        metric_type = metric_type_dropdown.value

        try:
            # Create test cases based on metric type
            if metric_type == "qa":
                test_cases = [
                    {
                        "name": "Exact Match",
                        "example": dspy.Example(question="What is 2+2?", answer="4"),
                        "prediction": dspy.Prediction(answer="4"),
                    },
                    {
                        "name": "Partial Match",
                        "example": dspy.Example(
                            question="What is the capital of France?",
                            answer="Paris is the capital",
                        ),
                        "prediction": dspy.Prediction(answer="Paris"),
                    },
                    {
                        "name": "No Match",
                        "example": dspy.Example(question="What is 2+2?", answer="4"),
                        "prediction": dspy.Prediction(answer="5"),
                    },
                ]
            elif metric_type == "sentiment":
                test_cases = [
                    {
                        "name": "Exact Sentiment",
                        "example": dspy.Example(
                            text="I love this!", sentiment="positive"
                        ),
                        "prediction": dspy.Prediction(sentiment="positive"),
                    },
                    {
                        "name": "Fuzzy Sentiment",
                        "example": dspy.Example(
                            text="Great product", sentiment="positive"
                        ),
                        "prediction": dspy.Prediction(sentiment="good"),
                    },
                    {
                        "name": "Wrong Sentiment",
                        "example": dspy.Example(text="Terrible", sentiment="negative"),
                        "prediction": dspy.Prediction(sentiment="positive"),
                    },
                ]
            else:  # composite
                test_cases = [
                    {
                        "name": "High Quality",
                        "example": dspy.Example(
                            question="Test", answer="correct answer"
                        ),
                        "prediction": dspy.Prediction(answer="correct answer"),
                    },
                    {
                        "name": "Medium Quality",
                        "example": dspy.Example(
                            question="Test", answer="correct answer"
                        ),
                        "prediction": dspy.Prediction(answer="correct response"),
                    },
                    {
                        "name": "Low Quality",
                        "example": dspy.Example(
                            question="Test", answer="correct answer"
                        ),
                        "prediction": dspy.Prediction(answer="wrong"),
                    },
                ]

            # Run tests
            test_results_list = []
            for test_case in test_cases:
                score = custom_metric(test_case["example"], test_case["prediction"])
                test_results_list.append(f"- **{test_case['name']}**: {score:.3f}")

            test_results = mo.md(
                cleandoc(
                    f"""
                    ## 🧪 Metric Test Results

                    **Metric Type:** {metric_type.title()}  
                    **Test Cases:** {len(test_cases)}  

                    ### 📊 Test Scores
                    {chr(10).join(test_results_list)}

                    ### 💡 Analysis
                    - **Score Range**: {min(custom_metric(tc["example"], tc["prediction"]) for tc in test_cases):.3f} - {max(custom_metric(tc["example"], tc["prediction"]) for tc in test_cases):.3f}  
                    - **Average Score**: {sum(custom_metric(tc["example"], tc["prediction"]) for tc in test_cases) / len(test_cases):.3f}  

                    Your metric is working correctly! Ready for optimization use.
                    """
                )
            )

        except Exception as e:
            test_results = mo.md(
                cleandoc(
                    f"""
                    ## ❌ Metric Test Failed

                    **Error:** {str(e)}  

                    Please check your metric configuration.
                    """
                )
            )

    output.replace(test_results)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell6_out = mo.md(
        cleandoc(
            """
            ## 🎯 Advanced Metric Strategies

            ### 🏆 Best Practices for Custom Metrics

            **Design Principles:**  
            - **Task Alignment** - Metrics should directly measure what matters for your task  
            - **Balanced Scoring** - Avoid metrics that are too easy or too hard to satisfy  
            - **Partial Credit** - Reward partially correct answers when appropriate  
            - **Robustness** - Handle edge cases and missing data gracefully  

            **Common Patterns:**  
            - **Exact + Fuzzy** - Combine exact matching with fuzzy similarity  
            - **Multi-Dimensional** - Evaluate different aspects (accuracy, quality, style)  
            - **Weighted Composite** - Balance multiple criteria with appropriate weights  
            - **Domain-Specific** - Include domain knowledge and business rules  

            ### ⚡ Performance Optimization

            **Speed Optimization:**  
            - **Early Returns** - Return immediately for obvious cases (exact matches)  
            - **Caching** - Cache expensive computations (embeddings, similarity scores)  
            - **Vectorization** - Use batch operations when possible  
            - **Lazy Evaluation** - Only compute what's needed  

            ### 🔧 Advanced Techniques

            **Adaptive Metrics:**  
            - **Difficulty Adjustment** - Scale scores based on question difficulty  
            - **Context Awareness** - Consider task context in evaluation  
            - **Learning Metrics** - Metrics that improve based on feedback  

            **Ensemble Metrics:**  
            - **Multiple Evaluators** - Combine different evaluation approaches  
            - **Voting Systems** - Use majority vote or weighted consensus  
            - **Confidence Weighting** - Weight scores by prediction confidence  

            ### 🚀 Production Deployment

            **Monitoring:**  
            - **Score Distributions** - Track metric score patterns over time  
            - **Performance Metrics** - Monitor evaluation speed and reliability  
            - **Edge Case Detection** - Identify unusual inputs or outputs  

            **A/B Testing:**  
            - **Metric Comparison** - Test different metrics on the same data  
            - **Business Impact** - Measure how metric changes affect outcomes  
            - **User Feedback** - Correlate metrics with user satisfaction  

            ### 💡 Next Steps

            **Apply Your Custom Metrics:**  
            1. **Integrate with Optimization** - Use in BootstrapFewShot and MIPRO  
            2. **Build Dashboards** - Create monitoring and analysis tools  
            3. **Scale to Production** - Deploy with proper monitoring  
            4. **Iterate and Improve** - Continuously refine based on feedback  

            Congratulations on mastering custom metrics creation! 🎉
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell6_out)
    return
