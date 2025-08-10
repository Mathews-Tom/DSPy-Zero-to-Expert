# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import difflib
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return (
        Any,
        cleandoc,
        difflib,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🎯 BootstrapFewShot Exercise Solutions

            **Complete reference implementations** for all BootstrapFewShot optimization exercises.

            ## 📚 Solutions Overview

            This notebook contains:  
            - **Exercise 1** - Sentiment Analysis Module with optimization  
            - **Exercise 2** - Custom Evaluation Metrics implementation  
            - **Exercise 3** - Parameter Tuning with grid search  
            - **Exercise 4** - Optimization Analysis Dashboard  

            Study these solutions to understand best practices and implementation patterns!
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
                ## ✅ Solutions Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**

                Ready to explore solutions!
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
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Solution 1: Sentiment Analysis Module

                **Complete implementation** of a sentiment analysis module optimized with BootstrapFewShot:
                """
            )
        )

        # Solution 1: Sentiment Analysis Module
        class SentimentSignature(dspy.Signature):
            """Analyze the sentiment of the given text."""

            text = dspy.InputField(desc="Text to analyze for sentiment")
            sentiment = dspy.OutputField(
                desc="Sentiment classification: positive, negative, or neutral"
            )

        class SentimentAnalyzer(dspy.Module):
            """Sentiment analysis module with chain of thought reasoning."""

            def __init__(self):
                super().__init__()
                self.analyze_sentiment = dspy.ChainOfThought(SentimentSignature)

            def forward(self, text):
                result = self.analyze_sentiment(text=text)
                return dspy.Prediction(
                    sentiment=result.sentiment, reasoning=result.rationale
                )

        # Training examples for sentiment analysis
        sentiment_training_examples = [
            dspy.Example(
                text="I absolutely love this product! It's amazing and works perfectly.",
                sentiment="positive",
            ).with_inputs("text"),
            dspy.Example(
                text="This is terrible. I hate it and want my money back.",
                sentiment="negative",
            ).with_inputs("text"),
            dspy.Example(
                text="The product is okay. Nothing special but it works.",
                sentiment="neutral",
            ).with_inputs("text"),
            dspy.Example(
                text="Fantastic service! Highly recommend to everyone.",
                sentiment="positive",
            ).with_inputs("text"),
            dspy.Example(
                text="Worst experience ever. Complete waste of time and money.",
                sentiment="negative",
            ).with_inputs("text"),
            dspy.Example(
                text="It's an average product. Does what it's supposed to do.",
                sentiment="neutral",
            ).with_inputs("text"),
        ]

        # Evaluation metric for sentiment analysis
        def sentiment_metric(example, pred, trace=None):
            """Evaluate sentiment prediction accuracy."""
            try:
                predicted_sentiment = pred.sentiment.strip().lower()
                expected_sentiment = example.sentiment.strip().lower()

                # Exact match
                if predicted_sentiment == expected_sentiment:
                    return 1.0

                # Partial credit for reasonable interpretations
                positive_terms = ["positive", "good", "great", "excellent", "happy"]
                negative_terms = ["negative", "bad", "terrible", "awful", "sad"]
                neutral_terms = ["neutral", "okay", "average", "mixed"]

                pred_category = None
                if any(term in predicted_sentiment for term in positive_terms):
                    pred_category = "positive"
                elif any(term in predicted_sentiment for term in negative_terms):
                    pred_category = "negative"
                elif any(term in predicted_sentiment for term in neutral_terms):
                    pred_category = "neutral"

                return 1.0 if pred_category == expected_sentiment else 0.0

            except Exception:
                return 0.0

        # Demonstrate optimization
        def optimize_sentiment_analyzer():
            """Optimize the sentiment analyzer with BootstrapFewShot."""
            # Create baseline module
            baseline_analyzer = SentimentAnalyzer()

            # Create optimizer
            optimizer = dspy.BootstrapFewShot(
                metric=sentiment_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=6,
                max_rounds=1,
            )

            # Optimize the module
            optimized_analyzer = optimizer.compile(
                baseline_analyzer, trainset=sentiment_training_examples
            )

            return baseline_analyzer, optimized_analyzer

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🎯 Solution 1 Complete

                **Implementation Features:**  
                - **SentimentSignature** - Clear input/output specification  
                - **SentimentAnalyzer** - ChainOfThought reasoning for better accuracy  
                - **Training Examples** - Diverse sentiment examples covering all categories  
                - **Smart Metric** - Handles exact matches and reasonable interpretations  
                - **Optimization Function** - Complete BootstrapFewShot setup  

                The solution demonstrates proper DSPy module structure and optimization!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        SentimentSignature = None
        SentimentAnalyzer = None
        sentiment_training_examples = []
        sentiment_metric = None
        optimize_sentiment_analyzer = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return


@app.cell
def _(available_providers, cleandoc, difflib, dspy, mo, output):
    if available_providers:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🔧 Solution 2: Custom Evaluation Metrics

                **Advanced evaluation metrics** for different optimization scenarios:
                """
            )
        )

        # Solution 2: Custom Evaluation Metrics
        def fuzzy_match_metric(example, pred, trace=None):
            """Fuzzy string matching metric with tolerance for minor differences."""
            try:
                predicted_text = pred.answer.strip().lower()
                expected_text = example.answer.strip().lower()

                # Use difflib for sequence matching
                similarity = difflib.SequenceMatcher(
                    None, predicted_text, expected_text
                ).ratio()

                # Return similarity score (0.0 to 1.0)
                return similarity

            except Exception:
                return 0.0

        def confidence_weighted_metric(example, pred, trace=None):
            """Metric that weights accuracy by prediction confidence."""
            try:
                # Calculate base accuracy
                predicted_answer = pred.answer.strip().lower()
                expected_answer = example.answer.strip().lower()
                accuracy = 1.0 if predicted_answer == expected_answer else 0.0

                # Extract confidence from reasoning or use length as proxy
                confidence = 1.0  # Default confidence

                if hasattr(pred, "reasoning") and pred.reasoning:
                    reasoning = pred.reasoning.lower()
                    # Look for confidence indicators
                    if any(
                        phrase in reasoning
                        for phrase in ["certain", "confident", "sure", "definitely"]
                    ):
                        confidence = 1.0
                    elif any(
                        phrase in reasoning
                        for phrase in ["maybe", "possibly", "might", "uncertain"]
                    ):
                        confidence = 0.7
                    elif any(
                        phrase in reasoning
                        for phrase in ["unsure", "don't know", "unclear"]
                    ):
                        confidence = 0.5

                # Weight accuracy by confidence
                return accuracy * confidence

            except Exception:
                return 0.0

        def multi_criteria_metric(example, pred, trace=None):
            """Multi-criteria evaluation combining accuracy, fluency, and relevance."""
            try:
                predicted_answer = pred.answer.strip()
                expected_answer = example.answer.strip()

                # Accuracy component (50%)
                accuracy = (
                    1.0
                    if predicted_answer.lower() == expected_answer.lower()
                    else difflib.SequenceMatcher(
                        None, predicted_answer.lower(), expected_answer.lower()
                    ).ratio()
                )

                # Fluency component (30%) - based on length and structure
                pred_words = predicted_answer.split()
                expected_words = expected_answer.split()

                # Penalize very short or very long answers
                length_ratio = len(pred_words) / max(len(expected_words), 1)
                fluency = 1.0 - abs(1.0 - length_ratio) * 0.5

                # Bonus for complete sentences
                if predicted_answer.endswith((".", "!", "?")):
                    fluency += 0.1

                fluency = min(1.0, max(0.0, fluency))

                # Relevance component (20%) - word overlap
                pred_words_set = {w.lower() for w in pred_words}
                expected_words_set = {w.lower() for w in expected_words}

                if expected_words_set:
                    relevance = len(
                        pred_words_set.intersection(expected_words_set)
                    ) / len(expected_words_set)
                else:
                    relevance = 0.0

                # Combined score
                final_score = accuracy * 0.5 + fluency * 0.3 + relevance * 0.2
                return min(1.0, max(0.0, final_score))

            except Exception:
                return 0.0

        # Test the metrics
        def test_custom_metrics():
            """Test all custom metrics with sample data."""
            test_example = dspy.Example(
                question="What is Python?",
                answer="Python is a programming language",
            )

            test_predictions = [
                dspy.Prediction(
                    answer="Python is a programming language",
                    reasoning="I am certain this is correct.",
                ),
                dspy.Prediction(
                    answer="Python is a high-level programming language",
                    reasoning="I think this is right but not completely sure.",
                ),
                dspy.Prediction(
                    answer="Python programming",
                    reasoning="I'm unsure about the complete answer.",
                ),
            ]

            results = []
            for i, pred in enumerate(test_predictions):
                fuzzy_score = fuzzy_match_metric(test_example, pred)
                confidence_score = confidence_weighted_metric(test_example, pred)
                multi_score = multi_criteria_metric(test_example, pred)

                results.append(
                    {
                        "prediction": pred.answer,
                        "fuzzy_match": fuzzy_score,
                        "confidence_weighted": confidence_score,
                        "multi_criteria": multi_score,
                    }
                )

            return results

        metric_test_results = test_custom_metrics()

        cell4_content = mo.md(
            cleandoc(
                f"""
                ### 🔧 Solution 2 Complete

                **Metric Implementations:**  
                - **Fuzzy Match** - Uses difflib for sequence similarity  
                - **Confidence Weighted** - Extracts confidence from reasoning  
                - **Multi-Criteria** - Combines accuracy (50%), fluency (30%), relevance (20%)  

                **Test Results:**  
                {chr(10).join([f"- Prediction: '{r['prediction']}' → Fuzzy: {r['fuzzy_match']:.3f}, Confidence: {r['confidence_weighted']:.3f}, Multi: {r['multi_criteria']:.3f}" for r in metric_test_results])}

                These metrics provide nuanced evaluation beyond simple exact matching!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        fuzzy_match_metric = None
        confidence_weighted_metric = None
        multi_criteria_metric = None
        test_custom_metrics = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output, time):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## ⚙️ Solution 3: Optimization Parameter Tuning

                **Systematic parameter exploration** with grid search and analysis:
                """
            )
        )

        # Solution 3: Parameter Tuning with Grid Search
        class SimpleQASignature(dspy.Signature):
            """Answer questions based on context."""

            context = dspy.InputField(desc="Context information")
            question = dspy.InputField(desc="Question to answer")
            answer = dspy.OutputField(desc="Answer based on context")

        class SimpleQAModule(dspy.Module):
            """Simple question answering module."""

            def __init__(self):
                super().__init__()
                self.generate_answer = dspy.ChainOfThought(SimpleQASignature)

            def forward(self, context, question):
                result = self.generate_answer(context=context, question=question)
                return dspy.Prediction(answer=result.answer, reasoning=result.rationale)

        # Training data for QA
        qa_training_data = [
            dspy.Example(
                context="Python is a high-level programming language known for its simplicity.",
                question="What is Python known for?",
                answer="Python is known for its simplicity",
            ).with_inputs("context", "question"),
            dspy.Example(
                context="Machine learning uses algorithms to find patterns in data.",
                question="What does machine learning use?",
                answer="Machine learning uses algorithms",
            ).with_inputs("context", "question"),
            dspy.Example(
                context="DSPy provides systematic optimization of language model prompts.",
                question="What does DSPy provide?",
                answer="DSPy provides systematic optimization",
            ).with_inputs("context", "question"),
        ]

        # Simple QA metric
        def qa_metric(example, pred, trace=None):
            """Simple QA evaluation metric."""
            try:
                pred_words = set(pred.answer.lower().split())
                expected_words = set(example.answer.lower().split())
                if expected_words:
                    return len(pred_words.intersection(expected_words)) / len(
                        expected_words
                    )
                return 0.0
            except Exception as _:
                return 0.0

        # Parameter grid search implementation
        def parameter_grid_search(module_class, training_data, param_grid):
            """Run optimization with different parameter combinations."""
            results = []

            for i, params in enumerate(param_grid):
                print(f"Testing configuration {i+1}/{len(param_grid)}: {params}")
                start_time = time.time()

                try:
                    # Create fresh module
                    module = module_class()

                    # Create optimizer with current parameters
                    optimizer = dspy.BootstrapFewShot(
                        metric=qa_metric,
                        max_bootstrapped_demos=params["max_bootstrapped_demos"],
                        max_labeled_demos=params["max_labeled_demos"],
                        max_rounds=params["max_rounds"],
                    )

                    # Optimize module
                    optimized_module = optimizer.compile(module, trainset=training_data)

                    # Evaluate performance
                    total_score = 0
                    for example in training_data[:2]:  # Use subset for validation
                        pred = optimized_module(
                            context=example.context, question=example.question
                        )
                        score = qa_metric(example, pred)
                        total_score += score

                    avg_score = total_score / min(2, len(training_data))
                    optimization_time = time.time() - start_time

                    results.append(
                        {
                            "parameters": params.copy(),
                            "performance": avg_score,
                            "optimization_time": optimization_time,
                            "success": True,
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "parameters": params.copy(),
                            "performance": 0.0,
                            "optimization_time": time.time() - start_time,
                            "success": False,
                            "error": str(e),
                        }
                    )

            return results

        # Parameter combinations to test
        param_combinations = [
            {"max_bootstrapped_demos": 2, "max_labeled_demos": 4, "max_rounds": 1},
            {"max_bootstrapped_demos": 3, "max_labeled_demos": 6, "max_rounds": 1},
            {"max_bootstrapped_demos": 4, "max_labeled_demos": 8, "max_rounds": 1},
            {"max_bootstrapped_demos": 2, "max_labeled_demos": 8, "max_rounds": 2},
            {"max_bootstrapped_demos": 3, "max_labeled_demos": 12, "max_rounds": 1},
        ]

        # Run optimization experiment
        def run_optimization_experiment():
            """Run the complete optimization experiment."""
            print("Starting parameter grid search...")

            # Run grid search
            results = parameter_grid_search(
                SimpleQAModule, qa_training_data, param_combinations
            )

            # Find best configuration
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                best_result = max(successful_results, key=lambda x: x["performance"])

                print(f"\nBest Configuration:")
                print(f"Parameters: {best_result['parameters']}")
                print(f"Performance: {best_result['performance']:.3f}")
                print(f"Optimization Time: {best_result['optimization_time']:.2f}s")

                # Show all results
                print(f"\nAll Results:")
                for i, result in enumerate(results):
                    status = "✅" if result["success"] else "❌"
                    print(
                        f"{status} Config {i+1}: {result['parameters']} → "
                        f"Score: {result['performance']:.3f}, "
                        f"Time: {result['optimization_time']:.2f}s"
                    )

                return results
            else:
                print("No successful optimizations found.")
                return results

        cell5_content = mo.md(
            cleandoc(
                """
                ### ⚙️ Solution 3 Complete

                **Grid Search Features:**  
                - **SimpleQAModule** - Clean QA implementation for testing  
                - **Parameter Grid** - Systematic exploration of parameter space  
                - **Performance Tracking** - Records scores and optimization times  
                - **Best Configuration** - Automatically identifies optimal settings  
                - **Error Handling** - Gracefully handles optimization failures  

                The solution provides a systematic approach to parameter optimization!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        SimpleQAModule = None
        SimpleQASignature = None
        qa_training_data = []
        parameter_grid_search = None
        run_optimization_experiment = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return


@app.cell
def _(Any, available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 📊 Solution 4: Optimization Analysis Dashboard

                **Comprehensive analysis system** for tracking and visualizing optimization results:
                """
            )
        )

        # Solution 4: Optimization Analysis Dashboard
        class OptimizationTracker:
            """Track and analyze multiple optimization runs."""

            def __init__(self):
                self.runs = []

            def add_run(
                self, parameters: dict, performance: float, metadata: dict = None
            ):
                """Add an optimization run to tracking."""
                run_data = {
                    "timestamp": time.time(),
                    "parameters": parameters.copy(),
                    "performance": performance,
                    "metadata": metadata or {},
                    "run_id": len(self.runs) + 1,
                }
                self.runs.append(run_data)

            def get_performance_trends(self) -> dict[str, Any]:
                """Analyze performance trends across runs."""
                if not self.runs:
                    return {"message": "No runs to analyze"}

                performances = [run["performance"] for run in self.runs]
                timestamps = [run["timestamp"] for run in self.runs]

                # Calculate trend
                if len(performances) > 1:
                    # Simple linear trend
                    x_vals = list(range(len(performances)))
                    n = len(x_vals)
                    sum_x = sum(x_vals)
                    sum_y = sum(performances)
                    sum_xy = sum(x * y for x, y in zip(x_vals, performances))
                    sum_x2 = sum(x * x for x in x_vals)

                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    trend = (
                        "improving"
                        if slope > 0.01
                        else "declining" if slope < -0.01 else "stable"
                    )
                else:
                    trend = "insufficient_data"

                return {
                    "total_runs": len(self.runs),
                    "average_performance": sum(performances) / len(performances),
                    "best_performance": max(performances),
                    "worst_performance": min(performances),
                    "performance_range": max(performances) - min(performances),
                    "trend": trend,
                    "latest_performance": performances[-1],
                }

            def compare_configurations(
                self, config1: dict, config2: dict
            ) -> dict[str, Any]:
                """Compare two different configurations."""
                # Find runs matching each configuration
                config1_runs = [
                    run
                    for run in self.runs
                    if all(run["parameters"].get(k) == v for k, v in config1.items())
                ]
                config2_runs = [
                    run
                    for run in self.runs
                    if all(run["parameters"].get(k) == v for k, v in config2.items())
                ]

                if not config1_runs or not config2_runs:
                    return {
                        "message": "Insufficient data for comparison",
                        "config1_runs": len(config1_runs),
                        "config2_runs": len(config2_runs),
                    }

                config1_perf = [run["performance"] for run in config1_runs]
                config2_perf = [run["performance"] for run in config2_runs]

                return {
                    "config1": {
                        "parameters": config1,
                        "runs": len(config1_runs),
                        "average_performance": sum(config1_perf) / len(config1_perf),
                        "best_performance": max(config1_perf),
                    },
                    "config2": {
                        "parameters": config2,
                        "runs": len(config2_runs),
                        "average_performance": sum(config2_perf) / len(config2_perf),
                        "best_performance": max(config2_perf),
                    },
                    "winner": (
                        "config1"
                        if sum(config1_perf) / len(config1_perf)
                        > sum(config2_perf) / len(config2_perf)
                        else "config2"
                    ),
                }

            def generate_insights(self) -> dict[str, Any]:
                """Generate insights and recommendations."""
                if len(self.runs) < 3:
                    return {"message": "Need at least 3 runs for meaningful insights"}

                insights = {
                    "parameter_analysis": {},
                    "recommendations": [],
                    "patterns": [],
                }

                # Analyze parameter impact
                param_performance = {}
                for run in self.runs:
                    for param, value in run["parameters"].items():
                        if param not in param_performance:
                            param_performance[param] = {}
                        if value not in param_performance[param]:
                            param_performance[param][value] = []
                        param_performance[param][value].append(run["performance"])

                # Calculate average performance for each parameter value
                for param, values in param_performance.items():
                    insights["parameter_analysis"][param] = {}
                    best_value = None
                    best_avg = -1

                    for value, performances in values.items():
                        avg_perf = sum(performances) / len(performances)
                        insights["parameter_analysis"][param][str(value)] = {
                            "average_performance": avg_perf,
                            "count": len(performances),
                        }

                        if avg_perf > best_avg:
                            best_avg = avg_perf
                            best_value = value

                    if best_value is not None:
                        insights["recommendations"].append(
                            f"For {param}, use value {best_value} (avg performance: {best_avg:.3f})"
                        )

                # Identify patterns
                sorted_runs = sorted(
                    self.runs, key=lambda x: x["performance"], reverse=True
                )
                top_runs = sorted_runs[: min(3, len(sorted_runs))]

                # Find common parameters in top runs
                if len(top_runs) >= 2:
                    common_params = {}
                    for run in top_runs:
                        for param, value in run["parameters"].items():
                            if param not in common_params:
                                common_params[param] = []
                            common_params[param].append(value)

                    for param, values in common_params.items():
                        if len(set(values)) == 1:  # All top runs use same value
                            insights["patterns"].append(
                                f"All top-performing runs use {param}={values[0]}"
                            )

                return insights

            def visualize_results(self) -> dict[str, Any]:
                """Create visualization data for optimization results."""
                if not self.runs:
                    return {"message": "No data to visualize"}

                # Prepare data for visualization
                viz_data = {
                    "performance_over_time": {
                        "run_ids": [run["run_id"] for run in self.runs],
                        "performances": [run["performance"] for run in self.runs],
                        "timestamps": [run["timestamp"] for run in self.runs],
                    },
                    "parameter_impact": {},
                    "summary_stats": self.get_performance_trends(),
                }

                # Parameter impact data
                param_performance = {}
                for run in self.runs:
                    for param, value in run["parameters"].items():
                        if param not in param_performance:
                            param_performance[param] = {}
                        if value not in param_performance[param]:
                            param_performance[param][value] = []
                        param_performance[param][value].append(run["performance"])

                for param, values in param_performance.items():
                    viz_data["parameter_impact"][param] = {
                        "values": list(values.keys()),
                        "avg_performances": [
                            sum(perfs) / len(perfs) for perfs in values.values()
                        ],
                    }

                return viz_data

        # Helper functions for analysis
        def create_performance_plot(tracker: OptimizationTracker) -> str:
            """Create a text-based performance plot."""
            if not tracker.runs:
                return "No data to plot"

            performances = [run["performance"] for run in tracker.runs]
            run_ids = [run["run_id"] for run in tracker.runs]

            # Simple text-based plot
            plot_lines = ["Performance Over Runs:", ""]

            max_perf = max(performances)
            min_perf = min(performances)
            range_perf = max_perf - min_perf if max_perf != min_perf else 1

            for i, (run_id, perf) in enumerate(
                zip(run_ids, performances, strict=False)
            ):
                # Normalize performance to 0-20 scale for text plot
                normalized = int((perf - min_perf) / range_perf * 20)
                bar = "█" * normalized + "░" * (20 - normalized)
                plot_lines.append(f"Run {run_id:2d}: {bar} {perf:.3f}")

            return "\n".join(plot_lines)

        def analyze_parameter_impact(
            tracker: OptimizationTracker, parameter_name: str
        ) -> dict[str, Any]:
            """Analyze the impact of a specific parameter."""
            param_data = {}

            for run in tracker.runs:
                if parameter_name in run["parameters"]:
                    value = run["parameters"][parameter_name]
                    if value not in param_data:
                        param_data[value] = []
                    param_data[value].append(run["performance"])

            if not param_data:
                return {"message": f"No data found for parameter '{parameter_name}'"}

            analysis = {}
            for value, performances in param_data.items():
                analysis[str(value)] = {
                    "count": len(performances),
                    "average": sum(performances) / len(performances),
                    "max": max(performances),
                    "min": min(performances),
                }

            # Find best value
            best_value = max(analysis.keys(), key=lambda k: analysis[k]["average"])

            return {
                "parameter": parameter_name,
                "analysis": analysis,
                "best_value": best_value,
                "best_average": analysis[best_value]["average"],
            }

        # Test the tracker
        def test_optimization_tracker():
            """Test the optimization tracking system."""
            tracker = OptimizationTracker()

            # Add sample runs
            sample_runs = [
                ({"max_bootstrapped_demos": 2, "max_labeled_demos": 8}, 0.75),
                ({"max_bootstrapped_demos": 4, "max_labeled_demos": 8}, 0.82),
                ({"max_bootstrapped_demos": 2, "max_labeled_demos": 16}, 0.78),
                ({"max_bootstrapped_demos": 4, "max_labeled_demos": 16}, 0.85),
                ({"max_bootstrapped_demos": 6, "max_labeled_demos": 8}, 0.80),
            ]

            for params, perf in sample_runs:
                tracker.add_run(params, perf, {"test_run": True})

            # Generate analysis
            trends = tracker.get_performance_trends()
            insights = tracker.generate_insights()
            viz_data = tracker.visualize_results()

            return tracker, trends, insights, viz_data

        # Run test
        test_tracker, test_trends, test_insights, test_viz = test_optimization_tracker()

        cell6_content = mo.md(
            cleandoc(
                f"""
                ### 📊 Solution 4 Complete

                **Dashboard Features:**  
                - **OptimizationTracker** - Complete tracking system with run management  
                - **Performance Trends** - Statistical analysis of optimization progress  
                - **Configuration Comparison** - Side-by-side parameter comparison  
                - **Insight Generation** - Automated recommendations and pattern detection  
                - **Visualization Support** - Data preparation for charts and plots  

                **Test Results:**  
                - **Total Runs:** {test_trends['total_runs']}  
                - **Best Performance:** {test_trends['best_performance']:.3f}  
                - **Trend:** {test_trends['trend']}  
                - **Recommendations:** {len(test_insights['recommendations'])} generated  

                The dashboard provides comprehensive optimization analysis and insights!
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        OptimizationTracker = None
        test_optimization_tracker = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell7_out = mo.md(
        cleandoc(
            """
            ## 🎓 Solutions Summary and Learning Points

            ### ✅ Key Implementation Patterns

            **1. Module Structure:**  
            - Clear signature definitions with descriptive fields  
            - ChainOfThought for better reasoning capabilities  
            - Proper prediction objects with additional metadata  

            **2. Evaluation Metrics:**  
            - Handle edge cases with try-catch blocks  
            - Provide partial credit for reasonable answers  
            - Combine multiple evaluation criteria for robustness  

            **3. Optimization Strategy:**  
            - Start with small parameter values and scale up  
            - Use systematic grid search for parameter exploration  
            - Track all experiments for comparison and analysis  

            **4. Analysis and Insights:**  
            - Store comprehensive metadata for each run  
            - Generate actionable recommendations from data  
            - Visualize trends and patterns for better understanding  

            ### 🚀 Advanced Techniques Demonstrated

            **Fuzzy Matching:**  
            - Uses `difflib.SequenceMatcher` for similarity scoring  
            - Handles minor variations in expected answers  
            - Provides continuous scores instead of binary pass/fail  

            **Confidence Weighting:**  
            - Extracts confidence signals from model reasoning  
            - Weights accuracy by prediction confidence  
            - Encourages models to express uncertainty appropriately  

            **Multi-Criteria Evaluation:**  
            - Combines accuracy, fluency, and relevance  
            - Uses weighted scoring for balanced evaluation  
            - Considers answer structure and completeness  

            **Parameter Impact Analysis:**  
            - Groups runs by parameter values  
            - Calculates statistical summaries for each group  
            - Identifies optimal parameter combinations  

            ### 💡 Best Practices Learned

            **Data Preparation:**  
            - Use diverse, high-quality training examples  
            - Ensure examples cover edge cases and variations  
            - Balance example distribution across categories  

            **Optimization Process:**  
            - Run multiple optimizations with different parameters  
            - Use held-out validation data for unbiased evaluation  
            - Track optimization time and resource usage  

            **Analysis and Monitoring:**  
            - Store all optimization runs for comparison  
            - Generate insights automatically from historical data  
            - Create visualizations to understand trends  

            **Error Handling:**  
            - Gracefully handle optimization failures  
            - Provide meaningful error messages  
            - Continue processing even when individual runs fail  

            ### 🎯 Next Steps

            **Apply These Solutions:**  
            1. **Adapt to Your Domain** - Modify signatures and metrics for your use case  
            2. **Experiment with Parameters** - Use grid search to find optimal settings  
            3. **Build Your Dashboard** - Track and analyze your optimization experiments  
            4. **Scale Up** - Apply techniques to larger datasets and more complex modules  

            **Explore Advanced Topics:**  
            - **MIPRO Optimization** - Multi-stage instruction and prompt optimization  
            - **Custom Teleprompters** - Build specialized optimization strategies  
            - **Production Deployment** - Scale optimized models for real applications  

            Excellent work mastering BootstrapFewShot optimization! 🎉
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
