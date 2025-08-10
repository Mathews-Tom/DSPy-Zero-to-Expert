# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    import random
    import statistics
    import sys
    import time
    from dataclasses import dataclass, field
    from enum import Enum
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
        Enum,
        cleandoc,
        dataclass,
        dspy,
        field,
        get_config,
        math,
        mo,
        output,
        random,
        setup_dspy_environment,
        statistics,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🧪 A/B Testing Framework

            **Duration:** 2-3 hours  
            **Prerequisites:** Completed evaluation framework modules  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will master:  
            - **Statistical A/B Testing** - Design and execute rigorous A/B tests  
            - **Significance Testing** - Apply statistical methods to determine significance  
            - **Power Analysis** - Calculate required sample sizes for reliable results  
            - **Multi-Variant Testing** - Compare multiple systems simultaneously  
            - **Automated Reporting** - Generate comprehensive A/B test reports  

            ## 📊 A/B Testing Overview

            **Why A/B Testing Matters:**  
            - **Scientific Rigor** - Make data-driven decisions with statistical confidence  
            - **Risk Mitigation** - Test changes before full deployment  
            - **Performance Optimization** - Systematically improve system performance  
            - **Objective Comparison** - Remove bias from system comparisons  

            **Framework Components:**  
            - **Experimental Design** - Proper randomization and control groups  
            - **Statistical Analysis** - Hypothesis testing and confidence intervals  
            - **Power Analysis** - Sample size calculation and effect size estimation  
            - **Result Interpretation** - Actionable insights from statistical results  

            Let's build a comprehensive A/B testing framework!
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
                ## ✅ A/B Testing Environment Ready

                **Configuration:**  
                - Provider: **{provider}**  
                - Model: **{config.default_model}**  
                - Available Providers: **{', '.join(available_providers)}**

                Ready to build statistical A/B testing frameworks!
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
    Any,
    Enum,
    available_providers,
    cleandoc,
    dataclass,
    dspy,
    field,
    math,
    mo,
    output,
    random,
    statistics,
    time,
):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Statistical A/B Testing Core

                **Building the foundation** for rigorous statistical A/B testing:
                """
            )
        )

        class TestStatus(Enum):
            """Status of an A/B test."""

            DESIGNED = "designed"
            RUNNING = "running"
            COMPLETED = "completed"
            STOPPED = "stopped"
            FAILED = "failed"

        @dataclass
        class ABTestResult:
            """Result of an A/B test comparison."""

            variant_a_name: str
            variant_b_name: str
            variant_a_scores: list[float]
            variant_b_scores: list[float]

            # Statistical measures
            mean_a: float = field(init=False)
            mean_b: float = field(init=False)
            std_a: float = field(init=False)
            std_b: float = field(init=False)

            # Test statistics
            effect_size: float = field(init=False)
            t_statistic: float = field(init=False)
            p_value: float = field(init=False)
            confidence_interval: tuple[float, float] = field(init=False)

            # Test metadata
            sample_size_a: int = field(init=False)
            sample_size_b: int = field(init=False)
            degrees_of_freedom: int = field(init=False)

            def __post_init__(self):
                """Calculate statistical measures after initialization."""
                self.sample_size_a = len(self.variant_a_scores)
                self.sample_size_b = len(self.variant_b_scores)

                if self.sample_size_a > 0:
                    self.mean_a = statistics.mean(self.variant_a_scores)
                    self.std_a = (
                        statistics.stdev(self.variant_a_scores)
                        if self.sample_size_a > 1
                        else 0.0
                    )
                else:
                    self.mean_a = 0.0
                    self.std_a = 0.0

                if self.sample_size_b > 0:
                    self.mean_b = statistics.mean(self.variant_b_scores)
                    self.std_b = (
                        statistics.stdev(self.variant_b_scores)
                        if self.sample_size_b > 1
                        else 0.0
                    )
                else:
                    self.mean_b = 0.0
                    self.std_b = 0.0

                # Calculate effect size (Cohen's d)
                if self.sample_size_a > 1 and self.sample_size_b > 1:
                    pooled_std = math.sqrt(
                        (
                            (self.sample_size_a - 1) * self.std_a**2
                            + (self.sample_size_b - 1) * self.std_b**2
                        )
                        / (self.sample_size_a + self.sample_size_b - 2)
                    )
                    self.effect_size = (
                        (self.mean_b - self.mean_a) / pooled_std
                        if pooled_std > 0
                        else 0.0
                    )
                else:
                    self.effect_size = 0.0

                # Calculate t-test statistics
                self._calculate_t_test()

            def _calculate_t_test(self):
                """Calculate t-test statistics."""
                if self.sample_size_a < 2 or self.sample_size_b < 2:
                    self.t_statistic = 0.0
                    self.p_value = 1.0
                    self.confidence_interval = (0.0, 0.0)
                    self.degrees_of_freedom = 0
                    return

                # Welch's t-test (unequal variances)
                se_a = self.std_a**2 / self.sample_size_a
                se_b = self.std_b**2 / self.sample_size_b
                se_diff = math.sqrt(se_a + se_b)

                if se_diff > 0:
                    self.t_statistic = (self.mean_b - self.mean_a) / se_diff

                    # Welch-Satterthwaite equation for degrees of freedom
                    if se_a > 0 and se_b > 0:
                        self.degrees_of_freedom = int(
                            (se_a + se_b) ** 2
                            / (
                                se_a**2 / (self.sample_size_a - 1)
                                + se_b**2 / (self.sample_size_b - 1)
                            )
                        )
                    else:
                        self.degrees_of_freedom = (
                            self.sample_size_a + self.sample_size_b - 2
                        )

                    # Approximate p-value calculation (simplified)
                    # In production, use scipy.stats.t.sf
                    abs_t = abs(self.t_statistic)
                    if abs_t > 3.0:
                        self.p_value = 0.001
                    elif abs_t > 2.5:
                        self.p_value = 0.01
                    elif abs_t > 2.0:
                        self.p_value = 0.05
                    elif abs_t > 1.5:
                        self.p_value = 0.1
                    else:
                        self.p_value = 0.2

                    # 95% confidence interval for difference in means
                    # Using t-critical value approximation
                    t_critical = 2.0 if self.degrees_of_freedom > 30 else 2.5
                    margin_of_error = t_critical * se_diff
                    diff = self.mean_b - self.mean_a
                    self.confidence_interval = (
                        diff - margin_of_error,
                        diff + margin_of_error,
                    )
                else:
                    self.t_statistic = 0.0
                    self.p_value = 1.0
                    self.confidence_interval = (0.0, 0.0)
                    self.degrees_of_freedom = 0

        class ABTestFramework:
            """Comprehensive A/B testing framework for system comparison."""

            def __init__(self):
                self.tests = {}
                self.test_counter = 0
                self.default_alpha = 0.05  # Significance level
                self.default_power = 0.8  # Statistical power

            def design_test(
                self,
                test_name: str,
                variant_a: Any,
                variant_b: Any,
                metric: Any,
                test_examples: list[dspy.Example],
                alpha: float = None,
                power: float = None,
                expected_effect_size: float = 0.2,
            ) -> str:
                """Design an A/B test with proper statistical planning."""

                test_id = f"test_{self.test_counter}_{int(time.time())}"
                self.test_counter += 1

                alpha = alpha or self.default_alpha
                power = power or self.default_power

                # Calculate required sample size
                required_sample_size = self._calculate_sample_size(
                    expected_effect_size, alpha, power
                )

                test_design = {
                    "id": test_id,
                    "name": test_name,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "metric": metric,
                    "test_examples": test_examples,
                    "status": TestStatus.DESIGNED,
                    "created_at": time.time(),
                    "config": {
                        "alpha": alpha,
                        "power": power,
                        "expected_effect_size": expected_effect_size,
                        "required_sample_size": required_sample_size,
                    },
                    "results": None,
                    "logs": [],
                }

                self.tests[test_id] = test_design
                self._log_test_event(test_id, f"Test designed: {test_name}")

                return test_id

            def run_test(self, test_id: str, randomize: bool = True) -> ABTestResult:
                """Run an A/B test with proper randomization."""

                if test_id not in self.tests:
                    raise ValueError(f"Test {test_id} not found")

                test = self.tests[test_id]
                test["status"] = TestStatus.RUNNING
                test["started_at"] = time.time()

                self._log_test_event(test_id, "Test execution started")

                try:
                    # Get test components
                    variant_a = test["variant_a"]
                    variant_b = test["variant_b"]
                    metric = test["metric"]
                    test_examples = test["test_examples"]

                    # Randomize examples if requested
                    if randomize:
                        examples_copy = test_examples.copy()
                        random.shuffle(examples_copy)
                        test_examples = examples_copy

                    # Split examples between variants (50/50 split)
                    split_point = len(test_examples) // 2
                    examples_a = test_examples[:split_point]
                    examples_b = test_examples[split_point:]

                    # Generate predictions for variant A
                    predictions_a = []
                    for example in examples_a:
                        if hasattr(variant_a, "forward"):
                            inputs = example.inputs()
                            pred = variant_a.forward(**inputs)
                        else:
                            pred = variant_a(example)
                        predictions_a.append(pred)

                    # Generate predictions for variant B
                    predictions_b = []
                    for example in examples_b:
                        if hasattr(variant_b, "forward"):
                            inputs = example.inputs()
                            pred = variant_b.forward(**inputs)
                        else:
                            pred = variant_b(example)
                        predictions_b.append(pred)

                    # Evaluate with metric
                    scores_a = []
                    for example, prediction in zip(
                        examples_a, predictions_a, strict=False
                    ):
                        if hasattr(metric, "evaluate"):
                            result = metric.evaluate(example, prediction)
                            score = (
                                result.score
                                if hasattr(result, "score")
                                else float(result)
                            )
                        else:
                            score = float(metric(example, prediction))
                        scores_a.append(score)

                    scores_b = []
                    for example, prediction in zip(
                        examples_b, predictions_b, strict=False
                    ):
                        if hasattr(metric, "evaluate"):
                            result = metric.evaluate(example, prediction)
                            score = (
                                result.score
                                if hasattr(result, "score")
                                else float(result)
                            )
                        else:
                            score = float(metric(example, prediction))
                        scores_b.append(score)

                    # Create test result
                    result = ABTestResult(
                        variant_a_name=getattr(variant_a, "name", "Variant A"),
                        variant_b_name=getattr(variant_b, "name", "Variant B"),
                        variant_a_scores=scores_a,
                        variant_b_scores=scores_b,
                    )

                    # Store results
                    test["results"] = result
                    test["status"] = TestStatus.COMPLETED
                    test["completed_at"] = time.time()
                    test["duration"] = test["completed_at"] - test["started_at"]

                    self._log_test_event(
                        test_id, "Test execution completed successfully"
                    )

                    return result

                except Exception as e:
                    test["status"] = TestStatus.FAILED
                    test["error"] = str(e)
                    self._log_test_event(test_id, f"Test execution failed: {str(e)}")
                    raise

            def _calculate_sample_size(
                self, effect_size: float, alpha: float, power: float
            ) -> int:
                """Calculate required sample size for given parameters."""
                # Simplified sample size calculation
                # In production, use more sophisticated methods

                # Z-scores for alpha and power
                z_alpha = 1.96 if alpha == 0.05 else 2.58  # Simplified
                z_power = 0.84 if power == 0.8 else 1.28  # Simplified

                # Sample size per group
                n = 2 * ((z_alpha + z_power) / effect_size) ** 2

                return max(10, int(math.ceil(n)))  # Minimum 10 samples

            def interpret_results(self, test_id: str) -> dict[str, Any]:
                """Interpret A/B test results with statistical significance."""

                if test_id not in self.tests:
                    return {"error": "Test not found"}

                test = self.tests[test_id]

                if test["status"] != TestStatus.COMPLETED:
                    return {
                        "error": "Test not completed",
                        "status": test["status"].value,
                    }

                result = test["results"]
                alpha = test["config"]["alpha"]

                # Statistical interpretation
                is_significant = result.p_value < alpha

                # Effect size interpretation
                if abs(result.effect_size) < 0.2:
                    effect_magnitude = "small"
                elif abs(result.effect_size) < 0.5:
                    effect_magnitude = "medium"
                else:
                    effect_magnitude = "large"

                # Practical significance
                practical_difference = abs(result.mean_b - result.mean_a)

                # Winner determination
                if is_significant:
                    if result.mean_b > result.mean_a:
                        winner = result.variant_b_name
                        improvement = (
                            ((result.mean_b - result.mean_a) / result.mean_a) * 100
                            if result.mean_a > 0
                            else 0
                        )
                    else:
                        winner = result.variant_a_name
                        improvement = (
                            ((result.mean_a - result.mean_b) / result.mean_b) * 100
                            if result.mean_b > 0
                            else 0
                        )
                else:
                    winner = "No significant difference"
                    improvement = 0

                interpretation = {
                    "test_id": test_id,
                    "test_name": test["name"],
                    "statistical_significance": {
                        "is_significant": is_significant,
                        "p_value": result.p_value,
                        "alpha": alpha,
                        "confidence_level": (1 - alpha) * 100,
                    },
                    "effect_size": {
                        "cohens_d": result.effect_size,
                        "magnitude": effect_magnitude,
                        "interpretation": self._interpret_effect_size(
                            result.effect_size
                        ),
                    },
                    "practical_significance": {
                        "mean_difference": result.mean_b - result.mean_a,
                        "relative_improvement": improvement,
                        "confidence_interval": result.confidence_interval,
                    },
                    "winner": winner,
                    "recommendation": self._generate_recommendation(
                        result, is_significant, effect_magnitude
                    ),
                    "sample_sizes": {
                        "variant_a": result.sample_size_a,
                        "variant_b": result.sample_size_b,
                        "total": result.sample_size_a + result.sample_size_b,
                    },
                }

                return interpretation

            def _interpret_effect_size(self, effect_size: float) -> str:
                """Interpret Cohen's d effect size."""
                abs_effect = abs(effect_size)
                if abs_effect < 0.2:
                    return "Negligible effect"
                elif abs_effect < 0.5:
                    return "Small effect"
                elif abs_effect < 0.8:
                    return "Medium effect"
                else:
                    return "Large effect"

            def _generate_recommendation(
                self, result: ABTestResult, is_significant: bool, effect_magnitude: str
            ) -> str:
                """Generate actionable recommendation based on test results."""

                if not is_significant:
                    return "No significant difference detected. Consider collecting more data or testing larger changes."

                winner = (
                    result.variant_b_name
                    if result.mean_b > result.mean_a
                    else result.variant_a_name
                )

                if effect_magnitude == "large":
                    return f"Strong evidence favors {winner}. Recommend immediate implementation."
                elif effect_magnitude == "medium":
                    return f"Moderate evidence favors {winner}. Consider implementation with monitoring."
                else:
                    return f"Weak evidence favors {winner}. Consider additional testing or larger sample size."

            def _log_test_event(self, test_id: str, message: str):
                """Log an event for a test."""
                if test_id in self.tests:
                    self.tests[test_id]["logs"].append(
                        {"timestamp": time.time(), "message": message}
                    )

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Statistical A/B Testing Core Created

                **Core Components:**  
                - **TestStatus** - Enumeration of test lifecycle states  
                - **ABTestResult** - Comprehensive statistical result analysis  
                - **ABTestFramework** - Complete A/B testing orchestration  

                **Statistical Features:**  
                - **Effect Size Calculation** - Cohen's d for practical significance  
                - **T-Test Analysis** - Welch's t-test for unequal variances  
                - **Sample Size Planning** - Power analysis for reliable results  
                - **Confidence Intervals** - Uncertainty quantification  

                **Key Capabilities:**  
                - **Proper Randomization** - Unbiased assignment to test groups  
                - **Statistical Rigor** - Hypothesis testing with significance levels  
                - **Result Interpretation** - Actionable insights from statistical analysis  
                - **Comprehensive Logging** - Complete audit trail of test execution  

                Ready to build interactive A/B testing interfaces!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        ABTestFramework = None
        ABTestResult = None
        TestStatus = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (ABTestFramework,)


@app.cell
def _(ABTestFramework, available_providers, cleandoc, dspy, mo, output):
    if available_providers and ABTestFramework:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🎛️ Step 2: Interactive A/B Testing Interface

                **Design and execute A/B tests** with interactive controls:
                """
            )
        )

        # Create A/B testing framework instance
        ab_framework = ABTestFramework()

        # Sample systems for A/B testing
        class SystemA:
            """High-accuracy system for A/B testing."""

            def __init__(self):
                self.name = "System A (High Accuracy)"
                self.answers = {
                    "What is the capital of France?": "Paris",
                    "Who wrote Romeo and Juliet?": "William Shakespeare",
                    "What is 2 + 2?": "4",
                    "What is the largest planet?": "Jupiter",
                    "What year did WWII end?": "1945",
                }

            def forward(self, question):
                answer = self.answers.get(question, "I don't know")
                return dspy.Prediction(answer=answer)

        class SystemB:
            """Moderate-accuracy system for A/B testing."""

            def __init__(self):
                self.name = "System B (Moderate Accuracy)"
                self.answers = {
                    "What is the capital of France?": "Paris, France",  # Verbose
                    "Who wrote Romeo and Juliet?": "Shakespeare",  # Abbreviated
                    "What is 2 + 2?": "Four",  # Different format
                    "What is the largest planet?": "Jupiter is the largest",  # Verbose
                    "What year did WWII end?": "1945",
                }

            def forward(self, question):
                answer = self.answers.get(question, "I'm not sure")
                return dspy.Prediction(answer=answer)

        # Simple evaluation metric
        class SimpleMetric:
            """Simple exact match metric for A/B testing."""

            def __init__(self):
                self.name = "Exact Match"

            def evaluate(self, example, prediction):
                expected = getattr(example, "answer", "").lower().strip()
                predicted = (
                    getattr(prediction, "answer", str(prediction)).lower().strip()
                )
                return 1.0 if expected == predicted else 0.0

        # Sample test data
        ab_test_examples = [
            dspy.Example(
                question="What is the capital of France?", answer="Paris"
            ).with_inputs("question"),
            dspy.Example(
                question="Who wrote Romeo and Juliet?", answer="William Shakespeare"
            ).with_inputs("question"),
            dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
            dspy.Example(
                question="What is the largest planet?", answer="Jupiter"
            ).with_inputs("question"),
            dspy.Example(question="What year did WWII end?", answer="1945").with_inputs(
                "question"
            ),
        ] * 4  # Repeat for larger sample size

        # Interactive controls
        test_name_input = mo.ui.text(value="QA System Comparison", label="Test Name")

        alpha_slider = mo.ui.slider(
            start=0.01,
            stop=0.10,
            value=0.05,
            step=0.01,
            label="Significance Level (α)",
            show_value=True,
        )

        power_slider = mo.ui.slider(
            start=0.7,
            stop=0.95,
            value=0.8,
            step=0.05,
            label="Statistical Power",
            show_value=True,
        )

        effect_size_slider = mo.ui.slider(
            start=0.1,
            stop=1.0,
            value=0.2,
            step=0.1,
            label="Expected Effect Size",
            show_value=True,
        )

        randomize_checkbox = mo.ui.checkbox(value=True, label="Randomize Example Order")

        design_test_button = mo.ui.run_button(
            label="📋 Design A/B Test", kind="success"
        )

        run_test_button = mo.ui.run_button(label="🚀 Run A/B Test", kind="info")

        interpret_results_button = mo.ui.run_button(
            label="📊 Interpret Results", kind="info"
        )

        ab_testing_controls = mo.vstack(
            [
                mo.md("### 🧪 A/B Testing Configuration"),
                test_name_input,
                mo.md("**Statistical Parameters:**"),
                alpha_slider,
                power_slider,
                effect_size_slider,
                mo.md("**Test Options:**"),
                randomize_checkbox,
                mo.md("---"),
                mo.hstack(
                    [design_test_button, run_test_button, interpret_results_button]
                ),
            ]
        )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🎛️ Interactive A/B Testing Interface Created

                **Interface Components:**  
                - **Test Configuration** - Set test name and statistical parameters  
                - **Statistical Controls** - Configure significance level, power, and effect size  
                - **Test Options** - Control randomization and other test settings  
                - **Execution Controls** - Design, run, and interpret A/B tests  

                **Sample Systems:**  
                - **System A** - High-accuracy system with exact answers  
                - **System B** - Moderate-accuracy system with varied response formats  

                **Test Data:**  
                - **20 Examples** - Repeated QA examples for sufficient sample size  
                - **Balanced Design** - Equal allocation between test variants  

                Configure your A/B test parameters and run the statistical comparison!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        ab_testing_controls = mo.md("")
        cell4_content = mo.md("")
        ab_framework = None
        design_test_button = None
        run_test_button = None
        interpret_results_button = None
        ab_test_examples = None

    cell4_out = mo.vstack([cell4_desc, ab_testing_controls, cell4_content])
    output.replace(cell4_out)
    return (
        SimpleMetric,
        SystemA,
        SystemB,
        ab_framework,
        ab_test_examples,
        alpha_slider,
        design_test_button,
        effect_size_slider,
        interpret_results_button,
        power_slider,
        randomize_checkbox,
        run_test_button,
        test_name_input,
    )


@app.cell
def _(
    SimpleMetric,
    SystemA,
    SystemB,
    ab_framework,
    ab_test_examples,
    alpha_slider,
    cleandoc,
    design_test_button,
    effect_size_slider,
    interpret_results_button,
    mo,
    output,
    power_slider,
    randomize_checkbox,
    run_test_button,
    test_name_input,
):
    # Handle A/B testing interactions
    import __main__

    ab_test_results_display = mo.md("")

    if design_test_button.value:
        # Design A/B test
        test_name = test_name_input.value
        alpha = alpha_slider.value
        power = power_slider.value
        expected_effect_size = effect_size_slider.value

        # Create systems and metric
        system_a = SystemA()
        system_b = SystemB()
        metric = SimpleMetric()

        # Design the test
        test_id = ab_framework.design_test(
            test_name=test_name,
            variant_a=system_a,
            variant_b=system_b,
            metric=metric,
            test_examples=ab_test_examples,
            alpha=alpha,
            power=power,
            expected_effect_size=expected_effect_size,
        )

        # Store test ID in global variable
        __main__.current_test_id = test_id

        # Get test details
        test_details = ab_framework.tests[test_id]
        required_sample_size = test_details["config"]["required_sample_size"]

        ab_test_results_display = mo.md(
            cleandoc(
                f"""
                ## 📋 A/B Test Designed Successfully

                **Test ID:** {test_id}  
                **Test Name:** {test_name}  

                ### 🔬 Statistical Design

                **Parameters:**  
                - **Significance Level (α):** {alpha:.3f}  
                - **Statistical Power:** {power:.1%}  
                - **Expected Effect Size:** {expected_effect_size:.1f}  
                - **Required Sample Size:** {required_sample_size} per group  

                **Test Setup:**  
                - **Variant A:** {system_a.name}  
                - **Variant B:** {system_b.name}  
                - **Metric:** {metric.name}  
                - **Available Examples:** {len(ab_test_examples)}  

                ### 📊 Power Analysis

                {'✅ Sufficient sample size available' if len(ab_test_examples) >= required_sample_size * 2 else '⚠️ Consider collecting more examples for adequate power'}

                ### 📋 Next Steps

                Click "🚀 Run A/B Test" to execute the statistical comparison.
                """
            )
        )

    elif run_test_button.value and __main__.current_test_id:
        # Run A/B test
        randomize = randomize_checkbox.value

        try:
            result = ab_framework.run_test(
                __main__.current_test_id, randomize=randomize
            )

            # Display basic results
            ab_test_results_display = mo.md(
                cleandoc(
                    f"""
                    ## 🚀 A/B Test Execution Complete

                    **Test ID:** {__main__.current_test_id}  
                    **Randomization:** {'Enabled' if randomize else 'Disabled'}  

                    ### 📊 Raw Results

                    **{result.variant_a_name}:**  
                    - Sample Size: {result.sample_size_a}  
                    - Mean Score: {result.mean_a:.3f}  
                    - Standard Deviation: {result.std_a:.3f}  

                    **{result.variant_b_name}:**  
                    - Sample Size: {result.sample_size_b}  
                    - Mean Score: {result.mean_b:.3f}  
                    - Standard Deviation: {result.std_b:.3f}  

                    ### 📈 Statistical Measures

                    - **Effect Size (Cohen's d):** {result.effect_size:.3f}  
                    - **T-Statistic:** {result.t_statistic:.3f}  
                    - **P-Value:** {result.p_value:.3f}  
                    - **95% Confidence Interval:** [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]  

                    ### 📋 Next Steps

                    Click "📊 Interpret Results" for detailed statistical interpretation and recommendations.
                    """
                )
            )

        except Exception as e:
            ab_test_results_display = mo.md(
                cleandoc(
                    f"""
                    ## ❌ A/B Test Execution Failed

                    **Error:** {str(e)}

                    Please check your test configuration and try again.
                    """
                )
            )

    elif interpret_results_button.value and __main__.current_test_id:
        # Interpret results
        interpretation = ab_framework.interpret_results(__main__.current_test_id)

        if "error" not in interpretation:
            # Format interpretation for display
            significance = interpretation["statistical_significance"]
            effect = interpretation["effect_size"]
            practical = interpretation["practical_significance"]

            # Significance indicator
            significance_indicator = (
                "✅ Statistically Significant"
                if significance["is_significant"]
                else "❌ Not Statistically Significant"
            )

            # Effect size interpretation
            effect_indicator = (
                f"📊 {effect['magnitude'].title()} Effect ({effect['interpretation']})"
            )

            # Winner declaration
            winner_text = (
                f"🏆 Winner: {interpretation['winner']}"
                if interpretation["winner"] != "No significant difference"
                else "🤝 No Clear Winner"
            )

            ab_test_results_display = mo.md(
                cleandoc(
                    f"""
                    ## 📊 A/B Test Statistical Interpretation

                    **Test:** {interpretation["test_name"]}  
                    **Test ID:** {interpretation["test_id"]}  

                    ### 🎯 Key Findings

                    {winner_text}  
                    {significance_indicator}  
                    {effect_indicator}  

                    ### 📈 Statistical Significance

                    - **P-Value:** {significance["p_value"]:.4f}  
                    - **Significance Level:** {significance["alpha"]:.3f}  
                    - **Confidence Level:** {significance["confidence_level"]:.1f}%  
                    - **Result:** {'Reject null hypothesis' if significance["is_significant"] else 'Fail to reject null hypothesis'}  

                    ### 📊 Effect Size Analysis

                    - **Cohen's d:** {effect["cohens_d"]:.3f}  
                    - **Magnitude:** {effect["magnitude"].title()}  
                    - **Interpretation:** {effect["interpretation"]}  

                    ### 💡 Practical Significance

                    - **Mean Difference:** {practical["mean_difference"]:.3f}  
                    - **Relative Improvement:** {practical["relative_improvement"]:.1f}%  
                    - **95% Confidence Interval:** [{practical["confidence_interval"][0]:.3f}, {practical["confidence_interval"][1]:.3f}]  

                    ### 📋 Sample Size Analysis

                    - **Variant A:** {interpretation["sample_sizes"]["variant_a"]} examples  
                    - **Variant B:** {interpretation["sample_sizes"]["variant_b"]} examples  
                    - **Total:** {interpretation["sample_sizes"]["total"]} examples  

                    ### 🎯 Recommendation

                    **{interpretation["recommendation"]}**

                    ### 💡 Interpretation Guide

                    - **P-Value < 0.05:** Strong evidence against null hypothesis  
                    - **Effect Size > 0.5:** Practically meaningful difference  
                    - **Confidence Interval:** Range of plausible values for true difference  

                    The statistical analysis provides {'strong' if significance["is_significant"] and abs(effect["cohens_d"]) > 0.5 else 'moderate' if significance["is_significant"] else 'weak'} evidence for system performance differences.
                    """
                )
            )
        else:
            ab_test_results_display = mo.md(f"## ❌ Error: {interpretation['error']}")

    # Debug information
    elif run_test_button.value and not __main__.current_test_id:
        ab_test_results_display = mo.md(
            "## ⚠️ No test designed yet. Please design a test first by clicking 'Design A/B Test'."
        )

    elif interpret_results_button.value and not __main__.current_test_id:
        ab_test_results_display = mo.md(
            "## ⚠️ No test results available. Please design and run a test first."
        )

    output.replace(ab_test_results_display)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell6_out = mo.md(
        cleandoc(
            """
            ## 🎯 Advanced A/B Testing Strategies

            ### 🏆 Statistical Best Practices

            **Experimental Design Principles:**  
            - **Randomization** - Ensure unbiased assignment to test groups  
            - **Sample Size Planning** - Calculate required samples for adequate power  
            - **Control Variables** - Account for confounding factors  
            - **Blinding** - Minimize bias in evaluation when possible  

            **Statistical Rigor:**  
            - **Hypothesis Testing** - Clearly define null and alternative hypotheses  
            - **Significance Levels** - Choose appropriate α levels (typically 0.05)  
            - **Multiple Comparisons** - Adjust for multiple testing when needed  
            - **Effect Size** - Report practical significance alongside statistical significance  

            ### ⚡ Advanced Testing Techniques

            **Multi-Variant Testing:**  
            - **A/B/C Testing** - Compare multiple variants simultaneously  
            - **Factorial Design** - Test multiple factors and their interactions  
            - **Sequential Testing** - Stop tests early when significance is reached  
            - **Bayesian A/B Testing** - Use Bayesian methods for continuous monitoring  

            **Specialized Designs:**  
            - **Stratified Randomization** - Ensure balance across important subgroups  
            - **Cluster Randomization** - Randomize groups rather than individuals  
            - **Crossover Design** - Each unit receives multiple treatments  
            - **Adaptive Testing** - Modify test parameters based on interim results  

            ### 🔧 Implementation Considerations

            **Sample Size and Power:**  
            - **Power Analysis** - Calculate required sample sizes before testing  
            - **Effect Size Estimation** - Use domain knowledge to estimate expected effects  
            - **Minimum Detectable Effect** - Determine smallest meaningful difference  
            - **Interim Analysis** - Plan for early stopping rules  

            **Bias Prevention:**  
            - **Selection Bias** - Ensure representative sampling  
            - **Measurement Bias** - Use consistent evaluation procedures  
            - **Survivorship Bias** - Account for dropouts and missing data  
            - **Confirmation Bias** - Pre-register analysis plans  

            ### 🚀 Production A/B Testing

            **Infrastructure Requirements:**  
            - **Traffic Splitting** - Reliable randomization and assignment  
            - **Data Collection** - Comprehensive logging and monitoring  
            - **Real-time Analysis** - Continuous monitoring of test metrics  
            - **Rollback Capability** - Quick reversion if issues arise  

            **Operational Excellence:**  
            - **Test Registry** - Centralized tracking of all experiments  
            - **Guardrail Metrics** - Monitor for unintended consequences  
            - **Ramp-up Strategy** - Gradual increase in test traffic  
            - **Post-test Analysis** - Long-term impact assessment  

            ### 💡 Common Pitfalls and Solutions

            **Statistical Pitfalls:**  
            - **Peeking Problem** - Don't check results repeatedly without adjustment  
            - **Simpson's Paradox** - Consider segment-level effects  
            - **Regression to Mean** - Account for natural variation  
            - **Novelty Effects** - Consider temporary behavior changes  

            **Practical Challenges:**  
            - **Network Effects** - Handle interactions between users  
            - **Seasonal Variations** - Account for time-based patterns  
            - **Technical Issues** - Monitor for implementation bugs  
            - **Business Constraints** - Balance statistical rigor with business needs  

            ### 📊 Advanced Metrics and Analysis

            **Beyond Simple Metrics:**  
            - **Ratio Metrics** - Handle metrics with denominators carefully  
            - **Long-term Effects** - Measure sustained impact over time  
            - **Heterogeneous Effects** - Analyze different user segments  
            - **Causal Inference** - Establish causal relationships  

            **Statistical Methods:**
            - **Bootstrap Confidence Intervals** - Non-parametric uncertainty estimation  
            - **Permutation Tests** - Distribution-free significance testing  
            - **Regression Analysis** - Control for covariates and confounders  
            - **Machine Learning** - Use ML for heterogeneous treatment effects  

            ### 💡 Next Steps

            **Enhance Your A/B Testing:**  
            1. **Implement Sequential Testing** - Add early stopping capabilities  
            2. **Build Multi-Variant Support** - Extend to A/B/C/D testing  
            3. **Add Bayesian Methods** - Implement Bayesian A/B testing  
            4. **Create Automated Reporting** - Generate comprehensive test reports  

            Excellent work mastering statistical A/B testing! 🎉
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell6_out)
    return


if __name__ == "__main__":
    app.run()
