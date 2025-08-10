# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return cleandoc, get_config, mo, output, setup_dspy_environment


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🏋️ Evaluation & Metrics Exercises

            **Practice building comprehensive evaluation systems** with advanced metrics and frameworks.

            ## 📚 Exercise Overview

            These exercises will help you master:
            - Creating custom evaluation metrics for specific domains
            - Building comprehensive evaluation frameworks
            - Implementing statistical A/B testing systems
            - Designing interactive evaluation dashboards

            Complete each exercise to build your evaluation expertise!
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
                ## ✅ Exercise Environment Ready

                **Configuration:**
                - Provider: **{provider}**
                - Model: **{config.default_model}**
                - Available Providers: **{', '.join(available_providers)}**

                Ready to start evaluation exercises!
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
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell3_out = mo.md(
            cleandoc(
                """
                ## 🎯 Exercise 1: Domain-Specific Evaluation Metrics

                **Task:** Create evaluation metrics for a code review assistant system.

                **Requirements:**
                1. Create a `CodeReviewMetric` that evaluates code review quality
                2. Implement metrics for helpfulness, accuracy, and actionability
                3. Build a composite metric combining multiple dimensions
                4. Test the metrics with sample code review data

                **Your Implementation:**
                """
            )
        )

        # Exercise 1 Template
        exercise1_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 1: Domain-Specific Evaluation Metrics

                from abc import ABC, abstractmethod
                from dataclasses import dataclass
                from typing import Any, Dict, List
                import dspy

                @dataclass
                class MetricResult:
                    metric_name: str
                    score: float
                    confidence: float = 1.0
                    metadata: Dict[str, Any] = None
                    explanation: str = ""

                class EvaluationMetric(ABC):
                    def __init__(self, name: str, description: str = ""):
                        self.name = name
                        self.description = description
                    
                    @abstractmethod
                    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
                        pass

                # TODO: Create CodeReviewHelpfulnessMetric
                class CodeReviewHelpfulnessMetric(EvaluationMetric):
                    \"\"\"Evaluates how helpful a code review comment is.\"\"\"
                    
                    def __init__(self):
                        super().__init__("code_review_helpfulness", "Measures helpfulness of code review comments")
                    
                    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
                        # TODO: Implement helpfulness evaluation
                        # Consider: constructive feedback, specific suggestions, clarity
                        pass

                # TODO: Create CodeReviewAccuracyMetric
                class CodeReviewAccuracyMetric(EvaluationMetric):
                    \"\"\"Evaluates accuracy of identified issues.\"\"\"
                    
                    def __init__(self):
                        super().__init__("code_review_accuracy", "Measures accuracy of issue identification")
                    
                    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
                        # TODO: Implement accuracy evaluation
                        # Consider: correct issue identification, false positives/negatives
                        pass

                # TODO: Create CodeReviewActionabilityMetric
                class CodeReviewActionabilityMetric(EvaluationMetric):
                    \"\"\"Evaluates how actionable the review comments are.\"\"\"
                    
                    def __init__(self):
                        super().__init__("code_review_actionability", "Measures actionability of review comments")
                    
                    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
                        # TODO: Implement actionability evaluation
                        # Consider: clear instructions, specific changes, implementability
                        pass

                # TODO: Create CompositeCodeReviewMetric
                class CompositeCodeReviewMetric(EvaluationMetric):
                    \"\"\"Combines multiple code review metrics.\"\"\"
                    
                    def __init__(self, weights: Dict[str, float] = None):
                        super().__init__("composite_code_review", "Combined code review evaluation")
                        self.weights = weights or {"helpfulness": 0.4, "accuracy": 0.4, "actionability": 0.2}
                        # TODO: Initialize sub-metrics
                    
                    def evaluate(self, example: dspy.Example, prediction: Any) -> MetricResult:
                        # TODO: Implement composite evaluation
                        # Combine results from all sub-metrics with weights
                        pass

                # TODO: Create test data and test your metrics
                def test_code_review_metrics():
                    \"\"\"Test the code review metrics with sample data.\"\"\"
                    # TODO: Create sample code review examples
                    # TODO: Test each metric individually
                    # TODO: Test composite metric
                    # TODO: Analyze results
                    pass

                if __name__ == "__main__":
                    test_code_review_metrics()
                """
            ),
            language="python",
            label="Exercise 1 Code",
        )

        exercise1_ui = mo.vstack([cell3_out, exercise1_code])
    else:
        exercise1_ui = mo.md("")

    output.replace(exercise1_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell4_out = mo.md(
            cleandoc(
                """
                ## 🔧 Exercise 2: Statistical Evaluation Framework

                **Task:** Build a statistical evaluation framework with confidence intervals and significance testing.

                **Requirements:**
                1. Create a `StatisticalEvaluator` class with bootstrap confidence intervals
                2. Implement significance testing for metric comparisons
                3. Build statistical reporting with effect sizes
                4. Create visualization of statistical results

                **Your Implementation:**
                """
            )
        )

        # Exercise 2 Template
        exercise2_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 2: Statistical Evaluation Framework

                import statistics
                import random
                from typing import List, Tuple, Dict, Any
                import dspy

                class StatisticalEvaluator:
                    \"\"\"Statistical evaluation framework with confidence intervals and significance testing.\"\"\"
                    
                    def __init__(self, confidence_level: float = 0.95):
                        self.confidence_level = confidence_level
                        self.alpha = 1 - confidence_level
                    
                    def bootstrap_confidence_interval(self, scores: List[float], n_bootstrap: int = 1000) -> Tuple[float, float]:
                        \"\"\"Calculate bootstrap confidence interval for mean score.\"\"\"
                        # TODO: Implement bootstrap resampling
                        # 1. Generate bootstrap samples
                        # 2. Calculate mean for each sample
                        # 3. Return confidence interval
                        pass
                    
                    def calculate_effect_size(self, scores_a: List[float], scores_b: List[float]) -> float:
                        \"\"\"Calculate Cohen's d effect size.\"\"\"
                        # TODO: Implement Cohen's d calculation
                        # Cohen's d = (mean_b - mean_a) / pooled_standard_deviation
                        pass
                    
                    def permutation_test(self, scores_a: List[float], scores_b: List[float], 
                                       n_permutations: int = 1000) -> float:
                        \"\"\"Perform permutation test for significance.\"\"\"
                        # TODO: Implement permutation test
                        # 1. Calculate observed difference
                        # 2. Generate permuted samples
                        # 3. Calculate p-value
                        pass
                    
                    def evaluate_with_statistics(self, system_a: Any, system_b: Any, 
                                               test_examples: List[dspy.Example], 
                                               metric: Any) -> Dict[str, Any]:
                        \"\"\"Comprehensive statistical evaluation of two systems.\"\"\"
                        # TODO: Implement statistical evaluation
                        # 1. Generate predictions for both systems
                        # 2. Calculate scores with the metric
                        # 3. Compute statistical measures
                        # 4. Return comprehensive results
                        pass
                    
                    def generate_statistical_report(self, results: Dict[str, Any]) -> str:
                        \"\"\"Generate a comprehensive statistical report.\"\"\"
                        # TODO: Create formatted statistical report
                        # Include: means, confidence intervals, effect size, p-value, interpretation
                        pass

                # TODO: Create sample systems and test the statistical evaluator
                def test_statistical_evaluator():
                    \"\"\"Test the statistical evaluation framework.\"\"\"
                    # TODO: Create test systems with known performance differences
                    # TODO: Run statistical evaluation
                    # TODO: Generate and display report
                    pass

                if __name__ == "__main__":
                    test_statistical_evaluator()
                """
            ),
            language="python",
            label="Exercise 2 Code",
        )

        exercise2_ui = mo.vstack([cell4_out, exercise2_code])
    else:
        exercise2_ui = mo.md("")

    output.replace(exercise2_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_out = mo.md(
            cleandoc(
                """
                ## ⚡ Exercise 3: Multi-System Evaluation Dashboard

                **Task:** Create an evaluation dashboard that can compare multiple systems across multiple metrics.

                **Requirements:**
                1. Create a `MultiSystemEvaluator` class
                2. Implement ranking and comparison algorithms
                3. Build visualization of results with charts
                4. Create export functionality for results

                **Your Implementation:**
                """
            )
        )

        # Exercise 3 Template
        exercise3_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 3: Multi-System Evaluation Dashboard

                from collections import defaultdict
                from typing import Dict, List, Any, Tuple
                import json
                import dspy

                class MultiSystemEvaluator:
                    \"\"\"Evaluation dashboard for comparing multiple systems across multiple metrics.\"\"\"
                    
                    def __init__(self):
                        self.systems = {}
                        self.metrics = {}
                        self.evaluation_results = {}
                        self.rankings = {}
                    
                    def add_system(self, name: str, system: Any) -> None:
                        \"\"\"Add a system to evaluate.\"\"\"
                        # TODO: Add system to the evaluator
                        pass
                    
                    def add_metric(self, name: str, metric: Any, weight: float = 1.0) -> None:
                        \"\"\"Add a metric with optional weight.\"\"\"
                        # TODO: Add metric to the evaluator
                        pass
                    
                    def evaluate_all_systems(self, test_examples: List[dspy.Example]) -> Dict[str, Any]:
                        \"\"\"Evaluate all systems with all metrics.\"\"\"
                        # TODO: Implement comprehensive evaluation
                        # 1. For each system, generate predictions
                        # 2. For each metric, evaluate predictions
                        # 3. Store results in organized structure
                        # 4. Calculate rankings
                        pass
                    
                    def calculate_rankings(self) -> Dict[str, List[Tuple[str, float]]]:
                        \"\"\"Calculate rankings for each metric and overall.\"\"\"
                        # TODO: Implement ranking calculation
                        # 1. Rank systems for each individual metric
                        # 2. Calculate weighted overall ranking
                        # 3. Return rankings dictionary
                        pass
                    
                    def generate_comparison_matrix(self) -> Dict[str, Dict[str, float]]:
                        \"\"\"Generate pairwise comparison matrix.\"\"\"
                        # TODO: Create pairwise comparison matrix
                        # Show how each system compares to every other system
                        pass
                    
                    def create_visualization_data(self) -> Dict[str, Any]:
                        \"\"\"Create data structure for visualization.\"\"\"
                        # TODO: Prepare data for charts and graphs
                        # Include: bar charts, radar charts, heatmaps
                        pass
                    
                    def export_results(self, format_type: str = "json") -> str:
                        \"\"\"Export evaluation results in specified format.\"\"\"
                        # TODO: Implement export functionality
                        # Support JSON, CSV, and formatted text reports
                        pass
                    
                    def generate_insights(self) -> List[str]:
                        \"\"\"Generate insights from evaluation results.\"\"\"
                        # TODO: Analyze results and generate insights
                        # Identify best performers, patterns, recommendations
                        pass

                # TODO: Create sample systems and metrics for testing
                def test_multi_system_evaluator():
                    \"\"\"Test the multi-system evaluation dashboard.\"\"\"
                    # TODO: Create multiple test systems with different characteristics
                    # TODO: Add various metrics
                    # TODO: Run comprehensive evaluation
                    # TODO: Generate visualizations and insights
                    pass

                if __name__ == "__main__":
                    test_multi_system_evaluator()
                """
            ),
            language="python",
            label="Exercise 3 Code",
        )

        exercise3_ui = mo.vstack([cell5_out, exercise3_code])
    else:
        exercise3_ui = mo.md("")

    output.replace(exercise3_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell6_out = mo.md(
            cleandoc(
                """
                ## 📊 Exercise 4: Advanced A/B Testing System

                **Task:** Build an advanced A/B testing system with sequential testing and Bayesian analysis.

                **Requirements:**
                1. Create a `SequentialABTest` class with early stopping
                2. Implement Bayesian A/B testing with posterior distributions
                3. Build multi-armed bandit testing capability
                4. Create comprehensive test reporting and visualization

                **Your Implementation:**
                """
            )
        )

        # Exercise 4 Template
        exercise4_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 4: Advanced A/B Testing System

                import math
                import random
                from typing import List, Dict, Any, Tuple, Optional
                from dataclasses import dataclass
                from enum import Enum
                import dspy

                class TestDecision(Enum):
                    CONTINUE = "continue"
                    STOP_VARIANT_A_WINS = "stop_a_wins"
                    STOP_VARIANT_B_WINS = "stop_b_wins"
                    STOP_NO_DIFFERENCE = "stop_no_difference"

                @dataclass
                class BayesianResult:
                    posterior_a_alpha: float
                    posterior_a_beta: float
                    posterior_b_alpha: float
                    posterior_b_beta: float
                    probability_b_better: float
                    expected_loss: float

                class SequentialABTest:
                    \"\"\"Advanced A/B testing with sequential analysis and early stopping.\"\"\"
                    
                    def __init__(self, alpha: float = 0.05, power: float = 0.8, 
                                 min_sample_size: int = 100):
                        self.alpha = alpha
                        self.power = power
                        self.min_sample_size = min_sample_size
                        self.test_history = []
                    
                    def calculate_sequential_boundaries(self, max_n: int) -> Dict[str, List[float]]:
                        \"\"\"Calculate sequential testing boundaries (O'Brien-Fleming).\"\"\"
                        # TODO: Implement O'Brien-Fleming boundaries
                        # Calculate upper and lower boundaries for early stopping
                        pass
                    
                    def sequential_test_decision(self, scores_a: List[float], scores_b: List[float]) -> TestDecision:
                        \"\"\"Make sequential testing decision.\"\"\"
                        # TODO: Implement sequential decision logic
                        # 1. Check if minimum sample size reached
                        # 2. Calculate current test statistic
                        # 3. Compare against sequential boundaries
                        # 4. Return appropriate decision
                        pass
                    
                    def run_sequential_test(self, variant_a: Any, variant_b: Any, 
                                          test_examples: List[dspy.Example], metric: Any,
                                          max_samples: int = 1000) -> Dict[str, Any]:
                        \"\"\"Run sequential A/B test with early stopping.\"\"\"
                        # TODO: Implement sequential testing
                        # 1. Gradually add samples
                        # 2. Check stopping criteria at each step
                        # 3. Stop when decision can be made
                        # 4. Return comprehensive results
                        pass

                class BayesianABTest:
                    \"\"\"Bayesian A/B testing with posterior distributions.\"\"\"
                    
                    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
                        self.prior_alpha = prior_alpha
                        self.prior_beta = prior_beta
                    
                    def update_posterior(self, successes: int, failures: int) -> Tuple[float, float]:
                        \"\"\"Update Beta posterior distribution.\"\"\"
                        # TODO: Implement Bayesian updating
                        # Beta(alpha + successes, beta + failures)
                        pass
                    
                    def calculate_probability_b_better(self, result: BayesianResult) -> float:
                        \"\"\"Calculate probability that variant B is better than A.\"\"\"
                        # TODO: Implement Monte Carlo integration
                        # Sample from both posterior distributions and compare
                        pass
                    
                    def calculate_expected_loss(self, result: BayesianResult) -> float:
                        \"\"\"Calculate expected loss of choosing wrong variant.\"\"\"
                        # TODO: Implement expected loss calculation
                        pass
                    
                    def run_bayesian_test(self, scores_a: List[float], scores_b: List[float]) -> BayesianResult:
                        \"\"\"Run Bayesian A/B test.\"\"\"
                        # TODO: Implement Bayesian analysis
                        # 1. Convert scores to successes/failures
                        # 2. Update posterior distributions
                        # 3. Calculate probability B is better
                        # 4. Calculate expected loss
                        pass

                class MultiArmedBandit:
                    \"\"\"Multi-armed bandit testing for multiple variants.\"\"\"
                    
                    def __init__(self, epsilon: float = 0.1):
                        self.epsilon = epsilon  # Exploration rate
                        self.arm_counts = {}
                        self.arm_rewards = {}
                    
                    def select_arm(self, available_arms: List[str]) -> str:
                        \"\"\"Select arm using epsilon-greedy strategy.\"\"\"
                        # TODO: Implement epsilon-greedy arm selection
                        # 1. With probability epsilon, explore randomly
                        # 2. Otherwise, exploit best arm
                        pass
                    
                    def update_arm(self, arm: str, reward: float) -> None:
                        \"\"\"Update arm statistics with new reward.\"\"\"
                        # TODO: Update arm counts and average rewards
                        pass
                    
                    def run_bandit_test(self, variants: Dict[str, Any], 
                                       test_examples: List[dspy.Example], 
                                       metric: Any, n_rounds: int = 1000) -> Dict[str, Any]:
                        \"\"\"Run multi-armed bandit test.\"\"\"
                        # TODO: Implement bandit testing
                        # 1. For each round, select arm
                        # 2. Get reward from selected variant
                        # 3. Update arm statistics
                        # 4. Return final results
                        pass

                # TODO: Test all advanced A/B testing methods
                def test_advanced_ab_testing():
                    \"\"\"Test the advanced A/B testing system.\"\"\"
                    # TODO: Create test scenarios
                    # TODO: Test sequential testing
                    # TODO: Test Bayesian analysis
                    # TODO: Test multi-armed bandit
                    # TODO: Compare results across methods
                    pass

                if __name__ == "__main__":
                    test_advanced_ab_testing()
                """
            ),
            language="python",
            label="Exercise 4 Code",
        )

        exercise4_ui = mo.vstack([cell6_out, exercise4_code])
    else:
        exercise4_ui = mo.md("")

    output.replace(exercise4_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell7_out = mo.md(
            cleandoc(
                """
                ## 🎓 Exercise Completion Guide

                ### ✅ Exercise Checklist

                **Exercise 1: Domain-Specific Evaluation Metrics**
                - [ ] Created CodeReviewHelpfulnessMetric with constructive feedback evaluation
                - [ ] Implemented CodeReviewAccuracyMetric with issue identification assessment
                - [ ] Built CodeReviewActionabilityMetric with implementability scoring
                - [ ] Created CompositeCodeReviewMetric combining all dimensions

                **Exercise 2: Statistical Evaluation Framework**
                - [ ] Implemented bootstrap confidence intervals for robust uncertainty estimation
                - [ ] Created effect size calculation (Cohen's d) for practical significance
                - [ ] Built permutation testing for distribution-free significance testing
                - [ ] Generated comprehensive statistical reports with interpretation

                **Exercise 3: Multi-System Evaluation Dashboard**
                - [ ] Built MultiSystemEvaluator for comparing multiple systems
                - [ ] Implemented ranking algorithms across multiple metrics
                - [ ] Created visualization data structures for charts and graphs
                - [ ] Added export functionality and insight generation

                **Exercise 4: Advanced A/B Testing System**
                - [ ] Implemented sequential A/B testing with early stopping
                - [ ] Created Bayesian A/B testing with posterior distributions
                - [ ] Built multi-armed bandit testing for multiple variants
                - [ ] Generated comprehensive test reports and visualizations

                ### 🚀 Next Steps

                After completing these exercises:
                1. **Review Solutions** - Check the solutions directory for reference implementations
                2. **Apply to Real Projects** - Use these frameworks in actual evaluation scenarios
                3. **Extend Functionality** - Add domain-specific metrics for your use cases
                4. **Build Production Systems** - Scale these frameworks for production use

                ### 💡 Advanced Evaluation Best Practices

                - **Statistical Rigor** - Always use proper statistical methods for comparisons
                - **Domain Expertise** - Incorporate domain knowledge into metric design
                - **Multi-Dimensional** - Evaluate systems across multiple quality dimensions
                - **Continuous Monitoring** - Track evaluation performance over time
                - **Actionable Insights** - Generate insights that lead to system improvements

                Excellent work mastering advanced evaluation and metrics! 🎉
                """
            )
        )
    else:
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
