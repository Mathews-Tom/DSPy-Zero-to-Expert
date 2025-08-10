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
            # 🏋️ BootstrapFewShot Optimization Exercises

            **Practice implementing and optimizing DSPy modules** with BootstrapFewShot techniques.

            ## 📚 Exercise Overview

            These exercises will help you master:
            - Creating optimizable DSPy modules
            - Designing effective evaluation metrics
            - Configuring BootstrapFewShot parameters
            - Analyzing optimization results

            Complete each exercise to build your optimization expertise!
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
                ## ✅ Exercise Environment Ready

                **Configuration:**
                - Provider: **{config.default_provider}**
                - Model: **{config.default_model}**

                Ready to start exercises!
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
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell3_out = mo.md(
            cleandoc(
                """
                ## 🎯 Exercise 1: Create a Sentiment Analysis Module

                **Task:** Build a DSPy module for sentiment analysis that can be optimized with BootstrapFewShot.

                **Requirements:**  
                1. Create a `SentimentSignature` with text input and sentiment output  
                2. Implement a `SentimentAnalyzer` module using ChainOfThought  
                3. Create training examples with diverse text samples  
                4. Design an evaluation metric for sentiment accuracy  

                **Your Implementation:**  
                """
            )
        )

        # Exercise 1 Template
        exercise1_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 1: Sentiment Analysis Module

                # TODO: Create SentimentSignature
                class SentimentSignature(dspy.Signature):
                    \"\"\"Analyze the sentiment of the given text.\"\"\"

                    # TODO: Add input and output fields
                    pass

                # TODO: Create SentimentAnalyzer module
                class SentimentAnalyzer(dspy.Module):
                    \"\"\"Sentiment analysis module with chain of thought reasoning.\"\"\"

                    def __init__(self):
                        super().__init__()
                        # TODO: Initialize the module
                        pass

                    def forward(self, text):
                        # TODO: Implement forward method
                        pass

                # TODO: Create training examples
                training_examples = [
                    # Add diverse sentiment examples here
                ]

                # TODO: Create evaluation metric
                def sentiment_metric(example, pred, trace=None):
                    \"\"\"Evaluate sentiment prediction accuracy.\"\"\"
                    # TODO: Implement metric logic
                    pass

                # Test your implementation
                if __name__ == "__main__":
                    analyzer = SentimentAnalyzer()
                    result = analyzer(text="I love this product!")
                    print(f"Sentiment: {result}")
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
                ## 🔧 Exercise 2: Custom Evaluation Metrics

                **Task:** Design and implement custom evaluation metrics for different types of tasks.

                **Requirements:**
                1. Create a `fuzzy_match_metric` for approximate string matching
                2. Implement a `confidence_weighted_metric` that considers prediction confidence
                3. Build a `multi_criteria_metric` that combines multiple evaluation aspects
                4. Test each metric with sample predictions

                **Your Implementation:**
                """
            )
        )

        # Exercise 2 Template
        exercise2_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 2: Custom Evaluation Metrics

                def fuzzy_match_metric(example, pred, trace=None):
                    \"\"\"Fuzzy string matching metric with tolerance for minor differences.\"\"\"
                    # TODO: Implement fuzzy matching logic
                    # Hint: Consider using string similarity algorithms
                    pass

                def confidence_weighted_metric(example, pred, trace=None):
                    \"\"\"Metric that weights accuracy by prediction confidence.\"\"\"
                    # TODO: Implement confidence-weighted evaluation
                    # Hint: Extract confidence from prediction or trace
                    pass

                def multi_criteria_metric(example, pred, trace=None):
                    \"\"\"Multi-criteria evaluation combining accuracy, fluency, and relevance.\"\"\"
                    # TODO: Implement multi-criteria evaluation
                    # Consider: accuracy (0.5), fluency (0.3), relevance (0.2)
                    pass

                # Test your metrics
                def test_metrics():
                    # TODO: Create test examples and predictions
                    test_example = dspy.Example(
                        question="What is Python?",
                        answer="Python is a programming language"
                    )

                    test_pred = dspy.Prediction(
                        answer="Python is a high-level programming language"
                    )

                    # TODO: Test each metric
                    fuzzy_score = fuzzy_match_metric(test_example, test_pred)
                    confidence_score = confidence_weighted_metric(test_example, test_pred)
                    multi_score = multi_criteria_metric(test_example, test_pred)

                    print(f"Fuzzy Match: {fuzzy_score}")
                    print(f"Confidence Weighted: {confidence_score}")
                    print(f"Multi-Criteria: {multi_score}")

                if __name__ == "__main__":
                    test_metrics()
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
                ## ⚙️ Exercise 3: Optimization Parameter Tuning

                **Task:** Experiment with different BootstrapFewShot parameters to find optimal settings.

                **Requirements:**
                1. Create a simple QA module for optimization
                2. Implement a parameter grid search function
                3. Run optimizations with different parameter combinations
                4. Analyze results to find the best configuration

                **Your Implementation:**
                """
            )
        )

        # Exercise 3 Template
        exercise3_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 3: Optimization Parameter Tuning

                # TODO: Create a simple QA module
                class SimpleQASignature(dspy.Signature):
                    \"\"\"Answer questions based on context.\"\"\"
                    # TODO: Define signature fields
                    pass

                class SimpleQAModule(dspy.Module):
                    \"\"\"Simple question answering module.\"\"\"

                    def __init__(self):
                        super().__init__()
                        # TODO: Initialize module
                        pass

                    def forward(self, context, question):
                        # TODO: Implement forward method
                        pass

                # TODO: Create training data
                qa_training_data = [
                    # Add QA examples here
                ]

                # TODO: Implement parameter grid search
                def parameter_grid_search(module_class, training_data, param_grid):
                    \"\"\"Run optimization with different parameter combinations.\"\"\"
                    results = []

                    for params in param_grid:
                        # TODO: Run BootstrapFewShot with current parameters
                        # TODO: Evaluate performance
                        # TODO: Store results
                        pass

                    return results

                # TODO: Define parameter grid
                param_combinations = [
                    # Add different parameter combinations to test
                    {"max_bootstrapped_demos": 2, "max_labeled_demos": 8, "max_rounds": 1},
                    {"max_bootstrapped_demos": 4, "max_labeled_demos": 16, "max_rounds": 1},
                    # Add more combinations...
                ]

                # TODO: Run grid search and analyze results
                def run_optimization_experiment():
                    \"\"\"Run the complete optimization experiment.\"\"\"
                    # TODO: Execute grid search
                    # TODO: Find best parameters
                    # TODO: Display results
                    pass

                if __name__ == "__main__":
                    run_optimization_experiment()
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
                ## 📊 Exercise 4: Optimization Analysis Dashboard

                **Task:** Build an analysis dashboard to visualize and compare optimization results.

                **Requirements:**
                1. Create a function to track multiple optimization runs
                2. Implement visualization for performance trends
                3. Build a comparison function for different configurations
                4. Generate insights and recommendations

                **Your Implementation:**
                """
            )
        )

        # Exercise 4 Template
        exercise4_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 4: Optimization Analysis Dashboard

                class OptimizationTracker:
                    \"\"\"Track and analyze multiple optimization runs.\"\"\"

                    def __init__(self):
                        self.runs = []

                    def add_run(self, parameters, performance, metadata=None):
                        \"\"\"Add an optimization run to tracking.\"\"\"
                        # TODO: Store run data with timestamp
                        pass

                    def get_performance_trends(self):
                        \"\"\"Analyze performance trends across runs.\"\"\"
                        # TODO: Calculate trends and statistics
                        pass

                    def compare_configurations(self, config1, config2):
                        \"\"\"Compare two different configurations.\"\"\"
                        # TODO: Find runs with matching configurations
                        # TODO: Compare their performance
                        pass

                    def generate_insights(self):
                        \"\"\"Generate insights and recommendations.\"\"\"
                        # TODO: Analyze parameter impact
                        # TODO: Identify best practices
                        # TODO: Generate recommendations
                        pass

                    def visualize_results(self):
                        \"\"\"Create visualizations of optimization results.\"\"\"
                        # TODO: Create performance plots
                        # TODO: Show parameter impact
                        # TODO: Display trends
                        pass

                # TODO: Implement helper functions
                def create_performance_plot(tracker):
                    \"\"\"Create a plot showing performance over time.\"\"\"
                    # TODO: Extract data from tracker
                    # TODO: Create visualization
                    pass

                def analyze_parameter_impact(tracker, parameter_name):
                    \"\"\"Analyze the impact of a specific parameter.\"\"\"
                    # TODO: Group runs by parameter value
                    # TODO: Calculate average performance
                    # TODO: Return analysis results
                    pass

                # TODO: Test the tracker
                def test_optimization_tracker():
                    \"\"\"Test the optimization tracking system.\"\"\"
                    tracker = OptimizationTracker()

                    # TODO: Add sample runs
                    # TODO: Generate insights
                    # TODO: Create visualizations

                    print("Optimization analysis complete!")

                if __name__ == "__main__":
                    test_optimization_tracker()
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
    cell7_out = mo.md(
        cleandoc(
            """
            ## 🎓 Exercise Completion Guide

            ### ✅ Exercise Checklist

            **Exercise 1: Sentiment Analysis Module**  
            - [ ] Created SentimentSignature with proper fields  
            - [ ] Implemented SentimentAnalyzer with ChainOfThought  
            - [ ] Added diverse training examples  
            - [ ] Designed accurate evaluation metric  

            **Exercise 2: Custom Evaluation Metrics**  
            - [ ] Implemented fuzzy_match_metric  
            - [ ] Created confidence_weighted_metric  
            - [ ] Built multi_criteria_metric  
            - [ ] Tested all metrics with examples  

            **Exercise 3: Parameter Tuning**  
            - [ ] Created SimpleQAModule  
            - [ ] Implemented parameter grid search  
            - [ ] Tested multiple parameter combinations  
            - [ ] Analyzed results for best configuration  

            **Exercise 4: Analysis Dashboard**  
            - [ ] Built OptimizationTracker class  
            - [ ] Implemented performance trend analysis  
            - [ ] Created configuration comparison  
            - [ ] Generated insights and recommendations  

            ### 🚀 Next Steps

            After completing these exercises:  
            1. **Review Solutions** - Check the solutions directory for reference implementations  
            2. **Experiment Further** - Try different modules and optimization strategies  
            3. **Move to MIPRO** - Explore advanced optimization techniques  
            4. **Build Real Applications** - Apply optimization to your own projects  

            ### 💡 Tips for Success

            - **Start Simple** - Begin with basic implementations, then add complexity  
            - **Test Frequently** - Validate each component before moving forward  
            - **Analyze Results** - Always examine why certain parameters work better  
            - **Document Findings** - Keep notes on what works for different tasks  

            Great job working through these optimization exercises! 🎉
            """
        ) if available_providers else ""
    )

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
