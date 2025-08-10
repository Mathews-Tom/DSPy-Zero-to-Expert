# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import difflib
    import re
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

    return (
        cleandoc,
        difflib,
        dspy,
        get_config,
        mo,
        output,
        re,
        setup_dspy_environment,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 Custom Metrics Design Exercises

            **Master the art of creating domain-specific evaluation metrics** for DSPy optimization.

            ## 📚 Exercise Overview

            These exercises will help you master:  
            - Domain-specific metric design patterns  
            - Composite metric creation with weighted scoring  
            - Metric validation and testing frameworks  
            - Advanced evaluation strategies for different task types  

            Complete each exercise to become a metrics design expert!
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

                Ready to start custom metrics exercises!
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
        cell3_out = mo.md(
            cleandoc(
                """
                ## 🎯 Exercise 1: Scientific Paper Evaluation Metric

                **Task:** Create a comprehensive metric for evaluating scientific paper summaries.

                **Requirements:**  
                1. Create a metric that evaluates accuracy, completeness, and clarity  
                2. Include domain-specific criteria (methodology, results, conclusions)  
                3. Implement partial credit for different aspects  
                4. Handle edge cases gracefully  

                **Your Implementation:**
                """
            )
        )

        # Exercise 1 Template
        exercise1_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 1: Scientific Paper Evaluation Metric

                def scientific_paper_metric(example, pred, trace=None):
                    \"\"\"
                    Evaluate scientific paper summaries on multiple dimensions.

                    Evaluation criteria:
                    - Accuracy: Factual correctness of information
                    - Completeness: Coverage of key sections (intro, methods, results, conclusions)
                    - Clarity: Readability and coherence
                    - Technical precision: Proper use of scientific terminology
                    \"\"\"
                    try:
                        # TODO: Extract predicted and expected summaries
                        predicted_summary = ""  # Get from pred
                        expected_summary = ""   # Get from example

                        # TODO: Implement accuracy scoring (40% weight)
                        accuracy_score = 0.0

                        # TODO: Implement completeness scoring (30% weight)
                        # Check for presence of: introduction, methodology, results, conclusions
                        completeness_score = 0.0

                        # TODO: Implement clarity scoring (20% weight)
                        # Consider: sentence structure, flow, readability
                        clarity_score = 0.0

                        # TODO: Implement technical precision scoring (10% weight)
                        # Check for: proper terminology, scientific accuracy
                        technical_score = 0.0

                        # TODO: Calculate weighted final score
                        final_score = 0.0

                        return final_score

                    except Exception as e:
                        return 0.0

                # TODO: Create test examples for scientific papers
                scientific_paper_examples = [
                    # Add examples with paper abstracts and expected summaries
                ]

                # Test your metric
                def test_scientific_metric():
                    \"\"\"Test the scientific paper evaluation metric.\"\"\"
                    # TODO: Create test cases
                    # TODO: Run metric evaluation
                    # TODO: Analyze results
                    print("Scientific paper metric test complete!")

                if __name__ == "__main__":
                    test_scientific_metric()
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
                ## 🔧 Exercise 2: Code Review Quality Metric

                **Task:** Design a metric for evaluating automated code review comments.

                **Requirements:**  
                1. Evaluate helpfulness, accuracy, and actionability of comments  
                2. Consider different types of issues (bugs, style, performance, security)  
                3. Implement severity weighting for different issue types  
                4. Include false positive detection  

                **Your Implementation:**
                """
            )
        )

        # Exercise 2 Template
        exercise2_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 2: Code Review Quality Metric

                def code_review_metric(example, pred, trace=None):
                    \"\"\"
                    Evaluate the quality of automated code review comments.

                    Evaluation criteria:
                    - Helpfulness: How useful is the comment for improving code
                    - Accuracy: Is the identified issue actually present
                    - Actionability: Can the developer act on the feedback
                    - Severity appropriateness: Is the severity level appropriate
                    \"\"\"
                    try:
                        # TODO: Extract review comment and expected feedback
                        predicted_comment = ""  # Get from pred
                        expected_issues = []    # Get from example

                        # TODO: Implement helpfulness scoring (35% weight)
                        # Consider: constructive feedback, specific suggestions
                        helpfulness_score = 0.0

                        # TODO: Implement accuracy scoring (30% weight)
                        # Check if identified issues are real issues
                        accuracy_score = 0.0

                        # TODO: Implement actionability scoring (25% weight)
                        # Can the developer understand and fix the issue?
                        actionability_score = 0.0

                        # TODO: Implement severity appropriateness (10% weight)
                        # Is the severity level (critical, major, minor) appropriate?
                        severity_score = 0.0

                        # TODO: Calculate weighted final score
                        final_score = 0.0

                        return final_score

                    except Exception as e:
                        return 0.0

                # TODO: Create test examples for code reviews
                code_review_examples = [
                    # Add examples with code snippets and expected review comments
                ]

                # TODO: Test your metric
                def test_code_review_metric():
                    \"\"\"Test the code review evaluation metric.\"\"\"
                    # TODO: Create test cases with different code issues
                    # TODO: Run metric evaluation
                    # TODO: Analyze results
                    print("Code review metric test complete!")

                if __name__ == "__main__":
                    test_code_review_metric()
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
                ## ⚡ Exercise 3: Multi-Language Translation Quality Metric

                **Task:** Create a sophisticated metric for evaluating translation quality across multiple languages.

                **Requirements:**  
                1. Evaluate fluency, accuracy, and cultural appropriateness  
                2. Handle different language characteristics (word order, grammar)  
                3. Implement language-specific scoring adjustments  
                4. Consider context and domain-specific terminology  

                **Your Implementation:**
                """
            )
        )

        # Exercise 3 Template
        exercise3_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 3: Multi-Language Translation Quality Metric

                def translation_quality_metric(example, pred, trace=None):
                    \"\"\"
                    Evaluate translation quality across multiple languages.

                    Evaluation criteria:
                    - Fluency: Natural flow and grammar in target language
                    - Accuracy: Preservation of meaning from source
                    - Cultural appropriateness: Proper cultural context
                    - Terminology consistency: Consistent use of domain terms
                    \"\"\"
                    try:
                        # TODO: Extract translation details
                        source_text = ""        # Get from example
                        predicted_translation = ""  # Get from pred
                        expected_translation = ""   # Get from example
                        target_language = ""    # Get from example

                        # TODO: Implement fluency scoring (30% weight)
                        # Consider: grammar, natural flow, readability
                        fluency_score = 0.0

                        # TODO: Implement accuracy scoring (40% weight)
                        # Semantic similarity between source and translation
                        accuracy_score = 0.0

                        # TODO: Implement cultural appropriateness (20% weight)
                        # Proper cultural context and expressions
                        cultural_score = 0.0

                        # TODO: Implement terminology consistency (10% weight)
                        # Consistent use of technical/domain terms
                        terminology_score = 0.0

                        # TODO: Apply language-specific adjustments
                        language_adjustment = 1.0  # Adjust based on target_language

                        # TODO: Calculate weighted final score
                        final_score = 0.0

                        return final_score

                    except Exception as e:
                        return 0.0

                # TODO: Create test examples for translations
                translation_examples = [
                    # Add examples with source text, target language, and expected translations
                ]

                # TODO: Test your metric
                def test_translation_metric():
                    \"\"\"Test the translation quality evaluation metric.\"\"\"
                    # TODO: Create test cases for different languages
                    # TODO: Run metric evaluation
                    # TODO: Analyze language-specific performance
                    print("Translation quality metric test complete!")

                if __name__ == "__main__":
                    test_translation_metric()
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
                ## 📊 Exercise 4: Adaptive Composite Metric System

                **Task:** Build a system that automatically adjusts metric weights based on task characteristics.

                **Requirements:**  
                1. Create a `CompositeMetricSystem` class  
                2. Implement automatic weight adjustment based on task type  
                3. Build a metric validation and testing framework  
                4. Create performance analysis and reporting tools  

                **Your Implementation:**
                """
            )
        )

        # Exercise 4 Template
        exercise4_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 4: Adaptive Composite Metric System

                class CompositeMetricSystem:
                    \"\"\"
                    Adaptive composite metric system that adjusts weights based on task characteristics.
                    \"\"\"

                    def __init__(self):
                        self.base_metrics = {}
                        self.weight_profiles = {}
                        self.validation_results = []

                    def register_metric(self, name, metric_function, description=""):
                        \"\"\"Register a base metric function.\"\"\"
                        # TODO: Store metric function with metadata
                        pass

                    def create_weight_profile(self, profile_name, task_characteristics, weights):
                        \"\"\"Create a weight profile for specific task characteristics.\"\"\"
                        # TODO: Store weight profile with task characteristics
                        pass

                    def auto_select_weights(self, task_type, content_characteristics):
                        \"\"\"Automatically select appropriate weights based on task characteristics.\"\"\"
                        # TODO: Analyze task characteristics
                        # TODO: Select best matching weight profile
                        # TODO: Return optimized weights
                        pass

                    def create_composite_metric(self, task_type, content_characteristics=None):
                        \"\"\"Create a composite metric optimized for the given task.\"\"\"
                        # TODO: Get appropriate weights
                        # TODO: Create composite metric function
                        # TODO: Return configured metric
                        pass

                    def validate_metric(self, metric, test_examples, expected_scores=None):
                        \"\"\"Validate a metric against test examples.\"\"\"
                        # TODO: Run metric on test examples
                        # TODO: Compare with expected scores if provided
                        # TODO: Calculate validation metrics
                        # TODO: Store validation results
                        pass

                    def generate_metric_report(self, metric_name):
                        \"\"\"Generate a comprehensive report for a metric's performance.\"\"\"
                        # TODO: Analyze validation results
                        # TODO: Generate performance statistics
                        # TODO: Create recommendations
                        pass

                # TODO: Create test examples for different task types
                adaptive_metric_examples = {
                    "qa": [],           # Question answering examples
                    "summarization": [], # Text summarization examples
                    "classification": [], # Classification examples
                    "generation": []     # Text generation examples
                }

                # TODO: Test the adaptive metric system
                def test_adaptive_metrics():
                    \"\"\"Test the adaptive composite metric system.\"\"\"
                    system = CompositeMetricSystem()

                    # TODO: Register base metrics
                    # TODO: Create weight profiles for different tasks
                    # TODO: Test automatic weight selection
                    # TODO: Validate metrics on test data
                    # TODO: Generate performance reports

                    print("Adaptive metric system test complete!")

                if __name__ == "__main__":
                    test_adaptive_metrics()
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

                **Exercise 1: Scientific Paper Evaluation**  
                - [ ] Created multi-dimensional evaluation (accuracy, completeness, clarity)  
                - [ ] Implemented domain-specific criteria for scientific content  
                - [ ] Added partial credit system for different aspects  
                - [ ] Built robust error handling  

                **Exercise 2: Code Review Quality**  
                - [ ] Evaluated helpfulness, accuracy, and actionability  
                - [ ] Implemented severity weighting for different issue types  
                - [ ] Added false positive detection  
                - [ ] Created comprehensive test cases  

                **Exercise 3: Multi-Language Translation**  
                - [ ] Built fluency, accuracy, and cultural appropriateness evaluation  
                - [ ] Implemented language-specific scoring adjustments  
                - [ ] Added terminology consistency checking  
                - [ ] Created multi-language test examples  

                **Exercise 4: Adaptive Composite System**  
                - [ ] Built CompositeMetricSystem class  
                - [ ] Implemented automatic weight adjustment  
                - [ ] Created metric validation framework  
                - [ ] Built performance analysis and reporting  

                ### 🚀 Next Steps

                After completing these exercises:  
                1. **Review Solutions** - Check the solutions directory for reference implementations  
                2. **Apply to Real Tasks** - Use your custom metrics in actual optimization workflows  
                3. **Build Metric Libraries** - Create reusable metric collections for your domain  
                4. **Integrate with Optimization** - Use your metrics with BootstrapFewShot and MIPRO  

                ### 💡 Custom Metrics Best Practices

                - **Domain Expertise** - Incorporate domain knowledge into your metrics  
                - **Multi-Dimensional** - Evaluate multiple aspects, not just accuracy  
                - **Partial Credit** - Reward partially correct answers appropriately  
                - **Validation** - Always test your metrics on diverse examples  
                - **Iterative Improvement** - Refine metrics based on real-world performance  

                Excellent work mastering custom metrics design! 🎉
                """
            )
        )
    else:
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
