# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    from common import ComparisonViewer, get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        ComparisonViewer,
        cleandoc,
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
            r"""
            # 🔍 Module Comparison & Analysis Tools

            Understanding when to use different DSPy modules is crucial for building effective AI systems. This notebook provides comprehensive tools for comparing and analyzing DSPy module performance.

            ## 🎯 Learning Objectives

            - Compare Predict vs ChainOfThought vs other modules
            - Analyze performance characteristics (speed, quality, consistency)
            - Understand trade-offs between different approaches
            - Build systematic evaluation workflows
            - Make data-driven decisions about module selection

            ## 🧪 What We'll Explore

            1. **Module Types Overview** - Understanding each module's strengths
            2. **Performance Comparison** - Speed and quality analysis
            3. **Use Case Matching** - When to use which module
            4. **Interactive Analysis** - Real-time comparison tools
            5. **Best Practices** - Guidelines for module selection
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
                ## ✅ Environment Ready

                **Configuration:**
                - Provider: {config.default_provider}
                - Model: {config.default_model}
                - Ready for module comparison!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first.
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
                ## 📚 DSPy Module Types Overview

                Let's understand the different types of DSPy modules and their characteristics:
                """
            )
        )

        module_info = {
            "Predict": {
                "description": "Direct input → output mapping",
                "best_for": "Simple transformations, fast responses",
                "characteristics": "Fast, efficient, straightforward",
            },
            "ChainOfThought": {
                "description": "Adds explicit reasoning steps",
                "best_for": "Complex problems requiring reasoning",
                "characteristics": "More reliable, interpretable, slower",
            },
            "ReAct": {
                "description": "Reasoning + Acting with tool use",
                "best_for": "Tasks requiring external information",
                "characteristics": "Powerful, flexible, complex",
            },
            "ProgramOfThought": {
                "description": "Generates and executes code",
                "best_for": "Mathematical and computational tasks",
                "characteristics": "Precise, verifiable, specialized",
            },
        }

        # Display module information
        cell3_info_text = []
        for module, details in module_info.items():
            cell3_info_text.append(
                cleandoc(
                    f"""
                    **{module}**  
                    - *Description*: {details['description']}  
                    - *Best for*: {details['best_for']}  
                    - *Characteristics*: {details['characteristics']}  
                    """
                )
            )

        cell3_ui = mo.vstack([cell3_out, mo.md("\n".join(cell3_info_text))])
    else:
        cell3_ui = mo.md("")
        module_info = None

    output.replace(cell3_ui)
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell4_out = mo.md(
            cleandoc(
                """
                ## 🧪 Comparison Setup

                Let's create a signature and test it with different module types:
                """
            )
        )

        # Create a signature for comparison
        class ProblemSolver(dspy.Signature):
            """Solve a given problem with clear reasoning and a definitive answer."""

            problem = dspy.InputField(
                desc="A problem that needs to be solved (math, logic, analysis, etc.)"
            )
            answer = dspy.OutputField(desc="The solution or answer to the problem")

        # Create different module types
        predict_solver = dspy.Predict(ProblemSolver)
        cot_solver = dspy.ChainOfThought(ProblemSolver)

        cell4_signature_info = mo.md(
            cleandoc(
                """
                **Test Signature:**
                ```python
                class ProblemSolver(dspy.Signature):
                    \"\"\"Solve a given problem with clear reasoning and a definitive answer.\"\"\"

                    problem = dspy.InputField(desc="A problem that needs to be solved")
                    answer = dspy.OutputField(desc="The solution or answer to the problem")
                ```

                **Modules Created:**
                - `dspy.Predict(ProblemSolver)` - Direct problem solving
                - `dspy.ChainOfThought(ProblemSolver)` - Reasoning-based solving
                """
            )
        )

        cell4_ui = mo.vstack([cell4_out, cell4_signature_info])
    else:
        cell4_ui = mo.md("")
        predict_solver = None
        cot_solver = None

    output.replace(cell4_ui)
    return cot_solver, predict_solver


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_out = mo.md(
            cleandoc(
                """
                ## 🎯 Test Problem Selection

                Choose a problem type to compare how different modules handle it:
                """
            )
        )

        test_problems = {
            "Math Word Problem": "A store sells apples for $2 each and oranges for $3 each. If someone buys 5 apples and 3 oranges, how much do they spend in total?",
            "Logic Puzzle": "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
            "Analysis Task": "What are the main advantages and disadvantages of remote work for both employees and employers?",
            "Creative Problem": "How would you design a system to reduce food waste in restaurants?",
            "Technical Question": "Explain the difference between supervised and unsupervised machine learning.",
        }

        problem_selector = mo.ui.dropdown(
            options=list(test_problems.keys()),
            label="Select Problem Type",
            value="Math Word Problem",
        )

        cell5_ui = mo.vstack([cell5_out, problem_selector])
    else:
        cell5_ui = mo.md("")
        test_problems = None
        problem_selector = None

    output.replace(cell5_ui)
    return problem_selector, test_problems


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    problem_selector,
    test_problems,
):
    if available_providers and problem_selector is not None and problem_selector.value:
        selected_problem = test_problems[problem_selector.value]

        cell6_out = mo.md(
            cleandoc(
                f"""
                ### Selected Problem: {problem_selector.value}

                **Problem:** {selected_problem}

                Now let's compare how different modules handle this problem:
                """
            )
        )
    else:
        cell6_out = mo.md("")
        selected_problem = None

    output.replace(cell6_out)
    return (selected_problem,)


@app.cell
def _(available_providers, mo, output, selected_problem):
    if available_providers and selected_problem:
        # Comparison execution button
        run_comparison = mo.ui.run_button(label="🔍 Run Module Comparison")

        cell7_content = mo.vstack(
            [
                mo.md("### Execute Comparison"),
                mo.md(
                    "This will run both Predict and ChainOfThought modules on the selected problem."
                ),
                run_comparison,
            ]
        )
    else:
        cell7_content = mo.md("")
        run_comparison = None

    output.replace(cell7_content)
    return (run_comparison,)


@app.cell
def _(
    ComparisonViewer,
    available_providers,
    cot_solver,
    mo,
    output,
    predict_solver,
    run_comparison,
    selected_problem,
    time,
):
    if (
        available_providers
        and run_comparison is not None
        and run_comparison.value
        and selected_problem
    ):
        try:
            # Initialize comparison viewer
            comparison = ComparisonViewer()

            # Test Predict module
            start_time = time.time()
            predict_result = predict_solver(problem=selected_problem)
            predict_time = time.time() - start_time

            comparison.add_comparison(
                name="Predict Module",
                result=predict_result,
                metrics={"execution_time": predict_time, "has_reasoning": 0},
                config={"module_type": "Predict", "reasoning": "None"},
            )

            # Test ChainOfThought module
            start_time = time.time()
            cot_result = cot_solver(problem=selected_problem)
            cot_time = time.time() - start_time

            comparison.add_comparison(
                name="ChainOfThought Module",
                result=cot_result,
                metrics={"execution_time": cot_time, "has_reasoning": 1},
                config={"module_type": "ChainOfThought", "reasoning": "Explicit"},
            )

            # Display comparison
            cell8_out = mo.vstack(
                [
                    mo.md("### 📊 Module Comparison Results"),
                    comparison.render(),
                    mo.md(
                        f"""
                    ### 🔍 Performance Analysis

                    **Speed Comparison:**  
                    - Predict: {predict_time:.3f} seconds  
                    - ChainOfThought: {cot_time:.3f} seconds  
                    - Speed difference: {abs(cot_time - predict_time):.3f} seconds  

                    **Key Observations:**  
                    - ChainOfThought typically takes longer due to reasoning steps  
                    - Predict is faster but may be less reliable for complex problems  
                    - The reasoning in ChainOfThought helps with interpretability  
                    """
                    ),
                ]
            )

        except Exception as e:
            cell8_out = mo.md(
                f"""
                ### ❌ Comparison Error

                Error: `{str(e)}`

                This might be due to API issues. Try again in a moment.
                """
            )
    else:
        cell8_out = mo.md("*Click 'Run Module Comparison' to see the analysis.*")

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell9_out = mo.md(
            cleandoc(
                """
                ## 📈 Systematic Performance Analysis

                Let's run a more comprehensive analysis across multiple problem types:
                """
            )
        )

        # Multi-problem analysis setup
        analysis_problems = [
            ("Simple Math", "What is 15% of 240?"),
            (
                "Word Problem",
                "A train travels 120 miles in 2 hours. What is its average speed?",
            ),
            (
                "Logic",
                "If A implies B, and B implies C, what can we conclude about A and C?",
            ),
            ("Analysis", "List 3 pros and cons of electric vehicles."),
        ]

        run_full_analysis = mo.ui.run_button(label="🔬 Run Full Performance Analysis")

        cell9_ui = mo.vstack(
            [
                cell9_out,
                mo.md("**Test Problems:**"),
                mo.md(
                    "\n".join(
                        [
                            f"- **{name}**: {problem}"
                            for name, problem in analysis_problems
                        ]
                    )
                ),
                mo.md("---"),
                run_full_analysis,
            ]
        )
    else:
        cell9_ui = mo.md("")
        analysis_problems = None
        run_full_analysis = None

    output.replace(cell9_ui)
    return analysis_problems, run_full_analysis


@app.cell
def _(
    analysis_problems,
    available_providers,
    cot_solver,
    mo,
    output,
    predict_solver,
    run_full_analysis,
    time,
):
    if (
        available_providers
        and run_full_analysis is not None
        and run_full_analysis.value
        and analysis_problems
    ):
        try:
            # Run comprehensive analysis
            results_data = []

            for problem_name, problem_text in analysis_problems:
                # Test Predict
                analysis_start_time = time.time()
                analysis_predict_result = predict_solver(problem=problem_text)
                analysis_predict_time = time.time() - analysis_start_time

                # Test ChainOfThought
                analysis_start_time = time.time()
                analysis_cot_result = cot_solver(problem=problem_text)
                analysis_cot_time = time.time() - analysis_start_time

                results_data.append(
                    {
                        "Problem": problem_name,
                        "Predict Time (s)": f"{analysis_predict_time:.3f}",
                        "CoT Time (s)": f"{analysis_cot_time:.3f}",
                        "Speed Ratio": f"{analysis_cot_time/analysis_predict_time:.2f}x",
                        "Predict Answer": (
                            str(analysis_predict_result.answer)[:50] + "..."
                            if len(str(analysis_predict_result.answer)) > 50
                            else str(analysis_predict_result.answer)
                        ),
                        "CoT Answer": (
                            str(analysis_cot_result.answer)[:50] + "..."
                            if len(str(analysis_cot_result.answer)) > 50
                            else str(analysis_cot_result.answer)
                        ),
                    }
                )

            # Calculate averages
            avg_predict_time = sum(
                float(r["Predict Time (s)"]) for r in results_data
            ) / len(results_data)
            avg_cot_time = sum(float(r["CoT Time (s)"]) for r in results_data) / len(
                results_data
            )

            cell10_out = mo.vstack(
                [
                    mo.md("### 📊 Comprehensive Analysis Results"),
                    mo.ui.table(results_data),
                    mo.md(
                        f"""
                    ### 📈 Summary Statistics

                    **Average Execution Times:**  
                    - Predict Module: {avg_predict_time:.3f} seconds  
                    - ChainOfThought Module: {avg_cot_time:.3f} seconds  
                    - Average Speed Ratio: {avg_cot_time/avg_predict_time:.2f}x slower for CoT  

                    **Key Insights:**  
                    - ChainOfThought consistently takes longer but provides reasoning  
                    - Speed difference varies by problem complexity  
                    - Both modules can handle different types of problems effectively  
                    """
                    ),
                ]
            )

        except Exception as e:
            cell10_out = mo.md(f"Analysis error: {str(e)}")
    else:
        cell10_out = mo.md(
            "*Click 'Run Full Performance Analysis' to see comprehensive results.*"
        )

    output.replace(cell10_out)
    return


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        cell11_content = mo.md(
            """
            ## 🎯 Module Selection Guidelines

            Based on our analysis, here are guidelines for choosing the right module:
            """
        )

        # Interactive decision tree
        cell11_task_complexity = mo.ui.radio(
            options=["Simple", "Moderate", "Complex"],
            label="How complex is your task?",
            value="Simple",
        )
        cell11_need_reasoning = mo.ui.radio(
            options=["Yes", "No", "Maybe"],
            label="Do you need to see the reasoning process?",
            value="No",
        )
        cell11_speed_priority = mo.ui.radio(
            options=["High", "Medium", "Low"],
            label="How important is execution speed?",
            value="Medium",
        )
        cell11_accuracy_priority = mo.ui.radio(
            options=["High", "Medium", "Low"],
            label="How important is accuracy?",
            value="High",
        )

        cell11_ui = mo.vstack(
            [
                cell11_content,
                mo.md("### 🤔 Module Selection Helper"),
                mo.md("Answer these questions to get a recommendation:"),
                cell11_task_complexity,
                cell11_need_reasoning,
                cell11_speed_priority,
                cell11_accuracy_priority,
            ]
        )
    else:
        cell11_ui = mo.md("")
        cell11_task_complexity = None
        cell11_need_reasoning = None
        cell11_speed_priority = None
        cell11_accuracy_priority = None

    output.replace(cell11_ui)
    return (
        cell11_accuracy_priority,
        cell11_need_reasoning,
        cell11_speed_priority,
        cell11_task_complexity,
    )


@app.cell
def _(
    available_providers,
    cell11_accuracy_priority,
    cell11_need_reasoning,
    cell11_speed_priority,
    cell11_task_complexity,
    mo,
    output,
):
    if (
        available_providers
        and cell11_task_complexity is not None
        and cell11_task_complexity.value
    ):
        cell12_responses = {
            "task_complexity": cell11_task_complexity.value,
            "need_reasoning": cell11_need_reasoning.value,
            "speed_priority": cell11_speed_priority.value,
            "accuracy_priority": cell11_accuracy_priority.value,
        }

        # Decision logic
        if (
            cell12_responses["task_complexity"] == "Simple"
            and cell12_responses["speed_priority"] == "High"
        ):
            cell12_recommendation = (
                "**Predict Module** - Fast and efficient for simple tasks"
            )
            cell12_reasoning = (
                "Simple tasks with high speed priority are perfect for Predict modules"
            )
        elif (
            cell12_responses["need_reasoning"] == "Yes"
            or cell12_responses["task_complexity"] == "Complex"
        ):
            cell12_recommendation = (
                "**ChainOfThought Module** - Better for complex reasoning"
            )
            cell12_reasoning = (
                "Complex tasks or need for reasoning transparency favor ChainOfThought"
            )
        elif (
            cell12_responses["accuracy_priority"] == "High"
            and cell12_responses["task_complexity"] != "Simple"
        ):
            cell12_recommendation = (
                "**ChainOfThought Module** - More reliable for important tasks"
            )
            cell12_reasoning = (
                "High accuracy requirements benefit from explicit reasoning steps"
            )
        else:
            cell12_recommendation = (
                "**Predict Module** - Good balance for most use cases"
            )
            cell12_reasoning = "For moderate complexity with balanced priorities, Predict is often sufficient"

        cell12_content = mo.md(
            f"""
            ### 💡 Recommendation

            {cell12_recommendation}

            **Reasoning:** {cell12_reasoning}

            ### � cDecision Matrix

            | Factor | Your Choice | Impact on Module Selection |
            |--------|-------------|---------------------------|
            | Task Complexity | {cell12_responses["task_complexity"]} | {"Favors CoT" if cell12_responses["task_complexity"] == "Complex" else "Neutral" if cell12_responses["task_complexity"] == "Moderate" else "Favors Predict"} |
            | Need Reasoning | {cell12_responses["need_reasoning"]} | {"Favors CoT" if cell12_responses["need_reasoning"] == "Yes" else "Neutral" if cell12_responses["need_reasoning"] == "Maybe" else "Favors Predict"} |
            | Speed Priority | {cell12_responses["speed_priority"]} | {"Favors Predict" if cell12_responses["speed_priority"] == "High" else "Neutral"} |
            | Accuracy Priority | {cell12_responses["accuracy_priority"]} | {"Favors CoT" if cell12_responses["accuracy_priority"] == "High" else "Neutral"} |
            """
        )
    else:
        cell12_content = mo.md(
            "*Complete the form above to get a personalized recommendation.*"
        )

    output.replace(cell12_content)
    return


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        cell13_content = mo.md(
            """
            ## 🔧 Advanced Module Features

            Let's explore some advanced features and configurations:
            """
        )

        # Advanced features demonstration
        cell13_feature_selector = mo.ui.dropdown(
            options=[
                "Custom Reasoning Format",
                "Module Chaining",
                "Error Handling",
                "Performance Optimization",
            ],
            label="Select Advanced Feature to Explore",
        )

        cell13_ui = mo.vstack([cell13_content, cell13_feature_selector])
    else:
        cell13_ui = mo.md("")
        cell13_feature_selector = None

    output.replace(cell13_ui)
    return (cell13_feature_selector,)


@app.cell
def _(available_providers, cell13_feature_selector, mo, output):
    if (
        available_providers
        and cell13_feature_selector is not None
        and cell13_feature_selector.value
    ):
        cell14_feature = cell13_feature_selector.value

        if cell14_feature == "Custom Reasoning Format":
            cell14_demo_content = """
            ### Custom Reasoning Format

            You can customize how ChainOfThought presents reasoning:

            ```python
            class CustomReasoningSignature(dspy.Signature):
                \"\"\"Solve with structured reasoning.\"\"\"
                problem = dspy.InputField()
                reasoning = dspy.OutputField(desc="Step-by-step reasoning in numbered format")
                answer = dspy.OutputField(desc="Final answer")

            # This gives you more control over reasoning format
            custom_cot = dspy.ChainOfThought(CustomReasoningSignature)
            ```
            """

        elif cell14_feature == "Module Chaining":
            cell14_demo_content = """
            ### Module Chaining

            Combine multiple modules for complex workflows:

            ```python
            # Step 1: Analyze the problem
            analyzer = dspy.Predict(AnalyzeProblem)

            # Step 2: Solve based on analysis
            solver = dspy.ChainOfThought(SolveProblem)

            # Chain them together
            analysis = analyzer(problem=user_input)
            solution = solver(problem=user_input, context=analysis.analysis)
            ```
            """

        elif cell14_feature == "Error Handling":
            cell14_demo_content = """
            ### Error Handling

            Robust error handling for production use:

            ```python
            try:
                result = predictor(problem=problem_text)
                if not result.answer:
                    # Fallback to simpler approach
                    result = fallback_predictor(problem=problem_text)
            except Exception as e:
                # Log error and use default response
                logger.error(f"Prediction failed: {e}")
                result = default_response
            ```
            """

        else:  # Performance Optimization
            cell14_demo_content = """
            ### Performance Optimization

            Optimize for speed and cost:

            ```python
            # Use caching for repeated queries
            from functools import lru_cache

            @lru_cache(maxsize=100)
            def cached_prediction(problem_hash):
                return predictor(problem=problem_text)

            # Batch processing for multiple problems
            results = []
            for problem in problem_batch:
                results.append(predictor(problem=problem))
            ```
            """

        cell14_content = mo.md(cell14_demo_content)
    else:
        cell14_content = mo.md("*Select an advanced feature above to explore.*")

    output.replace(cell14_content)
    return


@app.cell
def _(available_providers, mo, output):
    cell15_content = (
        mo.md(
            """
        ## 📊 Module Comparison Summary

        ### 🎯 Key Takeaways

        **Predict Module:**  
        - ✅ Fast execution  
        - ✅ Simple to use  
        - ✅ Good for straightforward tasks  
        - ❌ No reasoning visibility  
        - ❌ May be less reliable for complex problems  

        **ChainOfThought Module:**  
        - ✅ Explicit reasoning steps  
        - ✅ Better for complex problems  
        - ✅ More interpretable results  
        - ✅ Higher reliability  
        - ❌ Slower execution  
        - ❌ Higher token usage/cost  

        ### 🔧 Selection Guidelines

        **Use Predict when:**  
        - Task is straightforward  
        - Speed is critical  
        - Cost optimization is important  
        - Reasoning transparency not needed  

        **Use ChainOfThought when:**  
        - Problem is complex  
        - Accuracy is critical  
        - Need to understand reasoning  
        - Debugging/validation required  

        ### 🚀 Next Steps

        Now that you understand module comparison, you're ready for:

        **Next Module: Interactive Signature Tester**  
        ```bash
        uv run marimo run 01-foundations/interactive_signature_tester.py
        ```

        **Coming Up:**  
        - Real-time signature testing  
        - Parameter optimization  
        - Result analysis tools  
        - Systematic validation workflows  
        """
        )
        if available_providers
        else mo.md("")
    )

    output.replace(cell15_content)
    return


if __name__ == "__main__":
    app.run()
