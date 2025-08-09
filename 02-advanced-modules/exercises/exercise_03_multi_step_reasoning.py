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
            # 🧪 Exercise 03: Multi-Step Reasoning

            **Duration:** 60-75 minutes  
            **Difficulty:** Advanced  
            **Prerequisites:** Completed Exercises 01 & 02

            ## 🎯 Exercise Objectives

            In this exercise, you will:  
            - ✅ Build a step planning system  
            - ✅ Create a step execution engine  
            - ✅ Implement context management across steps  
            - ✅ Build a complete reasoning pipeline  
            - ✅ Handle complex, multi-hop problems  

            ## 📋 Tasks Overview

            1. **Task 1:** Create a step planner signature
            2. **Task 2:** Build a step executor
            3. **Task 3:** Implement context management
            4. **Task 4:** Create a reasoning pipeline
            5. **Task 5:** Test with complex problems

            ## 🚀 Let's Get Started!

            You'll build a sophisticated multi-step reasoning system!
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
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  

                Ready to build multi-step reasoning systems!
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
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 1: Create a Step Planner

                **Your Challenge:** Create a signature that can decompose complex problems into logical steps.

                **Requirements:**  
                - Analyze the problem type and complexity  
                - Generate a numbered list of steps  
                - Estimate the number of steps needed  
                - Identify dependencies between steps  

                **Example Problem:** "Plan a birthday party for 20 people with a $500 budget"

                **Expected Output:**  
                1. Determine guest preferences and dietary restrictions  
                2. Create a budget breakdown for food, decorations, and entertainment  
                3. Book a venue or prepare home space  
                4. Order food and supplies  
                5. Set up decorations and prepare activities  
                """
            )
        )

        # Student implementation area
        step_planner_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                class StepPlannerSignature(dspy.Signature):  
                    \"\"\"Decompose complex problems into logical steps.\"\"\"  

                    problem = dspy.InputField(desc="The complex problem to solve")  
                    context = dspy.InputField(desc="Any relevant context or constraints")  

                    # Add your output fields here  
                    problem_type = dspy.OutputField(desc="...")  
                    complexity_level = dspy.OutputField(desc="...")  
                    step_plan = dspy.OutputField(desc="...")  
                    estimated_steps = dspy.OutputField(desc="...")  
                    dependencies = dspy.OutputField(desc="...")  
                """
            ),
            label="Create your step planner signature:",
            rows=12,
        )

        test_planner_button = mo.ui.run_button(label="📋 Test Step Planner")

        cell3_content = mo.vstack([step_planner_code, test_planner_button])
    else:
        cell3_desc = mo.md("")
        step_planner_code = None
        test_planner_button = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (test_planner_button,)


@app.cell
def _(available_providers, cleandoc, mo, output, test_planner_button):
    if available_providers and test_planner_button.value:
        test_problem = "Plan a birthday party for 20 people with a $500 budget"

        cell4_out = mo.md(
            cleandoc(
                f"""
                📋 **Step Planner Testing Results**

                **Test Problem:** {test_problem}

                *Your step planner results would appear here*

                **Expected Analysis:**  
                - **Problem Type:** Event planning  
                - **Complexity Level:** Medium  
                - **Estimated Steps:** 5-7 steps  
                - **Dependencies:** Budget must be allocated before purchasing  

                **Validation Checklist:**  
                - ✅ Identifies problem type correctly  
                - ✅ Assesses complexity appropriately  
                - ✅ Provides logical step sequence  
                - ✅ Estimates reasonable step count  
                - ✅ Identifies key dependencies  
                """
            )
        )
    else:
        cell4_out = mo.md("*Implement your step planner above and test it*")

    output.replace(cell4_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 2: Build a Step Executor

                **Your Challenge:** Create a signature that can execute individual steps in a reasoning process.

                **Requirements:**  
                - Take a step description and execute it  
                - Use context from previous steps  
                - Provide detailed reasoning for the step  
                - Generate guidance for the next step  
                - Assess confidence in the result  

                **Key Features:**  
                - Context awareness (what happened before)  
                - Step-specific reasoning  
                - Result generation  
                - Next-step guidance  
                """
            )
        )

        # Student implementation area
        step_executor_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                class StepExecutorSignature(dspy.Signature):  
                    \"\"\"Execute individual steps in a multi-step reasoning process.\"\"\"  

                    step_description = dspy.InputField(desc="Description of the current step")  
                    step_number = dspy.InputField(desc="Current step number")  
                    previous_results = dspy.InputField(desc="Results from previous steps")  

                    # Add your output fields here  
                    reasoning = dspy.OutputField(desc="...")  
                    step_result = dspy.OutputField(desc="...")  
                    confidence = dspy.OutputField(desc="...")  
                    next_step_guidance = dspy.OutputField(desc="...")  
                """
            ),
            label="Create your step executor signature:",
            rows=12,
        )

        test_executor_button = mo.ui.run_button(label="⚙️ Test Step Executor")

        cell5_content = mo.vstack([step_executor_code, test_executor_button])
    else:
        cell5_desc = mo.md("")
        step_executor_code = None
        test_executor_button = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (test_executor_button,)


@app.cell
def _(available_providers, mo, output, test_executor_button):
    if available_providers and test_executor_button.value:
        cell6_out = mo.md(
            """
            ⚙️ **Step Executor Testing Results**

            **Test Step:** "Create a budget breakdown for food, decorations, and entertainment"  
            **Previous Context:** "Determined 20 guests with mixed dietary preferences"  

            *Your step executor results would appear here*

            **Expected Output:**  
            - **Reasoning:** Detailed analysis of budget allocation  
            - **Step Result:** Specific budget breakdown ($300 food, $100 decorations, $100 entertainment)  
            - **Confidence:** 0.8-0.9 (high confidence)  
            - **Next Step Guidance:** Proceed to venue booking with budget constraints  

            **Validation Checklist:**  
            - ✅ Provides detailed reasoning  
            - ✅ Generates specific, actionable results  
            - ✅ Assesses confidence appropriately  
            - ✅ Offers helpful next-step guidance  
            """
        )
    else:
        cell6_out = mo.md("*Implement your step executor above and test it*")

    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 3: Implement Context Management

                **Your Challenge:** Create a system to manage context and state across reasoning steps.

                **Requirements:**  
                - Track results from each step  
                - Maintain context for future steps  
                - Handle information flow between steps  
                - Provide context summaries when needed  

                **Context Manager Features:**  
                - Store step results  
                - Retrieve relevant context  
                - Summarize context for efficiency  
                - Handle context overflow  
                """
            )
        )

        # Student implementation area
        context_manager_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                class ContextManager:  
                \"\"\"Manage context and state across reasoning steps.\"\"\"  

                def __init__(self):  
                    self.step_results = []  
                    self.context_summary = ""  

                def add_step_result(self, step_number: int, description: str, result: str, confidence: float):  
                    \"\"\"Add a step result to the context.\"\"\"  
                    # Your implementation here  
                    pass  

                def get_context_for_step(self, step_number: int) -> str:  
                    \"\"\"Get relevant context for a specific step.\"\"\"  
                    # Your implementation here  
                    pass  

                def get_full_context(self) -> str:  
                    \"\"\"Get complete context summary.\"\"\"  
                    # Your implementation here  
                    pass  

                def clear_context(self):  
                    \"\"\"Clear all context data.\"\"\"  
                    # Your implementation here  
                    pass  
            """
            ),
            label="Implement your context manager:",
            rows=15,
        )

        test_context_button = mo.ui.run_button(label="🧠 Test Context Manager")

        cell7_content = mo.vstack([context_manager_code, test_context_button])
    else:
        cell7_desc = mo.md("")
        context_manager_code = None
        test_context_button = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return (test_context_button,)


@app.cell
def _(available_providers, mo, output, test_context_button):
    if available_providers and test_context_button.value:
        cell8_out = mo.md(
            """
            🧠 **Context Manager Testing Results**

            *Your context manager implementation results would appear here*

            **Test Scenario:**  
            1. Add step 1 result: "Guest preferences identified"  
            2. Add step 2 result: "Budget allocated: $300 food, $100 decorations, $100 entertainment"  
            3. Get context for step 3  
            4. Test context summarization  

            **Expected Behaviors:**  
            - Stores step results with metadata  
            - Retrieves relevant context efficiently  
            - Provides appropriate context summaries  
            - Handles context growth gracefully  

            **Validation Checklist:**  
            - ✅ Stores step results correctly  
            - ✅ Retrieves context appropriately  
            - ✅ Summarizes context when needed  
            - ✅ Manages context size effectively  
            """
        )
    else:
        cell8_out = mo.md("*Implement your context manager above and test it*")

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell9_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 4: Create a Reasoning Pipeline

                **Your Challenge:** Build a complete pipeline that orchestrates multi-step reasoning.

                **Requirements:**  
                - Use your step planner to decompose problems  
                - Execute steps using your step executor  
                - Manage context throughout the process  
                - Handle errors and edge cases  
                - Provide comprehensive final results  

                **Pipeline Architecture:**  
                1. **Planning Phase:** Decompose problem into steps  
                2. **Execution Phase:** Execute steps sequentially  
                3. **Context Management:** Maintain state across steps  
                4. **Synthesis Phase:** Combine results into final answer  
                """
            )
        )

        # Student implementation area
        pipeline_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                class MultiStepReasoningPipeline:  
                    \"\"\"Complete pipeline for multi-step reasoning.\"\"\"  

                    def __init__(self, step_planner, step_executor, context_manager):  
                        self.step_planner = step_planner  
                        self.step_executor = step_executor  
                        self.context_manager = context_manager  

                    def execute(self, problem: str, context: str = "", max_steps: int = 10):  
                        \"\"\"Execute the complete reasoning pipeline.\"\"\"  
                        try:  
                            # Step 1: Plan the approach  
                            # Your implementation here  

                            # Step 2: Execute each step  
                            # Your implementation here  

                            # Step 3: Synthesize results  
                            # Your implementation here  

                            return {  
                                "success": True,  
                                "plan": None,  # Add your plan result  
                                "step_results": [],  # Add your step results  
                                "final_answer": "",  # Add your final answer  
                            }  
                        except Exception as e:  
                            return {"success": False, "error": str(e)}  
                """
            ),
            label="Implement your reasoning pipeline:",
            rows=18,
        )

        test_pipeline_button = mo.ui.run_button(label="🔄 Test Reasoning Pipeline")

        cell9_content = mo.vstack([pipeline_code, test_pipeline_button])
    else:
        cell9_desc = mo.md("")
        pipeline_code = None
        test_pipeline_button = None
        cell9_content = mo.md("")

    cell9_out = mo.vstack([cell9_desc, cell9_content])
    output.replace(cell9_out)
    return (test_pipeline_button,)


@app.cell
def _(available_providers, mo, output, test_pipeline_button):
    if available_providers and test_pipeline_button.value:
        cell10_out = mo.md(
            """
            🔄 **Reasoning Pipeline Testing Results**

            *Your pipeline implementation results would appear here*

            **Test Problem:** "Plan a birthday party for 20 people with a $500 budget"

            **Expected Pipeline Flow:**  
            1. **Planning:** Decompose into 5-7 logical steps  
            2. **Execution:** Execute each step with context  
            3. **Context Management:** Track results across steps  
            4. **Synthesis:** Combine into comprehensive plan  

            **Validation Checklist:**  
            - ✅ Successfully decomposes complex problems  
            - ✅ Executes steps in logical sequence  
            - ✅ Maintains context across steps  
            - ✅ Handles errors gracefully  
            - ✅ Provides comprehensive final results  
            """
        )
    else:
        cell10_out = mo.md("*Implement your reasoning pipeline above and test it*")

    output.replace(cell10_out)  
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell11_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 5: Test with Complex Problems

                **Your Challenge:** Test your pipeline with increasingly complex problems.

                **Test Problems:**  
                1. **Business Strategy:** "Develop a market entry strategy for a new product in the European market"  
                2. **Technical Planning:** "Design a scalable web application architecture for 1 million users"  
                3. **Research Project:** "Investigate the impact of remote work on team productivity and propose improvements"  

                **Evaluation Criteria:**  
                - Problem decomposition quality  
                - Step execution effectiveness  
                - Context management efficiency  
                - Final result comprehensiveness  
                """
            )
        )

        # Problem selector
        complex_problems = [
            "Develop a market entry strategy for a new product in the European market",
            "Design a scalable web application architecture for 1 million users",
            "Investigate the impact of remote work on team productivity and propose improvements",
        ]

        complex_problem_selector = mo.ui.dropdown(
            options=complex_problems,
            label="Select a complex problem:",
            value=complex_problems[0],
        )

        test_complex_button = mo.ui.run_button(label="🧩 Test Complex Problem")

        cell11_content = mo.vstack([complex_problem_selector, test_complex_button])
    else:
        cell11_desc = mo.md("")
        complex_problems = None
        complex_problem_selector = None
        test_complex_button = None
        cell11_content = mo.md("")

    cell11_out = mo.vstack([cell11_desc, cell11_content])
    output.replace(cell11_out)
    return complex_problem_selector, test_complex_button


@app.cell
def _(
    available_providers,
    complex_problem_selector,
    mo,
    output,
    test_complex_button,
):
    if available_providers and test_complex_button.value:
        selected_complex_problem = complex_problem_selector.value
        cell12_out = mo.md(
            f"""
            🧩 **Complex Problem Testing Results**

            **Problem:** {selected_complex_problem}

            *Your pipeline's complex problem-solving results would appear here*

            **Analysis Framework:**  
            - **Problem Decomposition:** How well did the planner break down the problem?  
            - **Step Quality:** Were individual steps well-reasoned and executed?  
            - **Context Flow:** Did information flow effectively between steps?  
            - **Final Result:** Is the final answer comprehensive and actionable?  

            **Performance Metrics:**  
            - Number of steps executed  
            - Average confidence per step  
            - Context management efficiency  
            - Overall solution quality  

            **Improvement Opportunities:**  
            - What aspects could be enhanced?  
            - Where did the reasoning struggle?  
            - How could context management be improved?  
            """
        )
    else:
        cell12_out = mo.md("*Select a complex problem and test your pipeline*")

    output.replace(cell12_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell13_desc = mo.md(
            cleandoc(
                """
                ## 🎉 Exercise Complete!

                **What You've Accomplished:**  
                - ✅ Built a sophisticated step planning system  
                - ✅ Created a robust step execution engine  
                - ✅ Implemented context management across steps  
                - ✅ Developed a complete reasoning pipeline  
                - ✅ Tested with complex, real-world problems  

                ## 📊 Final Analysis

                **Reflection Questions:**  
                1. What was the most challenging aspect of multi-step reasoning?  
                2. How did context management impact reasoning quality?  
                3. What patterns did you observe in complex problem solving?  
                4. How could your pipeline be improved for production use?  
                """
            )
        )

        # Final analysis
        analysis_input = mo.ui.text_area(
            placeholder="Analyze your multi-step reasoning system...",
            label="Your analysis of the multi-step reasoning exercise:",
            rows=6,
        )

        submit_analysis_button = mo.ui.run_button(label="📊 Submit Analysis")

        cell13_content = mo.vstack([analysis_input, submit_analysis_button])
    else:
        cell13_desc = mo.md("")
        analysis_input = None
        submit_analysis_button = None
        cell13_content = mo.md("")

    cell13_out = mo.vstack([cell13_desc, cell13_content])
    output.replace(cell13_out)
    return analysis_input, submit_analysis_button


@app.cell
def _(
    analysis_input,
    available_providers,
    cleandoc,
    mo,
    output,
    submit_analysis_button,
):
    if available_providers and submit_analysis_button.value:
        cell14_out = mo.md(
            cleandoc(
                f"""
                ## 📊 Your Analysis

                {analysis_input.value or "No analysis provided"}

                ## 🚀 Congratulations!

                You've successfully built a complete multi-step reasoning system! This is advanced AI engineering.

                **Key Achievements:**  
                - **System Architecture:** Designed a modular, extensible reasoning pipeline  
                - **Problem Decomposition:** Created intelligent step planning capabilities  
                - **Context Management:** Implemented sophisticated state management  
                - **Complex Problem Solving:** Handled real-world, multi-faceted challenges  

                ## 🔮 Advanced Challenges

                **Next Level Improvements:**  
                - **Parallel Step Execution:** Execute independent steps concurrently  
                - **Dynamic Replanning:** Adjust plans based on intermediate results  
                - **Tool Integration:** Combine with external tools and APIs  
                - **Learning from Experience:** Improve planning based on past executions  
                - **Uncertainty Handling:** Manage confidence and uncertainty propagation  

                **Production Considerations:**  
                - Performance optimization and caching  
                - Error recovery and retry mechanisms  
                - Monitoring and observability  
                - Scalability and resource management  

                You're now ready to tackle the most complex reasoning challenges! 🧠✨
                """
            )
        )
    else:
        cell14_out = mo.md("*Complete your analysis to finish the exercise*")

    output.replace(cell14_out)
    return


if __name__ == "__main__":
    app.run()
