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
            # 🧪 Exercise 01: ReAct Basics

            **Duration:** 30-45 minutes  
            **Difficulty:** Intermediate  
            **Prerequisites:** Completed Module 01 (DSPy Foundations)

            ## 🎯 Exercise Objectives

            In this exercise, you will:  
            - ✅ Create your first ReAct signature  
            - ✅ Build a simple reasoning agent  
            - ✅ Test the agent with different problem types  
            - ✅ Analyze the reasoning patterns  

            ## 📋 Tasks Overview

            1. **Task 1:** Create a basic ReAct signature
            2. **Task 2:** Implement a simple ReAct agent
            3. **Task 3:** Test with mathematical problems
            4. **Task 4:** Test with logical reasoning problems
            5. **Task 5:** Analyze and improve the agent

            ## 🚀 Let's Get Started!

            Complete each task step by step. Don't peek at the solutions until you've tried each task!
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

                Ready to start the ReAct exercises!
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
                ## 📝 Task 1: Create a Basic ReAct Signature

                **Your Challenge:** Create a DSPy signature that implements the ReAct pattern.

                **Requirements:**  
                - Include fields for: question, thought, action, observation, answer  
                - Add appropriate descriptions for each field  
                - Make it suitable for step-by-step problem solving  

                **Hint:** Think about the ReAct cycle: Think → Act → Observe → Repeat
                """
            )
        )

        # TODO: Students should implement this
        # Example solution (hidden):
        # """
        # class StudentReActSignature(dspy.Signature):
        #     \"\"\"Solve problems using reasoning and actions step by step.\"\"\"

        #     question = dspy.InputField(desc="The problem or question to solve")
        #     thought = dspy.OutputField(desc="Your reasoning about the current situation")
        #     action = dspy.OutputField(desc="The action you decide to take")
        #     observation = dspy.OutputField(desc="What you observe from the action")
        #     answer = dspy.OutputField(desc="Final answer when the problem is solved")
        # """

        cell3_content = mo.md(
            cleandoc(
                """
                ### 💡 Your Task

                Create your ReAct signature in the code cell below. Think about:
                - What inputs does the agent need?
                - What outputs should it produce at each step?
                - How can you make the descriptions clear and helpful?

                ```python
                # Your code here - create a class called StudentReActSignature
                class StudentReActSignature(dspy.Signature):
                    # Add your signature fields here
                    pass
                ```
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        # Student implementation area
        cell4_desc = mo.md("### 🔧 Your Implementation")

        # TODO: Students implement their signature here
        # Placeholder for student work
        student_signature_code = mo.ui.text_area(
            placeholder=cleandoc(
                """class StudentReActSignature(dspy.Signature):
                    \"\"\"Your signature description here\"\"\"

                    # Add your fields here
                    question = dspy.InputField(desc="...")
                    # Add more fields...
                """
            ),
            label="Implement your ReAct signature:",
            rows=10,
        )

        test_signature_button = mo.ui.run_button(label="✅ Test My Signature")

        cell4_content = mo.vstack([student_signature_code, test_signature_button])
    else:
        cell4_desc = mo.md("")
        student_signature_code = None
        test_signature_button = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return student_signature_code, test_signature_button


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    student_signature_code,
    test_signature_button,
):
    if available_providers and test_signature_button.value:
        try:
            # Basic validation of student code
            code = student_signature_code.value
            if "class" in code and "dspy.Signature" in code:
                cell5_out = mo.md(
                    cleandoc(
                        """
                        ✅ **Great start!** Your signature structure looks good.

                        **Next Steps:**
                        - Make sure you have all required fields: question, thought, action, observation, answer
                        - Check that your field descriptions are clear and helpful
                        - Ensure your class inherits from dspy.Signature

                        Ready for Task 2!
                        """
                    )
                )
            else:
                cell5_out = mo.md(
                    cleandoc(
                        """
                        ⚠️ **Check your code:**
                        - Make sure you define a class that inherits from dspy.Signature
                        - Include the required ReAct fields
                        - Add descriptive field descriptions

                        Try again!
                        """
                    )
                )
        except Exception as e:
            cell5_out = mo.md(f"❌ **Error in your code:** {str(e)}")
    else:
        cell5_out = mo.md(
            "*Implement your signature above and click 'Test My Signature'*"
        )

    output.replace(cell5_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 2: Build a Simple ReAct Agent

                **Your Challenge:** Create a ReAct agent using your signature.

                **Requirements:**
                - Use dspy.ChainOfThought with your signature
                - Test it with a simple mathematical problem
                - Analyze the agent's reasoning process

                **Test Problem:** "What is 15% of 240, and then what is 25% of that result?"
                """
            )
        )

        # Student implementation area
        agent_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                # Create your ReAct agent here
                # Example:
                # my_react_agent = dspy.ChainOfThought(YourSignatureClass)
                # result = my_react_agent(question="Your test question")
                """
            ),
            label="Create and test your ReAct agent:",
            rows=8,
        )

        test_agent_button = mo.ui.run_button(label="🤖 Test My Agent")

        cell6_content = mo.vstack([agent_code, test_agent_button])
    else:
        cell6_desc = mo.md("")
        agent_code = None
        test_agent_button = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return (test_agent_button,)


@app.cell
def _(available_providers, cleandoc, mo, output, test_agent_button):
    if available_providers and test_agent_button.value:
        cell7_out = mo.md(
            cleandoc(
                """
                📝 **Agent Testing Results**

                *This would show the results of your agent implementation*

                **Analysis Questions:**
                1. How did your agent break down the problem?
                2. What reasoning steps did it take?
                3. Was the final answer correct?
                4. How could you improve the reasoning process?

                **Expected Answer:** 15% of 240 = 36, then 25% of 36 = 9
                """
            )
        )
    else:
        cell7_out = mo.md("*Implement and test your agent above*")

    output.replace(cell7_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell8_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 3: Test with Different Problem Types

                **Your Challenge:** Test your agent with various problem types.

                **Test Problems:**
                1. **Logic:** "If all cats are animals, and Fluffy is a cat, what can we conclude about Fluffy?"
                2. **Planning:** "I need to prepare dinner for 6 people. I have chicken, rice, and vegetables. What steps should I take?"
                3. **Analysis:** "A store sells 100 items per day. If sales increase by 20% each month, how many items will they sell per day after 3 months?"

                **Your Task:** Test each problem and analyze the reasoning patterns.
                """
            )
        )

        # Problem selector
        test_problems = [
            "If all cats are animals, and Fluffy is a cat, what can we conclude about Fluffy?",
            "I need to prepare dinner for 6 people. I have chicken, rice, and vegetables. What steps should I take?",
            "A store sells 100 items per day. If sales increase by 20% each month, how many items will they sell per day after 3 months?",
        ]

        problem_selector = mo.ui.dropdown(
            options=test_problems,
            label="Select a test problem:",
            value=test_problems[0],
        )

        test_problem_button = mo.ui.run_button(label="🧪 Test Selected Problem")

        cell8_content = mo.vstack([problem_selector, test_problem_button])
    else:
        cell8_desc = mo.md("")
        test_problems = None
        problem_selector = None
        test_problem_button = None
        cell8_content = mo.md("")

    cell8_out = mo.vstack([cell8_desc, cell8_content])
    output.replace(cell8_out)
    return problem_selector, test_problem_button


@app.cell
def _(available_providers, mo, output, problem_selector, test_problem_button):
    if available_providers and test_problem_button.value:
        selected_problem = problem_selector.value
        cell9_out = mo.md(
            f"""
            🧪 **Testing Problem:** {selected_problem}

            *Results would appear here based on your agent implementation*

            **Reflection Questions:**
            - How did the agent approach this type of problem?
            - What reasoning patterns did you observe?
            - Were there any limitations in the agent's approach?
            - How might you improve the agent for this problem type?
            """
        )
    else:
        cell9_out = mo.md("*Select a problem and test it with your agent*")

    output.replace(cell9_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell10_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 4: Analysis and Improvement

                **Your Challenge:** Analyze your agent's performance and suggest improvements.

                **Analysis Framework:**
                1. **Strengths:** What did your agent do well?
                2. **Weaknesses:** Where did it struggle?
                3. **Patterns:** What reasoning patterns did you observe?
                4. **Improvements:** How could you enhance the agent?

                **Reflection Questions:**
                - Which problem types were handled best/worst?
                - How consistent was the reasoning quality?
                - What additional fields or instructions might help?
                """
            )
        )

        # Analysis form
        strengths_input = mo.ui.text_area(
            placeholder="What did your agent do well?",
            label="Strengths:",
            rows=3,
        )

        weaknesses_input = mo.ui.text_area(
            placeholder="Where did your agent struggle?",
            label="Weaknesses:",
            rows=3,
        )

        improvements_input = mo.ui.text_area(
            placeholder="How could you improve the agent?",
            label="Suggested Improvements:",
            rows=3,
        )

        submit_analysis_button = mo.ui.run_button(label="📊 Submit Analysis")

        cell10_content = mo.vstack(
            [
                strengths_input,
                weaknesses_input,
                improvements_input,
                submit_analysis_button,
            ]
        )
    else:
        cell10_desc = mo.md("")
        strengths_input = None
        weaknesses_input = None
        improvements_input = None
        submit_analysis_button = None
        cell10_content = mo.md("")

    cell10_out = mo.vstack([cell10_desc, cell10_content])
    output.replace(cell10_out)
    return (
        improvements_input,
        strengths_input,
        submit_analysis_button,
        weaknesses_input,
    )


@app.cell
def _(
    available_providers,
    cleandoc,
    improvements_input,
    mo,
    output,
    strengths_input,
    submit_analysis_button,
    weaknesses_input,
):
    if available_providers and submit_analysis_button.value:
        cell11_out = mo.md(
            cleandoc(
                f"""
                ## 📊 Your Analysis Summary

                **Strengths Identified:**
                {strengths_input.value or "No strengths noted"}

                **Weaknesses Identified:**
                {weaknesses_input.value or "No weaknesses noted"}

                **Improvement Suggestions:**
                {improvements_input.value or "No improvements suggested"}

                ## 🎉 Exercise Complete!

                **What You've Accomplished:**
                - ✅ Created a basic ReAct signature
                - ✅ Built and tested a ReAct agent
                - ✅ Tested with multiple problem types
                - ✅ Analyzed agent performance and identified improvements

                **Next Steps:**
                - Review the solution notebook to compare approaches
                - Try Exercise 02: Tool Integration
                - Experiment with more complex ReAct patterns

                Great work on completing your first ReAct exercise! 🚀
                """
            )
        )
    else:
        cell11_out = mo.md("*Complete your analysis above to finish the exercise*")

    output.replace(cell11_out)
    return


if __name__ == "__main__":
    app.run()
