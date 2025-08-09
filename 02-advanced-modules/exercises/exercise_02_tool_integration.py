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
            # 🧪 Exercise 02: Tool Integration

            **Duration:** 45-60 minutes  
            **Difficulty:** Intermediate-Advanced  
            **Prerequisites:** Completed Exercise 01 (ReAct Basics)  

            ## 🎯 Exercise Objectives

            In this exercise, you will:  
            - ✅ Create custom tools for your agents  
            - ✅ Build a tool-aware ReAct agent  
            - ✅ Implement tool selection logic  
            - ✅ Handle tool execution and error cases  
            - ✅ Create a multi-tool problem-solving system  

            ## 📋 Tasks Overview

            1. **Task 1:** Create a simple calculator tool  
            2. **Task 2:** Create a text analysis tool  
            3. **Task 3:** Build a tool-integrated agent  
            4. **Task 4:** Test multi-tool problem solving  
            5. **Task 5:** Handle errors and edge cases  

            ## 🚀 Let's Get Started!

            You'll build a complete tool integration system from scratch!
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

                Ready to build tool-integrated agents!
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
                ## 📝 Task 1: Create a Calculator Tool

                **Your Challenge:** Create a calculator tool that can perform basic mathematical operations.

                **Requirements:**  
                - Function should take an expression as input  
                - Support basic operations: +, -, *, /, **  
                - Return the result or an error message  
                - Handle invalid expressions gracefully  

                **Example Usage:**  
                ```python  
                calculator_tool("2 + 3 * 4")  # Should return 14  
                calculator_tool("10 / 0")     # Should handle division by zero  
                ```
                """
            )
        )

        # Student implementation area
        calculator_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                def calculator_tool(expression: str) -> str:  
                    \"\"\"Simple calculator tool for mathematical expressions.\"\"\"  
                    try:  
                        # Your implementation here  
                        # Hint: Use eval() carefully with safety checks  
                        pass  
                    except Exception as e:  
                        return f"Error: {str(e)}"  
                """
            ),
            label="Implement your calculator tool:",
            rows=10,
        )

        test_calculator_button = mo.ui.run_button(label="🧮 Test Calculator")

        cell3_content = mo.vstack([calculator_code, test_calculator_button])
    else:
        cell3_desc = mo.md("")
        calculator_code = None
        test_calculator_button = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (test_calculator_button,)


@app.cell
def _(available_providers, cleandoc, mo, output, test_calculator_button):
    if available_providers and test_calculator_button.value:
        # Test the calculator implementation
        test_expressions = [
            "2 + 3 * 4",
            "10 / 2",
            "2 ** 3",
            "10 / 0",  # Division by zero
            "invalid expression",  # Invalid syntax
        ]

        cell4_out = mo.md(
            cleandoc(
                f"""
                🧮 **Calculator Testing Results**

                **Test Expressions:**
                {'\n'.join([f"- `{expr}`  " for expr in test_expressions])}

                *Your implementation results would appear here*

                **Expected Behaviors:**  
                - `2 + 3 * 4` → `14`  
                - `10 / 2` → `5.0`  
                - `2 ** 3` → `8`  
                - `10 / 0` → Error message  
                - `invalid expression` → Error message  

                **Validation Checklist:**  
                - ✅ Handles basic arithmetic correctly  
                - ✅ Manages division by zero  
                - ✅ Returns appropriate error messages  
                - ✅ Uses safe evaluation methods  
                """
            )
        )
    else:
        cell4_out = mo.md("*Implement your calculator tool above and test it*")

    output.replace(cell4_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 2: Create a Text Analysis Tool

                **Your Challenge:** Create a text analysis tool that provides insights about text.

                **Requirements:**  
                - Count words, sentences, and characters  
                - Calculate average word length  
                - Find the most common words  
                - Return results as a dictionary  

                **Example Usage:**  
                ```python  
                text_analyzer("Hello world! This is a test.")  
                # Should return: {  
                #   "word_count": 6,  
                #   "sentence_count": 2,  
                #   "char_count": 29,  
                #   "avg_word_length": 4.0  
                # }  
                ```
                """
            )
        )

        # Student implementation area
        text_analyzer_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                def text_analyzer(text: str) -> dict:
                    \"\"\"Analyze text and return statistics.\"\"\"
                    try:
                        # Your implementation here
                        # Hint: Use string methods like split(), len(), etc.
                        words = text.split()
                        # Add more analysis...

                        return {
                            "word_count": 0,
                            "sentence_count": 0,
                            "char_count": 0,
                            "avg_word_length": 0.0
                        }
                    except Exception as e:
                        return {"error": str(e)}
                """
            ),
            label="Implement your text analyzer tool:",
            rows=12,
        )

        test_analyzer_button = mo.ui.run_button(label="📝 Test Text Analyzer")

        cell5_content = mo.vstack([text_analyzer_code, test_analyzer_button])
    else:
        cell5_desc = mo.md("")
        text_analyzer_code = None
        test_analyzer_button = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (test_analyzer_button,)


@app.cell
def _(available_providers, mo, output, test_analyzer_button):
    if available_providers and test_analyzer_button.value:
        test_text = "Hello world! This is a test. How are you doing today?"

        cell6_out = mo.md(
            f"""
            📝 **Text Analyzer Testing Results**

            **Test Text:** "{test_text}"

            *Your implementation results would appear here*

            **Expected Results:**  
            - Word count: 11  
            - Sentence count: 3  
            - Character count: 54  
            - Average word length: ~3.9  

            **Validation Checklist:**  
            - ✅ Counts words correctly  
            - ✅ Counts sentences correctly  
            - ✅ Counts characters correctly  
            - ✅ Calculates average word length  
            - ✅ Handles edge cases (empty text, etc.)  
            """
        )
    else:
        cell6_out = mo.md("*Implement your text analyzer tool above and test it*")

    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 3: Build a Tool-Integrated Agent

                **Your Challenge:** Create an agent that can intelligently select and use tools.

                **Requirements:**  
                - Create a signature that includes tool selection  
                - Agent should decide which tool to use based on the problem  
                - Handle tool execution and integrate results  
                - Provide reasoning for tool choices  

                **Agent Capabilities:**  
                - Solve math problems using calculator  
                - Analyze text using text analyzer  
                - Explain reasoning for tool selection  
                """
            )
        )

        # Student implementation area
        agent_signature_code = mo.ui.text_area(
            placeholder=cleandoc(
                """
                class ToolIntegratedSignature(dspy.Signature):  
                    \"\"\"Agent that can use tools to solve problems.\"\"\"  

                    problem = dspy.InputField(desc="The problem to solve")  
                    available_tools = dspy.InputField(desc="List of available tools")  

                    # Add your output fields here  
                    reasoning = dspy.OutputField(desc="...")  
                    tool_choice = dspy.OutputField(desc="...")  
                    # Add more fields...  
                """
            ),
            label="Create your tool-integrated agent signature:",
            rows=10,
        )

        agent_implementation_code = mo.ui.text_area(
            placeholder=cleandoc(
                """# Create your agent and tool execution logic  
                tool_agent = dspy.ChainOfThought(ToolIntegratedSignature)  

                def execute_with_tools(problem: str):  
                    \"\"\"Execute problem solving with tool integration.\"\"\"  
                    # Your implementation here  
                    pass  
                """
            ),
            label="Implement your agent execution logic:",
            rows=8,
        )

        test_tool_agent_button = mo.ui.run_button(label="🤖 Test Tool Agent")

        cell7_content = mo.vstack(
            [
                agent_signature_code,
                agent_implementation_code,
                test_tool_agent_button,
            ]
        )
    else:
        cell7_desc = mo.md("")
        agent_signature_code = None
        agent_implementation_code = None
        test_tool_agent_button = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return (test_tool_agent_button,)


@app.cell
def _(available_providers, mo, output, test_tool_agent_button):
    if available_providers and test_tool_agent_button.value:
        cell8_out = mo.md(
            """
            🤖 **Tool Agent Testing Results**

            *Your agent implementation results would appear here*

            **Test Cases:**  
            1. "Calculate 15% of 240"  
            2. "Analyze this text: 'The quick brown fox jumps over the lazy dog'"  
            3. "What is 2^10 and how many words are in this sentence?"  

            **Expected Behaviors:**  
            - Correctly identifies when to use calculator vs text analyzer  
            - Provides clear reasoning for tool selection  
            - Executes tools and integrates results  
            - Handles multi-tool problems appropriately  
            """
        )
    else:
        cell8_out = mo.md("*Implement your tool-integrated agent above and test it*")

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell9_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 4: Multi-Tool Problem Solving

                **Your Challenge:** Test your agent with problems that require multiple tools.

                **Test Problems:**  
                1. "Calculate 25% of 160, then analyze the text 'The result is important for our analysis'"  
                2. "How many characters are in the phrase 'Hello World' and what is 12 * 8?"  
                3. "Find the square root of 144, then count words in 'Mathematics is fascinating and useful'"  

                **Requirements:**  
                - Agent should handle sequential tool usage  
                - Combine results from multiple tools  
                - Provide comprehensive final answers  
                """
            )
        )

        # Problem selector
        multi_tool_problems = [
            "Calculate 25% of 160, then analyze the text 'The result is important for our analysis'",
            "How many characters are in the phrase 'Hello World' and what is 12 * 8?",
            "Find the square root of 144, then count words in 'Mathematics is fascinating and useful'",
        ]

        problem_selector = mo.ui.dropdown(
            options=multi_tool_problems,
            label="Select a multi-tool problem:",
            value=multi_tool_problems[0],
        )

        test_multitool_button = mo.ui.run_button(label="🔧 Test Multi-Tool Problem")

        cell9_content = mo.vstack([problem_selector, test_multitool_button])
    else:
        cell9_desc = mo.md("")
        multi_tool_problems = None
        problem_selector = None
        test_multitool_button = None
        cell9_content = mo.md("")

    cell9_out = mo.vstack([cell9_desc, cell9_content])
    output.replace(cell9_out)
    return problem_selector, test_multitool_button


@app.cell
def _(
    available_providers,
    mo,
    output,
    problem_selector,
    test_multitool_button,
):
    if available_providers and test_multitool_button.value:
        selected_problem = problem_selector.value
        cell10_out = mo.md(
            f"""
            🔧 **Multi-Tool Problem Results**

            **Problem:** {selected_problem}

            *Your agent's multi-tool execution results would appear here*

            **Analysis Questions:**  
            - Did the agent correctly identify the need for multiple tools?  
            - How did it sequence the tool usage?  
            - Were the results properly integrated?  
            - What could be improved in the approach?  

            **Expected Workflow:**  
            1. Parse problem to identify tool requirements  
            2. Execute tools in logical sequence  
            3. Combine results into coherent answer  
            4. Provide clear reasoning throughout  
            """
        )
    else:
        cell10_out = mo.md("*Select a problem and test your multi-tool agent*")

    output.replace(cell10_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell11_desc = mo.md(
            cleandoc(
                """
                ## 📝 Task 5: Error Handling and Edge Cases

                **Your Challenge:** Improve your agent to handle errors and edge cases gracefully.

                **Error Scenarios to Handle:**  
                1. Invalid mathematical expressions  
                2. Empty or null text inputs  
                3. Tool execution failures  
                4. Ambiguous tool selection scenarios  

                **Requirements:**  
                - Add error handling to your tools  
                - Update agent to handle tool failures  
                - Provide helpful error messages  
                - Implement fallback strategies  
                """
            )
        )

        # Error handling implementation
        error_handling_code = mo.ui.text_area(
            placeholder=cleandoc(
                """# Improve your tools and agent with error handling

                def safe_calculator_tool(expression: str) -> dict:
                    \"\"\"Calculator with comprehensive error handling.\"\"\"
                    try:
                        # Add validation and safety checks
                        # Return structured results with success/error status
                        pass
                    except Exception as e:
                        return {"success": False, "error": str(e)}

                def safe_text_analyzer(text: str) -> dict:
                    \"\"\"Text analyzer with error handling.\"\"\"
                    try:
                        # Add input validation
                        # Handle edge cases like empty text
                        pass
                    except Exception as e:
                        return {"success": False, "error": str(e)}

                # Update your agent to handle tool errors gracefully
                """
            ),
            label="Implement error handling for your tools and agent:",
            rows=15,
        )

        test_error_handling_button = mo.ui.run_button(label="⚠️ Test Error Handling")

        cell11_content = mo.vstack([error_handling_code, test_error_handling_button])
    else:
        cell11_desc = mo.md("")
        error_handling_code = None
        test_error_handling_button = None
        cell11_content = mo.md("")

    cell11_out = mo.vstack([cell11_desc, cell11_content])
    output.replace(cell11_out)
    return (test_error_handling_button,)


@app.cell
def _(available_providers, cleandoc, mo, output, test_error_handling_button):
    if available_providers and test_error_handling_button.value:
        error_test_cases = [
            "Calculate: invalid_expression",
            "Analyze text: ''",  # Empty string
            "What is 10 / 0?",
            "Analyze this: None",
        ]

        cell12_out = mo.md(
            cleandoc(
                f"""
                ⚠️ **Error Handling Test Results**

                **Test Cases:**
                {chr(10).join([f"- {case}" for case in error_test_cases])}

                *Your error handling results would appear here*

                **Validation Checklist:**  
                - ✅ Handles invalid mathematical expressions  
                - ✅ Manages empty or null text inputs  
                - ✅ Provides helpful error messages  
                - ✅ Agent continues functioning after errors  
                - ✅ Implements appropriate fallback strategies  

                **Good Error Handling Practices:**  
                - Clear, user-friendly error messages  
                - Graceful degradation when tools fail  
                - Logging for debugging purposes  
                - Fallback options when possible  
                """
            )
        )
    else:
        cell12_out = mo.md("*Implement error handling and test it*")

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
                - ✅ Created custom tools (calculator and text analyzer)  
                - ✅ Built a tool-integrated ReAct agent  
                - ✅ Implemented tool selection logic  
                - ✅ Tested multi-tool problem solving  
                - ✅ Added error handling and edge case management  

                ## 📊 Final Reflection

                **Reflection Questions:**  
                1. What was the most challenging part of tool integration?  
                2. How did your agent's reasoning improve with tools?  
                3. What additional tools would be useful?  
                4. How could you improve the tool selection process?  
                """
            )
        )

        # Final reflection
        reflection_input = mo.ui.text_area(
            placeholder="Share your thoughts on the tool integration exercise...",
            label="Your reflection on this exercise:",
            rows=5,
        )

        submit_reflection_button = mo.ui.run_button(label="📝 Submit Reflection")

        cell13_content = mo.vstack([reflection_input, submit_reflection_button])
    else:
        cell13_desc = mo.md("")
        reflection_input = None
        submit_reflection_button = None
        cell13_content = mo.md("")

    cell13_out = mo.vstack([cell13_desc, cell13_content])
    output.replace(cell13_out)
    return reflection_input, submit_reflection_button


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    reflection_input,
    submit_reflection_button,
):
    if available_providers and submit_reflection_button.value:
        cell14_out = mo.md(
            cleandoc(
                f"""
                ## 📝 Your Reflection

                {reflection_input.value or "No reflection provided"}

                ## 🚀 Next Steps

                **Congratulations!** You've successfully completed the Tool Integration exercise.

                **What's Next:**  
                - Review the solution notebook to compare approaches  
                - Try Exercise 03: Multi-Step Reasoning  
                - Experiment with creating additional tools  
                - Explore more complex tool integration patterns  

                **Advanced Challenges:**  
                - Create tools that call external APIs  
                - Implement tool caching and optimization  
                - Build a tool registry system  
                - Add tool usage analytics and monitoring  

                Keep building and experimenting! 🛠️
                """
            )
        )
    else:
        cell14_out = mo.md("*Complete your reflection to finish the exercise*")

    output.replace(cell14_out)
    return


if __name__ == "__main__":
    app.run()
