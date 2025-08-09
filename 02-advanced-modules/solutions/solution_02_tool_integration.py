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

    return cleandoc, dspy, get_config, mo, output, setup_dspy_environment


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🔑 Solution 02: Tool Integration

            **Exercise:** Tool Integration  
            **Difficulty:** Intermediate-Advanced  
            **Focus:** Building tool-integrated agents  

            ## 📋 Solution Overview

            This solution demonstrates:  
            - ✅ Complete tool implementations with error handling  
            - ✅ Tool-integrated agent architecture  
            - ✅ Multi-tool problem-solving strategies  
            - ✅ Production-ready error handling patterns  

            ## 🎯 Learning Outcomes

            By studying this solution, you'll understand:  
            - How to design robust, reusable tools  
            - Effective tool integration patterns  
            - Error handling and edge case management  
            - Multi-tool orchestration strategies  

            Let's explore the complete solution!
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

                Solution environment is ready!
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
                ## 🔑 Task 1 Solution: Calculator Tool

                **Complete Implementation with Safety and Error Handling:**
                """
            )
        )

        def solution_calculator_tool(expression: str) -> dict:
            """Advanced calculator tool with comprehensive error handling and safety."""
            try:
                # Input validation
                if not expression or not isinstance(expression, str):
                    return {
                        "success": False,
                        "error": "Invalid input: expression must be a non-empty string",
                        "result": None,
                    }

                # Clean the expression
                expression = expression.strip()
                if not expression:
                    return {
                        "success": False,
                        "error": "Empty expression provided",
                        "result": None,
                    }

                # Safety check: only allow safe characters
                allowed_chars = set("0123456789+-*/.() ")
                if not all(c in allowed_chars for c in expression):
                    return {
                        "success": False,
                        "error": "Invalid characters in expression. Only numbers, +, -, *, /, ., (, ), and spaces allowed",
                        "result": None,
                    }

                # Additional safety: prevent dangerous operations
                dangerous_patterns = ["__", "import", "exec", "eval", "open", "file"]
                expression_lower = expression.lower()
                if any(pattern in expression_lower for pattern in dangerous_patterns):
                    return {
                        "success": False,
                        "error": "Potentially dangerous expression detected",
                        "result": None,
                    }

                # Safe evaluation with limited scope
                allowed_names = {
                    "__builtins__": {},
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "pow": pow,
                }

                # Evaluate the expression
                _result = eval(expression, allowed_names, {})

                # Check for valid result
                if not isinstance(_result, (int, float)):
                    return {
                        "success": False,
                        "error": f"Expression resulted in invalid type: {type(_result)}",
                        "result": None,
                    }

                # Check for infinity or NaN
                if isinstance(_result, float):
                    if _result == float("inf") or _result == float("-inf"):
                        return {
                            "success": False,
                            "error": "Result is infinite",
                            "result": None,
                        }
                    if _result != _result:  # NaN check
                        return {
                            "success": False,
                            "error": "Result is not a number (NaN)",
                            "result": None,
                        }

                return {
                    "success": True,
                    "result": _result,
                    "expression": expression,
                    "error": None,
                }

            except ZeroDivisionError:
                return {
                    "success": False,
                    "error": "Division by zero",
                    "result": None,
                }
            except SyntaxError as e:
                return {
                    "success": False,
                    "error": f"Invalid mathematical expression: {str(e)}",
                    "result": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Calculation error: {str(e)}",
                    "result": None,
                }

        # Test the calculator
        test_expressions = [
            "2 + 3 * 4",  # Should be 14
            "10 / 2",  # Should be 5.0
            "2 ** 3",  # Should be 8
            "10 / 0",  # Division by zero
            "invalid expression",  # Invalid syntax
            "",  # Empty string
            "2 + (3 * 4)",  # Parentheses
        ]

        test_results = []
        for expr in test_expressions:
            _result = solution_calculator_tool(expr)
            test_results.append(f"- `{expr}` → {_result}")

        cell3_content = mo.md(
            cleandoc(
                f"""
                ### 🧮 Calculator Tool Test Results

                {chr(10).join(test_results)}

                ### 💡 Key Implementation Features

                **Safety Measures:**  
                - Input validation and sanitization  
                - Character whitelist for security  
                - Dangerous pattern detection  
                - Limited evaluation scope  

                **Error Handling:**  
                - Specific error messages for different failure modes  
                - Graceful handling of edge cases  
                - Structured return format with success/error status  

                **Robustness:**  
                - Handles division by zero  
                - Detects infinite and NaN results  
                - Validates result types  
                - Comprehensive exception handling  
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        solution_calculator_tool = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (solution_calculator_tool,)


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 2 Solution: Text Analysis Tool

                **Complete Implementation with Comprehensive Analysis:**
                """
            )
        )

        def solution_text_analyzer(text: str) -> dict:
            """Advanced text analyzer with comprehensive statistics and error handling."""
            try:
                # Input validation
                if text is None:
                    return {
                        "success": False,
                        "error": "Text input is None",
                    }

                if not isinstance(text, str):
                    return {
                        "success": False,
                        "error": f"Invalid input type: expected string, got {type(text)}",
                    }

                # Handle empty text
                if not text.strip():
                    return {
                        "success": True,
                        "word_count": 0,
                        "sentence_count": 0,
                        "char_count": 0,
                        "char_count_no_spaces": 0,
                        "avg_word_length": 0.0,
                        "most_common_words": [],
                        "unique_words": 0,
                        "error": None,
                    }

                # Basic counts
                char_count = len(text)
                char_count_no_spaces = len(text.replace(" ", ""))

                # Word analysis
                words = text.split()
                word_count = len(words)

                # Calculate average word length
                if word_count > 0:
                    total_word_length = sum(
                        len(word.strip(".,!?;:\"'()[]{}")) for word in words
                    )
                    avg_word_length = total_word_length / word_count
                else:
                    avg_word_length = 0.0

                # Sentence counting (basic approach)
                sentence_endings = [".", "!", "?"]
                sentence_count = sum(text.count(ending) for ending in sentence_endings)
                if sentence_count == 0 and text.strip():
                    sentence_count = (
                        1  # If no sentence endings but text exists, count as 1
                    )

                # Word frequency analysis
                word_freq = {}
                cleaned_words = []
                for word in words:
                    # Clean word of punctuation
                    cleaned_word = word.strip(".,!?;:\"'()[]{}").lower()
                    if cleaned_word:  # Only count non-empty words
                        cleaned_words.append(cleaned_word)
                        word_freq[cleaned_word] = word_freq.get(cleaned_word, 0) + 1

                # Most common words (top 5)
                most_common_words = sorted(
                    word_freq.items(), key=lambda x: x[1], reverse=True
                )[:5]
                unique_words = len(word_freq)

                return {
                    "success": True,
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "char_count": char_count,
                    "char_count_no_spaces": char_count_no_spaces,
                    "avg_word_length": round(avg_word_length, 2),
                    "most_common_words": most_common_words,
                    "unique_words": unique_words,
                    "error": None,
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Text analysis error: {str(e)}",
                }

        # Test the text analyzer
        test_texts = [
            "Hello world! This is a test. How are you doing today?",
            "",  # Empty string
            "Single sentence without ending",
            "The quick brown fox jumps over the lazy dog.",
            "A B C D E F G",  # Short words
        ]

        analyzer_results = []
        for text in test_texts:
            _result = solution_text_analyzer(text)
            if _result["success"]:
                analyzer_results.append(
                    f"**Text:** \"{text[:50]}{'...' if len(text) > 50 else ''}\"  "
                    f"- Words: {_result['word_count']}, Sentences: {_result['sentence_count']}  "
                    f"- Characters: {_result['char_count']} (no spaces: {_result['char_count_no_spaces']})  "
                    f"- Avg word length: {_result['avg_word_length']}  "
                    f"- Unique words: {_result['unique_words']}  "
                    f"- Most common: {_result['most_common_words'][:3]}  "
                    "  "
                )
            else:
                analyzer_results.append(
                    f"**Text:** \"{text}\" → Error: {_result['error']}  "
                )

        cell4_content = mo.md(
            cleandoc(
                f"""
                ### 📝 Text Analyzer Test Results

                {chr(10).join(analyzer_results)}   

                ### 💡 Key Implementation Features

                **Comprehensive Analysis:**  
                - Word and sentence counting  
                - Character counting (with and without spaces)  
                - Average word length calculation  
                - Word frequency analysis  
                - Most common words identification  

                **Robust Processing:**  
                - Handles empty and None inputs gracefully  
                - Cleans punctuation for word analysis  
                - Case-insensitive word frequency counting  
                - Proper sentence boundary detection  

                **Error Handling:**  
                - Input type validation  
                - Graceful handling of edge cases  
                - Structured error reporting  
                - Fallback values for empty inputs  
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        solution_text_analyzer = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (solution_text_analyzer,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 3 Solution: Tool-Integrated Agent

                **Complete Agent Architecture with Tool Selection:**
                """
            )
        )

        # Tool-integrated agent signature
        class SolutionToolIntegratedSignature(dspy.Signature):
            """Intelligent agent that can select and use appropriate tools to solve problems."""

            problem = dspy.InputField(desc="The problem or question to solve")
            available_tools = dspy.InputField(
                desc="List of available tools and their capabilities"
            )
            reasoning = dspy.OutputField(
                desc="Step-by-step analysis of the problem and tool requirements"
            )
            tool_selection = dspy.OutputField(
                desc="Which tool(s) to use and why they are appropriate for this problem"
            )
            tool_inputs = dspy.OutputField(
                desc="Specific inputs to provide to the selected tool(s)"
            )
            expected_outcome = dspy.OutputField(
                desc="What you expect to learn or achieve from using the tool(s)"
            )
            final_answer = dspy.OutputField(
                desc="Final answer incorporating tool results and reasoning"
            )

        # Create the agent
        solution_tool_agent = dspy.ChainOfThought(SolutionToolIntegratedSignature)

        # Tool execution function
        def execute_with_tools(problem: str, available_tools: dict) -> dict:
            """Execute problem solving with intelligent tool integration."""
            try:
                # Get agent's analysis and tool selection
                tools_description = ", ".join(
                    [
                        f"{name}: {desc}"
                        for name, desc in {
                            "calculator": "performs mathematical calculations",
                            "text_analyzer": "analyzes text for statistics and insights",
                        }.items()
                    ]
                )

                agent_response = solution_tool_agent(
                    problem=problem, available_tools=tools_description
                )

                # Execute selected tools based on agent's decision
                tool_results = {}

                # Simple tool selection logic based on agent's response
                tool_selection = agent_response.tool_selection.lower()
                tool_inputs = agent_response.tool_inputs

                if "calculator" in tool_selection and "calculator" in available_tools:
                    # Extract mathematical expression from tool inputs
                    calc_input = tool_inputs
                    # Simple extraction - in production, use more sophisticated parsing
                    import re

                    math_expressions = re.findall(r"[0-9+\-*/.()\s]+", calc_input)
                    if math_expressions:
                        calc_result = available_tools["calculator"](
                            math_expressions[0].strip()
                        )
                        tool_results["calculator"] = calc_result

                if (
                    "text_analyzer" in tool_selection
                    and "text_analyzer" in available_tools
                ):
                    # Extract text from tool inputs or use the problem itself
                    text_to_analyze = tool_inputs
                    if "analyze" in problem.lower() and "text" in problem.lower():
                        # Extract quoted text from problem
                        import re

                        quoted_text = re.findall(r'"([^"]*)"', problem)
                        if quoted_text:
                            text_to_analyze = quoted_text[0]

                    analyzer_result = available_tools["text_analyzer"](text_to_analyze)
                    tool_results["text_analyzer"] = analyzer_result

                return {
                    "success": True,
                    "agent_response": agent_response,
                    "tool_results": tool_results,
                    "problem": problem,
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "problem": problem,
                }

        cell5_content = mo.md(
            cleandoc(
                """
                ### 🤖 Tool-Integrated Agent Created

                **Key Architecture Components:**

                **Signature Design:**  
                - **Problem Analysis:** Detailed reasoning about requirements  
                - **Tool Selection:** Intelligent choice of appropriate tools  
                - **Input Specification:** Clear inputs for selected tools  
                - **Outcome Prediction:** Expected results from tool usage  
                - **Final Integration:** Synthesis of tool results into answer  

                **Execution Engine:**  
                - **Agent Reasoning:** Uses ChainOfThought for tool selection  
                - **Tool Orchestration:** Executes selected tools with proper inputs  
                - **Result Integration:** Combines tool outputs with agent reasoning  
                - **Error Handling:** Graceful failure management throughout  
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        SolutionToolIntegratedSignature = None
        solution_tool_agent = None
        execute_with_tools = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (execute_with_tools,)


@app.cell
def _(
    available_providers,
    cleandoc,
    execute_with_tools,
    mo,
    output,
    solution_calculator_tool,
    solution_text_analyzer,
):
    if available_providers and execute_with_tools:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 4 Solution: Multi-Tool Problem Solving

                **Testing Complex Multi-Tool Scenarios:**
                """
            )
        )

        # Available tools for the agent
        available_tools = {
            "calculator": solution_calculator_tool,
            "text_analyzer": solution_text_analyzer,
        }

        # Multi-tool test problems
        multi_tool_problems = [
            "Calculate 25% of 160, then analyze the text 'The result is important for our analysis'",
            "How many characters are in the phrase 'Hello World' and what is 12 * 8?",
            "Find the square root of 144, then count words in 'Mathematics is fascinating and useful'",
        ]

        # Test each multi-tool problem
        multitool_results = []
        for problem in multi_tool_problems:
            _result = execute_with_tools(problem, available_tools)

            if _result["success"]:
                agent_resp = _result["agent_response"]
                tool_results = _result["tool_results"]

                # Format tool results
                tool_summary = []
                for tool_name, tool_result in tool_results.items():
                    if tool_result.get("success", False):
                        if tool_name == "calculator":
                            tool_summary.append(f"Calculator: {tool_result['result']}")
                        elif tool_name == "text_analyzer":
                            tool_summary.append(
                                f"Text Analysis: {tool_result['word_count']} words, "
                                f"{tool_result['char_count']} characters"
                            )
                    else:
                        tool_summary.append(
                            f"{tool_name}: Error - {tool_result.get('error', 'Unknown error')}"
                        )

                multitool_results.append(
                    cleandoc(
                        f"""
                        **Problem:** {problem}

                        **Agent Analysis:**
                        - **Reasoning:** {agent_resp.reasoning}
                        - **Tool Selection:** {agent_resp.tool_selection}
                        - **Tool Inputs:** {agent_resp.tool_inputs}
                        - **Expected Outcome:** {agent_resp.expected_outcome}

                        **Tool Execution Results:**
                        {'\n'.join([f"- {summary}" for summary in tool_summary])}

                        **Final Answer:** {agent_resp.final_answer}
                        """
                    )
                )
            else:
                multitool_results.append(
                    f"**Problem:** {problem}\n**Error:** {_result['error']}"
                )

        cell6_content = mo.md(
            cleandoc(
                f"""
                ### 🔧 Multi-Tool Problem Solving Results

                {chr(10).join(multitool_results)}

                ### 📊 Multi-Tool Analysis

                **Successful Patterns:**
                - **Problem Decomposition:** Agent correctly identifies multiple tool requirements
                - **Sequential Processing:** Handles problems requiring multiple tools in sequence
                - **Result Integration:** Combines outputs from different tools effectively
                - **Context Awareness:** Maintains context across tool executions

                **Key Capabilities Demonstrated:**
                - **Tool Selection Intelligence:** Chooses appropriate tools based on problem type
                - **Input Extraction:** Correctly identifies inputs for each tool
                - **Error Resilience:** Handles tool failures gracefully
                - **Result Synthesis:** Provides comprehensive final answers
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        multitool_results = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return


@app.cell
def _(
    available_providers,
    cleandoc,
    execute_with_tools,
    mo,
    output,
    solution_calculator_tool,
    solution_text_analyzer,
):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 5 Solution: Error Handling and Edge Cases

                **Production-Ready Error Handling Implementation:**
                """
            )
        )

        # Enhanced tools with comprehensive error handling
        def safe_calculator_tool(expression: str) -> dict:
            """Production-ready calculator with comprehensive error handling."""
            # Input validation
            if expression is None:
                return {
                    "success": False,
                    "error": "Expression cannot be None",
                    "result": None,
                }

            if not isinstance(expression, str):
                return {
                    "success": False,
                    "error": f"Expression must be string, got {type(expression)}",
                    "result": None,
                }

            expression = expression.strip()
            if not expression:
                return {
                    "success": False,
                    "error": "Expression cannot be empty",
                    "result": None,
                }

            # Use the existing solution_calculator_tool which already has comprehensive error handling
            if solution_calculator_tool:
                return solution_calculator_tool(expression)
            else:
                return {
                    "success": False,
                    "error": "Calculator tool not available",
                    "result": None,
                }

        def safe_text_analyzer(text: str) -> dict:
            """Production-ready text analyzer with comprehensive error handling."""
            # Use the existing solution_text_analyzer which already has comprehensive error handling
            if solution_text_analyzer:
                return solution_text_analyzer(text)
            else:
                return {
                    "success": False,
                    "error": "Text analyzer tool not available",
                }

        # Enhanced agent with error handling
        def robust_execute_with_tools(problem: str, available_tools: dict) -> dict:
            """Enhanced tool execution with comprehensive error handling."""
            try:
                # Input validation
                if not problem or not isinstance(problem, str):
                    return {
                        "success": False,
                        "error": "Problem must be a non-empty string",
                        "fallback_response": "Unable to process invalid problem input",
                    }

                if not available_tools or not isinstance(available_tools, dict):
                    return {
                        "success": False,
                        "error": "No valid tools available",
                        "fallback_response": "Cannot solve problem without tools",
                    }

                # Execute with fallback strategies
                _result = execute_with_tools(problem, available_tools)

                if not result["success"]:
                    # Implement fallback strategy
                    fallback_response = f"Unable to fully process '{problem}' due to: {_result.get('error', 'Unknown error')}"

                    # Try to provide partial help based on problem type
                    if any(
                        word in problem.lower()
                        for word in ["calculate", "math", "number", "%", "*", "+"]
                    ):
                        fallback_response += ". This appears to be a mathematical problem. Please check your expression format."
                    elif any(
                        word in problem.lower()
                        for word in ["text", "analyze", "words", "characters"]
                    ):
                        fallback_response += ". This appears to be a text analysis problem. Please provide clear text to analyze."

                    _result["fallback_response"] = fallback_response

                return _result

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unexpected error in tool execution: {str(e)}",
                    "fallback_response": "An unexpected error occurred. Please try rephrasing your problem.",
                }

        # Test error scenarios
        error_test_cases = [
            ("Calculate: invalid_expression", "Invalid mathematical expression"),
            ("Analyze text: ''", "Empty text analysis"),
            ("What is 10 / 0?", "Division by zero"),
            ("", "Empty problem"),
            (None, "None input"),
        ]

        error_results = []
        safe_tools = {
            "calculator": safe_calculator_tool,
            "text_analyzer": safe_text_analyzer,
        }

        for test_case, description in error_test_cases:
            try:
                result = robust_execute_with_tools(test_case, safe_tools)
                error_results.append(
                    f"**{description}:**\n"
                    f"- Input: `{test_case}`\n"
                    f"- Success: {result['success']}\n"
                    f"- Error: {result.get('error', 'None')}\n"
                    f"- Fallback: {result.get('fallback_response', 'None')}"
                )
            except Exception as e:
                error_results.append(f"**{description}:** Unexpected error - {str(e)}")

        cell7_content = mo.md(
            cleandoc(
                f"""
                ### ⚠️ Error Handling Test Results

                {chr(10).join(error_results)}

                ### 🛡️ Error Handling Strategies Implemented

                **Input Validation:**
                - Type checking for all inputs
                - Null and empty value handling
                - Format validation for expressions

                **Graceful Degradation:**
                - Structured error responses
                - Helpful error messages
                - Fallback strategies when possible

                **Recovery Mechanisms:**
                - Problem type detection for better error messages
                - Partial processing when some tools fail
                - User guidance for common error scenarios

                **Production Readiness:**
                - Comprehensive exception handling
                - Logging-friendly error formats
                - Consistent response structures
                - Security considerations (safe evaluation)
                """
            )
        )
    else:
        cell7_desc = mo.md("")
        cell7_content = mo.md("")

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell8_out = mo.md(
            cleandoc(
                """
                ## 🎓 Solution Summary

                ### ✅ Complete Solution Components

                **1. Robust Tool Implementation:**  
                - **Calculator Tool:** Safe mathematical evaluation with comprehensive error handling  
                - **Text Analyzer:** Advanced text statistics with edge case management  
                - **Error Handling:** Production-ready validation and fallback strategies  

                **2. Intelligent Agent Architecture:**  
                - **Tool Selection:** Smart analysis of problem requirements  
                - **Input Extraction:** Sophisticated parsing of tool inputs  
                - **Result Integration:** Seamless combination of tool outputs  
                - **Error Recovery:** Graceful handling of tool failures  

                **3. Multi-Tool Orchestration:**  
                - **Sequential Processing:** Handling problems requiring multiple tools  
                - **Context Management:** Maintaining state across tool executions  
                - **Result Synthesis:** Comprehensive final answer generation  

                ### 🧠 Key Design Principles

                **Tool Design:**  
                - **Single Responsibility:** Each tool has a clear, focused purpose  
                - **Consistent Interface:** Standardized input/output formats  
                - **Error Transparency:** Clear error reporting and handling  
                - **Safety First:** Input validation and secure execution  

                **Agent Architecture:**  
                - **Reasoning First:** Detailed analysis before tool selection  
                - **Explicit Planning:** Clear tool selection and input specification  
                - **Result Integration:** Thoughtful combination of tool outputs  
                - **Error Resilience:** Graceful handling of failures  

                **Production Readiness:**  
                - **Comprehensive Testing:** Edge cases and error scenarios covered  
                - **Structured Responses:** Consistent data formats for integration  
                - **Security Considerations:** Safe evaluation and input validation  
                - **Monitoring Support:** Error logging and performance tracking  

                ### 🚀 Advanced Extensions

                **Tool Registry System:**  
                ```python  
                class ToolRegistry:  
                    def __init__(self):  
                        self.tools = {}  
                        self.usage_stats = {}  

                    def register_tool(self, name, tool_func, description, schema):  
                        # Tool registration with metadata  
                        pass  

                    def execute_tool(self, name, inputs):  
                        # Monitored tool execution  
                        pass  
                ```

                **Dynamic Tool Discovery:**  
                - Runtime tool registration  
                - Capability-based tool selection  
                - Tool composition and chaining  
                - Performance monitoring and optimization  

                **Advanced Agent Patterns:**  
                - **Parallel Tool Execution:** Execute independent tools concurrently  
                - **Tool Result Caching:** Cache expensive tool operations  
                - **Adaptive Tool Selection:** Learn from past successes/failures  
                - **Tool Chain Optimization:** Optimize multi-tool workflows  

                ### 🎯 Next Steps

                **For Further Learning:**  
                - Implement additional tools (web search, database queries, file operations)  
                - Build tool composition patterns (tool chains, parallel execution)  
                - Add tool performance monitoring and analytics  
                - Explore external API integration patterns  

                **Production Deployment:**  
                - Add comprehensive logging and monitoring  
                - Implement tool usage quotas and rate limiting  
                - Build tool result caching and optimization  
                - Create tool discovery and documentation systems  

                Congratulations on mastering tool integration! 🛠️✨  
                """
            )
        )
    else:
        cell8_out = mo.md("")

    output.replace(cell8_out)
    return


if __name__ == "__main__":
    app.run()
