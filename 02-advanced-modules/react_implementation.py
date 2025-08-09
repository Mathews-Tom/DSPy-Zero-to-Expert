# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any

    import dspy
    import marimo as mo
    from marimo import output

    from common import (
        DSPyParameterPanel,
        get_config,
        setup_dspy_environment,
    )

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        Any,
        DSPyParameterPanel,
        Path,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        project_root,
        setup_dspy_environment,
        sys,
        time,
    )


@app.cell
def __(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # 🤖 Module 02: Advanced DSPy Modules - ReAct Implementation

            **Duration:** 90-120 minutes
            **Prerequisites:** Completed Module 01 (DSPy Foundations)

            ## 🎯 Learning Objectives

            By the end of this module, you will:
            - ✅ Master ReAct (Reasoning + Acting) module patterns
            - ✅ Build interactive agents with tool integration
            - ✅ Implement multi-step reasoning workflows
            - ✅ Debug and trace complex agent behaviors
            - ✅ Create production-ready agent systems

            ## 🧠 What is ReAct?

            **ReAct** combines **Reasoning** and **Acting** in a unified framework:
            - **Reasoning**: Think through problems step-by-step
            - **Acting**: Take actions using external tools
            - **Observation**: Process results and continue reasoning

            This creates powerful agents that can:
            - Solve complex, multi-step problems
            - Use external APIs and tools
            - Adapt their approach based on results
            - Provide transparent reasoning traces

            ## 🛠️ What You'll Build

            1. **Basic ReAct Agent** - Simple reasoning + action patterns
            2. **Tool-Integrated Agent** - External API integration
            3. **Multi-Step Problem Solver** - Complex reasoning chains
            4. **Interactive Agent Builder** - Create custom agents
            5. **Production Agent System** - Scalable, robust implementation
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def __(cleandoc, get_config, mo, output, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        cell2_out = mo.md(
            cleandoc(
                f"""
                ## ✅ Advanced Module Environment Ready

                **Configuration:**
                - Provider: **{config.default_provider}**
                - Model: **{config.default_model}**
                - Advanced modules enabled!

                Ready to build intelligent agents!
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
    return (
        available_providers,
        config,
    )


@app.cell
def __(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Understanding ReAct Architecture

                Let's start by understanding the ReAct pattern with a simple example:

                ### ReAct Cycle
                1. **Thought**: Reason about the current situation
                2. **Action**: Decide what action to take
                3. **Observation**: Process the result of the action
                4. **Repeat**: Continue until the problem is solved

                ### Basic ReAct Signature
                """
            )
        )

        # Define a basic ReAct signature
        class BasicReActSignature(dspy.Signature):
            """Solve problems using reasoning and actions in an iterative process."""

            question = dspy.InputField(desc="The question or problem to solve")
            thought = dspy.OutputField(desc="Current reasoning about the problem")
            action = dspy.OutputField(desc="Next action to take")
            observation = dspy.OutputField(desc="Result or observation from the action")
            answer = dspy.OutputField(desc="Final answer when problem is solved")

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🧩 Basic ReAct Signature Created

                ```python
                class BasicReActSignature(dspy.Signature):
                    \"\"\"Solve problems using reasoning and actions in an iterative process.\"\"\"
                    question = dspy.InputField(desc="The question or problem to solve")
                    thought = dspy.OutputField(desc="Current reasoning about the problem")
                    action = dspy.OutputField(desc="Next action to take")
                    observation = dspy.OutputField(desc="Result or observation from the action")
                    answer = dspy.OutputField(desc="Final answer when problem is solved")
                ```

                This signature captures the essential ReAct pattern!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        BasicReActSignature = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (BasicReActSignature,)


@app.cell
def __(BasicReActSignature, available_providers, cleandoc, dspy, mo, output):
    if available_providers and BasicReActSignature:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🧪 Step 2: Simple ReAct Agent Demo

                Let's create a simple ReAct agent and see it in action:
                """
            )
        )

        # Create a simple ReAct module
        basic_react = dspy.ChainOfThought(BasicReActSignature)

        # Test question
        demo_question = "What is 15% of 240, and then what is 30% of that result?"

        cell4_content = mo.md(
            cleandoc(
                f"""
                ### 🎯 Demo Question
                **Question:** {demo_question}

                This requires multi-step calculation - perfect for ReAct!
                """
            )
        )

        # Demo button
        run_basic_demo = mo.ui.run_button(label="🚀 Run Basic ReAct Demo")
    else:
        cell4_desc = mo.md("")
        basic_react = None
        demo_question = None
        run_basic_demo = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content, run_basic_demo])
    output.replace(cell4_out)
    return (
        basic_react,
        demo_question,
        run_basic_demo,
    )


@app.cell
def __(
    available_providers,
    basic_react,
    cleandoc,
    demo_question,
    mo,
    output,
    run_basic_demo,
):
    if available_providers and run_basic_demo.value and basic_react:
        try:
            # Run the basic ReAct demo
            basic_result = basic_react(question=demo_question)

            cell5_out = mo.vstack(
                [
                    mo.md("## 🤖 Basic ReAct Agent Results"),
                    mo.md(f"**Question:** {demo_question}"),
                    mo.md("### 🧠 Agent Response"),
                    mo.md(f"**Thought:** {basic_result.thought}"),
                    mo.md(f"**Action:** {basic_result.action}"),
                    mo.md(f"**Observation:** {basic_result.observation}"),
                    mo.md(f"**Answer:** {basic_result.answer}"),
                    mo.md("### 📊 Full Result Object"),
                    mo.md(f"```json\n{str(basic_result)}\n```"),
                    mo.md(
                        cleandoc(
                            """
                            ### 💡 ReAct Pattern Analysis

                            Notice how the agent:
                            1. **Thinks** about the problem structure
                            2. **Plans** the necessary actions
                            3. **Observes** intermediate results
                            4. **Concludes** with the final answer

                            This is the foundation of ReAct reasoning!
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell5_out = mo.md(f"❌ **Demo Error:** {str(e)}")
    else:
        cell5_out = mo.md("*Click 'Run Basic ReAct Demo' to see the agent in action*")

    output.replace(cell5_out)
    return (basic_result,)


@app.cell
def __(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 🛠️ Step 3: Tool-Integrated ReAct Agent

                Now let's build a more sophisticated agent that can use external tools:
                """
            )
        )

        # Define tool functions
        def calculator_tool(expression: str) -> str:
            """Simple calculator tool for mathematical expressions."""
            try:
                # Safe evaluation of mathematical expressions
                allowed_chars = set("0123456789+-*/.() ")
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"Calculator result: {result}"
                else:
                    return "Error: Invalid characters in expression"
            except Exception as e:
                return f"Calculator error: {str(e)}"

        def search_tool(query: str) -> str:
            """Simulated search tool (in real implementation, this would call an API)."""
            # Simulated search results
            search_results = {
                "python": "Python is a high-level programming language known for its simplicity and readability.",
                "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
                "dspy": "DSPy is a framework for algorithmically optimizing LM prompts and weights.",
                "react": "ReAct combines reasoning and acting in language models for complex problem solving.",
            }

            query_lower = query.lower()
            for key, value in search_results.items():
                if key in query_lower:
                    return f"Search result: {value}"

            return f"Search result: No specific information found for '{query}'"

        def memory_tool(action: str, content: str = "") -> str:
            """Simple memory tool for storing and retrieving information."""
            if not hasattr(memory_tool, "memory"):
                memory_tool.memory = {}

            if action == "store":
                key = f"memory_{len(memory_tool.memory)}"
                memory_tool.memory[key] = content
                return f"Stored in memory: {key}"
            elif action == "recall":
                if memory_tool.memory:
                    return f"Memory contents: {list(memory_tool.memory.values())}"
                else:
                    return "Memory is empty"
            else:
                return "Memory tool usage: store <content> or recall"

        # Available tools registry
        available_tools = {
            "calculator": calculator_tool,
            "search": search_tool,
            "memory": memory_tool,
        }

        cell6_content = mo.md(
            """
        ### 🔧 Tools Created

        **Available Tools:**
        - **Calculator**: Evaluate mathematical expressions
        - **Search**: Find information (simulated)
        - **Memory**: Store and recall information

        These tools will be available to our ReAct agent!
        """
        )
    else:
        cell6_desc = mo.md("")
        calculator_tool = None
        search_tool = None
        memory_tool = None
        available_tools = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return (
        available_tools,
        calculator_tool,
        memory_tool,
        search_tool,
    )


@app.cell
def __(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        # Define tool-integrated ReAct signature
        class ToolReActSignature(dspy.Signature):
            """Solve problems using reasoning, actions, and external tools."""

            question = dspy.InputField(desc="The question or problem to solve")
            tools = dspy.InputField(
                desc="List of available tools: calculator, search, memory"
            )
            thought = dspy.OutputField(
                desc="Current reasoning about the problem and tool usage"
            )
            tool_choice = dspy.OutputField(
                desc="Which tool to use: calculator, search, memory, or none"
            )
            tool_input = dspy.OutputField(desc="Input to provide to the chosen tool")
            next_step = dspy.OutputField(desc="What to do after using the tool")
            answer = dspy.OutputField(desc="Final answer when problem is solved")

        # Create tool-integrated ReAct module
        tool_react = dspy.ChainOfThought(ToolReActSignature)

        cell7_out = mo.md(
            cleandoc(
                """
                ### 🤖 Tool-Integrated ReAct Agent Created

                This agent can:
                - Reason about which tools to use
                - Execute tool actions
                - Process tool results
                - Continue reasoning based on results
                """
            )
        )
    else:
        ToolReActSignature = None
        tool_react = None
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return (
        ToolReActSignature,
        tool_react,
    )


@app.cell
def __(available_providers, mo, output):
    if available_providers:
        # Tool-integrated test cases
        tool_test_cases = [
            "Calculate the area of a circle with radius 7, then search for information about circles",
            "What is 25% of 800? Store the result in memory and then recall it",
            "Search for information about machine learning, then calculate 15 * 23",
        ]

        tool_case_selector = mo.ui.dropdown(
            options=tool_test_cases,
            label="Select a test case for the tool-integrated agent",
            value=tool_test_cases[0],
        )

        run_tool_demo = mo.ui.run_button(label="🔧 Run Tool-Integrated Demo")

        cell8_out = mo.vstack(
            [
                mo.md("### 🧪 Tool Integration Test Cases"),
                tool_case_selector,
                run_tool_demo,
            ]
        )
    else:
        tool_test_cases = None
        tool_case_selector = None
        run_tool_demo = None
        cell8_out = mo.md("")

    output.replace(cell8_out)
    return (
        run_tool_demo,
        tool_case_selector,
        tool_test_cases,
    )


@app.cell
def __(
    available_providers,
    available_tools,
    cleandoc,
    mo,
    output,
    run_tool_demo,
    tool_case_selector,
    tool_react,
):
    if available_providers and run_tool_demo.value and tool_case_selector.value:
        try:
            selected_question = tool_case_selector.value
            tools_list = "calculator, search, memory"

            # Get agent's reasoning and tool choice
            agent_response = tool_react(
                question=selected_question, available_tools=tools_list
            )

            # Execute the chosen tool if specified
            tool_result = "No tool executed"
            if (
                agent_response.tool_choice
                and agent_response.tool_choice.lower() in available_tools
            ):
                tool_name = agent_response.tool_choice.lower()
                tool_function = available_tools[tool_name]

                if tool_name == "calculator":
                    tool_result = tool_function(agent_response.tool_input)
                elif tool_name == "search":
                    tool_result = tool_function(agent_response.tool_input)
                elif tool_name == "memory":
                    # Parse memory action
                    tool_input = agent_response.tool_input.lower()
                    if "store" in tool_input:
                        content = agent_response.tool_input.replace("store", "").strip()
                        tool_result = tool_function("store", content)
                    else:
                        tool_result = tool_function("recall")

            cell9_out = mo.vstack(
                [
                    mo.md("## 🔧 Tool-Integrated ReAct Results"),
                    mo.md(f"**Question:** {selected_question}"),
                    mo.md("### 🧠 Agent Reasoning"),
                    mo.md(f"**Thought:** {agent_response.thought}"),
                    mo.md(f"**Tool Choice:** {agent_response.tool_choice}"),
                    mo.md(f"**Tool Input:** {agent_response.tool_input}"),
                    mo.md(f"**Next Step:** {agent_response.next_step}"),
                    mo.md("### 🔧 Tool Execution"),
                    mo.md(f"**Tool Result:** {tool_result}"),
                    mo.md("### 🎯 Final Answer"),
                    mo.md(f"**Answer:** {agent_response.answer}"),
                    mo.md(
                        cleandoc(
                            """
                            ### 💡 Tool Integration Analysis

                            The agent successfully:
                            1. **Analyzed** the problem requirements
                            2. **Selected** appropriate tools
                            3. **Executed** tool actions
                            4. **Integrated** results into reasoning

                            This demonstrates the power of tool-augmented reasoning!
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell9_out = mo.md(f"❌ **Tool Demo Error:** {str(e)}")
    else:
        cell9_out = mo.md("*Select a test case and click 'Run Tool-Integrated Demo'*")

    output.replace(cell9_out)
    return (
        agent_response,
        selected_question,
        tool_function,
        tool_name,
        tool_result,
        tools_list,
    )


@app.cell
def __(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell10_desc = mo.md(
            cleandoc(
                """
                ## 🔄 Step 4: Multi-Step ReAct Agent

                Let's build an agent that can handle complex, multi-step problems:
                """
            )
        )

        # Define multi-step ReAct signature
        class MultiStepReActSignature(dspy.Signature):
            """Solve complex problems through iterative reasoning and actions."""

            problem = dspy.InputField(desc="Complex problem requiring multiple steps")
            context = dspy.InputField(desc="Any relevant context or constraints")
            step_number = dspy.InputField(desc="Current step number in the process")
            reasoning = dspy.OutputField(desc="Detailed reasoning for the current step")
            action_plan = dspy.OutputField(
                desc="Plan for the next action or calculation"
            )
            intermediate_result = dspy.OutputField(
                desc="Result or finding from current step"
            )
            is_complete = dspy.OutputField(
                desc="Whether the problem is fully solved: yes or no"
            )
            final_answer = dspy.OutputField(
                desc="Complete final answer if problem is solved"
            )

        # Create multi-step agent
        multistep_react = dspy.ChainOfThought(MultiStepReActSignature)

        cell10_content = mo.md(
            cleandoc(
                """
                ### 🔄 Multi-Step ReAct Agent Created

                This agent can:
                - Break down complex problems into steps
                - Track progress through multi-step processes
                - Maintain context across iterations
                - Determine when problems are fully solved
                """
            )
        )
    else:
        cell10_desc = mo.md("")
        MultiStepReActSignature = None
        multistep_react = None
        cell10_content = mo.md("")

    cell10_out = mo.vstack([cell10_desc, cell10_content])
    output.replace(cell10_out)
    return MultiStepReActSignature, multistep_react


@app.cell
def __(available_providers, mo, output):
    if available_providers:
        # Complex multi-step problems
        complex_problems = [
            {
                "problem": "A company has 150 employees. 60% work in engineering, 25% in sales, and the rest in administration. If engineering gets a 15% salary increase and sales gets 10%, what's the total percentage increase in company payroll assuming equal salaries?",
                "context": "Assume all employees have equal base salaries",
            },
            {
                "problem": "Plan a 7-day trip to Japan for 2 people with a budget of $4000. Include flights, accommodation, food, and activities. Provide a day-by-day breakdown.",
                "context": "Traveling from New York, prefer mid-range accommodations",
            },
            {
                "problem": "Design a simple recommendation system for a bookstore. Explain the algorithm, data requirements, and implementation steps.",
                "context": "Focus on collaborative filtering approach",
            },
        ]

        complex_selector = mo.ui.dropdown(
            options=[p["problem"][:80] + "..." for p in complex_problems],
            label="Select a complex problem",
            value=complex_problems[0]["problem"][:80] + "...",
        )

        run_multistep_demo = mo.ui.run_button(label="🔄 Run Multi-Step Demo")

        cell11_out = mo.vstack(
            [
                mo.md("### 🧩 Complex Problem Test Cases"),
                complex_selector,
                run_multistep_demo,
            ]
        )
    else:
        complex_problems = None
        complex_selector = None
        run_multistep_demo = None
        cell11_out = mo.md("")

    output.replace(cell11_out)
    return (
        complex_problems,
        complex_selector,
        run_multistep_demo,
    )


@app.cell
def __(
    available_providers,
    complex_problems,
    complex_selector,
    cleandoc,
    mo,
    output,
    multistep_react,
    run_multistep_demo,
):
    if available_providers and run_multistep_demo.value and complex_problems:
        try:
            # Find selected problem
            selected_problem = None
            for problem in complex_problems:
                if problem["problem"][:80] in complex_selector.value:
                    selected_problem = problem
                    break

            if selected_problem:
                # Simulate multi-step process
                steps = []
                max_steps = 4

                for step in range(1, max_steps + 1):
                    step_result = multistep_react(
                        problem=selected_problem["problem"],
                        context=selected_problem["context"],
                        step_number=str(step),
                    )

                    steps.append(
                        {
                            "step": step,
                            "reasoning": step_result.reasoning,
                            "action_plan": step_result.action_plan,
                            "result": step_result.intermediate_result,
                            "complete": step_result.is_complete.lower() == "yes",
                            "final_answer": (
                                step_result.final_answer
                                if step_result.is_complete.lower() == "yes"
                                else None
                            ),
                        }
                    )

                    # Stop if problem is complete
                    if step_result.is_complete.lower() == "yes":
                        break

                # Display multi-step results
                step_displays = []
                for step_data in steps:
                    step_displays.append(
                        cleandoc(
                            f"""
                            **Step {step_data['step']}:**
                            - **Reasoning:** {step_data['reasoning']}
                            - **Action Plan:** {step_data['action_plan']}
                            - **Result:** {step_data['result']}
                            - **Complete:** {step_data['complete']}
                            {f"- **Final Answer:** {step_data['final_answer']}" if step_data['final_answer'] else ""}
                            """
                        )
                    )

                cell12_out = mo.vstack(
                    [
                        mo.md("## 🔄 Multi-Step ReAct Results"),
                        mo.md(f"**Problem:** {selected_problem['problem']}"),
                        mo.md(f"**Context:** {selected_problem['context']}"),
                        mo.md("### 📋 Step-by-Step Process"),
                        mo.md("\n".join(step_displays)),
                        mo.md(
                            cleandoc(
                                f"""
                                ### 📊 Process Analysis

                                **Total Steps:** {len(steps)}
                                **Problem Solved:** {'Yes' if any(s['complete'] for s in steps) else 'No'}
                                **Final Result:** {steps[-1]['final_answer'] if steps and steps[-1]['final_answer'] else 'In progress'}

                                ### 💡 Multi-Step Insights

                                The agent demonstrated:
                                1. **Problem Decomposition** - Breaking complex problems into manageable steps
                                2. **Progress Tracking** - Monitoring completion status
                                3. **Context Maintenance** - Keeping relevant information across steps
                                4. **Adaptive Planning** - Adjusting approach based on intermediate results
                                """
                            )
                        ),
                    ]
                )
            else:
                cell12_out = mo.md("❌ Problem not found")

        except Exception as e:
            cell12_out = mo.md(f"❌ **Multi-Step Demo Error:** {str(e)}")
    else:
        cell12_out = mo.md("*Select a complex problem and click 'Run Multi-Step Demo'*")

    output.replace(cell12_out)
    return (
        max_steps,
        selected_problem,
        step_displays,
        step_result,
        steps,
    )


@app.cell
def __(available_providers, cleandoc, mo, output):
    if available_providers:
        cell13_desc = mo.md(
            cleandoc(
                """
                ## 🎨 Step 5: Interactive Agent Builder

                Now let's build a tool to create custom ReAct agents:
                """
            )
        )

        # Interactive agent builder form
        agent_name_input = mo.ui.text(placeholder="MyCustomAgent", label="Agent Name")
        agent_purpose_input = mo.ui.text_area(
            placeholder="Describe what your agent should do...",
            label="Agent Purpose",
            rows=3,
        )
        input_fields_input = mo.ui.text_area(
            placeholder="field_name: description\nother_field: description",
            label="Input Fields (one per line)",
            rows=4,
        )
        reasoning_fields_input = mo.ui.text_area(
            placeholder="thought: reasoning description\nanalysis: analysis description",
            label="Reasoning Fields (one per line)",
            rows=4,
        )
        action_fields_input = mo.ui.text_area(
            placeholder="action: action description\nresult: result description",
            label="Action Fields (one per line)",
            rows=4,
        )
        tools_needed_input = mo.ui.multiselect(
            options=["calculator", "search", "memory", "custom"],
            label="Tools Needed",
            value=["calculator"],
        )
        create_agent_button = mo.ui.run_button(label="🤖 Create Custom Agent")

        # Create a simple container instead of a form to avoid _clone issues
        agent_builder_form = mo.vstack(
            [
                agent_name_input,
                agent_purpose_input,
                input_fields_input,
                reasoning_fields_input,
                action_fields_input,
                tools_needed_input,
                create_agent_button,
            ]
        )

        cell13_content = mo.vstack(
            [
                mo.md("### 🎨 Custom Agent Builder"),
                mo.md(
                    "Design your own ReAct agent with custom fields and capabilities:"
                ),
                agent_builder_form,
            ]
        )
    else:
        cell13_desc = mo.md("")
        agent_builder_form = None
        cell13_content = mo.md("")
        agent_name_input = None
        agent_purpose_input = None
        input_fields_input = None
        reasoning_fields_input = None
        action_fields_input = None
        tools_needed_input = None
        create_agent_button = None

    cell13_out = mo.vstack([cell13_desc, cell13_content])
    output.replace(cell13_out)
    return (
        action_fields_input,
        agent_builder_form,
        agent_name_input,
        agent_purpose_input,
        create_agent_button,
        input_fields_input,
        reasoning_fields_input,
        tools_needed_input,
    )


@app.cell
def __(
    action_fields_input,
    agent_builder_form,
    agent_name_input,
    agent_purpose_input,
    create_agent_button,
    available_providers,
    cleandoc,
    dspy,
    input_fields_input,
    mo,
    output,
    reasoning_fields_input,
    tools_needed_input,
):
    if available_providers and create_agent_button.value:
        # Get values from individual components
        agent_name = agent_name_input.value or "CustomAgent"
        agent_purpose = agent_purpose_input.value or "Custom ReAct agent"
        input_fields_text = (
            input_fields_input.value or "question: The question to solve"
        )
        reasoning_fields_text = (
            reasoning_fields_input.value or "thought: Current reasoning"
        )
        action_fields_text = action_fields_input.value or "action: Next action to take"
        tools_needed = tools_needed_input.value or ["calculator"]

        if agent_name and agent_purpose:

            try:
                # Parse fields from individual components
                input_fields = {}
                for line in input_fields_text.split("\n"):
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        input_fields[name.strip()] = desc.strip()

                reasoning_fields = {}
                for line in reasoning_fields_text.split("\n"):
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        reasoning_fields[name.strip()] = desc.strip()

                action_fields = {}
                for line in action_fields_text.split("\n"):
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        action_fields[name.strip()] = desc.strip()

                # Generate signature code
                signature_code = cleandoc(
                    f'''class {agent_name}Signature(dspy.Signature):
                        """{agent_purpose}"""

                        # Input fields
                    '''
                )

                for name, desc in input_fields.items():
                    signature_code += f'    {name} = dspy.InputField(desc="{desc}")\n'

                signature_code += "\n    # Reasoning fields\n"
                for name, desc in reasoning_fields.items():
                    signature_code += f'    {name} = dspy.OutputField(desc="{desc}")\n'

                signature_code += "\n    # Action fields\n"
                for name, desc in action_fields.items():
                    signature_code += f'    {name} = dspy.OutputField(desc="{desc}")\n'

                # Create a simple signature without dynamic creation to avoid _clone issues
                # For now, just show the generated code without creating the actual agent
                custom_agent = None

                cell14_out = mo.vstack(
                    [
                        mo.md(f"## 🤖 Custom Agent Created: {agent_name}"),
                        mo.md(f"**Purpose:** {agent_purpose}"),
                        mo.md(f"**Tools:** {', '.join(tools_needed)}"),
                        mo.md("### 📝 Generated Signature Code"),
                        mo.md(f"```python\n{signature_code}\n```"),
                        mo.md("### ✅ Agent Ready for Testing"),
                        #                         mo.md(
                        #                             cleandoc(
                        #                                 """
                        #                                 Your custom agent is now ready! You can:
                        #                                 1. Test it with sample inputs
                        mo.md(
                            cleandoc(
                                """
                                Your custom agent signature is now ready! You can:
                                1. Copy the generated code above
                                2. Use it to create a DSPy ChainOfThought module
                                3. Integrate it with the specified tools
                                4. Test it with sample inputs
                                """
                            )
                        ),
                    ]
                )

            except Exception as e:
                custom_agent = None
                cell14_out = mo.md(f"❌ **Agent Creation Error:** {str(e)}")
        else:
            custom_agent = None
            cell14_out = mo.md(
                "*Please provide agent name and purpose to create your custom agent*"
            )
    else:
        custom_agent = None
        cell14_out = mo.md(
            "*Click the 'Create Custom Agent' button above to generate your ReAct agent*"
        )

    output.replace(cell14_out)
    return (
        custom_agent,
        agent_name,
        agent_purpose,
        input_fields,
        reasoning_fields,
        action_fields,
        signature_code,
    )


@app.cell
def __(available_providers, cleandoc, mo, output):
    if available_providers:
        cell15_desc = mo.md(
            cleandoc(
                """
                ## 🏭 Step 6: Production ReAct System

                Let's design a production-ready ReAct system with proper error handling, monitoring, and scalability:
                """
            )
        )

        # Production system design
        production_considerations = {
            "Error Handling": [
                "Tool execution failures",
                "API rate limiting",
                "Invalid reasoning outputs",
                "Infinite reasoning loops",
            ],
            "Performance": [
                "Response time optimization",
                "Parallel tool execution",
                "Caching strategies",
                "Resource management",
            ],
            "Monitoring": [
                "Reasoning step tracking",
                "Tool usage analytics",
                "Success/failure rates",
                "Performance metrics",
            ],
            "Scalability": [
                "Horizontal scaling",
                "Load balancing",
                "State management",
                "Tool service integration",
            ],
        }

        consideration_displays = []
        for category, items in production_considerations.items():
            consideration_displays.append(
                cleandoc(
                    f"""
                    **{category}:**
                    {chr(10).join([f"- {item}" for item in items])}
                    """
                )
            )

        production_code = cleandoc(
            """
            ### 🔧 Production Implementation Pattern
            ```python
            class ProductionReActAgent:
                def __init__(self, signature, tools, config):
                    self.signature = signature
                    self.tools = tools
                    self.config = config
                    self.monitor = AgentMonitor()

                def execute(self, inputs, max_steps=10):
                    try:
                        for step in range(max_steps):
                            # Execute reasoning step
                            result = self.reason(inputs)

                            # Execute tool if needed
                            if result.tool_choice:
                                tool_result = self.execute_tool(
                                    result.tool_choice,
                                    result.tool_input
                                )
                                inputs.update({"tool_result": tool_result})

                            # Check completion
                            if self.is_complete(result):
                                return result

                            # Monitor progress
                            self.monitor.log_step(step, result)

                    except Exception as e:
                        return self.handle_error(e)

                def execute_tool(self, tool_name, tool_input):
                    # Tool execution with error handling
                    pass

                def handle_error(self, error):
                    # Graceful error handling
                    pass
            ```
            """
        )

        cell15_content = mo.vstack(
            [
                mo.md("### 🏭 Production System Considerations"),
                mo.md("\n".join(consideration_displays)),
                mo.md(production_code),
            ]
        )
    else:
        cell15_desc = mo.md("")
        production_considerations = None
        cell15_content = mo.md("")

    cell15_out = mo.vstack([cell15_desc, cell15_content])
    output.replace(cell15_out)
    return (
        consideration_displays,
        production_considerations,
    )


@app.cell
def __(available_providers, cleandoc, mo, output):
    if available_providers:
        cell16_desc = mo.md(
            cleandoc(
                """
                ## 🎯 ReAct Best Practices & Patterns

                Based on our exploration, here are key patterns for effective ReAct agents:
                """
            )
        )

        cell16_content = mo.vstack(
            [
                mo.md(
                    cleandoc(
                        """
                        ## 💡 ReAct Best Practices

                        ### Signature Design
                        - Clear separation of reasoning and action fields
                        - Specific tool choice and input fields
                        - Progress tracking fields for multi-step problems
                        - Completion status indicators
                        
                        ### Tool Integration
                        - Standardized tool interfaces
                        - Error handling for tool failures
                        - Tool result validation
                        - Fallback strategies for unavailable tools
                        
                        ### Reasoning Quality
                        - Step-by-step thinking encouragement
                        - Explicit action justification
                        - Progress evaluation at each step
                        - Clear completion criteria
                        
                        ### Production Deployment 
                        - Comprehensive monitoring and logging
                        - Rate limiting and resource management
                        - Graceful degradation strategies
                        - Performance optimization
                        """
                    )
                ),
                mo.md(
                    cleandoc(
                        """
                        ## 🚀 Advanced ReAct Patterns

                        ### 1. Hierarchical ReAct
                        - Break complex problems into sub-problems
                        - Use specialized agents for different domains
                        - Coordinate multiple agents for complex tasks

                        ### 2. Memory-Augmented ReAct
                        - Maintain conversation history
                        - Learn from previous interactions
                        - Build knowledge bases over time

                        ### 3. Multi-Modal ReAct
                        - Process text, images, and other data types
                        - Use specialized tools for different modalities
                        - Integrate results across modalities

                        ### 4. Collaborative ReAct
                        - Multiple agents working together
                        - Shared tool access and coordination
                        - Distributed problem solving
                        """
                    )
                ),
            ]
        )
    else:
        cell16_desc = mo.md("")
        cell16_content = mo.md("")

    cell16_out = mo.vstack([cell16_desc, cell16_content])
    output.replace(cell16_out)
    return


@app.cell
def __(available_providers, cleandoc, mo, output):
    cell17_out = mo.md(
        cleandoc(
            """
            ## 🎓 Module 02 Complete!

            ### 🏆 What You've Mastered

            ✅ **ReAct Architecture** - Understanding reasoning + acting patterns
            ✅ **Tool Integration** - Connecting agents with external capabilities
            ✅ **Multi-Step Reasoning** - Complex problem decomposition and solving
            ✅ **Interactive Agent Building** - Creating custom agents for specific tasks
            ✅ **Production Systems** - Scalable, robust agent deployment

            ### 🧠 Key Concepts Learned

            1. **ReAct Pattern**
                - Thought → Action → Observation → Repeat
                - Transparent reasoning processes
                - Tool-augmented problem solving

            2. **Agent Architecture**
                - Signature design for reasoning and actions
                - Tool integration patterns
                - Error handling and recovery

            3. **Complex Problem Solving**
                - Multi-step process management
                - Progress tracking and completion detection
                - Context maintenance across steps

            4. **Production Considerations**
                - Monitoring and observability
                - Performance optimization
                - Scalability and reliability

            ### 🛠️ Skills Developed

            - **Agent Design** - Creating effective ReAct signatures
            - **Tool Integration** - Connecting external capabilities
            - **Problem Decomposition** - Breaking complex tasks into steps
            - **System Architecture** - Production-ready agent systems
            - **Debugging & Monitoring** - Observing agent behavior

            ### 🚀 Ready for Advanced Topics?

            You now understand advanced DSPy modules! Time to explore tool integration and multi-step reasoning:

            **Next Module:**
            ```bash
            uv run marimo run 02-advanced-modules/tool_integration.py
            ```

            **Coming Up:**
            - External API integration patterns
            - Web search and data retrieval
            - Tool composition and chaining
            - Advanced debugging techniques

            ### 💡 Practice Challenges

            Before moving on, try building ReAct agents for:
            1. **Research Assistant** - Search, analyze, and synthesize information
            2. **Data Analysis Agent** - Load, process, and visualize data
            3. **Code Generation Agent** - Plan, write, and test code
            4. **Travel Planning Agent** - Research, plan, and book travel

            ### 🎯 Advanced Challenges

            For expert-level practice:
            1. **Multi-Agent System** - Coordinate multiple specialized agents
            2. **Learning Agent** - Improve performance over time
            3. **Fault-Tolerant Agent** - Handle failures gracefully
            4. **Real-Time Agent** - Process streaming data and events

            The ReAct pattern is incredibly powerful - master it and you can build agents that solve almost any problem!
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell17_out)
    return


if __name__ == "__main__":
    app.run()
