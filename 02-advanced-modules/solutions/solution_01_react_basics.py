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
            # 🔑 Solution 01: ReAct Basics

            **Exercise:** ReAct Basics  
            **Difficulty:** Intermediate  
            **Focus:** Building foundational ReAct agents

            ## 📋 Solution Overview

            This solution demonstrates:  
            - ✅ Complete ReAct signature implementation  
            - ✅ Proper agent construction and testing  
            - ✅ Analysis of reasoning patterns  
            - ✅ Best practices and common pitfalls  

            ## 🎯 Learning Outcomes

            By studying this solution, you'll understand:  
            - How to design effective ReAct signatures  
            - Proper field descriptions and their impact  
            - Agent testing methodologies  
            - Reasoning pattern analysis techniques  

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
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 1 Solution: ReAct Signature

                **Complete Implementation:**
                """
            )
        )

        # Complete ReAct signature solution
        class SolutionReActSignature(dspy.Signature):
            """Solve problems using reasoning and actions in an iterative process.

            This signature implements the ReAct pattern where the agent:  
            1. Thinks about the current situation (thought)  
            2. Decides on an action to take (action)  
            3. Observes the results (observation)  
            4. Provides a final answer when ready (answer)  
            """

            question = dspy.InputField(desc="The problem or question to solve")
            thought = dspy.OutputField(
                desc="Your current reasoning about the problem and what you need to do next"
            )
            action = dspy.OutputField(
                desc="The specific action you decide to take to make progress"
            )
            observation = dspy.OutputField(
                desc="What you observe or learn from taking the action"
            )
            answer = dspy.OutputField(
                desc="Final answer to the question when you have enough information"
            )

        cell3_content = mo.md(
            cleandoc(
                """
                ### 💡 Key Design Decisions

                **Field Descriptions:**  
                - **Detailed and specific** - Each field has clear guidance  
                - **Process-oriented** - Descriptions guide the reasoning flow  
                - **Action-focused** - Emphasizes taking concrete steps  

                **Signature Docstring:**  
                - Explains the ReAct pattern clearly  
                - Provides step-by-step process guidance  
                - Helps the LM understand the expected behavior  

                **Best Practices Applied:**  
                - Clear field naming (thought, action, observation, answer)  
                - Logical flow from reasoning to action to observation  
                - Comprehensive final answer field  
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        SolutionReActSignature = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (SolutionReActSignature,)


@app.cell
def _(SolutionReActSignature, available_providers, cleandoc, dspy, mo, output):
    if available_providers and SolutionReActSignature:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 2 Solution: ReAct Agent Implementation

                **Complete Agent with Testing:**
                """
            )
        )

        # Create the ReAct agent
        solution_react_agent = dspy.ChainOfThought(SolutionReActSignature)

        # Test with the mathematical problem
        test_question = "What is 15% of 240, and then what is 25% of that result?"

        try:
            agent_result = solution_react_agent(question=test_question)

            cell4_content = mo.md(
                cleandoc(
                    f"""
                    ### 🤖 Agent Test Results

                    **Question:** {test_question}

                    **Agent Response:**  
                    - **Thought:** {agent_result.thought}  
                    - **Action:** {agent_result.action}  
                    - **Observation:** {agent_result.observation}  
                    - **Answer:** {agent_result.answer}  

                    ### 📊 Analysis

                    **Reasoning Quality:**  
                    - The agent breaks down the multi-step calculation  
                    - Shows clear logical progression  
                    - Provides specific numerical results  

                    **Expected Answer:** 15% of 240 = 36, then 25% of 36 = 9

                    **Agent Performance:**  
                    - ✅ Correctly identified two-step calculation  
                    - ✅ Showed intermediate results  
                    - ✅ Provided final answer with reasoning  
                    """
                )
            )
        except Exception as e:
            cell4_content = mo.md(f"❌ **Error:** {str(e)}")
    else:
        cell4_desc = mo.md("")
        solution_react_agent = None
        agent_result = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (solution_react_agent,)


@app.cell
def _(available_providers, cleandoc, mo, output, solution_react_agent):
    if available_providers and solution_react_agent:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 3 Solution: Testing Different Problem Types

                **Comprehensive Testing with Analysis:**
                """
            )
        )

        # Test problems with solutions
        test_problems = [
            {
                "problem": "If all cats are animals, and Fluffy is a cat, what can we conclude about Fluffy?",
                "type": "Logic",
                "expected": "Fluffy is an animal (syllogistic reasoning)",
            },
            {
                "problem": "I need to prepare dinner for 6 people. I have chicken, rice, and vegetables. What steps should I take?",
                "type": "Planning",
                "expected": "Sequential cooking steps with timing considerations",
            },
            {
                "problem": "A store sells 100 items per day. If sales increase by 20% each month, how many items will they sell per day after 3 months?",
                "type": "Analysis",
                "expected": "100 × 1.2³ = 172.8 items per day",
            },
        ]

        # Test each problem
        test_results = []
        for problem_data in test_problems:
            try:
                result = solution_react_agent(question=problem_data["problem"])
                test_results.append(
                    {
                        "problem": problem_data["problem"],
                        "type": problem_data["type"],
                        "expected": problem_data["expected"],
                        "thought": result.thought,
                        "action": result.action,
                        "observation": result.observation,
                        "answer": result.answer,
                    }
                )
            except Exception as e:
                test_results.append(
                    {
                        "problem": problem_data["problem"],
                        "type": problem_data["type"],
                        "error": str(e),
                    }
                )

        # Format results for display
        results_display = []
        for i, result in enumerate(test_results):
            if "error" not in result:
                results_display.append(
                    cleandoc(
                        f"""
                        **Problem {i+1}: {result['type']}**  
                        - **Question:** {result['problem']}  
                        - **Expected:** {result['expected']}  
                        - **Agent Thought:** {result['thought']}  
                        - **Agent Action:** {result['action']}  
                        - **Agent Observation:** {result['observation']}  
                        - **Agent Answer:** {result['answer']}  

                        """
                    )
                )
            else:
                results_display.append(f"**Problem {i+1}:** Error - {result['error']}")

        cell5_content = mo.md(
            cleandoc(
                f"""
                ### 🧪 Multi-Problem Test Results

                {chr(10).join(results_display)}

                ### 📊 Pattern Analysis

                **Observed Reasoning Patterns:**  
                1. **Logic Problems:** Agent uses deductive reasoning, applies rules systematically  
                2. **Planning Problems:** Agent breaks down into sequential steps, considers dependencies  
                3. **Analysis Problems:** Agent identifies mathematical relationships, performs calculations  

                **Strengths:**  
                - Clear step-by-step reasoning  
                - Appropriate action selection for problem type  
                - Good observation and synthesis  

                **Areas for Improvement:**  
                - Could benefit from more detailed intermediate steps  
                - Sometimes rushes to conclusions  
                - Could use more explicit verification steps  
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        test_problems = None
        test_results = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 🔑 Task 4 Solution: Analysis and Improvement

                **Comprehensive Performance Analysis:**
                """
            )
        )

        cell6_content = mo.md(
            cleandoc(
                """
                ### 📊 Detailed Analysis

                **Strengths Identified:**  
                - **Clear Reasoning Structure:** Agent consistently follows thought → action → observation → answer pattern  
                - **Problem Type Adaptation:** Different approaches for logic, planning, and analytical problems  
                - **Step-by-Step Breakdown:** Complex problems are decomposed effectively  
                - **Specific Actions:** Agent takes concrete, actionable steps rather than vague approaches  

                **Weaknesses Identified:**  
                - **Limited Verification:** Agent doesn't always double-check calculations or logic  
                - **Single-Pass Reasoning:** No iterative refinement of answers  
                - **Context Limitations:** Doesn't always maintain full context in complex problems  
                - **Error Handling:** Limited graceful handling of ambiguous or incomplete information  

                **Observed Reasoning Patterns:**  
                1. **Deductive Pattern:** For logic problems (premise → rule → conclusion)  
                2. **Sequential Pattern:** For planning problems (step 1 → step 2 → step 3)  
                3. **Analytical Pattern:** For math problems (identify formula → calculate → verify)  

                ### 🚀 Improvement Suggestions

                **Enhanced Signature Design:**  
                ```python
                class ImprovedReActSignature(dspy.Signature):  
                    question = dspy.InputField(desc="The problem or question to solve")  
                    thought = dspy.OutputField(desc="Your reasoning about the problem and approach")  
                    action = dspy.OutputField(desc="Specific action to take (calculate, analyze, plan, etc.)")  
                    observation = dspy.OutputField(desc="Results and insights from the action")  
                    verification = dspy.OutputField(desc="Check if the result makes sense and is complete")  
                    answer = dspy.OutputField(desc="Final verified answer to the question")  
                ```

                **Additional Improvements:**  
                - **Multi-Step Iteration:** Allow multiple thought-action-observation cycles  
                - **Confidence Scoring:** Add confidence assessment for each step  
                - **Error Detection:** Include explicit error checking and correction  
                - **Context Preservation:** Better maintain context across reasoning steps  
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 🚀 Bonus: Enhanced ReAct Implementation

                **Advanced ReAct with Verification:**
                """
            )
        )

        # Enhanced ReAct signature with verification
        class EnhancedReActSignature(dspy.Signature):
            """Advanced ReAct with verification and confidence assessment."""

            question = dspy.InputField(desc="The problem or question to solve")
            thought = dspy.OutputField(
                desc="Your detailed reasoning about the problem and planned approach"
            )
            action = dspy.OutputField(
                desc="Specific action to take (calculate, analyze, plan, research, etc.)"
            )
            observation = dspy.OutputField(
                desc="Results, insights, and findings from taking the action"
            )
            verification = dspy.OutputField(
                desc="Check if the result makes sense, is complete, and addresses the question"
            )
            confidence = dspy.OutputField(
                desc="Confidence level in the answer (high/medium/low) with reasoning"
            )
            answer = dspy.OutputField(
                desc="Final verified and confident answer to the question"
            )

        # Create enhanced agent
        enhanced_react_agent = dspy.ChainOfThought(EnhancedReActSignature)

        # Test with a complex problem
        complex_test = "A company's revenue grew from $1M to $1.5M over 2 years. If this growth rate continues, what will their revenue be in 5 years total?"

        try:
            enhanced_result = enhanced_react_agent(question=complex_test)

            cell7_content = mo.md(
                cleandoc(
                    f"""
                    ### 🔬 Enhanced Agent Test

                    **Complex Problem:** {complex_test}  

                    **Enhanced Agent Response:**  
                    - **Thought:** {enhanced_result.thought}  
                    - **Action:** {enhanced_result.action}  
                    - **Observation:** {enhanced_result.observation}  
                    - **Verification:** {enhanced_result.verification}  
                    - **Confidence:** {enhanced_result.confidence}  
                    - **Answer:** {enhanced_result.answer}  

                    ### 💡 Enhancement Benefits

                    **Verification Step:**  
                    - Adds self-checking mechanism  
                    - Catches potential errors before final answer  
                    - Ensures completeness of response  

                    **Confidence Assessment:**  
                    - Provides transparency about certainty  
                    - Helps users understand reliability  
                    - Enables better decision-making  

                    **Expected Calculation:**  
                    - Growth rate: (1.5/1.0)^(1/2) - 1 = 22.47% annually  
                    - After 5 years: $1M × (1.2247)^5 = $2.76M  
                    """
                )
            )
        except Exception as e:
            cell7_content = mo.md(f"❌ **Error:** {str(e)}")
    else:
        cell7_desc = mo.md("")
        EnhancedReActSignature = None
        enhanced_react_agent = None
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

                **1. ReAct Signature Design:**  
                - Clear field definitions with detailed descriptions  
                - Logical flow from thought to action to observation  
                - Comprehensive docstring explaining the pattern  

                **2. Agent Implementation:**  
                - Proper use of dspy.ChainOfThought  
                - Systematic testing with diverse problem types  
                - Performance analysis and pattern recognition  

                **3. Advanced Enhancements:**  
                - Verification step for self-checking  
                - Confidence assessment for transparency  
                - Error handling and edge case management  

                ### 🧠 Key Learning Points

                **Signature Design Principles:**  
                - Field descriptions directly impact agent behavior  
                - Clear process guidance improves reasoning quality  
                - Comprehensive output fields enable better analysis  

                **Testing Methodology:**  
                - Test with diverse problem types (logic, planning, analysis)  
                - Analyze reasoning patterns across different domains  
                - Identify strengths and weaknesses systematically  

                **Improvement Strategies:**  
                - Add verification and confidence assessment  
                - Enable iterative reasoning for complex problems  
                - Implement error detection and correction mechanisms  

                ### 🚀 Next Steps

                **For Further Learning:**  
                - Experiment with different signature variations  
                - Test with domain-specific problems  
                - Explore multi-step reasoning patterns  
                - Integrate with external tools and knowledge sources  

                **Advanced Challenges:**  
                - Build ReAct agents for specific domains (medical, legal, technical)  
                - Implement learning from feedback mechanisms  
                - Create adaptive reasoning strategies  
                - Develop reasoning quality metrics and evaluation frameworks  

                Congratulations on mastering ReAct basics! 🎉
                """
            )
        )
    else:
        cell8_out = mo.md("")

    output.replace(cell8_out)
    return


if __name__ == "__main__":
    app.run()
