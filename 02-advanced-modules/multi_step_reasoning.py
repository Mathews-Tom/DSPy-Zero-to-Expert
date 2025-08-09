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

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return cleandoc, dspy, get_config, mo, output, setup_dspy_environment, time


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # 🧠 Multi-Step Reasoning Pipeline

            **Duration:** 90-120 minutes  
            **Prerequisites:** Completed ReAct and Tool Integration

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Build complex multi-step reasoning pipelines  
            - ✅ Implement multi-hop question answering systems  
            - ✅ Create reasoning step tracking and visualization  
            - ✅ Debug complex reasoning workflows  
            - ✅ Optimize reasoning chains for performance and accuracy  

            ## 🧩 What is Multi-Step Reasoning?

            Multi-step reasoning involves:  
            - **Problem Decomposition** - Breaking complex problems into manageable steps  
            - **Sequential Processing** - Executing steps in logical order  
            - **Context Maintenance** - Preserving information across steps  
            - **Dynamic Planning** - Adapting the approach based on intermediate results  
            - **Result Synthesis** - Combining step outputs into final answers  

            ## 🏗️ Pipeline Architecture

            Our multi-step reasoning system includes:  
            1. **Step Planner** - Decomposes problems into logical steps  
            2. **Step Executor** - Executes individual reasoning steps  
            3. **Context Manager** - Maintains state across steps  
            4. **Progress Tracker** - Monitors and visualizes progress  
            5. **Result Synthesizer** - Combines outputs into final answers  

            Let's build sophisticated reasoning pipelines!
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
                ## ✅ Multi-Step Reasoning Environment Ready

                **Configuration:**
                - Provider: **{config.default_provider}**
                - Model: **{config.default_model}**
                - Advanced reasoning enabled!

                Ready to build complex reasoning systems!
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
                ## 🧩 Step 1: Step Planning Architecture

                Let's start by building a system that can decompose complex problems into steps:
                """
            )
        )

        # Step Planner Signature
        class StepPlannerSignature(dspy.Signature):
            """Decompose complex problems into logical, sequential steps."""

            problem = dspy.InputField(desc="Complex problem or question to solve")
            context = dspy.InputField(desc="Any relevant context or constraints")
            problem_type = dspy.OutputField(
                desc="Type of problem: analytical, computational, research, creative"
            )
            complexity_level = dspy.OutputField(
                desc="Complexity level: low, medium, high, very_high"
            )
            step_plan = dspy.OutputField(
                desc="Numbered list of logical steps to solve the problem"
            )
            estimated_steps = dspy.OutputField(desc="Estimated number of steps needed")
            dependencies = dspy.OutputField(
                desc="Dependencies between steps or required resources"
            )

        # Step Executor Signature
        class StepExecutorSignature(dspy.Signature):
            """Execute individual steps in a multi-step reasoning process."""

            step_description = dspy.InputField(
                desc="Description of the current step to execute"
            )
            step_number = dspy.InputField(desc="Current step number in the sequence")
            previous_results = dspy.InputField(
                desc="Results and context from previous steps"
            )
            reasoning = dspy.OutputField(desc="Detailed reasoning for this step")
            step_result = dspy.OutputField(
                desc="Result or output from executing this step"
            )
            confidence = dspy.OutputField(
                desc="Confidence in the step result (0.0-1.0)"
            )
            next_step_guidance = dspy.OutputField(
                desc="Guidance for the next step based on current results"
            )

        # Result Synthesizer Signature
        class ResultSynthesizerSignature(dspy.Signature):
            """Synthesize results from multiple reasoning steps into a final answer."""

            original_problem = dspy.InputField(desc="The original problem or question")
            all_step_results = dspy.InputField(desc="Results from all executed steps")
            step_confidences = dspy.InputField(desc="Confidence scores from each step")
            final_answer = dspy.OutputField(
                desc="Comprehensive final answer to the original problem"
            )
            reasoning_summary = dspy.OutputField(
                desc="Summary of the reasoning process"
            )
            confidence_assessment = dspy.OutputField(
                desc="Overall confidence in the final answer"
            )
            alternative_approaches = dspy.OutputField(
                desc="Alternative approaches that could be considered"
            )

        # Create modules
        step_planner = dspy.ChainOfThought(StepPlannerSignature)
        step_executor = dspy.ChainOfThought(StepExecutorSignature)
        result_synthesizer = dspy.ChainOfThought(ResultSynthesizerSignature)

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🧩 Multi-Step Reasoning Components Created

                **Components:**  
                - **Step Planner** - Decomposes problems into logical steps  
                - **Step Executor** - Executes individual reasoning steps  
                - **Result Synthesizer** - Combines step outputs into final answers  

                Each component uses ChainOfThought for detailed reasoning!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        StepPlannerSignature = None
        StepExecutorSignature = None
        ResultSynthesizerSignature = None
        step_planner = None
        step_executor = None
        result_synthesizer = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return result_synthesizer, step_executor, step_planner


@app.cell
def _(available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🔄 Step 2: Multi-Step Pipeline Implementation

                Now let's build a complete pipeline that orchestrates the reasoning process:
                """
            )
        )

        # Multi-Step Reasoning Pipeline
        class MultiStepReasoningPipeline:
            """Complete pipeline for multi-step reasoning problems."""

            def __init__(self, step_planner, step_executor, result_synthesizer):
                self.step_planner = step_planner
                self.step_executor = step_executor
                self.result_synthesizer = result_synthesizer
                self.execution_log = []

            def execute(self, problem: str, context: str = "", max_steps: int = 10):
                """Execute the complete multi-step reasoning pipeline."""
                self.execution_log = []

                try:
                    # Step 1: Plan the approach
                    self._log("Planning", "Decomposing problem into steps")
                    plan_result = self.step_planner(problem=problem, context=context)

                    self._log(
                        "Planning Complete",
                        {
                            "problem_type": plan_result.problem_type,
                            "complexity": plan_result.complexity_level,
                            "estimated_steps": plan_result.estimated_steps,
                            "plan": plan_result.step_plan,
                        },
                    )

                    # Step 2: Execute each step
                    step_results = []
                    step_confidences = []
                    previous_results = "Starting multi-step reasoning process."

                    # Parse steps from plan (simplified parsing)
                    steps = self._parse_steps(plan_result.step_plan)
                    actual_steps = min(len(steps), max_steps)

                    for i, step_desc in enumerate(steps[:actual_steps]):
                        self._log(f"Step {i+1}", f"Executing: {step_desc}")

                        step_result = self.step_executor(
                            step_description=step_desc,
                            step_number=str(i + 1),
                            previous_results=previous_results,
                        )

                        step_results.append(
                            {
                                "step_number": i + 1,
                                "description": step_desc,
                                "reasoning": step_result.reasoning,
                                "result": step_result.step_result,
                                "confidence": step_result.confidence,
                                "guidance": step_result.next_step_guidance,
                            }
                        )

                        step_confidences.append(step_result.confidence)
                        previous_results += (
                            f"\n\nStep {i+1} Result: {step_result.step_result}"
                        )

                        self._log(
                            f"Step {i+1} Complete",
                            {
                                "result": step_result.step_result,
                                "confidence": step_result.confidence,
                            },
                        )

                    # Step 3: Synthesize final answer
                    self._log("Synthesis", "Combining step results into final answer")

                    final_result = self.result_synthesizer(
                        original_problem=problem,
                        all_step_results=str(step_results),
                        step_confidences=str(step_confidences),
                    )

                    self._log(
                        "Complete",
                        {
                            "final_answer": final_result.final_answer,
                            "confidence": final_result.confidence_assessment,
                        },
                    )

                    return {
                        "success": True,
                        "plan": plan_result,
                        "step_results": step_results,
                        "final_result": final_result,
                        "execution_log": self.execution_log,
                        "total_steps": len(step_results),
                    }

                except Exception as e:
                    self._log("Error", f"Pipeline execution failed: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "execution_log": self.execution_log,
                    }

            def _parse_steps(self, step_plan: str) -> list[str]:
                """Parse step plan into individual steps."""
                steps = []
                lines = step_plan.split("\n")
                for line in lines:
                    line = line.strip()
                    if line and (
                        line[0].isdigit()
                        or line.startswith("-")
                        or line.startswith("•")
                    ):
                        # Remove numbering and clean up
                        clean_step = line
                        for prefix in [
                            "1.",
                            "2.",
                            "3.",
                            "4.",
                            "5.",
                            "6.",
                            "7.",
                            "8.",
                            "9.",
                            "10.",
                            "-",
                            "•",
                        ]:
                            if clean_step.startswith(prefix):
                                clean_step = clean_step[len(prefix) :].strip()
                                break
                        if clean_step:
                            steps.append(clean_step)
                return steps

            def _log(self, phase: str, message):
                """Log execution progress."""
                self.execution_log.append(
                    {"timestamp": time.time(), "phase": phase, "message": message}
                )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🔄 Multi-Step Pipeline Created

                **Pipeline Features:**  
                - **Problem Decomposition** - Automatic step planning  
                - **Sequential Execution** - Step-by-step processing  
                - **Context Preservation** - Information flow between steps  
                - **Progress Logging** - Detailed execution tracking  
                - **Error Handling** - Graceful failure management  

                The pipeline is ready to handle complex reasoning tasks!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        MultiStepReasoningPipeline = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (MultiStepReasoningPipeline,)


@app.cell
def _(
    MultiStepReasoningPipeline,
    available_providers,
    mo,
    output,
    result_synthesizer,
    step_executor,
    step_planner,
):
    if available_providers and MultiStepReasoningPipeline:
        # Create pipeline instance
        reasoning_pipeline = MultiStepReasoningPipeline(
            step_planner, step_executor, result_synthesizer
        )

        # Complex reasoning problems
        complex_reasoning_problems = [
            {
                "problem": "A company wants to expand internationally. They have $2M budget, 50 employees, and expertise in software development. Analyze the best expansion strategy considering market research, legal requirements, staffing, and financial projections.",
                "context": "Technology company, B2B SaaS product, currently US-based",
            },
            {
                "problem": "Design a sustainable urban transportation system for a city of 500,000 people. Consider environmental impact, cost-effectiveness, user adoption, and integration with existing infrastructure.",
                "context": "Mid-sized city, current public transport is limited, environmental concerns are high priority",
            },
            {
                "problem": "Investigate why a machine learning model's accuracy dropped from 95% to 78% after deployment. Provide a systematic approach to identify and fix the issue.",
                "context": "Image classification model, deployed 2 months ago, training data from 6 months ago",
            },
            {
                "problem": "Plan a comprehensive cybersecurity strategy for a healthcare organization with 10,000 patients, 200 staff, and multiple locations. Address compliance, training, technology, and incident response.",
                "context": "HIPAA compliance required, recent increase in healthcare cyberattacks, limited IT budget",
            },
        ]

        problem_selector = mo.ui.dropdown(
            options=[p["problem"][:80] + "..." for p in complex_reasoning_problems],
            label="Select a complex reasoning problem",
            value=complex_reasoning_problems[0]["problem"][:80] + "...",
        )

        run_pipeline_demo = mo.ui.run_button(label="🧠 Run Multi-Step Reasoning")

        cell5_out = mo.vstack(
            [
                mo.md("### 🧩 Complex Reasoning Problems"),
                mo.md(
                    "Select a problem that requires sophisticated multi-step analysis:"
                ),
                problem_selector,
                run_pipeline_demo,
            ]
        )
    else:
        reasoning_pipeline = None
        complex_reasoning_problems = None
        problem_selector = None
        run_pipeline_demo = None
        cell5_out = mo.md("")

    output.replace(cell5_out)
    return (
        complex_reasoning_problems,
        problem_selector,
        reasoning_pipeline,
        run_pipeline_demo,
    )


@app.cell
def _(
    available_providers,
    cleandoc,
    complex_reasoning_problems,
    mo,
    output,
    problem_selector,
    reasoning_pipeline,
    run_pipeline_demo,
    time,
):
    if available_providers and run_pipeline_demo.value and reasoning_pipeline:
        try:
            # Find selected problem
            selected_problem_data = None
            for problem_data in complex_reasoning_problems:
                if problem_data["problem"][:80] in problem_selector.value:
                    selected_problem_data = problem_data
                    break

            if selected_problem_data:
                # Execute the reasoning pipeline
                start_time = time.time()
                pipeline_result = reasoning_pipeline.execute(
                    problem=selected_problem_data["problem"],
                    context=selected_problem_data["context"],
                    max_steps=8,
                )
                execution_time = time.time() - start_time

                if pipeline_result["success"]:
                    # Display comprehensive results
                    plan = pipeline_result["plan"]
                    step_results = pipeline_result["step_results"]
                    final_result = pipeline_result["final_result"]

                    # Format step results
                    step_displays = []
                    for step in step_results:
                        step_displays.append(
                            cleandoc(
                                f"""
                                **Step {step['step_number']}: {step['description']}**
                                - **Reasoning:** {step['reasoning']}
                                - **Result:** {step['result']}
                                - **Confidence:** {step['confidence']}
                                - **Next Step Guidance:** {step['guidance']}
                                """
                            )
                        )

                    # Format execution log
                    log_displays = []
                    for log_entry in pipeline_result["execution_log"]:
                        timestamp = time.strftime(
                            "%H:%M:%S", time.localtime(log_entry["timestamp"])
                        )
                        log_displays.append(
                            f"**{timestamp} - {log_entry['phase']}:** {log_entry['message']}"
                        )

                    cell6_out = mo.vstack(
                        [
                            mo.md("## 🧠 Multi-Step Reasoning Results"),
                            mo.md(f"**Problem:** {selected_problem_data['problem']}"),
                            mo.md(f"**Context:** {selected_problem_data['context']}"),
                            mo.md(f"**Execution Time:** {execution_time:.2f} seconds"),
                            mo.md("### 📋 Problem Analysis & Planning"),
                            mo.md(f"**Problem Type:** {plan.problem_type}"),
                            mo.md(f"**Complexity Level:** {plan.complexity_level}"),
                            mo.md(f"**Estimated Steps:** {plan.estimated_steps}"),
                            mo.md(f"**Dependencies:** {plan.dependencies}"),
                            mo.md("**Step Plan:**"),
                            mo.md(plan.step_plan),
                            mo.md("### 🔄 Step-by-Step Execution"),
                            mo.md("\n".join(step_displays)),
                            mo.md("### 🎯 Final Synthesis"),
                            mo.md(f"**Final Answer:** {final_result.final_answer}"),
                            mo.md(
                                f"**Reasoning Summary:** {final_result.reasoning_summary}"
                            ),
                            mo.md(
                                f"**Confidence Assessment:** {final_result.confidence_assessment}"
                            ),
                            mo.md(
                                f"**Alternative Approaches:** {final_result.alternative_approaches}"
                            ),
                            mo.md("### 📊 Execution Analysis"),
                            mo.md(
                                f"**Total Steps Executed:** {pipeline_result['total_steps']}"
                            ),
                            mo.md(
                                f"**Average Step Confidence:** {sum(float(s.get('confidence', '0.5')) for s in step_results) / len(step_results):.2f}"
                            ),
                            mo.md("**Pipeline Success:** ✅ Yes"),
                            mo.md("### 📝 Execution Log"),
                            mo.md(
                                "\n".join(log_displays[:10])
                            ),  # Show first 10 log entries
                        ]
                    )
                else:
                    cell6_out = mo.md(
                        cleandoc(
                            f"""
                            ## ❌ Pipeline Execution Failed

                            **Error:** {pipeline_result['error']}

                            **Execution Log:**
                            {chr(10).join([f"- {entry['phase']}: {entry['message']}" for entry in pipeline_result['execution_log']])}
                            """
                        )
                    )
            else:
                mo.md("❌ Problem not found")

        except Exception as e:
            cell6_out = mo.md(f"❌ **Pipeline Demo Error:** {str(e)}")
    else:
        cell6_out = mo.md(
            "*Select a problem and click 'Run Multi-Step Reasoning' to see the pipeline in action*"
        )

    output.replace(cell6_out)
    return (pipeline_result,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
                """
                ## 🔍 Step 3: Multi-Hop Question Answering

                Let's build a specialized system for multi-hop question answering:
                """
            )
        )

        # Multi-Hop QA Signature
        class MultiHopQASignature(dspy.Signature):
            """Answer complex questions that require multiple reasoning hops."""

            question = dspy.InputField(
                desc="Complex question requiring multi-hop reasoning"
            )
            context_sources = dspy.InputField(
                desc="Available information sources or context"
            )
            hop_analysis = dspy.OutputField(desc="Analysis of reasoning hops needed")
            hop_1_question = dspy.OutputField(desc="First sub-question to answer")
            hop_1_answer = dspy.OutputField(desc="Answer to first sub-question")
            hop_2_question = dspy.OutputField(desc="Second sub-question based on hop 1")
            hop_2_answer = dspy.OutputField(desc="Answer to second sub-question")
            hop_3_question = dspy.OutputField(desc="Third sub-question if needed")
            hop_3_answer = dspy.OutputField(
                desc="Answer to third sub-question if applicable"
            )
            final_answer = dspy.OutputField(
                desc="Final comprehensive answer to original question"
            )
            reasoning_chain = dspy.OutputField(
                desc="Complete reasoning chain showing all hops"
            )

        # Create multi-hop QA module
        multihop_qa = dspy.ChainOfThought(MultiHopQASignature)

        cell7_content = mo.md(
            cleandoc(
                """
                ### 🔍 Multi-Hop QA System Created

                This system can:
                - **Decompose Questions** - Break complex questions into sub-questions
                - **Sequential Reasoning** - Answer sub-questions in logical order
                - **Information Integration** - Combine answers across reasoning hops
                - **Chain Visualization** - Show complete reasoning process
                """
            )
        )
    else:
        cell7_desc = mo.md("")
        MultiHopQASignature = None
        multihop_qa = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack(
        [
            cell7_desc,
            cell7_content,
        ]
    )
    output.replace(cell7_out)
    return (multihop_qa,)


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Multi-hop questions
        multihop_questions = [
            {
                "question": "What is the population of the capital city of the country where the 2024 Olympics were held?",
                "context": "The 2024 Olympics were held in Paris, France. Paris is the capital of France with a population of approximately 2.1 million in the city proper.",
            },
            {
                "question": "How many years after the invention of the telephone was the company founded that currently owns the most popular search engine?",
                "context": "The telephone was invented by Alexander Graham Bell in 1876. Google was founded in 1998 and owns the most popular search engine.",
            },
            {
                "question": "What is the GDP per capita of the country that produces the most coffee in the world?",
                "context": "Brazil produces the most coffee in the world. Brazil's GDP per capita is approximately $8,700 as of recent data.",
            },
            {
                "question": "In what year was the university founded where the creator of the World Wide Web studied for his undergraduate degree?",
                "context": "Tim Berners-Lee created the World Wide Web. He studied at Oxford University for his undergraduate degree. Oxford University was founded in 1096.",
            },
        ]

        multihop_selector = mo.ui.dropdown(
            options=[q["question"] for q in multihop_questions],
            label="Select a multi-hop question",
            value=multihop_questions[0]["question"],
        )

        run_multihop_demo = mo.ui.run_button(label="🔍 Run Multi-Hop QA")

        cell8_out = mo.vstack(
            [
                mo.md("### 🔍 Multi-Hop Question Answering"),
                mo.md("Select a question that requires multiple reasoning steps:"),
                multihop_selector,
                run_multihop_demo,
            ]
        )
    else:
        multihop_questions = None
        multihop_selector = None
        run_multihop_demo = None
        cell8_out = mo.md("")

    output.replace(cell8_out)
    return multihop_questions, multihop_selector, run_multihop_demo


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    multihop_qa,
    multihop_questions,
    multihop_selector,
    output,
    run_multihop_demo,
):
    if available_providers and run_multihop_demo.value and multihop_qa:
        try:
            # Find selected question
            selected_qa = None
            for qa_data in multihop_questions:
                if qa_data["question"] == multihop_selector.value:
                    selected_qa = qa_data
                    break

            if selected_qa:
                # Execute multi-hop reasoning
                qa_result = multihop_qa(
                    question=selected_qa["question"],
                    context_sources=selected_qa["context"],
                )

                cell9_out = mo.vstack(
                    [
                        mo.md("## 🔍 Multi-Hop QA Results"),
                        mo.md(f"**Original Question:** {selected_qa['question']}"),
                        mo.md(f"**Available Context:** {selected_qa['context']}"),
                        mo.md("### 🧩 Hop Analysis"),
                        mo.md(f"**Reasoning Hops Needed:** {qa_result.hop_analysis}"),
                        mo.md("### 🔄 Reasoning Chain"),
                        mo.md(f"**Hop 1 Question:** {qa_result.hop_1_question}"),
                        mo.md(f"**Hop 1 Answer:** {qa_result.hop_1_answer}"),
                        mo.md("---"),
                        mo.md(f"**Hop 2 Question:** {qa_result.hop_2_question}"),
                        mo.md(f"**Hop 2 Answer:** {qa_result.hop_2_answer}"),
                        mo.md("---"),
                        (
                            mo.md(f"**Hop 3 Question:** {qa_result.hop_3_question}")
                            if qa_result.hop_3_question.strip()
                            else mo.md("")
                        ),
                        (
                            mo.md(f"**Hop 3 Answer:** {qa_result.hop_3_answer}")
                            if qa_result.hop_3_answer.strip()
                            else mo.md("")
                        ),
                        mo.md("### 🎯 Final Result"),
                        mo.md(f"**Final Answer:** {qa_result.final_answer}"),
                        mo.md(
                            f"**Complete Reasoning Chain:** {qa_result.reasoning_chain}"
                        ),
                        mo.md(
                            cleandoc(
                                """
                                ### 💡 Multi-Hop Analysis

                                The system successfully:
                                1. **Decomposed** the complex question into sub-questions
                                2. **Sequenced** the reasoning hops logically
                                3. **Integrated** information across multiple steps
                                4. **Synthesized** a comprehensive final answer

                                This demonstrates the power of structured multi-hop reasoning!
                                """
                            )
                        ),
                    ]
                )
            else:
                cell9_out = mo.md("❌ Question not found")

        except Exception as e:
            cell9_out = mo.md(f"❌ **Multi-Hop QA Error:** {str(e)}")
    else:
        cell9_out = mo.md(
            "*Select a question and click 'Run Multi-Hop QA' to see multi-hop reasoning in action*"
        )

    output.replace(cell9_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell10_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 4: Reasoning Visualization & Debugging

                Let's build tools to visualize and debug complex reasoning processes:
                """
            )
        )

        # Reasoning Visualizer
        class ReasoningVisualizer:
            """Visualize and analyze reasoning processes."""

            def __init__(self):
                self.reasoning_traces = []

            def add_trace(self, trace_data):
                """Add a reasoning trace for analysis."""
                self.reasoning_traces.append(
                    {"timestamp": time.time(), "trace": trace_data}
                )

            def visualize_pipeline(self, pipeline_result):
                """Create visualization for pipeline execution."""
                if not pipeline_result.get("success"):
                    return "❌ Pipeline failed - no visualization available"

                visualization = []

                # Problem Analysis
                plan = pipeline_result["plan"]
                visualization.append(
                    cleandoc(
                        f"""
                        **🎯 Problem Analysis**
                        - Type: {plan.problem_type}
                        - Complexity: {plan.complexity_level}
                        - Estimated Steps: {plan.estimated_steps}
                        """
                    )
                )

                # Step Flow
                step_results = pipeline_result["step_results"]
                visualization.append("**🔄 Reasoning Flow**")

                for _, step in enumerate(step_results):
                    confidence_bar = "🟩" * int(float(step.get("confidence", 0.5)) * 10)
                    visualization.append(
                        cleandoc(
                            f"""
                            Step {step['step_number']}: {step['description']}
                            Confidence: {confidence_bar} ({step.get('confidence', 'N/A')})
                            Result: {step['result'][:100]}{'...' if len(step['result']) > 100 else ''}
                            """
                        )
                    )

                # Final Synthesis
                final_result = pipeline_result["final_result"]
                visualization.append(
                    cleandoc(
                        f"""
                        **🎯 Final Synthesis**
                        Answer: {final_result.final_answer[:200]}{'...' if len(final_result.final_answer) > 200 else ''}
                        Confidence: {final_result.confidence_assessment}
                        """
                    )
                )

                return "\n".join(visualization)

            def analyze_reasoning_quality(self, pipeline_result):
                """Analyze the quality of reasoning process."""
                if not pipeline_result.get("success"):
                    return {"error": "Cannot analyze failed pipeline"}

                step_results = pipeline_result["step_results"]

                # Calculate metrics
                confidences = [
                    float(step.get("confidence", 0.5)) for step in step_results
                ]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )
                min_confidence = min(confidences) if confidences else 0

                # Analyze step quality
                step_quality = []
                for step in step_results:
                    quality_score = (
                        len(step.get("reasoning", "")) / 100
                    )  # Simple heuristic
                    step_quality.append(min(quality_score, 1.0))

                avg_quality = (
                    sum(step_quality) / len(step_quality) if step_quality else 0
                )

                return {
                    "total_steps": len(step_results),
                    "avg_confidence": avg_confidence,
                    "min_confidence": min_confidence,
                    "avg_reasoning_quality": avg_quality,
                    "consistency_score": (
                        1.0 - (max(confidences) - min(confidences))
                        if confidences
                        else 0
                    ),
                    "recommendations": self._generate_recommendations(
                        avg_confidence, min_confidence, avg_quality
                    ),
                }

            def _generate_recommendations(self, avg_conf, min_conf, avg_quality):
                """Generate improvement recommendations."""
                recommendations = []

                if avg_conf < 0.7:
                    recommendations.append(
                        "Consider breaking down steps further for higher confidence"
                    )
                if min_conf < 0.5:
                    recommendations.append(
                        "Review low-confidence steps for potential issues"
                    )
                if avg_quality < 0.6:
                    recommendations.append(
                        "Enhance reasoning detail in step descriptions"
                    )

                if not recommendations:
                    recommendations.append("Reasoning quality looks good!")

                return recommendations

        # Create visualizer
        reasoning_visualizer = ReasoningVisualizer()

        cell10_content = mo.md(
            cleandoc(
                """
                ### 📊 Reasoning Visualizer Created

                **Visualization Features:**  
                - **Pipeline Flow** - Visual representation of reasoning steps  
                - **Confidence Tracking** - Monitor confidence across steps  
                - **Quality Analysis** - Assess reasoning quality metrics  
                - **Improvement Recommendations** - Suggestions for optimization  

                The visualizer helps debug and optimize reasoning processes!
                """
            )
        )
    else:
        cell10_desc = mo.md("")
        ReasoningVisualizer = None
        reasoning_visualizer = None
        cell10_content = mo.md("")

    cell10_out = mo.vstack([cell10_desc, cell10_content])
    output.replace(cell10_out)
    return (reasoning_visualizer,)


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    pipeline_result,
    reasoning_visualizer,
):
    if available_providers and reasoning_visualizer and "pipeline_result" in locals():
        try:
            # Visualize the last pipeline execution
            visualization = reasoning_visualizer.visualize_pipeline(pipeline_result)
            quality_analysis = reasoning_visualizer.analyze_reasoning_quality(
                pipeline_result
            )

            cell11_out = mo.vstack(
                [
                    mo.md("## 📊 Reasoning Process Visualization"),
                    mo.md(visualization),
                    mo.md("### 📈 Quality Analysis"),
                    mo.md(
                        cleandoc(
                            f"""
                            **Performance Metrics:**  
                            - Total Steps: {quality_analysis['total_steps']}  
                            - Average Confidence: {quality_analysis['avg_confidence']:.2f}  
                            - Minimum Confidence: {quality_analysis['min_confidence']:.2f}  
                            - Reasoning Quality: {quality_analysis['avg_reasoning_quality']:.2f}  
                            - Consistency Score: {quality_analysis['consistency_score']:.2f}  

                            **Recommendations:**  
                            {'\n'.join([f"- {rec}" for rec in quality_analysis['recommendations']])}  
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell11_out = mo.md(f"❌ **Visualization Error:** {str(e)}")
    else:
        cell11_out = mo.md(
            "*Run a multi-step reasoning pipeline first to see visualization*"
        )

    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell12_out = mo.md(
        cleandoc(
            """
            ## 🎓 Multi-Step Reasoning Module Complete!

            ### 🏆 What You've Mastered

            ✅ **Multi-Step Pipeline Architecture** - Complete reasoning pipeline system  
            ✅ **Problem Decomposition** - Breaking complex problems into manageable steps  
            ✅ **Sequential Execution** - Step-by-step reasoning with context preservation  
            ✅ **Multi-Hop Question Answering** - Complex question decomposition and answering  
            ✅ **Reasoning Visualization** - Debugging and optimization tools  

            ### 🧠 Key Components Built

            1. **Step Planner**
                - Problem type classification
                - Complexity assessment
                - Step decomposition and planning

            2. **Step Executor**
                - Individual step reasoning
                - Context-aware execution
                - Confidence tracking

            3. **Result Synthesizer**
                - Multi-step result integration
                - Final answer generation
                - Alternative approach consideration

            4. **Multi-Hop QA System**
                - Complex question decomposition
                - Sequential sub-question answering
                - Information integration across hops

            5. **Reasoning Visualizer**
                - Pipeline execution visualization
                - Quality analysis and metrics
                - Improvement recommendations

            ### 🎯 Skills Developed

            - **System Architecture** - Designing complex reasoning pipelines
            - **Problem Decomposition** - Breaking down complex problems systematically
            - **Context Management** - Maintaining information flow across steps
            - **Quality Assessment** - Evaluating reasoning process effectiveness
            - **Debugging Techniques** - Identifying and fixing reasoning issues

            ### 🚀 Ready for Advanced Debugging?

            You now understand sophisticated reasoning patterns! Time to master debugging techniques:

            **Next Module:**
            ```bash
            uv run marimo run 02-advanced-modules/debugging_dashboard.py
            ```

            **Coming Up:**
            - Interactive debugging tools for complex agents
            - Step-by-step execution visualization
            - Performance profiling and optimization
            - Advanced error diagnosis and resolution

            ### 💡 Practice Challenges

            Before moving on, try building reasoning systems for:

            1. **Research Analysis**
                - Multi-source information synthesis
                - Evidence evaluation and ranking
                - Conclusion generation with uncertainty

            2. **Strategic Planning**
                - Goal decomposition and prioritization
                - Resource allocation optimization
                - Risk assessment and mitigation

            3. **Technical Troubleshooting**
                - Problem diagnosis methodology
                - Solution evaluation and testing
                - Implementation planning

            4. **Creative Problem Solving**
                - Ideation and brainstorming
                - Concept evaluation and refinement
                - Implementation feasibility analysis

            ### 🏭 Production Considerations

            When deploying multi-step reasoning systems:  
            - **Performance**: Optimize step execution for speed and accuracy  
            - **Reliability**: Implement robust error handling and recovery  
            - **Scalability**: Design for parallel step execution where possible  
            - **Monitoring**: Track reasoning quality and performance metrics  
            - **Maintenance**: Plan for reasoning pattern updates and improvements  

            Master these multi-step reasoning patterns and you can build agents that solve the most complex problems systematically!
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell12_out)
    return


if __name__ == "__main__":
    app.run()
