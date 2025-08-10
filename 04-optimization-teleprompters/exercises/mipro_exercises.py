# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo mipro

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
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
        dspy,
        get_config,
        mo,
        output,
        random,
        setup_dspy_environment,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🎯 MIPRO Optimization Exercises

            **Practice advanced multi-stage optimization** with MIPRO techniques.

            ## 📚 Exercise Overview

            These exercises will help you master:  
            - Multi-stage instruction optimization  
            - Prompt optimization with temperature control  
            - Advanced evaluation metrics for complex tasks  
            - Strategy comparison and effectiveness measurement  

            Complete each exercise to build your MIPRO expertise!
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

                Ready to start MIPRO exercises!
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
                ## 🎯 Exercise 1: Multi-Stage Reasoning Module

                **Task:** Build a complex reasoning module that benefits from MIPRO's multi-stage optimization.

                **Requirements:**  
                1. Create a `ComplexReasoningSignature` with multiple output fields  
                2. Implement a `MultiStageReasoningModule` with chain-of-thought  
                3. Create training examples for complex reasoning tasks  
                4. Design evaluation metrics that assess reasoning quality  

                **Your Implementation:**
                """
            )
        )

        # Exercise 1 Template
        exercise1_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 1: Multi-Stage Reasoning Module

                # TODO: Create ComplexReasoningSignature
                class ComplexReasoningSignature(dspy.Signature):
                    \"\"\"Perform complex multi-step reasoning with detailed analysis.\"\"\"

                    # TODO: Add input and output fields for complex reasoning
                    pass

                # TODO: Create MultiStageReasoningModule
                class MultiStageReasoningModule(dspy.Module):
                    \"\"\"Multi-stage reasoning module optimized for MIPRO.\"\"\"

                    def __init__(self):
                        super().__init__()
                        # TODO: Initialize the reasoning module
                        pass

                    def forward(self, problem, context=""):
                        # TODO: Implement multi-stage reasoning
                        pass

                # TODO: Create training examples for complex reasoning
                complex_reasoning_examples = [
                    # Add complex reasoning examples here
                ]

                # TODO: Create evaluation metric for reasoning quality
                def reasoning_quality_metric(example, pred, trace=None):
                    \"\"\"Evaluate the quality of multi-stage reasoning.\"\"\"
                    # TODO: Implement reasoning evaluation logic
                    pass

                # Test your implementation
                if __name__ == "__main__":
                    module = MultiStageReasoningModule()
                    result = module(problem="Analyze the economic impact of remote work", context="Post-pandemic workplace trends")
                    print(f"Reasoning: {result}")
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
                ## 🔧 Exercise 2: Instruction Candidate Generation

                **Task:** Implement a system for generating and evaluating instruction candidates for MIPRO.

                **Requirements:**  
                1. Create an `InstructionGenerator` class  
                2. Implement methods for generating instruction variations  
                3. Build an evaluation system for instruction effectiveness  
                4. Create a selection mechanism for the best instructions  

                **Your Implementation:**
                """
            )
        )

        # Exercise 2 Template
        exercise2_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 2: Instruction Candidate Generation

                class InstructionGenerator:
                    \"\"\"Generate and evaluate instruction candidates for MIPRO optimization.\"\"\"

                    def __init__(self, base_instructions=None):
                        # TODO: Initialize with base instructions
                        pass

                    def generate_candidates(self, num_candidates=10, task_type="reasoning"):
                        \"\"\"Generate instruction candidates for the given task type.\"\"\"
                        # TODO: Implement instruction generation logic
                        pass

                    def evaluate_instruction(self, instruction, test_examples, module_class):
                        \"\"\"Evaluate an instruction's effectiveness.\"\"\"
                        # TODO: Implement instruction evaluation
                        pass

                    def select_best_instruction(self, candidates, test_examples, module_class):
                        \"\"\"Select the best instruction from candidates.\"\"\"
                        # TODO: Implement selection logic
                        pass

                # TODO: Create test examples for instruction evaluation
                instruction_test_examples = [
                    # Add test examples here
                ]

                # TODO: Test instruction generation
                def test_instruction_generation():
                    \"\"\"Test the instruction generation system.\"\"\"
                    generator = InstructionGenerator()

                    # TODO: Generate candidates
                    # TODO: Evaluate candidates
                    # TODO: Select best instruction

                    print("Instruction generation test complete!")

                if __name__ == "__main__":
                    test_instruction_generation()
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
                ## ⚡ Exercise 3: Temperature-Based Prompt Optimization

                **Task:** Implement prompt optimization with temperature control for creative tasks.

                **Requirements:**  
                1. Create a `PromptOptimizer` class with temperature control  
                2. Implement prompt variation generation based on temperature  
                3. Build evaluation metrics for creative output quality  
                4. Create a temperature scheduling system  

                **Your Implementation:**
                """
            )
        )

        # Exercise 3 Template
        exercise3_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 3: Temperature-Based Prompt Optimization

                class PromptOptimizer:
                    \"\"\"Optimize prompts with temperature-based variation generation.\"\"\"

                    def __init__(self, base_prompts=None):
                        # TODO: Initialize with base prompts
                        pass

                    def generate_prompt_variations(self, base_prompt, temperature=1.0, num_variations=5):
                        \"\"\"Generate prompt variations based on temperature.\"\"\"
                        # TODO: Implement temperature-based prompt generation
                        pass

                    def evaluate_prompt_creativity(self, prompt, test_examples, module_class):
                        \"\"\"Evaluate a prompt's effectiveness for creative tasks.\"\"\"
                        # TODO: Implement creativity evaluation
                        pass

                    def optimize_with_temperature_schedule(self, initial_temp=2.0, final_temp=0.5, steps=5):
                        \"\"\"Optimize prompts using a temperature schedule.\"\"\"
                        # TODO: Implement temperature scheduling
                        pass

                # TODO: Create creative task examples
                creative_task_examples = [
                    # Add creative writing examples here
                ]

                # TODO: Create creativity evaluation metric
                def creativity_metric(example, pred, trace=None):
                    \"\"\"Evaluate creativity and quality of generated content.\"\"\"
                    # TODO: Implement creativity assessment
                    pass

                # TODO: Test temperature-based optimization
                def test_temperature_optimization():
                    \"\"\"Test the temperature-based prompt optimization.\"\"\"
                    optimizer = PromptOptimizer()

                    # TODO: Test different temperatures
                    # TODO: Evaluate results
                    # TODO: Find optimal temperature

                    print("Temperature optimization test complete!")

                if __name__ == "__main__":
                    test_temperature_optimization()
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
                ## 📊 Exercise 4: MIPRO vs BootstrapFewShot Comparison

                **Task:** Build a comprehensive comparison system between MIPRO and BootstrapFewShot.

                **Requirements:**  
                1. Create a `OptimizationComparator` class  
                2. Implement side-by-side optimization runs  
                3. Build detailed performance analysis  
                4. Generate insights about when to use each method  

                **Your Implementation:**
                """
            )
        )

        # Exercise 4 Template
        exercise4_code = mo.ui.code_editor(
            value=cleandoc(
                """# Exercise 4: MIPRO vs BootstrapFewShot Comparison

                class OptimizationComparator:
                    \"\"\"Compare MIPRO and BootstrapFewShot optimization strategies.\"\"\"

                    def __init__(self):
                        self.comparison_results = []

                    def run_bootstrap_optimization(self, module_class, training_data, metric):
                        \"\"\"Run BootstrapFewShot optimization.\"\"\"
                        # TODO: Implement BootstrapFewShot optimization
                        pass

                    def run_mipro_optimization(self, module_class, training_data, metric):
                        \"\"\"Run MIPRO optimization.\"\"\"
                        # TODO: Implement MIPRO optimization
                        pass

                    def compare_strategies(self, module_class, training_data, validation_data, metric):
                        \"\"\"Compare both optimization strategies side-by-side.\"\"\"
                        # TODO: Run both optimizations
                        # TODO: Compare results
                        # TODO: Generate analysis
                        pass

                    def generate_recommendations(self, task_characteristics):
                        \"\"\"Generate recommendations for which strategy to use.\"\"\"
                        # TODO: Analyze task characteristics
                        # TODO: Provide strategy recommendations
                        pass

                # TODO: Create comparison test data
                comparison_test_data = [
                    # Add test examples for comparison
                ]

                # TODO: Test the comparison system
                def test_optimization_comparison():
                    \"\"\"Test the optimization comparison system.\"\"\"
                    comparator = OptimizationComparator()

                    # TODO: Run comparisons
                    # TODO: Analyze results
                    # TODO: Generate recommendations

                    print("Optimization comparison test complete!")

                if __name__ == "__main__":
                    test_optimization_comparison()
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

                **Exercise 1: Multi-Stage Reasoning Module**  
                - [ ] Created ComplexReasoningSignature with multiple outputs  
                - [ ] Implemented MultiStageReasoningModule with chain-of-thought  
                - [ ] Added complex reasoning training examples  
                - [ ] Designed reasoning quality evaluation metric  

                **Exercise 2: Instruction Candidate Generation**  
                - [ ] Built InstructionGenerator class  
                - [ ] Implemented instruction variation generation  
                - [ ] Created instruction effectiveness evaluation  
                - [ ] Built selection mechanism for best instructions  

                **Exercise 3: Temperature-Based Prompt Optimization**  
                - [ ] Created PromptOptimizer with temperature control  
                - [ ] Implemented temperature-based prompt variations  
                - [ ] Built creativity evaluation metrics  
                - [ ] Created temperature scheduling system  

                **Exercise 4: Strategy Comparison**  
                - [ ] Built OptimizationComparator class  
                - [ ] Implemented side-by-side optimization runs  
                - [ ] Created detailed performance analysis  
                - [ ] Generated strategy selection insights  

                ### 🚀 Next Steps

                After completing these exercises:  
                1. **Review Solutions** - Check the solutions directory for reference implementations  
                2. **Experiment with Real Tasks** - Apply MIPRO to your own complex reasoning tasks  
                3. **Combine Techniques** - Try using MIPRO and BootstrapFewShot together  
                4. **Build Production Systems** - Scale your optimized models for real applications  

                ### 💡 MIPRO Best Practices

                - **Use for Complex Tasks** - MIPRO excels with multi-step reasoning and creative tasks  
                - **Instruction Quality Matters** - Invest time in generating good instruction candidates  
                - **Temperature Tuning** - Experiment with different temperatures for different task types  
                - **Multi-Stage Benefits** - The separate optimization stages provide fine-grained control  

                Excellent work mastering MIPRO optimization! 🎉
                """
            )
        )
    else:
        cell7_out = mo.md("")

    output.replace(cell7_out)
    return


if __name__ == "__main__":
    app.run()
