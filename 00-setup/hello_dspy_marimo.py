# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import sys
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    from common import (
        DSPyParameterPanel,
        SignatureBuilder,
        get_config,
        setup_dspy_environment,
    )

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    return (
        DSPyParameterPanel,
        SignatureBuilder,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = cleandoc(
        r"""
        # 🎉 Welcome to DSPy Zero-to-Expert!

        This is your first interactive DSPy learning experience using Marimo reactive notebooks.

        ## What You'll Learn Today

        - **DSPy Basics**: Understanding signatures, modules, and predictions
        - **Marimo Integration**: How reactive notebooks enhance DSPy development
        - **Interactive Development**: Real-time parameter tuning and result visualization
        - **Foundation Concepts**: Building blocks for advanced DSPy applications

        Let's start by checking your environment and configuration!
        """
    )
    output.replace(mo.md(cell1_out))
    return


@app.cell
def _(cleandoc, get_config, mo, output):
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        cell2_out = cleandoc(
            f"""
            ## ✅ Environment Status: Ready!

            **Configuration Details:**  
            - **Available LLM Providers**: [{', '.join(available_providers)}]  
            - **Default Provider**: {config.default_provider}  
            - **Default Model**: {config.default_model}  
            - **Cache Enabled**: {config.enable_cache}  
            """
        )
    else:
        cell2_out = cleandoc(
            """
            ## ⚠️ Configuration Required

            It looks like you haven't configured any LLM providers yet. Please:

            1. Edit your `.env` file
            2. Add at least one API key:
                - `OPENAI_API_KEY` for OpenAI
                - `ANTHROPIC_API_KEY` for Anthropic
                - `COHERE_API_KEY` for Cohere
            3. Restart this notebook

            Need help? Run: `uv run 00-setup/setup_environment.py`
            """
        )
    output.replace(mo.md(cell2_out))
    return available_providers, config


@app.cell
def _(
    available_providers,
    cleandoc,
    config,
    mo,
    output,
    setup_dspy_environment,
):
    if available_providers:
        setup_dspy_environment(
            provider=config.default_provider, model=config.default_model
        )
        cell3_out = cleandoc(
            f"""
            ## 🔧 DSPy Configuration

            DSPy has been configured with:  
            - **Provider**: {config.default_provider}  
            - **Model**: {config.default_model}  

            You're now ready to create your first DSPy signature!
            """
        )
    else:
        cell3_out = "Please configure your API keys first (see above)."
    output.replace(mo.md(cell3_out))
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell4_out = ""
    if available_providers:
        cell4_out = cleandoc(
            """
            ## 📝 Your First DSPy Signature

            A **DSPy Signature** defines the input and output structure for your AI tasks.
            Think of it as a contract that specifies what goes in and what comes out.

            Let's create a simple signature for a question-answering task:
            """
        )
    output.replace(mo.md(cell4_out))
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        # Define our first DSPy signature
        class SimpleQA(dspy.Signature):
            """Answer questions with helpful and accurate responses."""

            question = dspy.InputField(desc="The question to answer")
            answer = dspy.OutputField(desc="A helpful and accurate answer")

        cell5_out = cleandoc(
            """
            ### Signature Definition

            ```python
            class SimpleQA(dspy.Signature):
                \"\"\"Answer questions with helpful and accurate responses.\"\"\"

                question = dspy.InputField(desc="The question to answer")
                answer = dspy.OutputField(desc="A helpful and accurate answer")
            ```

            **Key Components:**  
            - **Docstring**: Describes what the signature does  
            - **InputField**: Defines input parameters with descriptions  
            - **OutputField**: Defines expected outputs with descriptions  

            Now let's create a predictor using this signature!
            """
        )
    else:
        cell5_out = ""
        SimpleQA = None
    output.replace(mo.md(cell5_out))
    return (SimpleQA,)


@app.cell
def _(
    DSPyParameterPanel,
    SimpleQA,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
):
    if available_providers and SimpleQA:
        # Create a predictor
        predictor = dspy.Predict(SimpleQA)

        # Create parameter panel for interactive control
        param_panel = DSPyParameterPanel(
            show_model_selection=False,  # Keep it simple for first example
            show_provider_selection=False,
        )

        cell6_out = cleandoc(
            """
            ### Interactive Parameter Control

            Use the controls below to adjust the prediction parameters:
            """
        )
    else:
        cell6_out = ""
        predictor = None
        param_panel = None
    output.replace(mo.md(cell6_out))
    return param_panel, predictor


@app.cell
def _(available_providers, mo, output, param_panel):
    cell7_out = ""
    if available_providers and param_panel:
        # Display the parameter panel
        param_panel.render()
    else:
        cell7_out = ""
    output.replace(mo.md(cell7_out))
    return


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Create input field for questions
        question_input = mo.ui.text_area(
            placeholder="Ask any question you'd like...",
            label="Your Question",
            value="What is the capital of France?",
        )
        cell8_out = mo.vstack([mo.md("### Ask a Question"), question_input])
    else:
        question_input = None
        cell8_out = ""
    output.replace(cell8_out)
    return (question_input,)


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    param_panel,
    predictor,
    question_input,
):
    if available_providers and predictor and question_input.value and param_panel:
        if question_input.value.strip():
            try:
                params = param_panel.get_values()
                result = predictor(question=question_input.value)

                cell9_out = mo.vstack(
                    [
                        mo.md("### 🤖 DSPy Response"),
                        mo.md(f"**Question:** {question_input.value}"),
                        mo.md(f"**Answer:** {result.answer}"),
                    ]
                )
            except Exception as e:
                cell9_out = mo.vstack(
                    [
                        mo.md("### ⚠️ Prediction Error"),
                        mo.md(
                            cleandoc(
                                f"""
                            An error occurred while making the prediction: `{str(e)}`

                            This might be due to:  
                            - API key issues  
                            - Network connectivity  
                            - Rate limiting  

                            Please check your configuration and try again.
                            """
                            )
                        ),
                    ]
                )
        else:
            cell9_out = mo.md("*Enter a question above to see DSPy in action!*")
    else:
        cell9_out = mo.md("*Waiting for input or setup...*")
    output.replace(cell9_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell10_out = ""
    if available_providers:
        cell10_out = cleandoc(
            """
            ## 🔄 Understanding Reactivity

            Notice how the response updates automatically when you:
            - Change your question
            - Adjust the model parameters

            This is the power of **Marimo's reactive programming model** combined with **DSPy**!

            ### What Just Happened?

            1. **Signature Definition**: We defined what our AI task looks like
            2. **Predictor Creation**: We created a `dspy.Predict` module
            3. **Interactive Parameters**: We used Marimo UI to control parameters
            4. **Reactive Execution**: Changes automatically trigger new predictions
            5. **Result Display**: Outputs are formatted and displayed in real-time
            """
        )
    output.replace(mo.md(cell10_out))
    return


@app.cell
def _(SignatureBuilder, available_providers, mo, output):
    if available_providers:
        # Interactive signature builder
        signature_builder = SignatureBuilder()

        cell11_out = mo.vstack(
            [
                mo.md("## 🛠️ Interactive Signature Builder"),
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                signature_builder.signature_name,
                                signature_builder.docstring,
                            ]
                        ),
                        mo.vstack(
                            [
                                mo.md("**Input Fields**"),
                                signature_builder.input_fields,
                                mo.md("**Output Fields**"),
                                signature_builder.output_fields,
                            ]
                        ),
                    ]
                ),
            ]
        )
    else:
        cell11_out = mo.md("")
        signature_builder = None
    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell12_out = ""
    if available_providers:
        cell12_out = cleandoc(
            """
            ## 🧠 DSPy Module Types

            DSPy provides different types of modules for different reasoning patterns:

            ### 1. Predict
            Direct input → output mapping (what we just used)

            ### 2. ChainOfThought
            Adds reasoning steps before the final answer

            ### 3. ReAct
            Combines reasoning with actions/tool use

            Let's compare `Predict` vs `ChainOfThought`:
            """
        )
    output.replace(mo.md(cell12_out))
    return


@app.cell
def _(SimpleQA, available_providers, dspy, mo, output, question_input):
    if available_providers and SimpleQA and question_input.value:
        # Create both types of predictors
        predict_module = dspy.Predict(SimpleQA)
        cot_module = dspy.ChainOfThought(SimpleQA)

        if question_input.value.strip():
            try:
                # Get predictions from both
                predict_result = predict_module(question=question_input.value)
                cot_result = cot_module(question=question_input.value)

                cell13_out = mo.vstack(
                    [
                        mo.md("### 📊 Module Comparison"),
                        mo.hstack(
                            [
                                mo.vstack(
                                    [
                                        mo.md("**Predict Module**"),
                                        mo.md("*Direct Answer:*"),
                                        mo.md(f"{predict_result.answer}"),
                                    ]
                                ),
                                mo.vstack(
                                    [
                                        mo.md("**ChainOfThought Module**"),
                                        mo.md("*Reasoning:*"),
                                        mo.md(
                                            f"{getattr(cot_result, 'rationale', 'No reasoning shown')}"
                                        ),
                                        mo.md("*Answer:*"),
                                        mo.md(f"{cot_result.answer}"),
                                    ]
                                ),
                            ]
                        ),
                    ]
                )
            except Exception as e:
                cell13_out = f"Error comparing modules: {str(e)}"
        else:
            cell13_out = "*Enter a question above to see the module comparison!*"
    else:
        cell13_out = ""
        predict_module = None
        cot_module = None
    output.replace(cell13_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell14_out = ""
    if available_providers:
        cell14_out = cleandoc(
            """
            ## 🎯 Key Takeaways

            Congratulations! You've just experienced the fundamentals of DSPy with Marimo:

            ### ✅ What You've Learned

            1. **DSPy Signatures**: Define the structure of AI tasks
            2. **Predictors**: Execute signatures with language models
            3. **Module Types**: Different reasoning patterns (Predict, ChainOfThought)
            4. **Reactive Development**: Real-time parameter tuning and result updates
            5. **Interactive Components**: UI elements that enhance the development experience

            ### 🚀 What's Next?

            You're now ready to dive deeper into DSPy! Here's your learning path:

            - **Module 01**: DSPy Foundations - Signatures & Basic Modules
            - **Module 02**: Advanced Modules - ReAct, Tools & Multi-Step Reasoning
            - **Module 03**: Retrieval-Augmented Generation (RAG)
            - **Module 04**: Optimization & Teleprompters
            - **And much more!**

            ### 🛠️ Ready to Continue?

            Run the next module with:
            ```bash
            uv run marimo run 01-foundations/signatures_basics.py
            ```
            """
        )
    output.replace(mo.md(cell14_out))
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell15_out = ""
    if available_providers:
        cell15_out = cleandoc(
            """
            ## 💡 Pro Tips for DSPy Development

            As you continue your DSPy journey, keep these tips in mind:

            ### 🎨 Signature Design
            - Use clear, descriptive field names
            - Write helpful docstrings and field descriptions
            - Keep signatures focused on single responsibilities

            ### ⚡ Interactive Development
            - Use Marimo's reactivity to experiment quickly
            - Leverage parameter panels for systematic testing
            - Compare different module types side-by-side

            ### 📊 Evaluation & Optimization
            - Always evaluate your DSPy systems
            - Use metrics to guide improvements
            - Leverage DSPy's optimization capabilities

            ### 🔧 Best Practices
            - Start simple, then add complexity
            - Use version control for your signatures
            - Document your reasoning and design decisions

            Happy learning! 🎉
            """
        )
    output.replace(mo.md(cell15_out))
    return


if __name__ == "__main__":
    app.run()
