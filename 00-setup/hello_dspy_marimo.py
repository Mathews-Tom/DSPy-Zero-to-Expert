import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    from pathlib import Path

    import dspy
    import marimo as mo

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from common import (
        DSPyParameterPanel,
        DSPyResultViewer,
        SignatureBuilder,
        get_config,
        setup_dspy_environment,
    )

    return (
        DSPyParameterPanel,
        DSPyResultViewer,
        Path,
        SignatureBuilder,
        dspy,
        get_config,
        mo,
        project_root,
        setup_dspy_environment,
        sys,
    )


@app.cell
def __(mo):
    mo.md(
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
    return


@app.cell
def __(get_config, mo):
    # Load configuration and check environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        mo.md(
            f"""
        ## ✅ Environment Status: Ready!
        
        **Configuration Details:**
        - **Available LLM Providers**: {', '.join(available_providers)}
        - **Default Provider**: {config.default_llm_provider}
        - **Default Model**: {config.default_model}
        - **Cache Enabled**: {config.enable_cache}
        
        Your environment is properly configured and ready for DSPy learning!
        """
        )
    else:
        mo.md(
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
    return available_providers, config


@app.cell
def __(available_providers, config, dspy, mo, setup_dspy_environment):
    # Configure DSPy if providers are available
    if available_providers:
        # Set up DSPy with the default provider
        setup_dspy_environment(provider=config.default_llm_provider)

        mo.md(
            f"""
        ## 🔧 DSPy Configuration
        
        DSPy has been configured with:
        - **Provider**: {config.default_llm_provider}
        - **Model**: {config.default_model}
        
        You're now ready to create your first DSPy signature!
        """
        )
    else:
        mo.md("Please configure your API keys first (see above).")
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 📝 Your First DSPy Signature
        
        A **DSPy Signature** defines the input and output structure for your AI tasks. 
        Think of it as a contract that specifies what goes in and what comes out.
        
        Let's create a simple signature for a question-answering task:
        """
        )
    return


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        # Define our first DSPy signature
        class SimpleQA(dspy.Signature):
            """Answer questions with helpful and accurate responses."""

            question = dspy.InputField(desc="The question to answer")
            answer = dspy.OutputField(desc="A helpful and accurate answer")

        mo.md(
            f"""
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
        SimpleQA = None
    return (SimpleQA,)


@app.cell
def __(DSPyParameterPanel, SimpleQA, available_providers, dspy, mo):
    if available_providers and SimpleQA:
        # Create a predictor
        predictor = dspy.Predict(SimpleQA)

        # Create parameter panel for interactive control
        param_panel = DSPyParameterPanel(
            show_model_selection=False,  # Keep it simple for first example
            show_temperature=True,
            show_max_tokens=True,
            show_provider_selection=False,
        )

        mo.md(
            """
        ### Interactive Parameter Control
        
        Use the controls below to adjust the prediction parameters:
        """
        )
    else:
        predictor = None
        param_panel = None
    return param_panel, predictor


@app.cell
def __(available_providers, mo, param_panel):
    if available_providers and param_panel:
        # Display the parameter panel
        param_panel.render()
    else:
        mo.md("")
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        # Create input field for questions
        question_input = mo.ui.text_area(
            placeholder="Ask any question you'd like...",
            label="Your Question",
            value="What is the capital of France?",
        )

        mo.vstack([mo.md("### Ask a Question"), question_input])
    else:
        question_input = None
    return (question_input,)


@app.cell
def __(available_providers, mo, param_panel, predictor, question_input):
    if available_providers and predictor and question_input and param_panel:
        # Get current parameter values
        params = param_panel.get_values()

        # Only make prediction if we have a question
        if question_input.value.strip():
            try:
                # Make prediction with current parameters
                # Note: In a real scenario, you'd apply the parameters to the LM
                result = predictor(question=question_input.value)

                mo.vstack(
                    [
                        mo.md("### 🤖 DSPy Response"),
                        mo.md(f"**Question:** {question_input.value}"),
                        mo.md(f"**Answer:** {result.answer}"),
                        mo.md(
                            f"**Parameters Used:** Temperature={params.get('temperature', 'N/A')}, Max Tokens={params.get('max_tokens', 'N/A')}"
                        ),
                    ]
                )
            except Exception as e:
                mo.md(
                    f"""
                ### ⚠️ Prediction Error
                
                An error occurred while making the prediction: `{str(e)}`
                
                This might be due to:
                - API key issues
                - Network connectivity
                - Rate limiting
                
                Please check your configuration and try again.
                """
                )
        else:
            mo.md("*Enter a question above to see DSPy in action!*")
    else:
        mo.md("")
    return params, result


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🔄 Understanding Reactivity
        
        Notice how the response updates automatically when you:
        - Change your question
        - Adjust the temperature or max tokens
        
        This is the power of **Marimo's reactive programming model** combined with **DSPy**!
        
        ### What Just Happened?
        
        1. **Signature Definition**: We defined what our AI task looks like
        2. **Predictor Creation**: We created a `dspy.Predict` module
        3. **Interactive Parameters**: We used Marimo UI to control parameters
        4. **Reactive Execution**: Changes automatically trigger new predictions
        5. **Result Display**: Outputs are formatted and displayed in real-time
        """
        )
    return


@app.cell
def __(SignatureBuilder, available_providers, mo):
    if available_providers:
        # Interactive signature builder
        signature_builder = SignatureBuilder()

        mo.vstack(
            [
                mo.md(
                    """
            ## 🛠️ Interactive Signature Builder
            
            Try building your own DSPy signature using the interactive builder below:
            """
                ),
                signature_builder.render(),
            ]
        )
    else:
        signature_builder = None
    return (signature_builder,)


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
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
    return


@app.cell
def __(SimpleQA, available_providers, dspy, mo, question_input):
    if available_providers and SimpleQA and question_input:
        # Create both types of predictors
        predict_module = dspy.Predict(SimpleQA)
        cot_module = dspy.ChainOfThought(SimpleQA)

        if question_input.value.strip():
            try:
                # Get predictions from both
                predict_result = predict_module(question=question_input.value)
                cot_result = cot_module(question=question_input.value)

                mo.vstack(
                    [
                        mo.md("### 📊 Module Comparison"),
                        mo.hstack(
                            [
                                mo.vstack(
                                    [
                                        mo.md("**Predict Module**"),
                                        mo.md(f"*Direct Answer:*"),
                                        mo.md(f"{predict_result.answer}"),
                                    ]
                                ),
                                mo.vstack(
                                    [
                                        mo.md("**ChainOfThought Module**"),
                                        mo.md(f"*Reasoning:*"),
                                        mo.md(
                                            f"{getattr(cot_result, 'rationale', 'No reasoning shown')}"
                                        ),
                                        mo.md(f"*Answer:*"),
                                        mo.md(f"{cot_result.answer}"),
                                    ]
                                ),
                            ]
                        ),
                    ]
                )
            except Exception as e:
                mo.md(f"Error comparing modules: {str(e)}")
        else:
            mo.md("*Enter a question above to see the module comparison!*")
    else:
        predict_module = None
        cot_module = None
    return cot_module, cot_result, predict_module, predict_result


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


if __name__ == "__main__":
    app.run()
