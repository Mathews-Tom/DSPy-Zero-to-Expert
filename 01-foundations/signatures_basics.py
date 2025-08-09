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
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return cleandoc, dspy, get_config, mo, output, setup_dspy_environment


@app.cell
def _(cleandoc, mo, output):
    cell1_out = cleandoc(
        r"""
        # 🏗️ Module 01: DSPy Foundations - Signatures & Basic Modules

        Welcome to the core of DSPy! In this module, you'll master the fundamental building blocks of DSPy programming.

        ## 🎯 Learning Objectives

        By the end of this module, you'll be able to:  
        - Design sophisticated DSPy signatures with best practices  
        - Understand inline vs class-based signature patterns  
        - Use different DSPy modules (Predict, ChainOfThought, etc.)  
        - Compose signatures for complex tasks  
        - Test and validate signatures systematically  

        ## 📚 What We'll Cover

        1. **Signature Design Patterns** - Advanced techniques for effective signatures
        2. **Module Types** - When to use Predict vs ChainOfThought vs others
        3. **Signature Composition** - Building complex tasks from simple parts
        4. **Interactive Testing** - Systematic validation and debugging
        5. **Best Practices** - Professional DSPy development patterns
        """
    )
    output.replace(mo.md(cell1_out))
    return


@app.cell
def _(cleandoc, get_config, mo, output, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        cell2_out = cleandoc(
            f"""
            ## ✅ Environment Ready

            **Configuration:**
            - Provider: {config.default_provider}
            - Model: {config.default_model}
            - Cache: {'Enabled' if config.enable_cache else 'Disabled'}

            Let's dive into DSPy signatures!
            """
        )
    else:
        cell2_out = cleandoc(
            """
            ## ⚠️ Setup Required

            Please complete Module 00 setup first:
            ```bash
            uv run 00-setup/setup_environment.py
            ```
            """
        )

    output.replace(mo.md(cell2_out))
    return (available_providers,)


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell3_out = (
        cleandoc(
            """
            ## 📝 Signature Design Patterns

            DSPy signatures are more than just input/output definitions - they're contracts that guide LLM behavior. Let's explore different design patterns.

            ### Pattern 1: Simple Task Signature

            For straightforward transformations:
            """
        )
        if available_providers
        else ""
    )

    output.replace(mo.md(cell3_out))
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        # Simple task signature example
        class TextSummarizer(dspy.Signature):
            """Summarize the given text into a concise, informative summary."""

            text = dspy.InputField(
                desc="The text to summarize (articles, documents, etc.)"
            )
            summary = dspy.OutputField(
                desc="A concise summary capturing the main points"
            )

        cell4_out = cleandoc(
            """
            ```python
            class TextSummarizer(dspy.Signature):
                \"\"\"Summarize the given text into a concise, informative summary.\"\"\"

                text = dspy.InputField(desc="The text to summarize (articles, documents, etc.)")
                summary = dspy.OutputField(desc="A concise summary capturing the main points")
            ```

            **Key Elements:**  
            - Clear, action-oriented class name  
            - Descriptive docstring explaining the task  
            - Specific field descriptions with examples  
            """
        )
        text_summarizer_signature = TextSummarizer
    else:
        cell4_out = ""
        text_summarizer_signature = None

    output.replace(mo.md(cell4_out))
    return (text_summarizer_signature,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        # Multi-output signature example
        class DocumentAnalyzer(dspy.Signature):
            """Analyze a document and extract key information including topic, sentiment, and key points."""

            document = dspy.InputField(desc="The document text to analyze")
            topic = dspy.OutputField(desc="The main topic or subject of the document")
            sentiment = dspy.OutputField(
                desc="Overall sentiment: positive, negative, or neutral"
            )
            key_points = dspy.OutputField(desc="3-5 key points as a bulleted list")
            complexity = dspy.OutputField(
                desc="Reading complexity: beginner, intermediate, or advanced"
            )

        cell5_out = cleandoc(
            """
            ### Pattern 2: Multi-Output Analysis

            For comprehensive analysis tasks:

            ```python
            class DocumentAnalyzer(dspy.Signature):
                \"\"\"Analyze a document and extract key information including topic, sentiment, and key points.\"\"\"

                document = dspy.InputField(desc="The document text to analyze")
                topic = dspy.OutputField(desc="The main topic or subject of the document")
                sentiment = dspy.OutputField(desc="Overall sentiment: positive, negative, or neutral")
                key_points = dspy.OutputField(desc="3-5 key points as a bulleted list")
                complexity = dspy.OutputField(desc="Reading complexity: beginner, intermediate, or advanced")
            ```

            **Benefits:**  
            - Single LLM call for multiple insights  
            - Consistent analysis across dimensions  
            - Efficient for batch processing  
            """
        )
        document_analyzer_signature = DocumentAnalyzer
    else:
        cell5_out = ""
        document_analyzer_signature = None

    output.replace(mo.md(cell5_out))
    return (document_analyzer_signature,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        # Contextual signature example
        class ContextualQA(dspy.Signature):
            """Answer questions based on provided context, citing specific information from the context."""

            context = dspy.InputField(
                desc="Background information and context for answering the question"
            )
            question = dspy.InputField(desc="The specific question to answer")
            answer = dspy.OutputField(
                desc="A comprehensive answer based on the provided context"
            )
            confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
            citations = dspy.OutputField(
                desc="Specific quotes or references from the context that support the answer"
            )

        cell6_out = cleandoc(
            """
            ### Pattern 3: Contextual Processing

            For tasks requiring external information:

            ```python
            class ContextualQA(dspy.Signature):
                \"\"\"Answer questions based on provided context, citing specific information from the context.\"\"\"

                context = dspy.InputField(desc="Background information and context for answering the question")
                question = dspy.InputField(desc="The specific question to answer")
                answer = dspy.OutputField(desc="A comprehensive answer based on the provided context")
                confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
                citations = dspy.OutputField(desc="Specific quotes or references from the context that support the answer")
            ```

            **Use Cases:**  
            - RAG (Retrieval-Augmented Generation)  
            - Document-based Q&A  
            - Research assistance  
            """
        )
        contextual_qa_signature = ContextualQA
    else:
        cell6_out = ""
        contextual_qa_signature = None

    output.replace(mo.md(cell6_out))
    return (contextual_qa_signature,)


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        cell7_out = mo.md(
            """
            ## 🧪 Interactive Signature Testing

            Let's test these signature patterns with real examples. Choose a signature to experiment with:
            """
        )

        signature_selector = mo.ui.dropdown(
            options=["TextSummarizer", "DocumentAnalyzer", "ContextualQA"],
            label="Select Signature to Test",
            value="TextSummarizer",
        )

        cell7_ui = mo.vstack([cell7_out, signature_selector])
    else:
        cell7_ui = mo.md("")
        signature_selector = None

    output.replace(cell7_ui)
    return (signature_selector,)


@app.cell
def _(
    available_providers,
    cleandoc,
    contextual_qa_signature,
    document_analyzer_signature,
    dspy,
    mo,
    output,
    signature_selector,
    text_summarizer_signature,
):
    if (
        available_providers
        and signature_selector is not None
        and signature_selector.value
    ):
        selected_signature = signature_selector.value

        # Create predictor based on selection
        if selected_signature == "TextSummarizer":
            current_signature = text_summarizer_signature
            predictor = dspy.Predict(text_summarizer_signature)
            sample_input = {
                "text": cleandoc(
                    """
                    Artificial Intelligence (AI) has revolutionized numerous industries over the past decade. From healthcare to finance,
                    AI systems are now capable of performing complex tasks that were once thought to be exclusively human. Machine learning
                    algorithms can analyze vast amounts of data to identify patterns, make predictions, and automate decision-making processes.
                    However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and the need
                    for responsible AI development. As we move forward, it's crucial to balance innovation with ethical considerations to
                    ensure AI benefits society as a whole.
                    """
                )
            }
        elif selected_signature == "DocumentAnalyzer":
            current_signature = document_analyzer_signature
            predictor = dspy.Predict(document_analyzer_signature)
            sample_input = {
                "document": cleandoc(
                    """
                    The future of renewable energy looks incredibly promising. Solar panel efficiency has improved dramatically,
                    with new technologies achieving over 25% efficiency rates. Wind power has become the cheapest source of electricity
                    in many regions. Battery storage solutions are solving the intermittency problem, making renewable energy more
                    reliable than ever before. Governments worldwide are investing heavily in green infrastructure, and major
                    corporations are committing to carbon neutrality. This transition represents not just an environmental necessity,
                    but also a tremendous economic opportunity for innovation and job creation.
                    """
                )
            }
        else:  # ContextualQA
            current_signature = contextual_qa_signature
            predictor = dspy.Predict(contextual_qa_signature)
            sample_input = {
                "context": cleandoc(
                    """
                    Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991.
                    It emphasizes code readability with its notable use of significant whitespace. Python supports multiple programming
                    paradigms, including procedural, object-oriented, and functional programming. It has a comprehensive standard library
                    and a large ecosystem of third-party packages. Python is widely used in web development, data science, artificial
                    intelligence, scientific computing, and automation.
                    """
                ),
                "question": "When was Python first released and who created it?",
            }

        cell8_out = mo.md(
            cleandoc(
                f"""
            ### Testing: {selected_signature}

            **Sample Input:**
            """
            )
        )
    else:
        cell8_out = mo.md("*Select a signature to test above.*")
        current_signature = None
        predictor = None
        sample_input = None
        selected_signature = None

    output.replace(cell8_out)
    return predictor, sample_input, selected_signature


@app.cell
def _(available_providers, mo, output, sample_input):
    if available_providers and sample_input:
        # Display sample input in a nice format
        cell9_in_disp = []
        for key, value in sample_input.items():
            cell9_in_disp.append(f"**{key.title()}:** {value.strip()}")

        cell9_out = mo.md("\n\n".join(cell9_in_disp))
    else:
        cell9_out = mo.md("*No sample input available.*")

    output.replace(cell9_out)
    return


@app.cell
def _(available_providers, mo, output, predictor, sample_input):
    if available_providers and predictor and sample_input:
        # Run prediction button
        run_prediction = mo.ui.run_button(label="🔍 Run Prediction")
        cell10_out = mo.vstack([mo.md("### Execute Prediction"), run_prediction])
    else:
        run_prediction = None
        cell10_out = mo.md("*Configure signature and input first.*")

    output.replace(cell10_out)
    return (run_prediction,)


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    predictor,
    run_prediction,
    sample_input,
    selected_signature,
):
    if (
        available_providers
        and predictor
        and run_prediction is not None
        and run_prediction.value
        and sample_input
    ):
        try:
            # Execute the prediction
            cell11_result = predictor(**sample_input)

            cell11_out = mo.vstack(
                [
                    mo.md(f"### 📊 {selected_signature} Results"),
                    mo.md(f"```json\n{str(cell11_result)}\n```"),
                    mo.md("---"),
                    mo.md("**Analysis:**"),
                    mo.md(
                        "Notice how the signature's field descriptions guide the LLM's response format and content focus."
                    ),
                ]
            )
        except Exception as e:
            cell11_out = mo.md(
                cleandoc(
                    f"""
                    ### ❌ Prediction Error

                    Error: `{str(e)}`

                    This might be due to API issues or rate limiting. Try again in a moment.
                    """
                )
            )
    else:
        cell11_out = mo.md("*Click 'Run Prediction' to see the results.*")

    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell12_out = (
        cleandoc(
            """
        ## 🔄 Module Types: Predict vs ChainOfThought

        DSPy provides different module types for different reasoning patterns. Let's compare them:

        ### Predict Module
        - Direct input → output mapping
        - Fast and efficient
        - Good for straightforward tasks

        ### ChainOfThought Module
        - Adds explicit reasoning steps
        - Better for complex problems
        - More interpretable results

        Let's see the difference in action:
        """
        )
        if available_providers
        else ""
    )

    output.replace(mo.md(cell12_out))
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        # Create a signature for comparison
        class MathWordProblem(dspy.Signature):
            """Solve a math word problem step by step."""

            problem = dspy.InputField(desc="A math word problem to solve")
            answer = dspy.OutputField(desc="The numerical answer")

        # Create both types of predictors
        predict_solver = dspy.Predict(MathWordProblem)
        cot_solver = dspy.ChainOfThought(MathWordProblem)

        # Sample problem
        math_problem = "Sarah has 24 apples. She gives 1/3 of them to her friend and eats 2 apples herself. How many apples does she have left?"

        cell13_out = cleandoc(
            f"""
            ### Comparison Setup

            **Signature:**
            ```python
            class MathWordProblem(dspy.Signature):
                \"\"\"Solve a math word problem step by step.\"\"\"

                problem = dspy.InputField(desc="A math word problem to solve")
                answer = dspy.OutputField(desc="The numerical answer")
            ```

            **Test Problem:** {math_problem}
            """
        )
    else:
        cell13_out = ""
        predict_solver = None
        cot_solver = None
        math_problem = None

    output.replace(mo.md(cell13_out))
    return cot_solver, math_problem, predict_solver


@app.cell
def _(available_providers, cot_solver, mo, output, predict_solver):
    if available_providers and predict_solver and cot_solver:
        # Button to run comparison
        run_comparison = mo.ui.run_button(label="🔍 Compare Modules")
        cell14_content = mo.vstack([mo.md("### Run Module Comparison"), run_comparison])
    else:
        run_comparison = None
        cell14_content = mo.md("*Module comparison not available.*")

    output.replace(cell14_content)
    return (run_comparison,)


@app.cell
def _(
    available_providers,
    cleandoc,
    cot_solver,
    math_problem,
    mo,
    output,
    predict_solver,
    run_comparison,
):
    if (
        available_providers
        and run_comparison is not None
        and run_comparison.value
        and predict_solver
        and cot_solver
    ):
        try:
            # Run both predictors
            predict_result = predict_solver(problem=math_problem)
            cot_result = cot_solver(problem=math_problem)

            cell15_out = mo.vstack(
                [
                    mo.md("### 📊 Module Comparison Results"),
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    mo.md("**🎯 Predict Module**"),
                                    mo.md("*Direct approach*"),
                                    mo.md(f"**Response:** {predict_result.answer}"),
                                    mo.md("*Note: May include reasoning in response*"),
                                ]
                            ),
                            mo.vstack(
                                [
                                    mo.md("**🧠 ChainOfThought Module**"),
                                    mo.md("*Structured reasoning approach*"),
                                    mo.md(
                                        f"**Reasoning:** {getattr(cot_result, 'rationale', 'See reasoning in response')}"
                                    ),
                                    mo.md(f"**Response:** {cot_result.answer}"),
                                ]
                            ),
                        ]
                    ),
                    mo.md(
                        cleandoc(
                            """
                            ### 🔍 Analysis

                            **Key Observations:**
                            - Both modules can include reasoning in their responses
                            - ChainOfThought is designed to separate reasoning from final answers
                            - The actual behavior may vary based on the LLM and prompt structure
                            - In practice, test both approaches to see which works better for your use case

                            **Expected Differences:**
                            - **Predict**: Usually more direct, faster execution
                            - **ChainOfThought**: Designed for explicit reasoning steps, may be more reliable for complex problems
                            """
                        )
                    ),
                ]
            )
        except Exception as e:
            cell15_out = mo.md(f"Error running comparison: {str(e)}")
    else:
        cell15_out = mo.md("*Click 'Compare Modules' to see the results.*")

    output.replace(cell15_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell16_out = (
        cleandoc(
            """
        ## 🚀 Module Summary & Next Steps

        ### 🎯 What You've Learned

        ✅ **Signature Design Patterns**: Simple tasks, multi-output analysis, contextual processing  
        ✅ **Module Types**: When to use Predict vs ChainOfThought  
        ✅ **Interactive Testing**: How to validate signatures systematically  
        ✅ **Composition Patterns**: Breaking complex tasks into manageable parts  
        ✅ **Best Practices**: Professional DSPy development guidelines  

        ### 🔧 Key Skills Developed

        - Designing effective DSPy signatures
        - Choosing appropriate module types
        - Testing and validating signatures
        - Composing complex workflows
        - Following professional best practices

        ### 🎓 Ready for Advanced Topics?

        You now have a solid foundation in DSPy signatures and basic modules. You're ready to explore:

        **Next Module: Advanced DSPy Modules**  
        ```bash
        uv run marimo run 02-advanced-modules/react_implementation.py
        ```

        **Topics Coming Up:**  
        - ReAct (Reasoning + Acting) modules  
        - Tool integration and external APIs  
        - Multi-step reasoning pipelines  
        - Advanced debugging techniques  

        ### 💡 Practice Suggestions

        Before moving on, try creating signatures for:  
        - Email classification (spam/not spam)  
        - Code review and suggestions  
        - Recipe generation from ingredients  
        - Meeting notes summarization  

        The more you practice, the more intuitive signature design becomes!
        """
        )
        if available_providers
        else ""
    )
    output.replace(mo.md(cell16_out))
    return


if __name__ == "__main__":
    app.run()
