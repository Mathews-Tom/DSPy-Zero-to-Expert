import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    from pathlib import Path
    from typing import Any, Dict, List

    import dspy
    import marimo as mo

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from common import (
        DSPyParameterPanel,
        DSPyResultViewer,
        SignatureBuilder,
        SignatureTester,
        get_config,
        setup_dspy_environment,
    )

    return (
        Any,
        DSPyParameterPanel,
        DSPyResultViewer,
        Dict,
        List,
        Path,
        SignatureBuilder,
        SignatureTester,
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
    return


@app.cell
def __(get_config, mo, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        mo.md(
            f"""
        ## ✅ Environment Ready
        
        **Configuration:**
        - Provider: {config.default_llm_provider}
        - Model: {config.default_model}
        - Cache: {'Enabled' if config.enable_cache else 'Disabled'}
        
        Let's dive into DSPy signatures!
        """
        )
    else:
        mo.md(
            """
        ## ⚠️ Setup Required
        
        Please complete Module 00 setup first:
        ```bash
        uv run 00-setup/setup_environment.py
        ```
        """
        )
    return available_providers, config


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 📝 Signature Design Patterns
        
        DSPy signatures are more than just input/output definitions - they're contracts that guide LLM behavior. Let's explore different design patterns.
        
        ### Pattern 1: Simple Task Signature
        
        For straightforward transformations:
        """
        )
    return


@app.cell
def __(available_providers, dspy, mo):
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

        mo.md(
            f"""
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
    else:
        TextSummarizer = None
    return (TextSummarizer,)


@app.cell
def __(available_providers, dspy, mo):
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

        mo.md(
            f"""
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
    else:
        DocumentAnalyzer = None
    return (DocumentAnalyzer,)


@app.cell
def __(available_providers, dspy, mo):
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

        mo.md(
            f"""
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
    else:
        ContextualQA = None
    return (ContextualQA,)


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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

        signature_selector
    else:
        signature_selector = None
    return (signature_selector,)


@app.cell
def __(
    ContextualQA,
    DocumentAnalyzer,
    TextSummarizer,
    available_providers,
    dspy,
    mo,
    signature_selector,
):
    if available_providers and signature_selector.value:
        selected_sig = signature_selector.value

        # Create predictor based on selection
        if selected_sig == "TextSummarizer":
            current_signature = TextSummarizer
            predictor = dspy.Predict(TextSummarizer)
            sample_input = {
                "text": """
                Artificial Intelligence (AI) has revolutionized numerous industries over the past decade. From healthcare to finance, 
                AI systems are now capable of performing complex tasks that were once thought to be exclusively human. Machine learning 
                algorithms can analyze vast amounts of data to identify patterns, make predictions, and automate decision-making processes. 
                However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and the need 
                for responsible AI development. As we move forward, it's crucial to balance innovation with ethical considerations to 
                ensure AI benefits society as a whole.
                """
            }
        elif selected_sig == "DocumentAnalyzer":
            current_signature = DocumentAnalyzer
            predictor = dspy.Predict(DocumentAnalyzer)
            sample_input = {
                "document": """
                The future of renewable energy looks incredibly promising. Solar panel efficiency has improved dramatically, 
                with new technologies achieving over 25% efficiency rates. Wind power has become the cheapest source of electricity 
                in many regions. Battery storage solutions are solving the intermittency problem, making renewable energy more 
                reliable than ever before. Governments worldwide are investing heavily in green infrastructure, and major 
                corporations are committing to carbon neutrality. This transition represents not just an environmental necessity, 
                but also a tremendous economic opportunity for innovation and job creation.
                """
            }
        else:  # ContextualQA
            current_signature = ContextualQA
            predictor = dspy.Predict(ContextualQA)
            sample_input = {
                "context": """
                Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. 
                It emphasizes code readability with its notable use of significant whitespace. Python supports multiple programming 
                paradigms, including procedural, object-oriented, and functional programming. It has a comprehensive standard library 
                and a large ecosystem of third-party packages. Python is widely used in web development, data science, artificial 
                intelligence, scientific computing, and automation.
                """,
                "question": "When was Python first released and who created it?",
            }

        mo.md(
            f"""
        ### Testing: {selected_sig}
        
        **Sample Input:**
        """
        )
    else:
        current_signature = None
        predictor = None
        sample_input = None
    return current_signature, predictor, sample_input, selected_sig


@app.cell
def __(available_providers, mo, sample_input):
    if available_providers and sample_input:
        # Display sample input in a nice format
        input_display = []
        for key, value in sample_input.items():
            input_display.append(f"**{key.title()}:** {value.strip()}")

        mo.md("\n\n".join(input_display))
    return (input_display,)


@app.cell
def __(available_providers, mo, predictor, sample_input):
    if available_providers and predictor and sample_input:
        # Run prediction button
        run_prediction = mo.ui.button(label="🔍 Run Prediction")

        mo.vstack([mo.md("### Execute Prediction"), run_prediction])
    else:
        run_prediction = None
    return (run_prediction,)


@app.cell
def __(
    DSPyResultViewer,
    available_providers,
    mo,
    predictor,
    run_prediction,
    sample_input,
    selected_sig,
):
    if available_providers and predictor and run_prediction.value and sample_input:
        try:
            # Execute the prediction
            result = predictor(**sample_input)

            mo.vstack(
                [
                    mo.md(f"### 📊 {selected_sig} Results"),
                    DSPyResultViewer(result).render(),
                    mo.md("---"),
                    mo.md("**Analysis:**"),
                    mo.md(
                        "Notice how the signature's field descriptions guide the LLM's response format and content focus."
                    ),
                ]
            )
        except Exception as e:
            mo.md(
                f"""
            ### ❌ Prediction Error
            
            Error: `{str(e)}`
            
            This might be due to API issues or rate limiting. Try again in a moment.
            """
            )
    else:
        mo.md("*Click 'Run Prediction' to see the results.*")
    return (result,)


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


@app.cell
def __(available_providers, dspy, mo):
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

        mo.md(
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
        MathWordProblem = None
        predict_solver = None
        cot_solver = None
        math_problem = None
    return MathWordProblem, cot_solver, math_problem, predict_solver


@app.cell
def __(available_providers, mo, predict_solver, cot_solver, math_problem):
    if available_providers and predict_solver and cot_solver:
        # Button to run comparison
        run_comparison = mo.ui.button(label="🔍 Compare Modules")

        mo.vstack([mo.md("### Run Module Comparison"), run_comparison])
    else:
        run_comparison = None
    return (run_comparison,)


@app.cell
def __(
    available_providers,
    cot_solver,
    math_problem,
    mo,
    predict_solver,
    run_comparison,
):
    if available_providers and run_comparison.value and predict_solver and cot_solver:
        try:
            # Run both predictors
            predict_result = predict_solver(problem=math_problem)
            cot_result = cot_solver(problem=math_problem)

            mo.vstack(
                [
                    mo.md("### 📊 Module Comparison Results"),
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    mo.md("**🎯 Predict Module**"),
                                    mo.md("*Direct approach*"),
                                    mo.md(f"**Answer:** {predict_result.answer}"),
                                    mo.md("*No reasoning shown*"),
                                ]
                            ),
                            mo.vstack(
                                [
                                    mo.md("**🧠 ChainOfThought Module**"),
                                    mo.md("*Step-by-step reasoning*"),
                                    mo.md(
                                        f"**Reasoning:** {getattr(cot_result, 'rationale', 'No reasoning available')}"
                                    ),
                                    mo.md(f"**Answer:** {cot_result.answer}"),
                                ]
                            ),
                        ]
                    ),
                    mo.md(
                        """
                ### 🔍 Analysis
                
                **Predict Module:**
                - Faster execution
                - Direct answer
                - Less interpretable
                
                **ChainOfThought Module:**
                - Shows reasoning process
                - More reliable for complex problems
                - Better for debugging and validation
                """
                    ),
                ]
            )
        except Exception as e:
            mo.md(f"Error running comparison: {str(e)}")
    else:
        mo.md("*Click 'Compare Modules' to see the results.*")
    return cot_result, predict_result


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🔧 Interactive Signature Builder
        
        Now it's your turn! Use the interactive builder to create your own signature:
        """
        )

        # Interactive signature builder
        from common import SignatureBuilder

        signature_builder = SignatureBuilder()

        signature_builder.render()
    else:
        signature_builder = None
    return (signature_builder,)


@app.cell
def __(available_providers, dspy, mo, signature_builder):
    if available_providers and signature_builder:
        # Test the built signature
        if (
            signature_builder.signature_name.value
            and signature_builder.input_fields.value
            and signature_builder.output_fields.value
        ):

            test_custom_button = mo.ui.button(label="🧪 Test Your Signature")

            mo.vstack([mo.md("### Test Your Custom Signature"), test_custom_button])
        else:
            test_custom_button = None
            mo.md("*Complete the signature builder above to test your signature.*")
    else:
        test_custom_button = None
    return (test_custom_button,)


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Signature Composition Patterns
        
        For complex tasks, you can compose multiple signatures together. Here are common patterns:
        
        ### Pattern 1: Sequential Processing
        Break complex tasks into steps:
        """
        )
    return


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        # Sequential processing example
        class ExtractClaims(dspy.Signature):
            """Extract factual claims from a text."""

            text = dspy.InputField(desc="Text to analyze")
            claims = dspy.OutputField(desc="List of factual claims, one per line")

        class VerifyClaim(dspy.Signature):
            """Verify if a claim is factually accurate."""

            claim = dspy.InputField(desc="A factual claim to verify")
            verification = dspy.OutputField(
                desc="true if accurate, false if inaccurate, unknown if uncertain"
            )
            reasoning = dspy.OutputField(desc="Brief explanation of the verification")

        mo.md(
            f"""
        ```python
        # Step 1: Extract claims
        class ExtractClaims(dspy.Signature):
            \"\"\"Extract factual claims from a text.\"\"\"
            text = dspy.InputField(desc="Text to analyze")
            claims = dspy.OutputField(desc="List of factual claims, one per line")
        
        # Step 2: Verify each claim
        class VerifyClaim(dspy.Signature):
            \"\"\"Verify if a claim is factually accurate.\"\"\"
            claim = dspy.InputField(desc="A factual claim to verify")
            verification = dspy.OutputField(desc="true if accurate, false if inaccurate, unknown if uncertain")
            reasoning = dspy.OutputField(desc="Brief explanation of the verification")
        ```
        
        **Benefits:**
        - Modular and reusable
        - Easier to debug and optimize
        - Can be parallelized
        """
        )
    else:
        ExtractClaims = None
        VerifyClaim = None
    return ExtractClaims, VerifyClaim


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 📋 Best Practices Checklist
        
        Use this checklist when designing DSPy signatures:
        """
        )

        best_practices = mo.ui.form(
            {
                "clear_purpose": mo.ui.checkbox(
                    label="✅ Signature has a clear, single purpose"
                ),
                "descriptive_name": mo.ui.checkbox(
                    label="✅ Class name is descriptive and action-oriented"
                ),
                "good_docstring": mo.ui.checkbox(
                    label="✅ Docstring explains what the signature does"
                ),
                "specific_fields": mo.ui.checkbox(
                    label="✅ Field descriptions are specific and helpful"
                ),
                "appropriate_outputs": mo.ui.checkbox(
                    label="✅ Output format is clearly specified"
                ),
                "testable": mo.ui.checkbox(
                    label="✅ Signature is easy to test with examples"
                ),
                "composable": mo.ui.checkbox(
                    label="✅ Can be combined with other signatures if needed"
                ),
            }
        )

        mo.vstack([mo.md("### 📝 Signature Design Checklist"), best_practices])
    else:
        best_practices = None
    return (best_practices,)


@app.cell
def __(available_providers, best_practices, mo):
    if available_providers and best_practices.value:
        checked_items = sum(best_practices.value.values())
        total_items = len(best_practices.value)

        if checked_items == total_items:
            mo.md(
                """
            ## 🎉 Excellent Signature Design!
            
            You've followed all the best practices. Your signatures should be:
            - Clear and purposeful
            - Easy to understand and use
            - Reliable and predictable
            - Maintainable and extensible
            """
            )
        elif checked_items >= total_items * 0.7:
            mo.md(
                f"""
            ## 👍 Good Progress! ({checked_items}/{total_items})
            
            You're following most best practices. Review the unchecked items to improve your signature design.
            """
            )
        else:
            mo.md(
                f"""
            ## 📚 Keep Learning ({checked_items}/{total_items})
            
            Focus on the unchecked practices to improve your signature design skills.
            """
            )
    else:
        mo.md("*Use the checklist above to evaluate your signature design.*")
    return checked_items, total_items


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


if __name__ == "__main__":
    app.run()
