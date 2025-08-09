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

    from common import DSPyResultViewer, get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return (
        DSPyResultViewer,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # 🎯 Exercise 2: Your First DSPy Signature

            **Objective**: Create and test your first custom DSPy signature.

            ## What You'll Learn

            1. How to design effective DSPy signatures
            2. The importance of clear field descriptions
            3. How to test signatures interactively
            4. Best practices for signature design

            ## Your Mission

            You'll create a signature for a **sentiment analysis** task that can classify text as positive, negative, or neutral.
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
                ## ✅ DSPy Ready

                Using: **{config.default_provider}** with model **{config.default_model}**
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please configure your API keys first by completing Exercise 1.
                """
            )
        )
    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(available_providers, cleandoc, mo, output):

    cell3_out = mo.md(
        cleandoc(
            """
            ## 📝 Step 1: Design Your Signature

            Let's start by understanding what makes a good DSPy signature:

            ### Key Components
            1. **Clear docstring** - Explains what the signature does
            2. **Descriptive field names** - Use meaningful names
            3. **Helpful descriptions** - Guide the LLM's understanding
            4. **Appropriate field types** - Input vs Output fields

            ### Your Task
            Create a signature for sentiment analysis with:  
            - Input: text to analyze  
            - Output: sentiment classification (positive/negative/neutral)  
            - Output: confidence score (0.0 to 1.0)  
            """
        )
        if available_providers
        else ""
    )
    output.replace(cell3_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    # Template for the signature - students should modify this
    signature_template = cleandoc(
        '''
        class SentimentAnalysis(dspy.Signature):  
            \"\"\"TODO: Add a clear description of what this signature does.\"\"\"  

            \\# TODO: Add input field for the text to analyze  
            text = dspy.InputField(desc="TODO: Add description")  

            \\# TODO: Add output field for sentiment classification  
            sentiment = dspy.OutputField(desc="TODO: Add description")  

            \\# TODO: Add output field for confidence score  
            confidence = dspy.OutputField(desc="TODO: Add description")  
        '''
    )

    cell4_out = mo.md(
        cleandoc(
            f"""
            ### 🛠️ Your Signature Template

            Complete the TODO items in this signature:

            ```python
            {signature_template}
            ```

            **Hints:**
            - The docstring should explain the task clearly
            - Input description should specify what kind of text
            - Sentiment output should specify the possible values
            - Confidence should explain the scale (0.0 to 1.0)
            """
        )
        if available_providers
        else ""
    )
    output.replace(cell4_out)
    return


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Interactive signature builder
        docstring = mo.ui.text_area(
            placeholder="Analyze the sentiment of the given text...",
            label="Signature Description (Docstring)",
        )
        input_desc = mo.ui.text_area(
            placeholder="The text to analyze for sentiment...",
            label="Input Field Description",
        )
        sentiment_desc = mo.ui.text_area(
            placeholder="The sentiment classification: positive, negative, or neutral...",
            label="Sentiment Output Description",
        )
        confidence_desc = mo.ui.text_area(
            placeholder="Confidence score from 0.0 to 1.0...",
            label="Confidence Output Description",
        )
    else:
        docstring = None
        input_desc = None
        sentiment_desc = None
        confidence_desc = None
    cell5_out = mo.vstack(
        [
            mo.md("### ✏️ Build Your Signature"),
            mo.md("Fill in the descriptions for each part:"),
            docstring,
            input_desc,
            sentiment_desc,
            confidence_desc,
        ]
    )
    output.replace(cell5_out)
    return confidence_desc, docstring, input_desc, sentiment_desc


@app.cell
def _(
    available_providers,
    cleandoc,
    confidence_desc,
    docstring,
    dspy,
    input_desc,
    mo,
    output,
    sentiment_desc,
):
    if available_providers and docstring is not None and docstring.value:

        if all(
            [
                docstring.value,
                input_desc.value,
                sentiment_desc.value,
                confidence_desc.value,
            ]
        ):  # All fields filled
            # Create the signature class dynamically
            class SentimentAnalysis(dspy.Signature):
                __doc__ = docstring.value

                text = dspy.InputField(desc=input_desc.value)
                sentiment = dspy.OutputField(desc=sentiment_desc.value)
                confidence = dspy.OutputField(desc=confidence_desc.value)

            # Generate code representation
            signature_code = cleandoc(
                f'''class SentimentAnalysis(dspy.Signature):
                    """{docstring.value}"""

                    text = dspy.InputField(desc="{input_desc.value}")
                    sentiment = dspy.OutputField(desc="{sentiment_desc.value}")
                    confidence = dspy.OutputField(desc="{confidence_desc.value}")
                '''
            )
            cell6_out = mo.vstack(
                [
                    mo.md("### 🎉 Your Complete Signature"),
                    mo.md(f"```python\n{signature_code}\n```"),
                    mo.md("Great! Now let's test it."),
                ]
            )
        else:
            SentimentAnalysis = None
            cell6_out = mo.md(
                "*Fill in all the fields above to generate your signature.*"
            )
    else:
        SentimentAnalysis = None
        cell6_out = mo.md("")
    output.replace(cell6_out)
    return (SentimentAnalysis,)


@app.cell
def _(SentimentAnalysis, available_providers, cleandoc, dspy, mo, output):
    if available_providers and SentimentAnalysis:
        # Create predictor
        sentiment_predictor = dspy.Predict(SentimentAnalysis)

        cell7_out = mo.md(
            cleandoc(
                """
                ## 🧪 Step 2: Test Your Signature

                Now let's test your signature with some example texts:  
                """
            )
        )
    else:
        sentiment_predictor = None
        cell7_out = mo.md("")
    output.replace(cell7_out)
    return (sentiment_predictor,)


@app.cell
def _(available_providers, mo, output, sentiment_predictor):
    if available_providers and sentiment_predictor:
        # Test input form
        test_text = mo.ui.text_area(
            placeholder="Enter text to analyze...",
            label="Text to Analyze",
            value="I absolutely love this new restaurant! The food was amazing and the service was excellent.",
        )
        run_test = mo.ui.run_button(label="🔍 Analyze Sentiment")

        cell8_content = mo.vstack([test_text, run_test])
    else:
        test_text = None
        run_test = None
        cell8_content = mo.md("")

    output.replace(cell8_content)
    return run_test, test_text


@app.cell
def _(
    DSPyResultViewer,
    available_providers,
    cleandoc,
    mo,
    output,
    run_test,
    sentiment_predictor,
    test_text,
):
    if (
        available_providers
        and sentiment_predictor
        and run_test is not None
        and run_test.value
    ):
        cell9_test_text = test_text.value

        if cell9_test_text.strip():
            try:
                # Run the prediction
                result = sentiment_predictor(text=cell9_test_text)

                # Display results
                cell9_out = mo.vstack(
                    [
                        mo.md("### 📊 Analysis Results"),
                        mo.md(f"**Input Text:** {test_text}"),
                        mo.md(f"**Sentiment:** {result.sentiment}"),
                        mo.md(f"**Confidence:** {result.confidence}"),
                        mo.md("---"),
                        mo.md("**Full Result Object:**"),
                        DSPyResultViewer(result).render(),
                    ]
                )
            except Exception as e:
                cell9_out = mo.md(
                    cleandoc(
                        f"""
                        ### ❌ Error

                        An error occurred: `{str(e)}`

                        **Common issues:**
                        - API key problems
                        - Network connectivity
                        - Rate limiting

                        Try again or check your configuration.
                        """
                    )
                )
        else:
            cell9_out = mo.md("*Enter some text to analyze.*")
    else:
        cell9_out = mo.md(
            "*Click the 'Analyze Sentiment' button to test your signature.*"
        )
    output.replace(cell9_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell10_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Step 3: Test Multiple Examples

                Try testing your signature with different types of text:
                """
            )
        )

        # Predefined test cases
        test_cases = [
            "I hate waiting in long lines. This is so frustrating!",
            "The weather today is okay, nothing special.",
            "This movie was absolutely fantastic! Best film I've seen all year!",
            "The product works as expected. No complaints.",
            "Terrible customer service. I'm never shopping here again.",
        ]

        test_case_selector = mo.ui.dropdown(
            options=test_cases, label="Select a test case or use the form above"
        )

        cell10_out = mo.vstack(
            [cell10_desc, mo.md("**Quick Test Cases:**"), test_case_selector]
        )
    else:
        test_case_selector = None
        cell10_out = mo.md("")
    output.replace(cell10_out)
    return (test_case_selector,)


@app.cell
def _(
    available_providers,
    cleandoc,
    mo,
    output,
    sentiment_predictor,
    test_case_selector,
):
    if available_providers and sentiment_predictor and test_case_selector.value:
        selected_text = test_case_selector.value

        try:
            quick_result = sentiment_predictor(text=selected_text)

            cell11_out = mo.md(
                cleandoc(
                    f"""
                    ### 🔍 Quick Test Result

                    **Text:** "{selected_text}"
                    **Sentiment:** {quick_result.sentiment}
                    **Confidence:** {quick_result.confidence}
                    """
                )
            )
        except Exception as e:
            cell11_out = mo.md(f"Error with quick test: {str(e)}")
    else:
        cell11_out = mo.md("*Select a test case above for quick testing.*")
    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Exercise completion and reflection
        signature_quality = mo.ui.radio(
            options=["Excellent", "Good", "Needs improvement"],
            label="How would you rate your signature design?",
        )
        what_learned = mo.ui.text_area(
            placeholder="What did you learn about DSPy signatures?",
            label="Key Learnings",
        )
        improvements = mo.ui.text_area(
            placeholder="How could you improve your signature?",
            label="Potential Improvements",
        )
        ready_next = mo.ui.checkbox(label="I'm ready to move on to Module 01")

        cell12_out = mo.vstack(
            [
                mo.md("## 🤔 Reflection & Completion"),
                mo.md("Take a moment to reflect on what you've learned:"),
                signature_quality,
                what_learned,
                improvements,
                ready_next,
            ]
        )
    else:
        signature_quality = None
        what_learned = None
        improvements = None
        ready_next = None
        cell12_out = mo.md("")

    output.replace(cell12_out)
    return (ready_next,)


@app.cell
def _(available_providers, cleandoc, mo, output, ready_next):
    if available_providers and ready_next is not None and ready_next.value:
        cell13_out = mo.md(
            cleandoc(
                """
                ## 🎉 Exercise Complete!

                **Congratulations!** You've successfully:

                ✅ Designed your first custom DSPy signature
                ✅ Understood the importance of clear descriptions
                ✅ Tested your signature with multiple examples
                ✅ Reflected on the learning process

                ### 🚀 What's Next?

                You're now ready for **Module 01: DSPy Foundations**!

                **Key concepts you'll explore:**
                - Advanced signature patterns
                - Different module types (Predict, ChainOfThought, etc.)
                - Signature composition and reuse
                - Interactive development workflows

                **Start Module 01:**
                ```bash
                uv run marimo run 01-foundations/signatures_basics.py
                ```
                """
            )
        )
    else:
        cell13_out = mo.md(
            "*Complete the reflection form above to finish this exercise.*"
        )
    output.replace(cell13_out)
    return


@app.cell
def _(cleandoc, mo, output):
    cell14_out = mo.md(
        cleandoc(
            """
            ## 💡 Signature Design Tips

            **Best Practices for DSPy Signatures:**

            1. **Be Specific**: Vague descriptions lead to unpredictable results
            2. **Include Examples**: Mention expected formats in descriptions
            3. **Use Constraints**: Specify ranges, formats, or valid options
            4. **Think Like an LLM**: What context would help you understand the task?
            5. **Iterate**: Test and refine your signatures based on results

            **Common Mistakes to Avoid:**  
            - Generic field names like "input" or "output"  
            - Missing or unclear descriptions  
            - Too many fields in one signature  
            - Ambiguous output formats  

            **Advanced Techniques (for later):**  
            - Using type hints for structured outputs  
            - Signature composition and inheritance  
            - Dynamic signature generation  
            - Multi-step reasoning patterns  
            """
        )
    )
    output.replace(cell14_out)
    return


if __name__ == "__main__":
    app.run()
