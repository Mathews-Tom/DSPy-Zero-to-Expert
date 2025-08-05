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
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from common import DSPyResultViewer, get_config, setup_dspy_environment

    return (
        DSPyResultViewer,
        Path,
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
        ## ✅ DSPy Ready
        
        Using: **{config.default_llm_provider}** with model **{config.default_model}**
        """
        )
    else:
        mo.md(
            """
        ## ⚠️ Setup Required
        
        Please configure your API keys first by completing Exercise 1.
        """
        )
    return available_providers, config


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        # Template for the signature - students should modify this
        signature_template = '''class SentimentAnalysis(dspy.Signature):
    """TODO: Add a clear description of what this signature does."""
    
    # TODO: Add input field for the text to analyze
    text = dspy.InputField(desc="TODO: Add description")
    
    # TODO: Add output field for sentiment classification
    sentiment = dspy.OutputField(desc="TODO: Add description")
    
    # TODO: Add output field for confidence score
    confidence = dspy.OutputField(desc="TODO: Add description")'''

        mo.md(
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
    return (signature_template,)


@app.cell
def __(available_providers, mo):
    if available_providers:
        # Interactive signature builder
        signature_form = mo.ui.form(
            {
                "docstring": mo.ui.text_area(
                    placeholder="Analyze the sentiment of the given text...",
                    label="Signature Description (Docstring)",
                ),
                "input_desc": mo.ui.text_area(
                    placeholder="The text to analyze for sentiment...",
                    label="Input Field Description",
                ),
                "sentiment_desc": mo.ui.text_area(
                    placeholder="The sentiment classification: positive, negative, or neutral...",
                    label="Sentiment Output Description",
                ),
                "confidence_desc": mo.ui.text_area(
                    placeholder="Confidence score from 0.0 to 1.0...",
                    label="Confidence Output Description",
                ),
            }
        )

        mo.vstack(
            [
                mo.md("### ✏️ Build Your Signature"),
                mo.md("Fill in the descriptions for each part:"),
                signature_form,
            ]
        )
    else:
        signature_form = None
    return (signature_form,)


@app.cell
def __(available_providers, dspy, mo, signature_form):
    if available_providers and signature_form.value:
        # Generate the complete signature based on user input
        form_data = signature_form.value

        if all(form_data.values()):  # All fields filled
            # Create the signature class dynamically
            class SentimentAnalysis(dspy.Signature):
                __doc__ = form_data["docstring"]

                text = dspy.InputField(desc=form_data["input_desc"])
                sentiment = dspy.OutputField(desc=form_data["sentiment_desc"])
                confidence = dspy.OutputField(desc=form_data["confidence_desc"])

            # Generate code representation
            signature_code = f'''class SentimentAnalysis(dspy.Signature):
    """{form_data["docstring"]}"""
    
    text = dspy.InputField(desc="{form_data["input_desc"]}")
    sentiment = dspy.OutputField(desc="{form_data["sentiment_desc"]}")
    confidence = dspy.OutputField(desc="{form_data["confidence_desc"]}")'''

            mo.vstack(
                [
                    mo.md("### 🎉 Your Complete Signature"),
                    mo.md(f"```python\n{signature_code}\n```"),
                    mo.md("Great! Now let's test it."),
                ]
            )
        else:
            SentimentAnalysis = None
            mo.md("*Fill in all the fields above to generate your signature.*")
    else:
        SentimentAnalysis = None
    return SentimentAnalysis, form_data, signature_code


@app.cell
def __(SentimentAnalysis, available_providers, dspy, mo):
    if available_providers and SentimentAnalysis:
        # Create predictor
        sentiment_predictor = dspy.Predict(SentimentAnalysis)

        mo.md(
            """
        ## 🧪 Step 2: Test Your Signature
        
        Now let's test your signature with some example texts:
        """
        )
    else:
        sentiment_predictor = None
    return (sentiment_predictor,)


@app.cell
def __(available_providers, mo, sentiment_predictor):
    if available_providers and sentiment_predictor:
        # Test input form
        test_form = mo.ui.form(
            {
                "test_text": mo.ui.text_area(
                    placeholder="Enter text to analyze...",
                    label="Text to Analyze",
                    value="I absolutely love this new restaurant! The food was amazing and the service was excellent.",
                ),
                "run_test": mo.ui.button(label="🔍 Analyze Sentiment"),
            }
        )

        test_form
    else:
        test_form = None
    return (test_form,)


@app.cell
def __(DSPyResultViewer, available_providers, mo, sentiment_predictor, test_form):
    if (
        available_providers
        and sentiment_predictor
        and test_form.value
        and test_form.value["run_test"]
    ):
        test_text = test_form.value["test_text"]

        if test_text.strip():
            try:
                # Run the prediction
                result = sentiment_predictor(text=test_text)

                # Display results
                mo.vstack(
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
                mo.md(
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
        else:
            mo.md("*Enter some text to analyze.*")
    else:
        mo.md("*Click the 'Analyze Sentiment' button to test your signature.*")
    return result, test_text


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Step 3: Test Multiple Examples
        
        Try testing your signature with different types of text:
        """
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

        mo.vstack([mo.md("**Quick Test Cases:**"), test_case_selector])
    else:
        test_case_selector = None
    return test_case_selector, test_cases


@app.cell
def __(available_providers, mo, sentiment_predictor, test_case_selector):
    if available_providers and sentiment_predictor and test_case_selector.value:
        selected_text = test_case_selector.value

        try:
            quick_result = sentiment_predictor(text=selected_text)

            mo.md(
                f"""
            ### 🔍 Quick Test Result
            
            **Text:** "{selected_text}"
            **Sentiment:** {quick_result.sentiment}
            **Confidence:** {quick_result.confidence}
            """
            )
        except Exception as e:
            mo.md(f"Error with quick test: {str(e)}")
    else:
        mo.md("*Select a test case above for quick testing.*")
    return quick_result, selected_text


@app.cell
def __(available_providers, mo):
    if available_providers:
        # Exercise completion and reflection
        reflection_form = mo.ui.form(
            {
                "signature_quality": mo.ui.radio(
                    options=["Excellent", "Good", "Needs improvement"],
                    label="How would you rate your signature design?",
                ),
                "what_learned": mo.ui.text_area(
                    placeholder="What did you learn about DSPy signatures?",
                    label="Key Learnings",
                ),
                "improvements": mo.ui.text_area(
                    placeholder="How could you improve your signature?",
                    label="Potential Improvements",
                ),
                "ready_next": mo.ui.checkbox(label="I'm ready to move on to Module 01"),
            }
        )

        mo.vstack(
            [
                mo.md("## 🤔 Reflection & Completion"),
                mo.md("Take a moment to reflect on what you've learned:"),
                reflection_form,
            ]
        )
    else:
        reflection_form = None
    return (reflection_form,)


@app.cell
def __(available_providers, mo, reflection_form):
    if available_providers and reflection_form.value:
        reflection = reflection_form.value

        if reflection["ready_next"]:
            mo.md(
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
        else:
            mo.md(
                """
            ### 📝 Your Reflection
            
            Thanks for sharing your thoughts! Take some time to experiment more with your signature if needed.
            
            **Remember:** Good signature design is crucial for DSPy success. The clearer your descriptions, the better your results will be.
            """
            )
    else:
        mo.md("*Complete the reflection form above to finish this exercise.*")
    return reflection


@app.cell
def __(mo):
    mo.md(
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
    return


if __name__ == "__main__":
    app.run()
