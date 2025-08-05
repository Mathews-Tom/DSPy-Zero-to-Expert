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

    from common import get_config, setup_dspy_environment

    return Path, dspy, get_config, mo, project_root, setup_dspy_environment, sys


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🎯 Exercise 1: Signature Design Fundamentals
        
        **Objective:** Master the art of creating effective DSPy signatures through hands-on practice.
        
        ## 📚 What You'll Learn
        - Design signatures for different task types
        - Write clear and effective field descriptions
        - Understand the impact of signature design on LLM performance
        - Practice iterative signature improvement
        
        ## 🎮 Exercise Format
        This exercise includes:
        - **4 Progressive Challenges** - From basic to advanced
        - **Interactive Testing** - Test your signatures immediately
        - **Automated Validation** - Get instant feedback
        - **Solution Hints** - Guidance when you need it
        
        Let's start building better signatures!
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
        
        Ready to start the exercises!
        """
        )
    else:
        mo.md(
            """
        ## ⚠️ Setup Required
        
        Please complete Module 00 setup first.
        """
        )
    return available_providers, config


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 1: Basic Classification Signature
        
        **Task:** Create a signature for classifying customer support tickets.
        
        **Requirements:**
        - Input: Customer message and ticket category options
        - Output: Selected category and confidence score
        - Categories: "technical", "billing", "general", "complaint"
        
        **Success Criteria:**
        - Clear field descriptions that guide the LLM
        - Appropriate field names
        - Proper confidence score format specification
        """
        )

        # Challenge 1 form
        challenge1_form = mo.ui.form(
            {
                "signature_code": mo.ui.text_area(
                    placeholder="""class TicketClassifier(dspy.Signature):
    \"\"\"Your docstring here...\"\"\"
    # Your fields here...""",
                    label="Your Signature Code",
                    rows=8,
                ),
                "submit": mo.ui.button(label="🧪 Test Signature"),
            }
        )

        challenge1_form
    else:
        challenge1_form = None
    return (challenge1_form,)


@app.cell
def __(available_providers, challenge1_form, dspy, mo):
    if (
        available_providers
        and challenge1_form.value
        and challenge1_form.value["submit"]
    ):
        signature_code = challenge1_form.value["signature_code"]

        if signature_code.strip():
            try:
                # Execute the signature code
                exec_globals = {"dspy": dspy}
                exec(signature_code, exec_globals)

                # Find the signature class
                signature_class = None
                for name, obj in exec_globals.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, dspy.Signature)
                        and obj != dspy.Signature
                    ):
                        signature_class = obj
                        break

                if signature_class:
                    # Test the signature
                    predictor = dspy.Predict(signature_class)

                    # Test cases
                    test_cases = [
                        {
                            "message": "My internet connection keeps dropping every few minutes. I've tried restarting my router but it doesn't help.",
                            "categories": "technical, billing, general, complaint",
                        },
                        {
                            "message": "I was charged twice for my monthly subscription. Please refund the duplicate charge.",
                            "categories": "technical, billing, general, complaint",
                        },
                    ]

                    results = []
                    for i, test_case in enumerate(test_cases):
                        try:
                            # Get field names dynamically
                            input_fields = [
                                name
                                for name, field in signature_class.__annotations__.items()
                                if isinstance(
                                    getattr(signature_class, name, None),
                                    dspy.InputField,
                                )
                            ]

                            # Create input dict based on available fields
                            test_input = {}
                            for field_name in input_fields:
                                if "message" in field_name.lower():
                                    test_input[field_name] = test_case["message"]
                                elif "categor" in field_name.lower():
                                    test_input[field_name] = test_case["categories"]

                            result = predictor(**test_input)
                            results.append(f"**Test {i+1}:** {result}")
                        except Exception as e:
                            results.append(f"**Test {i+1} Error:** {str(e)}")

                    # Validation feedback
                    feedback = []

                    # Check docstring
                    if (
                        not signature_class.__doc__
                        or len(signature_class.__doc__.strip()) < 20
                    ):
                        feedback.append("❌ Add a more descriptive docstring")
                    else:
                        feedback.append("✅ Good docstring")

                    # Check field count
                    input_count = len(
                        [
                            name
                            for name, field in signature_class.__annotations__.items()
                            if isinstance(
                                getattr(signature_class, name, None), dspy.InputField
                            )
                        ]
                    )
                    output_count = len(
                        [
                            name
                            for name, field in signature_class.__annotations__.items()
                            if isinstance(
                                getattr(signature_class, name, None), dspy.OutputField
                            )
                        ]
                    )

                    if input_count >= 2:
                        feedback.append("✅ Good input field count")
                    else:
                        feedback.append("❌ Consider adding more input fields")

                    if output_count >= 2:
                        feedback.append("✅ Good output field count")
                    else:
                        feedback.append("❌ Add confidence score output")

                    mo.vstack(
                        [
                            mo.md("### 🧪 Test Results"),
                            mo.md("\n".join(results)),
                            mo.md("### 📊 Validation Feedback"),
                            mo.md("\n".join(feedback)),
                            mo.md(
                                "### 💡 Improvement Tips"
                                if len([f for f in feedback if f.startswith("❌")]) > 0
                                else "### 🎉 Great Job!"
                            ),
                            mo.md(
                                """
                            - Make field descriptions specific and actionable
                            - Include format requirements (e.g., "confidence: 0.0-1.0")
                            - Use descriptive field names that indicate purpose
                            - Test with edge cases to validate robustness
                            """
                                if len([f for f in feedback if f.startswith("❌")]) > 0
                                else "Your signature looks good! Ready for Challenge 2?"
                            ),
                        ]
                    )
                else:
                    mo.md(
                        "❌ No valid signature class found. Make sure to define a class that inherits from dspy.Signature"
                    )

            except Exception as e:
                mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            mo.md("*Enter your signature code and click 'Test Signature'*")
    else:
        mo.md("*Complete Challenge 1 above to see results*")
    return (
        exec_globals,
        feedback,
        input_count,
        input_fields,
        output_count,
        predictor,
        result,
        results,
        signature_class,
        signature_code,
        test_case,
        test_cases,
        test_input,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 2: Multi-Output Analysis Signature
        
        **Task:** Create a signature for analyzing product reviews.
        
        **Requirements:**
        - Input: Product review text and product category
        - Outputs: 
          - Sentiment (positive/negative/neutral)
          - Key features mentioned (3-5 items)
          - Overall rating prediction (1-5 stars)
          - Summary of main points
        
        **Success Criteria:**
        - Multiple structured outputs
        - Clear format specifications
        - Handles different product categories
        """
        )

        # Challenge 2 form
        challenge2_form = mo.ui.form(
            {
                "signature_code": mo.ui.text_area(
                    placeholder="""class ReviewAnalyzer(dspy.Signature):
    \"\"\"Your comprehensive docstring here...\"\"\"
    # Your input and output fields here...""",
                    label="Your Signature Code",
                    rows=10,
                ),
                "submit": mo.ui.button(label="🧪 Test Signature"),
            }
        )

        challenge2_form
    else:
        challenge2_form = None
    return (challenge2_form,)


@app.cell
def __(available_providers, challenge2_form, dspy, mo):
    if (
        available_providers
        and challenge2_form.value
        and challenge2_form.value["submit"]
    ):
        signature_code = challenge2_form.value["signature_code"]

        if signature_code.strip():
            try:
                # Execute the signature code
                exec_globals = {"dspy": dspy}
                exec(signature_code, exec_globals)

                # Find the signature class
                signature_class = None
                for name, obj in exec_globals.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, dspy.Signature)
                        and obj != dspy.Signature
                    ):
                        signature_class = obj
                        break

                if signature_class:
                    # Test the signature
                    predictor = dspy.Predict(signature_class)

                    # Test cases for review analysis
                    test_cases = [
                        {
                            "review": "This laptop is amazing! The battery life is incredible - lasts all day. The screen is crisp and bright. Only downside is it gets a bit warm during heavy use.",
                            "category": "Electronics",
                        },
                        {
                            "review": "Terrible experience with this restaurant. Food was cold, service was slow, and the place was dirty. Would not recommend.",
                            "category": "Restaurant",
                        },
                    ]

                    results = []
                    for i, test_case in enumerate(test_cases):
                        try:
                            # Get field names dynamically
                            input_fields = [
                                name
                                for name, field in signature_class.__annotations__.items()
                                if isinstance(
                                    getattr(signature_class, name, None),
                                    dspy.InputField,
                                )
                            ]

                            # Create input dict
                            test_input = {}
                            for field_name in input_fields:
                                if (
                                    "review" in field_name.lower()
                                    or "text" in field_name.lower()
                                ):
                                    test_input[field_name] = test_case["review"]
                                elif "categor" in field_name.lower():
                                    test_input[field_name] = test_case["category"]

                            result = predictor(**test_input)
                            results.append(f"**Test {i+1}:** {result}")
                        except Exception as e:
                            results.append(f"**Test {i+1} Error:** {str(e)}")

                    # Advanced validation
                    feedback = []

                    # Check output field count and types
                    output_fields = [
                        name
                        for name, field in signature_class.__annotations__.items()
                        if isinstance(
                            getattr(signature_class, name, None), dspy.OutputField
                        )
                    ]

                    if len(output_fields) >= 4:
                        feedback.append("✅ Good number of output fields")
                    else:
                        feedback.append("❌ Need at least 4 output fields")

                    # Check for specific field types
                    field_names = [name.lower() for name in output_fields]
                    if any("sentiment" in name for name in field_names):
                        feedback.append("✅ Sentiment field found")
                    else:
                        feedback.append("❌ Missing sentiment analysis field")

                    if any(
                        "feature" in name or "mention" in name for name in field_names
                    ):
                        feedback.append("✅ Features field found")
                    else:
                        feedback.append("❌ Missing features extraction field")

                    if any("rating" in name or "score" in name for name in field_names):
                        feedback.append("✅ Rating field found")
                    else:
                        feedback.append("❌ Missing rating prediction field")

                    mo.vstack(
                        [
                            mo.md("### 🧪 Test Results"),
                            mo.md("\n".join(results)),
                            mo.md("### 📊 Advanced Validation"),
                            mo.md("\n".join(feedback)),
                            mo.md(
                                "### 🎯 Challenge 2 Complete!"
                                if len([f for f in feedback if f.startswith("❌")]) == 0
                                else "### 💡 Keep Improving"
                            ),
                        ]
                    )
                else:
                    mo.md("❌ No valid signature class found")

            except Exception as e:
                mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            mo.md("*Enter your signature code and click 'Test Signature'*")
    else:
        mo.md("*Complete Challenge 2 above to see results*")
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 3: Complex Reasoning Signature
        
        **Task:** Create a signature for mathematical word problem solving.
        
        **Requirements:**
        - Input: Word problem text and difficulty level
        - Outputs:
          - Step-by-step solution process
          - Final numerical answer
          - Confidence in solution accuracy
          - Alternative solution methods (if applicable)
        
        **Success Criteria:**
        - Handles multi-step reasoning
        - Clear solution format specification
        - Appropriate for ChainOfThought module
        """
        )

        # Challenge 3 form
        challenge3_form = mo.ui.form(
            {
                "signature_code": mo.ui.text_area(
                    placeholder="""class MathProblemSolver(dspy.Signature):
    \"\"\"Your detailed docstring for math problem solving...\"\"\"
    # Your comprehensive field definitions here...""",
                    label="Your Signature Code",
                    rows=12,
                ),
                "test_with_cot": mo.ui.checkbox(
                    label="Test with ChainOfThought module", value=True
                ),
                "submit": mo.ui.button(label="🧪 Test Signature"),
            }
        )

        challenge3_form
    else:
        challenge3_form = None
    return (challenge3_form,)


@app.cell
def __(available_providers, challenge3_form, dspy, mo):
    if (
        available_providers
        and challenge3_form.value
        and challenge3_form.value["submit"]
    ):
        signature_code = challenge3_form.value["signature_code"]
        use_cot = challenge3_form.value["test_with_cot"]

        if signature_code.strip():
            try:
                # Execute the signature code
                exec_globals = {"dspy": dspy}
                exec(signature_code, exec_globals)

                # Find the signature class
                signature_class = None
                for name, obj in exec_globals.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, dspy.Signature)
                        and obj != dspy.Signature
                    ):
                        signature_class = obj
                        break

                if signature_class:
                    # Create predictor (Predict or ChainOfThought)
                    if use_cot:
                        predictor = dspy.ChainOfThought(signature_class)
                        module_type = "ChainOfThought"
                    else:
                        predictor = dspy.Predict(signature_class)
                        module_type = "Predict"

                    # Math problem test cases
                    test_cases = [
                        {
                            "problem": "Sarah has 24 apples. She gives away 1/3 of them to her friends and then buys 8 more apples. How many apples does she have now?",
                            "difficulty": "medium",
                        },
                        {
                            "problem": "A train travels 120 miles in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?",
                            "difficulty": "easy",
                        },
                    ]

                    results = []
                    for i, test_case in enumerate(test_cases):
                        try:
                            # Get field names dynamically
                            input_fields = [
                                name
                                for name, field in signature_class.__annotations__.items()
                                if isinstance(
                                    getattr(signature_class, name, None),
                                    dspy.InputField,
                                )
                            ]

                            # Create input dict
                            test_input = {}
                            for field_name in input_fields:
                                if "problem" in field_name.lower():
                                    test_input[field_name] = test_case["problem"]
                                elif "difficult" in field_name.lower():
                                    test_input[field_name] = test_case["difficulty"]

                            result = predictor(**test_input)
                            results.append(f"**Test {i+1}:** {result}")
                        except Exception as e:
                            results.append(f"**Test {i+1} Error:** {str(e)}")

                    # Expert-level validation
                    feedback = []

                    # Check for reasoning-appropriate fields
                    output_fields = [
                        name
                        for name, field in signature_class.__annotations__.items()
                        if isinstance(
                            getattr(signature_class, name, None), dspy.OutputField
                        )
                    ]

                    field_names = [name.lower() for name in output_fields]

                    if any(
                        "step" in name or "process" in name or "solution" in name
                        for name in field_names
                    ):
                        feedback.append("✅ Step-by-step reasoning field found")
                    else:
                        feedback.append("❌ Missing step-by-step solution field")

                    if any(
                        "answer" in name or "result" in name for name in field_names
                    ):
                        feedback.append("✅ Final answer field found")
                    else:
                        feedback.append("❌ Missing final answer field")

                    if any("confidence" in name for name in field_names):
                        feedback.append("✅ Confidence assessment field found")
                    else:
                        feedback.append("❌ Missing confidence field")

                    # Check docstring quality
                    if (
                        signature_class.__doc__
                        and "step" in signature_class.__doc__.lower()
                    ):
                        feedback.append("✅ Docstring mentions step-by-step approach")
                    else:
                        feedback.append(
                            "❌ Docstring should emphasize reasoning process"
                        )

                    mo.vstack(
                        [
                            mo.md(f"### 🧪 Test Results ({module_type} Module)"),
                            mo.md("\n".join(results)),
                            mo.md("### 📊 Expert Validation"),
                            mo.md("\n".join(feedback)),
                            mo.md(
                                "### 🏆 Challenge 3 Mastered!"
                                if len([f for f in feedback if f.startswith("❌")]) <= 1
                                else "### 🎯 Almost There!"
                            ),
                        ]
                    )
                else:
                    mo.md("❌ No valid signature class found")

            except Exception as e:
                mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            mo.md("*Enter your signature code and click 'Test Signature'*")
    else:
        mo.md("*Complete Challenge 3 above to see results*")
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 4: Creative Design Challenge
        
        **Task:** Design your own signature for a task of your choice!
        
        **Your Mission:**
        1. Choose a real-world problem you want to solve
        2. Design a comprehensive signature
        3. Test it with realistic inputs
        4. Iterate based on results
        
        **Suggested Domains:**
        - **Content Creation**: Blog post generator, social media optimizer
        - **Data Analysis**: Report summarizer, trend analyzer  
        - **Education**: Study guide creator, concept explainer
        - **Business**: Email composer, meeting scheduler
        - **Creative**: Story writer, recipe creator
        
        **Expert-Level Requirements:**
        - Minimum 3 input fields, 4 output fields
        - Clear format specifications in descriptions
        - Handles edge cases gracefully
        - Appropriate for your chosen domain
        """
        )

        # Challenge 4 form
        challenge4_form = mo.ui.form(
            {
                "domain": mo.ui.text(
                    placeholder="e.g., Content Creation, Data Analysis, Education...",
                    label="Problem Domain",
                ),
                "task_description": mo.ui.text_area(
                    placeholder="Describe the specific task your signature will solve...",
                    label="Task Description",
                    rows=3,
                ),
                "signature_code": mo.ui.text_area(
                    placeholder="""class YourSignature(dspy.Signature):
    \"\"\"Your creative and comprehensive docstring...\"\"\"
    # Your innovative field design here...""",
                    label="Your Creative Signature",
                    rows=15,
                ),
                "test_inputs": mo.ui.text_area(
                    placeholder="""Provide 2-3 realistic test cases in this format:
Test 1:
field1: value1
field2: value2

Test 2:
field1: value1
field2: value2""",
                    label="Your Test Cases",
                    rows=8,
                ),
                "submit": mo.ui.button(label="🚀 Test Creative Signature"),
            }
        )

        challenge4_form
    else:
        challenge4_form = None
    return (challenge4_form,)


@app.cell
def __(available_providers, challenge4_form, dspy, mo):
    if (
        available_providers
        and challenge4_form.value
        and challenge4_form.value["submit"]
    ):
        form_data = challenge4_form.value
        signature_code = form_data["signature_code"]
        test_inputs_text = form_data["test_inputs"]

        if signature_code.strip() and test_inputs_text.strip():
            try:
                # Execute the signature code
                exec_globals = {"dspy": dspy}
                exec(signature_code, exec_globals)

                # Find the signature class
                signature_class = None
                for name, obj in exec_globals.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, dspy.Signature)
                        and obj != dspy.Signature
                    ):
                        signature_class = obj
                        break

                if signature_class:
                    # Parse test inputs
                    test_cases = []
                    current_test = {}

                    for line in test_inputs_text.split("\n"):
                        line = line.strip()
                        if line.startswith("Test ") and current_test:
                            test_cases.append(current_test)
                            current_test = {}
                        elif ":" in line and not line.startswith("Test "):
                            key, value = line.split(":", 1)
                            current_test[key.strip()] = value.strip()

                    if current_test:
                        test_cases.append(current_test)

                    # Test both Predict and ChainOfThought
                    predict_predictor = dspy.Predict(signature_class)
                    cot_predictor = dspy.ChainOfThought(signature_class)

                    results = []
                    for i, test_case in enumerate(test_cases):
                        try:
                            # Test with Predict
                            predict_result = predict_predictor(**test_case)
                            results.append(
                                f"**Test {i+1} (Predict):** {predict_result}"
                            )

                            # Test with ChainOfThought
                            cot_result = cot_predictor(**test_case)
                            results.append(
                                f"**Test {i+1} (ChainOfThought):** {cot_result}"
                            )
                            results.append("---")
                        except Exception as e:
                            results.append(f"**Test {i+1} Error:** {str(e)}")

                    # Comprehensive validation
                    feedback = []

                    # Field analysis
                    input_fields = [
                        name
                        for name, field in signature_class.__annotations__.items()
                        if isinstance(
                            getattr(signature_class, name, None), dspy.InputField
                        )
                    ]
                    output_fields = [
                        name
                        for name, field in signature_class.__annotations__.items()
                        if isinstance(
                            getattr(signature_class, name, None), dspy.OutputField
                        )
                    ]

                    if len(input_fields) >= 3:
                        feedback.append("✅ Excellent input field count")
                    else:
                        feedback.append("❌ Need at least 3 input fields")

                    if len(output_fields) >= 4:
                        feedback.append("✅ Excellent output field count")
                    else:
                        feedback.append("❌ Need at least 4 output fields")

                    # Docstring quality
                    if (
                        signature_class.__doc__
                        and len(signature_class.__doc__.strip()) > 50
                    ):
                        feedback.append("✅ Comprehensive docstring")
                    else:
                        feedback.append("❌ Docstring needs more detail")

                    # Domain appropriateness
                    if form_data["domain"] and form_data["task_description"]:
                        feedback.append("✅ Clear domain and task definition")
                    else:
                        feedback.append("❌ Define your domain and task clearly")

                    # Calculate score
                    score = len([f for f in feedback if f.startswith("✅")])
                    total = len(feedback)

                    mo.vstack(
                        [
                            mo.md(f"### 🎨 Creative Challenge Results"),
                            mo.md(f"**Domain:** {form_data['domain']}"),
                            mo.md(f"**Task:** {form_data['task_description']}"),
                            mo.md("### 🧪 Test Results"),
                            mo.md("\n".join(results)),
                            mo.md("### 📊 Expert Evaluation"),
                            mo.md("\n".join(feedback)),
                            mo.md(f"### 🏆 Score: {score}/{total}"),
                            mo.md(
                                "### 🎉 Outstanding Work!"
                                if score == total
                                else (
                                    "### 🎯 Great Progress!"
                                    if score >= total * 0.75
                                    else "### 💪 Keep Improving!"
                                )
                            ),
                            mo.md(
                                """
                        **Next Steps:**
                        - Try your signature with more complex test cases
                        - Experiment with different parameter settings
                        - Consider how this signature could be optimized
                        - Think about production deployment requirements
                        """
                            ),
                        ]
                    )
                else:
                    mo.md("❌ No valid signature class found")

            except Exception as e:
                mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            mo.md("*Complete all fields and click 'Test Creative Signature'*")
    else:
        mo.md("*Complete Challenge 4 above to see results*")
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎓 Exercise Complete!
        
        ### 🏆 What You've Mastered
        
        ✅ **Basic Signature Design** - Classification with confidence scoring
        ✅ **Multi-Output Analysis** - Complex structured outputs  
        ✅ **Reasoning Signatures** - Step-by-step problem solving
        ✅ **Creative Design** - Custom signatures for real-world problems
        
        ### 🎯 Key Skills Developed
        
        - **Field Design Patterns** - Input/output field best practices
        - **Description Writing** - Clear, actionable field descriptions
        - **Format Specification** - Structured output requirements
        - **Testing Methodology** - Systematic signature validation
        - **Iterative Improvement** - Refining based on results
        
        ### 💡 Pro Tips You've Learned
        
        1. **Be Specific**: Detailed descriptions guide LLM behavior
        2. **Test Early**: Validate signatures with realistic inputs
        3. **Multiple Outputs**: Rich structured outputs provide more value
        4. **Consider Context**: Design for your specific use case
        5. **Iterate Often**: Refine based on actual performance
        
        ### 🚀 Ready for Advanced Topics?
        
        You now have solid signature design skills! Time to explore advanced DSPy modules:
        
        **Next Exercise:**
        ```bash
        uv run marimo run 01-foundations/exercises/exercise_02_module_comparison.py
        ```
        
        **Coming Up:**
        - Module performance comparison
        - Parameter optimization techniques
        - Advanced testing strategies
        - Production deployment considerations
        
        ### 🎯 Bonus Challenges
        
        Before moving on, try creating signatures for:
        1. **Code Review Assistant** - Analyze code quality and suggest improvements
        2. **Meeting Minutes Generator** - Extract action items and decisions
        3. **Recipe Nutritional Analyzer** - Calculate calories and nutritional info
        4. **Social Media Content Optimizer** - Improve engagement potential
        
        The more you practice, the better your DSPy systems will become!
        """
        )
    return


if __name__ == "__main__":
    app.run()
