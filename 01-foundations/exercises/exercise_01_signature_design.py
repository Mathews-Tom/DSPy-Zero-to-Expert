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
            - **Automated Validation** - Get instant _feedback  
            - **Solution Hints** - Guidance when you need it  

            Let's start building better signatures!
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
                - Provider: {config.default_provider}
                - Model: {config.default_model}

                Ready to start the exercises!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first.
                """
            )
        )
    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
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
        )

        # Challenge 1 form components
        signature_code_input_1 = mo.ui.text_area(
            placeholder=cleandoc(
                """class TicketClassifier(dspy.Signature):
                    \"\"\"Your docstring here...\"\"\"
                    # Your fields here...
                """
            ),
            label="Your Signature Code",
            rows=8,
        )
        submit_button_1 = mo.ui.run_button(label="🧪 Test Signature")

        challenge1_form = mo.vstack(
            [
                signature_code_input_1,
                submit_button_1,
            ]
        )
    else:
        challenge1_form = None
        signature_code_input_1 = None
        submit_button_1 = None
        cell3_desc = mo.md("")

    cell3_out = mo.vstack(
        [
            cell3_desc,
            challenge1_form,
        ]
    )
    output.replace(cell3_out)
    return (
        challenge1_form,
        signature_code_input_1,
        submit_button_1,
    )


@app.cell
def _(available_providers, signature_code_input_1, submit_button_1, dspy, mo, output):
    if available_providers and submit_button_1.value:
        _signature_code = signature_code_input_1.value or ""

        if _signature_code.strip():
            try:
                # Execute the signature code
                _exec_globals = {"dspy": dspy}
                exec(_signature_code, _exec_globals)

                # Find the signature class
                _signature_class = None
                for _name, _obj in _exec_globals.items():
                    if (
                        isinstance(_obj, type)
                        and issubclass(_obj, dspy.Signature)
                        and _obj != dspy.Signature
                    ):
                        _signature_class = _obj
                        break

                if _signature_class:
                    # Test the signature
                    _predictor = dspy.Predict(_signature_class)

                    # Test cases
                    _test_cases = [
                        {
                            "message": "My internet connection keeps dropping every few minutes. I've tried restarting my router but it doesn't help.",
                            "categories": "technical, billing, general, complaint",
                        },
                        {
                            "message": "I was charged twice for my monthly subscription. Please refund the duplicate charge.",
                            "categories": "technical, billing, general, complaint",
                        },
                    ]

                    _results = []
                    for _i, _test_case in enumerate(_test_cases):
                        try:
                            # Get field names dynamically
                            _input_fields = [
                                _name
                                for _name, field in _signature_class.__annotations__.items()
                                if isinstance(
                                    getattr(_signature_class, _name, None),
                                    dspy.InputField,
                                )
                            ]

                            # Create input dict based on available fields
                            _test_input = {}
                            for _field_name in _input_fields:
                                if "message" in _field_name.lower():
                                    _test_input[_field_name] = _test_case["message"]
                                elif "categor" in _field_name.lower():
                                    _test_input[_field_name] = _test_case["categories"]

                            _result = _predictor(**_test_input)
                            _results.append(f"**Test {_i+1}:** {_result}")
                        except Exception as e:
                            _results.append(f"**Test {_i+1} Error:** {str(e)}")

                    # Validation _feedback
                    cell4_feedback = []

                    # Check docstring
                    if (
                        not _signature_class.__doc__
                        or len(_signature_class.__doc__.strip()) < 20
                    ):
                        cell4_feedback.append("❌ Add a more descriptive docstring")
                    else:
                        cell4_feedback.append("✅ Good docstring")

                    # Check field count
                    input_count = len(
                        [
                            _name
                            for _name, field in _signature_class.__annotations__.items()
                            if isinstance(
                                getattr(_signature_class, _name, None), dspy.InputField
                            )
                        ]
                    )
                    output_count = len(
                        [
                            _name
                            for _name, field in _signature_class.__annotations__.items()
                            if isinstance(
                                getattr(_signature_class, _name, None), dspy.OutputField
                            )
                        ]
                    )

                    if input_count >= 2:
                        cell4_feedback.append("✅ Good input field count")
                    else:
                        cell4_feedback.append("❌ Consider adding more input fields")

                    if output_count >= 2:
                        cell4_feedback.append("✅ Good output field count")
                    else:
                        cell4_feedback.append("❌ Add confidence score output")

                    cell4_out = mo.vstack(
                        [
                            mo.md("### 🧪 Test Results"),
                            mo.md("\n".join(_results)),
                            mo.md("### 📊 Validation Feedback"),
                            mo.md("\n".join(cell4_feedback)),
                            mo.md(
                                "### 💡 Improvement Tips"
                                if len(
                                    [_f for _f in cell4_feedback if _f.startswith("❌")]
                                )
                                > 0
                                else "### 🎉 Great Job!"
                            ),
                            mo.md(
                                """
                            - Make field descriptions specific and actionable
                            - Include format requirements (e.g., "confidence: 0.0-1.0")
                            - Use descriptive field names that indicate purpose
                            - Test with edge cases to validate robustness
                            """
                                if len(
                                    [_f for _f in cell4_feedback if _f.startswith("❌")]
                                )
                                > 0
                                else "Your signature looks good! Ready for Challenge 2?"
                            ),
                        ]
                    )
                else:
                    cell4_out = mo.md(
                        "❌ No valid signature class found. Make sure to define a class that inherits from dspy.Signature"
                    )

            except Exception as e:
                cell4_out = mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            cell4_out = mo.md("*Enter your signature code and click 'Test Signature'*")
    else:
        cell4_out = mo.md("*Complete Challenge 1 above to see _results*")
    output.replace(cell4_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
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
        )

        # Challenge 2 form components
        challenge2_signature_input = mo.ui.text_area(
            placeholder=cleandoc(
                """
                class ReviewAnalyzer(dspy.Signature):
                    \"\"\"Your comprehensive docstring here...\"\"\"
                    # Your input and output fields here...
                """
            ),
            label="Your Signature Code",
            rows=10,
        )
        challenge2_submit_button = mo.ui.run_button(label="🧪 Test Signature")

        challenge2_form = mo.vstack(
            [
                challenge2_signature_input,
                challenge2_submit_button,
            ]
        )
    else:
        cell5_desc = mo.md("")
        challenge2_form = None
    cell5_out = mo.vstack(
        [
            cell5_desc,
            challenge2_form,
        ]
    )
    output.replace(cell5_out)
    return (challenge2_form, challenge2_signature_input, challenge2_submit_button)


@app.cell
def _(
    available_providers,
    challenge2_signature_input,
    challenge2_submit_button,
    dspy,
    mo,
    output,
):
    if available_providers and challenge2_submit_button.value:
        _signature_code = challenge2_signature_input.value or ""

        if _signature_code.strip():
            try:
                # Execute the signature code
                _exec_globals = {"dspy": dspy}
                exec(_signature_code, _exec_globals)

                # Find the signature class
                _signature_class = None
                for _name, _obj in _exec_globals.items():
                    if (
                        isinstance(_obj, type)
                        and issubclass(_obj, dspy.Signature)
                        and _obj != dspy.Signature
                    ):
                        _signature_class = _obj
                        break

                if _signature_class:
                    # Test the signature
                    _predictor = dspy.Predict(_signature_class)

                    # Test cases for review analysis
                    _test_cases = [
                        {
                            "review": "This laptop is amazing! The battery life is incredible - lasts all day. The screen is crisp and bright. Only downside is it gets a bit warm during heavy use.",
                            "category": "Electronics",
                        },
                        {
                            "review": "Terrible experience with this restaurant. Food was cold, service was slow, and the place was dirty. Would not recommend.",
                            "category": "Restaurant",
                        },
                    ]

                    _results = []
                    for _i, _test_case in enumerate(_test_cases):
                        try:
                            # Get field names dynamically
                            _input_fields = [
                                _name
                                for _name, field in _signature_class.__annotations__.items()
                                if isinstance(
                                    getattr(_signature_class, _name, None),
                                    dspy.InputField,
                                )
                            ]

                            # Create input dict
                            _test_input = {}
                            for _field_name in _input_fields:
                                if (
                                    "review" in _field_name.lower()
                                    or "text" in _field_name.lower()
                                ):
                                    _test_input[_field_name] = _test_case["review"]
                                elif "categor" in _field_name.lower():
                                    _test_input[_field_name] = _test_case["category"]

                            _result = _predictor(**_test_input)
                            _results.append(f"**Test {_i+1}:** {_result}")
                        except Exception as e:
                            _results.append(f"**Test {_i+1} Error:** {str(e)}")

                    # Advanced validation
                    _feedback = []

                    # Check output field count and types
                    _output_fields = [
                        _name
                        for _name, field in _signature_class.__annotations__.items()
                        if isinstance(
                            getattr(_signature_class, _name, None), dspy.OutputField
                        )
                    ]

                    if len(_output_fields) >= 4:
                        _feedback.append("✅ Good number of output fields")
                    else:
                        _feedback.append("❌ Need at least 4 output fields")

                    # Check for specific field types
                    _field_names = [_name.lower() for _name in _output_fields]
                    if any("sentiment" in _name for _name in _field_names):
                        _feedback.append("✅ Sentiment field found")
                    else:
                        _feedback.append("❌ Missing sentiment analysis field")

                    if any(
                        "feature" in _name or "mention" in _name
                        for _name in _field_names
                    ):
                        _feedback.append("✅ Features field found")
                    else:
                        _feedback.append("❌ Missing features extraction field")

                    if any(
                        "rating" in _name or "score" in _name for _name in _field_names
                    ):
                        _feedback.append("✅ Rating field found")
                    else:
                        _feedback.append("❌ Missing rating prediction field")

                    cell6_out = mo.vstack(
                        [
                            mo.md("### 🧪 Test Results"),
                            mo.md("\n".join(_results)),
                            mo.md("### 📊 Advanced Validation"),
                            mo.md("\n".join(_feedback)),
                            mo.md(
                                "### 🎯 Challenge 2 Complete!"
                                if len([f for f in _feedback if f.startswith("❌")])
                                == 0
                                else "### 💡 Keep Improving"
                            ),
                        ]
                    )
                else:
                    cell6_out = mo.md("❌ No valid signature class found")

            except Exception as e:
                cell6_out = mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            cell6_out = mo.md("*Enter your signature code and click 'Test Signature'*")
    else:
        cell6_out = mo.md("*Complete Challenge 2 above to see _results*")
    output.replace(cell6_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell7_desc = mo.md(
            cleandoc(
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
        )

        # Challenge 3 form components
        challenge3_signature_input = mo.ui.text_area(
            placeholder=cleandoc(
                """class MathProblemSolver(dspy.Signature):
                    \"\"\"Your detailed docstring for math problem solving...\"\"\"
                    # Your comprehensive field definitions here...
                """
            ),
            label="Your Signature Code",
            rows=12,
        )
        challenge3_cot_checkbox = mo.ui.checkbox(
            label="Test with ChainOfThought module", value=True
        )
        challenge3_submit_button = mo.ui.run_button(label="🧪 Test Signature")

        challenge3_form = mo.vstack(
            [
                challenge3_signature_input,
                challenge3_cot_checkbox,
                challenge3_submit_button,
            ]
        )
    else:
        cell7_desc = mo.md("")
        challenge3_form = None
    cell7_out = mo.vstack(
        [
            cell7_desc,
            challenge3_form,
        ]
    )
    output.replace(cell7_out)
    return (
        challenge3_form,
        challenge3_signature_input,
        challenge3_cot_checkbox,
        challenge3_submit_button,
    )


@app.cell
def _(
    available_providers,
    challenge3_signature_input,
    challenge3_cot_checkbox,
    challenge3_submit_button,
    dspy,
    mo,
    output,
):
    if available_providers and challenge3_submit_button.value:
        _signature_code = challenge3_signature_input.value or ""
        use_cot = challenge3_cot_checkbox.value

        if _signature_code.strip():
            try:
                # Execute the signature code
                exec_globals = {"dspy": dspy}
                exec(_signature_code, exec_globals)

                # Find the signature class
                _signature_class = None
                for _name, _obj in exec_globals.items():
                    if (
                        isinstance(_obj, type)
                        and issubclass(_obj, dspy.Signature)
                        and _obj != dspy.Signature
                    ):
                        _signature_class = _obj
                        break

                if _signature_class:
                    # Create _predictor (Predict or ChainOfThought)
                    if use_cot:
                        _predictor = dspy.ChainOfThought(_signature_class)
                        module_type = "ChainOfThought"
                    else:
                        _predictor = dspy.Predict(_signature_class)
                        module_type = "Predict"

                    # Math problem test cases
                    _test_cases = [
                        {
                            "problem": "Sarah has 24 apples. She gives away 1/3 of them to her friends and then buys 8 more apples. How many apples does she have now?",
                            "difficulty": "medium",
                        },
                        {
                            "problem": "A train travels 120 miles in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?",
                            "difficulty": "easy",
                        },
                    ]

                    _results = []
                    for _i, _test_case in enumerate(_test_cases):
                        try:
                            # Get field names dynamically
                            _input_fields = [
                                _name
                                for _name, field in _signature_class.__annotations__.items()
                                if isinstance(
                                    getattr(_signature_class, _name, None),
                                    dspy.InputField,
                                )
                            ]

                            # Create input dict
                            _test_input = {}
                            for _field_name in _input_fields:
                                if "problem" in _field_name.lower():
                                    _test_input[_field_name] = _test_case["problem"]
                                elif "difficult" in _field_name.lower():
                                    _test_input[_field_name] = _test_case["difficulty"]

                            _result = _predictor(**_test_input)
                            _results.append(f"**Test {_i+1}:** {_result}")
                        except Exception as e:
                            _results.append(f"**Test {_i+1} Error:** {str(e)}")

                    # Expert-level validation
                    _feedback = []

                    # Check for reasoning-appropriate fields
                    _output_fields = [
                        _name
                        for _name, field in _signature_class.__annotations__.items()
                        if isinstance(
                            getattr(_signature_class, _name, None), dspy.OutputField
                        )
                    ]

                    _field_names = [_name.lower() for _name in _output_fields]

                    if any(
                        "step" in _name or "process" in _name or "solution" in _name
                        for _name in _field_names
                    ):
                        _feedback.append("✅ Step-by-step reasoning field found")
                    else:
                        _feedback.append("❌ Missing step-by-step solution field")

                    if any(
                        "answer" in _name or "_result" in _name
                        for _name in _field_names
                    ):
                        _feedback.append("✅ Final answer field found")
                    else:
                        _feedback.append("❌ Missing final answer field")

                    if any("confidence" in _name for _name in _field_names):
                        _feedback.append("✅ Confidence assessment field found")
                    else:
                        _feedback.append("❌ Missing confidence field")

                    # Check docstring quality
                    if (
                        _signature_class.__doc__
                        and "step" in _signature_class.__doc__.lower()
                    ):
                        _feedback.append("✅ Docstring mentions step-by-step approach")
                    else:
                        _feedback.append(
                            "❌ Docstring should emphasize reasoning process"
                        )

                    cell8_out = mo.vstack(
                        [
                            mo.md(f"### 🧪 Test Results ({module_type} Module)"),
                            mo.md("\n".join(_results)),
                            mo.md("### 📊 Expert Validation"),
                            mo.md("\n".join(_feedback)),
                            mo.md(
                                "### 🏆 Challenge 3 Mastered!"
                                if len([f for f in _feedback if f.startswith("❌")])
                                <= 1
                                else "### 🎯 Almost There!"
                            ),
                        ]
                    )
                else:
                    cell8_out = mo.md("❌ No valid signature class found")

            except Exception as e:
                cell8_out = mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            cell8_out = mo.md("*Enter your signature code and click 'Test Signature'*")
    else:
        cell8_out = mo.md("*Complete Challenge 3 above to see _results*")
    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell9_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Challenge 4: Creative Design Challenge

                **Task:** Design your own signature for a task of your choice!

                **Your Mission:**
                1. Choose a real-world problem you want to solve
                2. Design a comprehensive signature
                3. Test it with realistic inputs
                4. Iterate based on _results

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
        )

        # Challenge 4 form components (individual components to avoid _clone issues)
        domain_input = mo.ui.text(
            placeholder="e.g., Content Creation, Data Analysis, Education...",
            label="Problem Domain",
        )
        task_description_input = mo.ui.text_area(
            placeholder="Describe the specific task your signature will solve...",
            label="Task Description",
            rows=3,
        )
        signature_code_input_2 = mo.ui.text_area(
            placeholder=cleandoc(
                """class YourSignature(dspy.Signature):
                    \"\"\"Your creative and comprehensive docstring...\"\"\"
                    # Your innovative field design here...
                """
            ),
            label="Your Creative Signature",
            rows=15,
        )
        test_inputs_input = mo.ui.text_area(
            placeholder=cleandoc(
                """
                Provide 2-3 realistic test cases in this format:
                Test 1:
                field1: value1
                field2: value2

                Test 2:
                field1: value1
                field2: value2
                """
            ),
            label="Your Test Cases",
            rows=8,
        )
        submit_button_2 = mo.ui.run_button(label="🚀 Test Creative Signature")

        challenge4_form = mo.vstack(
            [
                domain_input,
                task_description_input,
                signature_code_input_2,
                test_inputs_input,
                submit_button_2,
            ]
        )
    else:
        cell9_desc = mo.md("")
        challenge4_form = None
    cell9_out = mo.vstack(
        [
            cell9_desc,
            challenge4_form,
        ]
    )
    output.replace(cell9_out)
    return (
        challenge4_form,
        domain_input,
        signature_code_input_2,
        submit_button_2,
        task_description_input,
        test_inputs_input,
    )


@app.cell
def _(
    available_providers,
    domain_input,
    task_description_input,
    signature_code_input_2,
    test_inputs_input,
    submit_button_2,
    cleandoc,
    dspy,
    mo,
    output,
):
    if available_providers and submit_button_2.value:
        # Get values from individual components
        domain_value = domain_input.value or ""
        task_description_value = task_description_input.value or ""
        _signature_code = signature_code_input_2.value or ""
        test_inputs_text = test_inputs_input.value or ""

        if _signature_code.strip() and test_inputs_text.strip():
            try:
                # Execute the signature code
                _exec_globals = {"dspy": dspy}
                exec(_signature_code, _exec_globals)

                # Find the signature class
                _signature_class = None
                for _name, _obj in _exec_globals.items():
                    if (
                        isinstance(_obj, type)
                        and issubclass(_obj, dspy.Signature)
                        and _obj != dspy.Signature
                    ):
                        _signature_class = _obj
                        break

                if _signature_class:
                    # Parse test inputs
                    _test_cases = []
                    current_test = {}

                    for line in test_inputs_text.split("\n"):
                        line = line.strip()
                        if line.startswith("Test ") and current_test:
                            _test_cases.append(current_test)
                            current_test = {}
                        elif ":" in line and not line.startswith("Test "):
                            key, value = line.split(":", 1)
                            current_test[key.strip()] = value.strip()

                    if current_test:
                        _test_cases.append(current_test)

                    # Test both Predict and ChainOfThought
                    predict_predictor = dspy.Predict(_signature_class)
                    cot_predictor = dspy.ChainOfThought(_signature_class)

                    _results = []
                    for _i, _test_case in enumerate(_test_cases):
                        try:
                            # Test with Predict
                            predict_result = predict_predictor(**_test_case)
                            _results.append(
                                f"**Test {_i+1} (Predict):** {predict_result}"
                            )

                            # Test with ChainOfThought
                            cot_result = cot_predictor(**_test_case)
                            _results.append(
                                f"**Test {_i+1} (ChainOfThought):** {cot_result}"
                            )
                            _results.append("---")
                        except Exception as e:
                            _results.append(f"**Test {_i+1} Error:** {str(e)}")

                    # Comprehensive validation
                    _feedback = []

                    # Field analysis
                    _input_fields = [
                        _name
                        for _name, field in _signature_class.__annotations__.items()
                        if isinstance(
                            getattr(_signature_class, _name, None), dspy.InputField
                        )
                    ]
                    _output_fields = [
                        _name
                        for _name, field in _signature_class.__annotations__.items()
                        if isinstance(
                            getattr(_signature_class, _name, None), dspy.OutputField
                        )
                    ]

                    if len(_input_fields) >= 3:
                        _feedback.append("✅ Excellent input field count")
                    else:
                        _feedback.append("❌ Need at least 3 input fields")

                    if len(_output_fields) >= 4:
                        _feedback.append("✅ Excellent output field count")
                    else:
                        _feedback.append("❌ Need at least 4 output fields")

                    # Docstring quality
                    if (
                        _signature_class.__doc__
                        and len(_signature_class.__doc__.strip()) > 50
                    ):
                        _feedback.append("✅ Comprehensive docstring")
                    else:
                        _feedback.append("❌ Docstring needs more detail")

                    # Domain appropriateness
                    if domain_value and task_description_value:
                        _feedback.append("✅ Clear domain and task definition")
                    else:
                        _feedback.append("❌ Define your domain and task clearly")

                    # Calculate score
                    score = len([f for f in _feedback if f.startswith("✅")])
                    total = len(_feedback)

                    cell10_out = mo.vstack(
                        [
                            mo.md(f"### 🎨 Creative Challenge Results"),
                            mo.md(f"**Domain:** {domain_value}"),
                            mo.md(f"**Task:** {task_description_value}"),
                            mo.md("### 🧪 Test Results"),
                            mo.md("\n".join(_results)),
                            mo.md("### 📊 Expert Evaluation"),
                            mo.md("\n".join(_feedback)),
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
                                cleandoc(
                                    """
                                    **Next Steps:**
                                    - Try your signature with more complex test cases
                                    - Experiment with different parameter settings
                                    - Consider how this signature could be optimized
                                    - Think about production deployment requirements
                                    """
                                )
                            ),
                        ]
                    )
                else:
                    cell10_out = mo.md("❌ No valid signature class found")

            except Exception as e:
                cell10_out = mo.md(f"❌ **Code Error:** {str(e)}")
        else:
            cell10_out = mo.md(
                "*Complete all fields and click 'Test Creative Signature'*"
            )
    else:
        cell10_out = mo.md("*Complete Challenge 4 above to see _results*")
    output.replace(cell10_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell11_out = mo.md(
        cleandoc(
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
            - **Iterative Improvement** - Refining based on _results

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
        if available_providers
        else ""
    )
    output.replace(cell11_out)
    return


if __name__ == "__main__":
    app.run()
