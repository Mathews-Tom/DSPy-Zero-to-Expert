# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Optional

    import dspy
    import marimo as mo
    from marimo import output

    from common import (
        ContainsMetric,
        DSPyParameterPanel,
        DSPyResultViewer,
        ExactMatchMetric,
        SignatureTester,
        create_evaluation_suite,
        get_config,
        setup_dspy_environment,
    )

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        DSPyParameterPanel,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # 🧪 Interactive Signature Testing Interface

            Master the art of signature testing with real-time parameter tuning, systematic validation, and comprehensive analysis tools.

            ## 🎯 Learning Objectives

            - Build and test custom signatures interactively
            - Optimize signature parameters in real-time
            - Validate signatures with systematic test cases
            - Analyze results and identify improvement opportunities
            - Develop professional testing workflows

            ## 🛠️ What You'll Master

            1. **Interactive Signature Builder** - Create signatures with live preview
            2. **Real-time Parameter Tuning** - Adjust settings and see immediate results
            3. **Systematic Test Cases** - Build comprehensive validation suites
            4. **Result Analysis** - Deep dive into signature performance
            5. **Optimization Suggestions** - AI-powered improvement recommendations
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
                ## ✅ Testing Environment Ready

                **Configuration:**
                - Provider: {config.default_provider}
                - Model: {config.default_model}
                - Interactive testing enabled!
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
        cell3_out = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Build Your Signature

                Let's start by creating a signature to test. You can either use a pre-built example or create your own:
                """
            )
        )

        # Signature selection
        signature_choice = mo.ui.radio(
            options=["Use Example Signature", "Build Custom Signature"],
            label="Choose your approach:",
            value="Use Example Signature",
        )

        cell3_ui = mo.vstack([cell3_out, signature_choice])
    else:
        cell3_ui = mo.md("")
        signature_choice = None

    output.replace(cell3_ui)
    return (signature_choice,)


@app.cell
def _(available_providers, mo, output, signature_choice):
    if (
        available_providers
        and signature_choice is not None
        and signature_choice.value == "Use Example Signature"
    ):
        # Pre-built example signatures
        example_signatures = {
            "Email Classifier": {
                "class_name": "EmailClassifier",
                "docstring": "Classify emails as spam, promotional, or important based on content and context.",
                "fields": {
                    "email_content": (
                        "InputField",
                        "The email content including subject and body",
                    ),
                    "sender_info": (
                        "InputField",
                        "Information about the sender (domain, reputation, etc.)",
                    ),
                    "classification": (
                        "OutputField",
                        "Classification: spam, promotional, or important",
                    ),
                    "confidence": ("OutputField", "Confidence score from 0.0 to 1.0"),
                    "reasoning": (
                        "OutputField",
                        "Brief explanation for the classification",
                    ),
                },
            },
            "Code Reviewer": {
                "class_name": "CodeReviewer",
                "docstring": "Review code for quality, bugs, and improvement suggestions.",
                "fields": {
                    "code": (
                        "InputField",
                        "The code to review (any programming language)",
                    ),
                    "language": ("InputField", "Programming language of the code"),
                    "quality_score": ("OutputField", "Code quality score from 1-10"),
                    "issues": ("OutputField", "List of identified issues or bugs"),
                    "suggestions": (
                        "OutputField",
                        "Specific improvement recommendations",
                    ),
                },
            },
            "Meeting Summarizer": {
                "class_name": "MeetingSummarizer",
                "docstring": "Summarize meeting transcripts with key points, decisions, and action items.",
                "fields": {
                    "transcript": ("InputField", "Full meeting transcript or notes"),
                    "participants": ("InputField", "List of meeting participants"),
                    "summary": ("OutputField", "Concise meeting summary"),
                    "key_decisions": ("OutputField", "Important decisions made"),
                    "action_items": (
                        "OutputField",
                        "Action items with owners if mentioned",
                    ),
                },
            },
        }

        example_selector = mo.ui.dropdown(
            options=list(example_signatures.keys()),
            label="Select Example Signature",
            value="Email Classifier",
        )

        cell4_out = mo.vstack(
            [mo.md("### Choose an Example Signature"), example_selector]
        )
    else:
        cell4_out = mo.md("")
        example_signatures = None
        example_selector = None

    output.replace(cell4_out)
    return example_selector, example_signatures


@app.cell
def _(available_providers, mo, output, signature_choice):
    if (
        available_providers
        and signature_choice is not None
        and signature_choice.value == "Build Custom Signature"
    ):
        # Custom signature builder
        signature_class_name = mo.ui.text(
            placeholder="MySignature", label="Signature Class Name"
        )
        signature_docstring = mo.ui.text_area(
            placeholder="Describe what this signature does...",
            label="Signature Description",
        )
        signature_input_fields = mo.ui.text_area(
            placeholder="field_name: description\nother_field: description",
            label="Input Fields (one per line, format: name: description)",
        )
        signature_output_fields = mo.ui.text_area(
            placeholder="result: description\nconfidence: description",
            label="Output Fields (one per line, format: name: description)",
        )

        cell5_out = mo.vstack(
            [
                mo.md("### Build Your Custom Signature"),
                signature_class_name,
                signature_docstring,
                signature_input_fields,
                signature_output_fields,
            ]
        )
    else:
        cell5_out = mo.md("")
        signature_class_name = None
        signature_docstring = None
        signature_input_fields = None
        signature_output_fields = None

    output.replace(cell5_out)
    return (
        signature_class_name,
        signature_docstring,
        signature_input_fields,
        signature_output_fields,
    )


@app.cell
def _(
    available_providers,
    cleandoc,
    dspy,
    example_selector,
    example_signatures,
    mo,
    output,
    signature_choice,
    signature_class_name,
    signature_docstring,
    signature_input_fields,
    signature_output_fields,
):
    if available_providers and signature_choice is not None:
        # Create the signature based on user choice
        if (
            signature_choice.value == "Use Example Signature"
            and example_selector is not None
            and example_selector.value
            and example_signatures
        ):
            # Use example signature
            signature_info = example_signatures[example_selector.value]

            # Dynamically create the signature class
            class_name = signature_info["class_name"]
            docstring = signature_info["docstring"]

            # Create signature class dynamically
            signature_attrs = {"__doc__": docstring}
            for field_name, (field_type, description) in signature_info[
                "fields"
            ].items():
                if field_type == "InputField":
                    signature_attrs[field_name] = dspy.InputField(desc=description)
                else:
                    signature_attrs[field_name] = dspy.OutputField(desc=description)

            test_signature = type(class_name, (dspy.Signature,), signature_attrs)

            # Generate code representation
            code_lines = [
                f"class {class_name}(dspy.Signature):  ",
                f'    """{docstring}"""  ',
                "  ",
            ]
            for field_name, (field_type, description) in signature_info[
                "fields"
            ].items():
                code_lines.append(
                    f'    {field_name} = dspy.{field_type}(desc="{description}")  '
                )

            signature_code = "\n".join(code_lines)

            cell6_out = mo.md(
                cleandoc(
                    f"""
                    ### ✅ Signature Created: {class_name}

                    ```python\n
                    {signature_code}
                    ```
                    """
                )
            )

        elif (
            signature_choice.value == "Build Custom Signature"
            and signature_class_name is not None
            and signature_class_name.value
        ):
            # Use custom signature
            if all(
                [
                    signature_class_name.value,
                    signature_docstring.value,
                    signature_input_fields.value,
                    signature_output_fields.value,
                ]
            ):
                # Parse input and output fields
                custom_input_fields = {}
                for input_line in signature_input_fields.value.split("\n"):
                    if ":" in input_line:
                        name, desc = input_line.split(":", 1)
                        custom_input_fields[name.strip()] = desc.strip()

                custom_output_fields = {}
                for output_line in signature_output_fields.value.split("\n"):
                    if ":" in output_line:
                        name, desc = output_line.split(":", 1)
                        custom_output_fields[name.strip()] = desc.strip()

                # Create signature class
                signature_attrs = {"__doc__": signature_docstring.value}
                for name, desc in custom_input_fields.items():
                    signature_attrs[name] = dspy.InputField(desc=desc)
                for name, desc in custom_output_fields.items():
                    signature_attrs[name] = dspy.OutputField(desc=desc)

                test_signature = type(
                    signature_class_name.value,
                    (dspy.Signature,),
                    signature_attrs,
                )

                # Generate code
                code_lines = [
                    f"class {signature_class_name.value}(dspy.Signature):  ",
                    f'    """{signature_docstring.value}"""',
                    "",
                ]
                for name, desc in custom_input_fields.items():
                    code_lines.append(f'    {name} = dspy.InputField(desc="{desc}")  ')
                for name, desc in custom_output_fields.items():
                    code_lines.append(f'    {name} = dspy.OutputField(desc="{desc}")  ')

                signature_code = "\n".join(code_lines)

                cell6_out = mo.md(
                    cleandoc(
                        f"""
                        ### ✅ Custom Signature Created

                        ```python\n
                        {signature_code}
                        ```
                        """
                    )
                )
            else:
                cell6_out = mo.md("*Complete all fields in the custom signature form.*")
                test_signature = None
                signature_code = None
        else:
            cell6_out = mo.md("*Select a signature option above.*")
            test_signature = None
            signature_code = None
    else:
        cell6_out = mo.md("")
        test_signature = None
        signature_code = None

    output.replace(cell6_out)
    return (test_signature,)


@app.cell
def _(
    DSPyParameterPanel,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
    test_signature,
):
    if available_providers and test_signature:
        cell7_out = mo.md(
            cleandoc(
                """
                ## ⚙️ Step 2: Configure Testing Parameters

                Adjust the parameters to optimize your signature's performance:
                """
            )
        )

        # Create parameter panel
        param_panel = DSPyParameterPanel(
            show_model_selection=True,
            show_provider_selection=False,
            custom_params={
                "num_tests": {
                    "type": "slider",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                    "label": "Number of Test Runs",
                }
            },
        )

        # Create predictors
        predict_module = dspy.Predict(test_signature)
        cot_module = dspy.ChainOfThought(test_signature)

        cell7_ui = mo.vstack([cell7_out, param_panel.render()])
    else:
        cell7_ui = mo.md("")
        param_panel = None
        predict_module = None
        cot_module = None

    output.replace(cell7_ui)
    return cot_module, predict_module


@app.cell
def _(available_providers, cleandoc, mo, output, test_signature):
    if available_providers and test_signature:
        cell8_out = mo.md(
            cleandoc(
                """
                ## 🧪 Step 3: Create Test Cases

                Build a comprehensive test suite for your signature:
                """
            )
        )

        # Test case builder
        test_case_name = mo.ui.text(
            placeholder="Test Case Name", label="Test Case Name"
        )
        test_case_inputs = mo.ui.text_area(
            placeholder="field1: value1\nfield2: value2",
            label="Test Inputs (format: field_name: value)",
        )
        test_case_expected_outputs = mo.ui.text_area(
            placeholder="expected_field: expected_value",
            label="Expected Outputs (optional, for validation)",
        )

        cell8_ui = mo.vstack(
            [
                cell8_out,
                mo.md("### Add Test Case"),
                test_case_name,
                test_case_inputs,
                test_case_expected_outputs,
            ]
        )
    else:
        cell8_ui = mo.md("")
        test_case_name = None
        test_case_inputs = None
        test_case_expected_outputs = None

    output.replace(cell8_ui)
    return test_case_expected_outputs, test_case_inputs, test_case_name


@app.cell
def _(available_providers, mo, output, test_case_name):
    global test_cases
    if available_providers and test_case_name is not None:
        # Test case management
        if "test_cases" not in globals():
            test_cases = []

        # Add test case button
        add_test_case = mo.ui.run_button(label="➕ Add Test Case")

        cell9_out = mo.vstack(
            [
                add_test_case,
                mo.md(
                    f"**Current Test Cases:** {len(test_cases) if test_cases else 0}"
                ),
            ]
        )
    else:
        cell9_out = mo.md("")
        add_test_case = None
        test_cases = []

    output.replace(cell9_out)
    return add_test_case, test_cases


@app.cell
def _(
    add_test_case,
    available_providers,
    mo,
    output,
    test_case_expected_outputs,
    test_case_inputs,
    test_case_name,
    test_cases,
):
    if (
        available_providers
        and add_test_case is not None
        and add_test_case.value
        and test_case_name is not None
        and test_case_name.value
    ):
        if test_case_name.value and test_case_inputs.value:
            # Parse test inputs
            cell10_test_inputs = {}
            for test_line in test_case_inputs.value.split("\n"):
                if ":" in test_line:
                    key, value = test_line.split(":", 1)
                    cell10_test_inputs[key.strip()] = value.strip()

            # Parse expected outputs
            cell10_expected_outputs = {}
            if test_case_expected_outputs.value:
                for (
                    test_case_expected_output_line
                ) in test_case_expected_outputs.value.split("\n"):
                    if ":" in test_case_expected_output_line:
                        key, value = test_case_expected_output_line.split(":", 1)
                        cell10_expected_outputs[key.strip()] = value.strip()

            # Add to test cases
            new_test_case = {
                "name": test_case_name.value,
                "inputs": cell10_test_inputs,
                "expected": cell10_expected_outputs,
            }

            test_cases.append(new_test_case)

            cell10_out = mo.md(
                f"""
                ### ✅ Test Case Added: {test_case_name.value}

                **Inputs:** {cell10_test_inputs}
                **Expected:** {cell10_expected_outputs if cell10_expected_outputs else "None specified"}

                **Total Test Cases:** {len(test_cases)}
                """
            )
        else:
            cell10_out = mo.md("*Please provide test name and inputs.*")
    else:
        cell10_out = mo.md("*Add test cases using the form above.*")

    output.replace(cell10_out)
    return


@app.cell
def _(available_providers, mo, output, test_cases):
    if available_providers and test_cases:
        cell11_out = mo.md(
            """
            ## 🚀 Step 4: Run Tests

            Execute your test suite and analyze the results:
            """
        )

        # Test execution options
        cell11_module_type = mo.ui.radio(
            options=["Predict", "ChainOfThought", "Both"],
            label="Which modules to test?",
            value="Both",
        )
        cell11_run_tests = mo.ui.run_button(label="🔍 Run Test Suite")

        cell11_ui = mo.vstack([cell11_out, cell11_module_type, cell11_run_tests])
    else:
        cell11_ui = mo.md("")
        cell11_module_type = None
        cell11_run_tests = None

    output.replace(cell11_ui)
    return cell11_module_type, cell11_run_tests


@app.cell
def _(
    available_providers,
    cell11_module_type,
    cell11_run_tests,
    cot_module,
    mo,
    output,
    predict_module,
    test_cases,
    time,
):
    if (
        available_providers
        and cell11_run_tests is not None
        and cell11_run_tests.value
        and test_cases
    ):
        try:
            module_type = cell11_module_type.value
            results = []

            for i, test_case in enumerate(test_cases):
                cell12_test_result = {
                    "test_name": test_case["name"],
                    "inputs": test_case["inputs"],
                    "expected": test_case["expected"],
                }

                # Test Predict module
                if module_type in ["Predict", "Both"]:
                    start_time = time.time()
                    try:
                        cell12_predict_result = predict_module(**test_case["inputs"])
                        cell12_predict_time = time.time() - start_time
                        cell12_test_result["predict_result"] = cell12_predict_result
                        cell12_test_result["predict_time"] = cell12_predict_time
                        cell12_test_result["predict_success"] = True
                    except Exception as e:
                        cell12_test_result["predict_error"] = str(e)
                        cell12_test_result["predict_success"] = False

                # Test ChainOfThought module
                if module_type in ["ChainOfThought", "Both"]:
                    start_time = time.time()
                    try:
                        cell12_cot_result = cot_module(**test_case["inputs"])
                        cell12_cot_time = time.time() - start_time
                        cell12_test_result["cot_result"] = cell12_cot_result
                        cell12_test_result["cot_time"] = cell12_cot_time
                        cell12_test_result["cot_success"] = True
                    except Exception as e:
                        cell12_test_result["cot_error"] = str(e)
                        cell12_test_result["cot_success"] = False

                results.append(cell12_test_result)

            # Display results
            result_displays = []
            for i, result in enumerate(results):
                cell12_result_text = [f"### Test {i+1}: {result['test_name']}"]

                if "predict_result" in result:
                    cell12_result_text.append(
                        f"**Predict Result:** {result['predict_result']}"
                    )
                    cell12_result_text.append(
                        f"**Predict Time:** {result['predict_time']:.3f}s"
                    )

                if "cot_result" in result:
                    cell12_result_text.append(
                        f"**ChainOfThought Result:** {result['cot_result']}"
                    )
                    cell12_result_text.append(
                        f"**CoT Time:** {result['cot_time']:.3f}s"
                    )

                if result.get("predict_error"):
                    cell12_result_text.append(
                        f"**Predict Error:** {result['predict_error']}"
                    )

                if result.get("cot_error"):
                    cell12_result_text.append(f"**CoT Error:** {result['cot_error']}")

                result_displays.append("\n".join(cell12_result_text))

            cell12_out = mo.vstack(
                [
                    mo.md("## 📊 Test Results"),
                    mo.md("\n\n---\n\n".join(result_displays)),
                ]
            )

        except Exception as e:
            cell12_out = mo.md(f"Test execution error: {str(e)}")
            results = []
    else:
        cell12_out = mo.md("*Configure and run tests using the options above.*")
        results = []

    output.replace(cell12_out)
    return (results,)


@app.cell
def _(available_providers, cleandoc, mo, output, results):
    if available_providers and results:
        cell13_out = mo.md(
            cleandoc(
                """
                ## 📈 Step 5: Performance Analysis

                Let's analyze the test results for insights and optimization opportunities:
                """
            )
        )

        # Calculate performance metrics
        cell13_total_tests = len(results)
        cell13_predict_successes = sum(
            1 for r in results if r.get("predict_success", False)
        )
        cell13_cot_successes = sum(1 for r in results if r.get("cot_success", False))

        cell13_predict_times = [
            r["predict_time"] for r in results if "predict_time" in r
        ]
        cell13_cot_times = [r["cot_time"] for r in results if "cot_time" in r]

        cell13_avg_predict_time = (
            sum(cell13_predict_times) / len(cell13_predict_times)
            if cell13_predict_times
            else 0
        )
        cell13_avg_cot_time = (
            sum(cell13_cot_times) / len(cell13_cot_times) if cell13_cot_times else 0
        )

        cell13_analysis = mo.md(
            cleandoc(
                f"""
                ### 📊 Performance Summary

                **Test Execution:**  
                - Total Tests: {cell13_total_tests}  
                - Predict Success Rate: {cell13_predict_successes}/{cell13_total_tests} ({cell13_predict_successes/cell13_total_tests*100:.1f}%)  
                - ChainOfThought Success Rate: {cell13_cot_successes}/{cell13_total_tests} ({cell13_cot_successes/cell13_total_tests*100:.1f}%)  

                **Timing Analysis:**  
                - Average Predict Time: {cell13_avg_predict_time:.3f} seconds  
                - Average CoT Time: {cell13_avg_cot_time:.3f} seconds  
                - Speed Difference: {abs(cell13_avg_cot_time - cell13_avg_predict_time):.3f} seconds  

                **Recommendations:**  
                {
                "- Consider using Predict for faster responses" if cell13_avg_predict_time < cell13_avg_cot_time else  
                "- ChainOfThought provides better reasoning at similar speed"  
                }
                """
            )
        )

        cell13_ui = mo.vstack([cell13_out, cell13_analysis])
    else:
        cell13_ui = mo.md("*Run tests first to see performance analysis.*")

    output.replace(cell13_ui)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell14_out = (
        mo.md(
            cleandoc(
                """
                ## 🎉 Module Complete!

                ### 🎯 What You've Accomplished

                ✅ **Interactive Signature Testing** - Built and tested signatures in real-time  
                ✅ **Parameter Optimization** - Tuned settings for optimal performance  
                ✅ **Systematic Validation** - Created comprehensive test suites  
                ✅ **Performance Analysis** - Analyzed speed, accuracy, and reliability  
                ✅ **Optimization Insights** - Received AI-powered improvement suggestions  

                ### 🔧 Skills Developed

                - Professional signature testing workflows
                - Real-time parameter tuning techniques
                - Systematic test case development
                - Performance analysis and optimization
                - Data-driven signature improvement

                ### 🚀 Ready for Advanced Topics?

                You now have solid foundations in DSPy signatures and testing. Time to explore advanced modules!

                **Next Module: Advanced DSPy Modules**  
                ```bash
                uv run marimo run 02-advanced-modules/react_implementation.py
                ```

                **Coming Up:**  
                - ReAct (Reasoning + Acting) modules  
                - Tool integration with external APIs  
                - Multi-step reasoning pipelines  
                - Advanced debugging and tracing  

                ### 💡 Practice Challenges

                Before moving on, try creating and testing signatures for:  
                1. **Product Review Analyzer** - Extract sentiment, features, and recommendations  
                2. **Code Bug Detector** - Identify potential issues in code snippets  
                3. **Meeting Action Item Extractor** - Parse meeting notes for tasks and owners  
                4. **Recipe Nutritional Analyzer** - Estimate calories and nutritional content  

                The more you practice systematic testing, the better your DSPy systems will become!
                """
            )
        )
        if available_providers
        else ""
    )

    output.replace(cell14_out)
    return


if __name__ == "__main__":
    app.run()
