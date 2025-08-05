import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import sys
    import time
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    import dspy
    import marimo as mo

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

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

    return (
        Any,
        DSPyParameterPanel,
        DSPyResultViewer,
        Dict,
        ExactMatchMetric,
        List,
        Optional,
        Path,
        SignatureTester,
        ContainsMetric,
        create_evaluation_suite,
        dspy,
        get_config,
        json,
        mo,
        project_root,
        setup_dspy_environment,
        sys,
        time,
    )


@app.cell
def __(mo):
    mo.md(
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
        ## ✅ Testing Environment Ready
        
        **Configuration:**
        - Provider: {config.default_llm_provider}
        - Model: {config.default_model}
        - Interactive testing enabled!
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
        ## 🏗️ Step 1: Build Your Signature
        
        Let's start by creating a signature to test. You can either use a pre-built example or create your own:
        """
        )

        # Signature selection
        signature_choice = mo.ui.radio(
            options=["Use Example Signature", "Build Custom Signature"],
            label="Choose your approach:",
            value="Use Example Signature",
        )

        signature_choice
    else:
        signature_choice = None
    return (signature_choice,)


@app.cell
def __(available_providers, dspy, mo, signature_choice):
    if available_providers and signature_choice.value == "Use Example Signature":
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

        mo.vstack([mo.md("### Choose an Example Signature"), example_selector])
    else:
        example_signatures = None
        example_selector = None
    return example_selector, example_signatures


@app.cell
def __(available_providers, mo, signature_choice):
    if available_providers and signature_choice.value == "Build Custom Signature":
        # Custom signature builder
        custom_signature_form = mo.ui.form(
            {
                "class_name": mo.ui.text(
                    placeholder="MySignature", label="Signature Class Name"
                ),
                "docstring": mo.ui.text_area(
                    placeholder="Describe what this signature does...",
                    label="Signature Description",
                ),
                "input_fields": mo.ui.text_area(
                    placeholder="field_name: description\nother_field: description",
                    label="Input Fields (one per line, format: name: description)",
                ),
                "output_fields": mo.ui.text_area(
                    placeholder="result: description\nconfidence: description",
                    label="Output Fields (one per line, format: name: description)",
                ),
            }
        )

        mo.vstack([mo.md("### Build Your Custom Signature"), custom_signature_form])
    else:
        custom_signature_form = None
    return (custom_signature_form,)


@app.cell
def __(
    available_providers,
    custom_signature_form,
    dspy,
    example_selector,
    example_signatures,
    mo,
    signature_choice,
):
    if available_providers:
        # Create the signature based on user choice
        if (
            signature_choice.value == "Use Example Signature"
            and example_selector
            and example_selector.value
        ):
            # Use example signature
            sig_info = example_signatures[example_selector.value]

            # Dynamically create the signature class
            class_name = sig_info["class_name"]
            docstring = sig_info["docstring"]

            # Create signature class dynamically
            signature_attrs = {"__doc__": docstring}
            for field_name, (field_type, description) in sig_info["fields"].items():
                if field_type == "InputField":
                    signature_attrs[field_name] = dspy.InputField(desc=description)
                else:
                    signature_attrs[field_name] = dspy.OutputField(desc=description)

            TestSignature = type(class_name, (dspy.Signature,), signature_attrs)

            # Generate code representation
            code_lines = [
                f"class {class_name}(dspy.Signature):",
                f'    """{docstring}"""',
                "",
            ]
            for field_name, (field_type, description) in sig_info["fields"].items():
                code_lines.append(
                    f'    {field_name} = dspy.{field_type}(desc="{description}")'
                )

            signature_code = "\n".join(code_lines)

            mo.md(
                f"""
            ### ✅ Signature Created: {class_name}
            
            ```python
            {signature_code}
            ```
            """
            )

        elif (
            signature_choice.value == "Build Custom Signature"
            and custom_signature_form
            and custom_signature_form.value
        ):
            # Use custom signature
            form_data = custom_signature_form.value

            if all(
                [
                    form_data["class_name"],
                    form_data["docstring"],
                    form_data["input_fields"],
                    form_data["output_fields"],
                ]
            ):

                # Parse input and output fields
                input_fields = {}
                for line in form_data["input_fields"].split("\n"):
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        input_fields[name.strip()] = desc.strip()

                output_fields = {}
                for line in form_data["output_fields"].split("\n"):
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        output_fields[name.strip()] = desc.strip()

                # Create signature class
                signature_attrs = {"__doc__": form_data["docstring"]}
                for name, desc in input_fields.items():
                    signature_attrs[name] = dspy.InputField(desc=desc)
                for name, desc in output_fields.items():
                    signature_attrs[name] = dspy.OutputField(desc=desc)

                TestSignature = type(
                    form_data["class_name"], (dspy.Signature,), signature_attrs
                )

                # Generate code
                code_lines = [
                    f"class {form_data['class_name']}(dspy.Signature):",
                    f'    """{form_data["docstring"]}"""',
                    "",
                ]
                for name, desc in input_fields.items():
                    code_lines.append(f'    {name} = dspy.InputField(desc="{desc}")')
                for name, desc in output_fields.items():
                    code_lines.append(f'    {name} = dspy.OutputField(desc="{desc}")')

                signature_code = "\n".join(code_lines)

                mo.md(
                    f"""
                ### ✅ Custom Signature Created
                
                ```python
                {signature_code}
                ```
                """
                )
            else:
                TestSignature = None
                signature_code = None
                mo.md("*Complete all fields in the custom signature form.*")
        else:
            TestSignature = None
            signature_code = None
            mo.md("*Select a signature option above.*")
    else:
        TestSignature = None
        signature_code = None
    return TestSignature, class_name, code_lines, signature_attrs, signature_code


@app.cell
def __(DSPyParameterPanel, TestSignature, available_providers, dspy, mo):
    if available_providers and TestSignature:
        mo.md(
            """
        ## ⚙️ Step 2: Configure Testing Parameters
        
        Adjust the parameters to optimize your signature's performance:
        """
        )

        # Create parameter panel
        param_panel = DSPyParameterPanel(
            show_model_selection=True,
            show_temperature=True,
            show_max_tokens=True,
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
        predict_module = dspy.Predict(TestSignature)
        cot_module = dspy.ChainOfThought(TestSignature)

        param_panel.render()
    else:
        param_panel = None
        predict_module = None
        cot_module = None
    return cot_module, param_panel, predict_module


@app.cell
def __(TestSignature, available_providers, mo):
    if available_providers and TestSignature:
        mo.md(
            """
        ## 🧪 Step 3: Create Test Cases
        
        Build a comprehensive test suite for your signature:
        """
        )

        # Test case builder
        test_case_form = mo.ui.form(
            {
                "test_name": mo.ui.text(
                    placeholder="Test Case Name", label="Test Case Name"
                ),
                "test_inputs": mo.ui.text_area(
                    placeholder="field1: value1\nfield2: value2",
                    label="Test Inputs (format: field_name: value)",
                ),
                "expected_outputs": mo.ui.text_area(
                    placeholder="expected_field: expected_value",
                    label="Expected Outputs (optional, for validation)",
                ),
            }
        )

        mo.vstack([mo.md("### Add Test Case"), test_case_form])
    else:
        test_case_form = None
    return (test_case_form,)


@app.cell
def __(available_providers, mo, test_case_form):
    if available_providers and test_case_form:
        # Test case management
        if "test_cases" not in globals():
            test_cases = []

        # Add test case button
        add_test_case = mo.ui.button(label="➕ Add Test Case")

        mo.vstack([add_test_case, mo.md(f"**Current Test Cases:** {len(test_cases)}")])
    else:
        add_test_case = None
        test_cases = []
    return add_test_case, test_cases


@app.cell
def __(add_test_case, available_providers, mo, test_case_form, test_cases):
    if (
        available_providers
        and add_test_case
        and add_test_case.value
        and test_case_form.value
    ):
        form_data = test_case_form.value

        if form_data["test_name"] and form_data["test_inputs"]:
            # Parse test inputs
            test_inputs = {}
            for line in form_data["test_inputs"].split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    test_inputs[key.strip()] = value.strip()

            # Parse expected outputs
            expected_outputs = {}
            if form_data["expected_outputs"]:
                for line in form_data["expected_outputs"].split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        expected_outputs[key.strip()] = value.strip()

            # Add to test cases
            new_test_case = {
                "name": form_data["test_name"],
                "inputs": test_inputs,
                "expected": expected_outputs,
            }

            test_cases.append(new_test_case)

            mo.md(
                f"""
            ### ✅ Test Case Added: {form_data["test_name"]}
            
            **Inputs:** {test_inputs}
            **Expected:** {expected_outputs if expected_outputs else "None specified"}
            
            **Total Test Cases:** {len(test_cases)}
            """
            )
        else:
            mo.md("*Please provide test name and inputs.*")
    else:
        mo.md("*Add test cases using the form above.*")
    return expected_outputs, form_data, new_test_case, test_inputs


@app.cell
def __(available_providers, mo, test_cases):
    if available_providers and test_cases:
        mo.md(
            """
        ## 🚀 Step 4: Run Tests
        
        Execute your test suite and analyze the results:
        """
        )

        # Test execution options
        test_options = mo.ui.form(
            {
                "module_type": mo.ui.radio(
                    options=["Predict", "ChainOfThought", "Both"],
                    label="Which modules to test?",
                    value="Both",
                ),
                "run_tests": mo.ui.button(label="🔍 Run Test Suite"),
            }
        )

        test_options
    else:
        test_options = None
    return (test_options,)


@app.cell
def __(
    available_providers,
    cot_module,
    mo,
    predict_module,
    test_cases,
    test_options,
    time,
):
    if (
        available_providers
        and test_options
        and test_options.value["run_tests"]
        and test_cases
    ):
        try:
            module_type = test_options.value["module_type"]
            results = []

            for i, test_case in enumerate(test_cases):
                test_result = {
                    "test_name": test_case["name"],
                    "inputs": test_case["inputs"],
                    "expected": test_case["expected"],
                }

                # Test Predict module
                if module_type in ["Predict", "Both"]:
                    start_time = time.time()
                    try:
                        predict_result = predict_module(**test_case["inputs"])
                        predict_time = time.time() - start_time
                        test_result["predict_result"] = predict_result
                        test_result["predict_time"] = predict_time
                        test_result["predict_success"] = True
                    except Exception as e:
                        test_result["predict_error"] = str(e)
                        test_result["predict_success"] = False

                # Test ChainOfThought module
                if module_type in ["ChainOfThought", "Both"]:
                    start_time = time.time()
                    try:
                        cot_result = cot_module(**test_case["inputs"])
                        cot_time = time.time() - start_time
                        test_result["cot_result"] = cot_result
                        test_result["cot_time"] = cot_time
                        test_result["cot_success"] = True
                    except Exception as e:
                        test_result["cot_error"] = str(e)
                        test_result["cot_success"] = False

                results.append(test_result)

            # Display results
            result_displays = []
            for i, result in enumerate(results):
                result_text = [f"### Test {i+1}: {result['test_name']}"]

                if "predict_result" in result:
                    result_text.append(
                        f"**Predict Result:** {result['predict_result']}"
                    )
                    result_text.append(
                        f"**Predict Time:** {result['predict_time']:.3f}s"
                    )

                if "cot_result" in result:
                    result_text.append(
                        f"**ChainOfThought Result:** {result['cot_result']}"
                    )
                    result_text.append(f"**CoT Time:** {result['cot_time']:.3f}s")

                if result.get("predict_error"):
                    result_text.append(f"**Predict Error:** {result['predict_error']}")

                if result.get("cot_error"):
                    result_text.append(f"**CoT Error:** {result['cot_error']}")

                result_displays.append("\n".join(result_text))

            mo.vstack(
                [
                    mo.md("## 📊 Test Results"),
                    mo.md("\n\n---\n\n".join(result_displays)),
                ]
            )

        except Exception as e:
            mo.md(f"Test execution error: {str(e)}")
    else:
        results = []
        mo.md("*Configure and run tests using the options above.*")
    return i, result, result_displays, result_text, results, start_time, test_result


@app.cell
def __(available_providers, mo, results):
    if available_providers and results:
        mo.md(
            """
        ## 📈 Step 5: Performance Analysis
        
        Let's analyze the test results for insights and optimization opportunities:
        """
        )

        # Calculate performance metrics
        total_tests = len(results)
        predict_successes = sum(1 for r in results if r.get("predict_success", False))
        cot_successes = sum(1 for r in results if r.get("cot_success", False))

        predict_times = [r["predict_time"] for r in results if "predict_time" in r]
        cot_times = [r["cot_time"] for r in results if "cot_time" in r]

        avg_predict_time = (
            sum(predict_times) / len(predict_times) if predict_times else 0
        )
        avg_cot_time = sum(cot_times) / len(cot_times) if cot_times else 0

        mo.md(
            f"""
        ### 📊 Performance Summary
        
        **Test Execution:**
        - Total Tests: {total_tests}
        - Predict Success Rate: {predict_successes}/{total_tests} ({predict_successes/total_tests*100:.1f}%)
        - ChainOfThought Success Rate: {cot_successes}/{total_tests} ({cot_successes/total_tests*100:.1f}%)
        
        **Timing Analysis:**
        - Average Predict Time: {avg_predict_time:.3f} seconds
        - Average CoT Time: {avg_cot_time:.3f} seconds
        - Speed Difference: {abs(avg_cot_time - avg_predict_time):.3f} seconds
        
        **Recommendations:**
        {
        "- Consider using Predict for faster responses" if avg_predict_time < avg_cot_time else
        "- ChainOfThought provides better reasoning at similar speed"
        }
        """
        )
    else:
        mo.md("*Run tests first to see performance analysis.*")
    return (
        avg_cot_time,
        avg_predict_time,
        cot_successes,
        cot_times,
        predict_successes,
        predict_times,
        total_tests,
    )


@app.cell
def __(available_providers, mo, results):
    if available_providers and results:
        mo.md(
            """
        ## 🎯 Step 6: Optimization Suggestions
        
        Based on your test results, here are specific recommendations:
        """
        )

        # Generate optimization suggestions
        suggestions = []

        # Check for errors
        predict_errors = [r for r in results if r.get("predict_error")]
        cot_errors = [r for r in results if r.get("cot_error")]

        if predict_errors or cot_errors:
            suggestions.append(
                "🔧 **Error Handling**: Some tests failed. Consider adding input validation or error handling."
            )

        # Check timing
        if avg_predict_time > 2.0:
            suggestions.append(
                "⚡ **Speed Optimization**: Consider simplifying your signature or using a faster model."
            )

        if avg_cot_time > avg_predict_time * 2:
            suggestions.append(
                "🧠 **Reasoning Balance**: ChainOfThought is significantly slower. Evaluate if reasoning is worth the cost."
            )

        # Check consistency
        predict_results = [
            str(r.get("predict_result", "")) for r in results if "predict_result" in r
        ]
        cot_results = [
            str(r.get("cot_result", "")) for r in results if "cot_result" in r
        ]

        if len(set(predict_results)) == len(predict_results):
            suggestions.append(
                "🎯 **Consistency**: Results vary significantly. Consider refining your signature descriptions."
            )

        # General suggestions
        suggestions.extend(
            [
                "📝 **Field Descriptions**: Make field descriptions more specific to guide LLM behavior.",
                "🧪 **More Test Cases**: Add edge cases and boundary conditions to your test suite.",
                "📊 **Evaluation Metrics**: Consider adding automated evaluation metrics for objective assessment.",
            ]
        )

        mo.md("### 💡 Optimization Recommendations\n\n" + "\n".join(suggestions))
    else:
        mo.md("*Run tests first to get optimization suggestions.*")
    return cot_results, predict_errors, predict_results, suggestions


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎓 Testing Best Practices
        
        ### 📋 Signature Testing Checklist
        
        **Before Testing:**
        - ✅ Clear signature purpose and scope
        - ✅ Specific field descriptions with examples
        - ✅ Appropriate input/output field types
        - ✅ Comprehensive test cases covering edge cases
        
        **During Testing:**
        - ✅ Test both Predict and ChainOfThought modules
        - ✅ Vary parameters (temperature, max_tokens)
        - ✅ Monitor execution times and success rates
        - ✅ Document unexpected behaviors
        
        **After Testing:**
        - ✅ Analyze results for patterns and issues
        - ✅ Refine signature based on findings
        - ✅ Add more test cases for weak areas
        - ✅ Consider production deployment requirements
        
        ### 🚀 Advanced Testing Techniques
        
        **Batch Testing:**
        ```python
        # Test multiple inputs at once
        test_inputs = [input1, input2, input3]
        results = [predictor(**inputs) for inputs in test_inputs]
        ```
        
        **A/B Testing:**
        ```python
        # Compare different signature versions
        results_v1 = test_signature_v1(test_cases)
        results_v2 = test_signature_v2(test_cases)
        compare_results(results_v1, results_v2)
        ```
        
        **Automated Evaluation:**
        ```python
        # Use evaluation metrics
        from common import ExactMatchMetric
        metric = ExactMatchMetric()
        scores = [metric.evaluate(pred, expected) for pred, expected in zip(predictions, expected_outputs)]
        ```
        """
        )
    return


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


if __name__ == "__main__":
    app.run()
