#!/usr/bin/env python3
"""
DSPy Helper Utilities

This module provides utility functions for working with DSPy signatures, modules,
and common patterns. These helpers simplify setup, testing, and validation tasks
across all solution scripts.

Author: DSPy Learning Framework
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dspy_environment(provider: str = "auto") -> bool:
    """
    Set up DSPy environment with live models using proper configuration.

    Args:
        provider: "auto", "openai", "anthropic", or "fallback"

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Import the proper configuration
        import sys
        from pathlib import Path

        # Add the config module path
        config_path = Path(__file__).parent.parent.parent.parent / "08-custom-modules"
        sys.path.insert(0, str(config_path))

        from dspy_config import (
            configure_dspy_lm,
            get_configured_model_info,
            is_dspy_configured,
        )

        # Configure DSPy with live models
        success = configure_dspy_lm(provider)

        if success and is_dspy_configured():
            model_info = get_configured_model_info()
            logger.info(f"DSPy environment configured successfully: {model_info}")
            return True
        else:
            logger.warning("DSPy configuration failed, using fallback")
            return False

    except Exception as e:
        logger.error(f"Failed to setup DSPy environment: {e}")
        return False


def create_test_signature(
    input_fields: Dict[str, str],
    output_fields: Dict[str, str],
    signature_name: str = "TestSignature",
) -> Type[dspy.Signature]:
    """
    Dynamically create a DSPy signature for testing.

    Args:
        input_fields: Dictionary of input field names and descriptions
        output_fields: Dictionary of output field names and descriptions
        signature_name: Name for the signature class

    Returns:
        Type[dspy.Signature]: Dynamically created signature class
    """
    # Create signature attributes
    signature_attrs = {}

    # Add input fields
    for field_name, description in input_fields.items():
        signature_attrs[field_name] = dspy.InputField(desc=description)

    # Add output fields
    for field_name, description in output_fields.items():
        signature_attrs[field_name] = dspy.OutputField(desc=description)

    # Create the signature class
    signature_class = type(signature_name, (dspy.Signature,), signature_attrs)

    logger.info(f"Created test signature: {signature_name}")
    return signature_class


def run_signature_test(
    signature_class: Type[dspy.Signature],
    test_inputs: Dict[str, Any],
    module_type: str = "ChainOfThought",
) -> Dict[str, Any]:
    """
    Run a test with a DSPy signature and return results.

    Args:
        signature_class: The signature class to test
        test_inputs: Input values for the signature
        module_type: Type of DSPy module to use

    Returns:
        Dict containing test results and metadata
    """
    try:
        start_time = time.time()

        # Create the module
        if module_type == "ChainOfThought":
            module = dspy.ChainOfThought(signature_class)
        elif module_type == "Predict":
            module = dspy.Predict(signature_class)
        elif module_type == "ReAct":
            module = dspy.ReAct(signature_class)
        else:
            module = dspy.ChainOfThought(signature_class)

        # Run the module
        result = module(**test_inputs)

        execution_time = time.time() - start_time

        # Extract outputs
        outputs = {}
        for field_name in signature_class.__annotations__:
            if hasattr(result, field_name):
                outputs[field_name] = getattr(result, field_name)

        return {
            "success": True,
            "outputs": outputs,
            "execution_time": execution_time,
            "module_type": module_type,
            "inputs": test_inputs,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "module_type": module_type,
            "inputs": test_inputs,
        }


def benchmark_signature_performance(
    signature_class: Type[dspy.Signature],
    test_cases: List[Dict[str, Any]],
    module_types: List[str] = None,
    iterations: int = 3,
) -> Dict[str, Any]:
    """
    Benchmark performance of a signature across multiple test cases and module types.

    Args:
        signature_class: The signature class to benchmark
        test_cases: List of test input dictionaries
        module_types: List of module types to test
        iterations: Number of iterations per test

    Returns:
        Dict containing benchmark results
    """
    if module_types is None:
        module_types = ["ChainOfThought", "Predict"]

    results = {
        "signature_name": signature_class.__name__,
        "total_test_cases": len(test_cases),
        "iterations_per_test": iterations,
        "module_results": {},
        "summary": {},
    }

    for module_type in module_types:
        module_results = {
            "execution_times": [],
            "success_rate": 0,
            "average_time": 0,
            "total_time": 0,
            "errors": [],
        }

        successful_runs = 0
        total_runs = 0

        for test_case in test_cases:
            for _ in range(iterations):
                result = run_signature_test(signature_class, test_case, module_type)
                total_runs += 1

                if result["success"]:
                    successful_runs += 1
                    module_results["execution_times"].append(result["execution_time"])
                else:
                    module_results["errors"].append(result["error"])

        # Calculate statistics
        if module_results["execution_times"]:
            module_results["average_time"] = sum(
                module_results["execution_times"]
            ) / len(module_results["execution_times"])
            module_results["total_time"] = sum(module_results["execution_times"])

        module_results["success_rate"] = (
            successful_runs / total_runs if total_runs > 0 else 0
        )

        results["module_results"][module_type] = module_results

    # Generate summary
    best_module = max(
        results["module_results"].keys(),
        key=lambda x: results["module_results"][x]["success_rate"],
    )

    results["summary"] = {
        "best_performing_module": best_module,
        "best_success_rate": results["module_results"][best_module]["success_rate"],
        "best_average_time": results["module_results"][best_module]["average_time"],
    }

    logger.info(f"Benchmark completed for {signature_class.__name__}")
    return results


def validate_signature_output(
    signature_class: Type[dspy.Signature],
    test_inputs: Dict[str, Any],
    expected_outputs: Dict[str, Any] = None,
    validation_rules: Dict[str, callable] = None,
) -> Dict[str, Any]:
    """
    Validate signature output against expected results or validation rules.

    Args:
        signature_class: The signature class to validate
        test_inputs: Input values for the signature
        expected_outputs: Expected output values (optional)
        validation_rules: Dictionary of field names to validation functions

    Returns:
        Dict containing validation results
    """
    # Run the signature
    test_result = run_signature_test(signature_class, test_inputs)

    if not test_result["success"]:
        return {"valid": False, "error": test_result["error"], "validation_details": {}}

    outputs = test_result["outputs"]
    validation_details = {}
    overall_valid = True

    # Validate against expected outputs
    if expected_outputs:
        for field_name, expected_value in expected_outputs.items():
            if field_name in outputs:
                actual_value = outputs[field_name]
                is_match = (
                    str(actual_value).strip().lower()
                    == str(expected_value).strip().lower()
                )
                validation_details[f"{field_name}_expected"] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "match": is_match,
                }
                if not is_match:
                    overall_valid = False
            else:
                validation_details[f"{field_name}_missing"] = {
                    "error": f"Expected field {field_name} not found in outputs"
                }
                overall_valid = False

    # Validate with custom rules
    if validation_rules:
        for field_name, validation_func in validation_rules.items():
            if field_name in outputs:
                try:
                    is_valid = validation_func(outputs[field_name])
                    validation_details[f"{field_name}_rule"] = {
                        "valid": is_valid,
                        "value": outputs[field_name],
                    }
                    if not is_valid:
                        overall_valid = False
                except Exception as e:
                    validation_details[f"{field_name}_rule_error"] = {
                        "error": str(e),
                        "value": outputs[field_name],
                    }
                    overall_valid = False
            else:
                validation_details[f"{field_name}_rule_missing"] = {
                    "error": f"Field {field_name} not found for validation"
                }
                overall_valid = False

    return {
        "valid": overall_valid,
        "outputs": outputs,
        "validation_details": validation_details,
        "execution_time": test_result["execution_time"],
    }


def create_signature_from_template(
    template_name: str, **kwargs
) -> Type[dspy.Signature]:
    """
    Create a signature from a predefined template.

    Args:
        template_name: Name of the template to use
        **kwargs: Additional parameters for template customization

    Returns:
        Type[dspy.Signature]: Created signature class
    """
    templates = {
        "question_answering": {
            "inputs": {
                "question": "Question to answer",
                "context": "Context information",
            },
            "outputs": {
                "answer": "Answer to the question",
                "confidence": "Confidence score",
            },
        },
        "text_classification": {
            "inputs": {
                "text": "Text to classify",
                "categories": "Available categories",
            },
            "outputs": {
                "category": "Predicted category",
                "confidence": "Confidence score",
            },
        },
        "text_summarization": {
            "inputs": {"text": "Text to summarize", "length": "Desired summary length"},
            "outputs": {"summary": "Generated summary", "key_points": "Key points"},
        },
        "sentiment_analysis": {
            "inputs": {"text": "Text to analyze"},
            "outputs": {"sentiment": "Sentiment label", "score": "Sentiment score"},
        },
        "translation": {
            "inputs": {
                "text": "Text to translate",
                "target_language": "Target language",
            },
            "outputs": {
                "translation": "Translated text",
                "confidence": "Translation confidence",
            },
        },
    }

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")

    template = templates[template_name]

    # Allow customization through kwargs
    inputs = template["inputs"].copy()
    outputs = template["outputs"].copy()

    if "custom_inputs" in kwargs:
        inputs.update(kwargs["custom_inputs"])

    if "custom_outputs" in kwargs:
        outputs.update(kwargs["custom_outputs"])

    signature_name = kwargs.get("signature_name", f"{template_name.title()}Signature")

    return create_test_signature(inputs, outputs, signature_name)


def analyze_signature_complexity(
    signature_class: Type[dspy.Signature],
) -> Dict[str, Any]:
    """
    Analyze the complexity of a DSPy signature.

    Args:
        signature_class: The signature class to analyze

    Returns:
        Dict containing complexity analysis
    """
    analysis = {
        "signature_name": signature_class.__name__,
        "total_fields": 0,
        "input_fields": 0,
        "output_fields": 0,
        "field_details": {},
        "complexity_score": 0,
    }

    # Analyze fields
    for field_name, field_obj in signature_class.__annotations__.items():
        analysis["total_fields"] += 1

        if hasattr(signature_class, field_name):
            field_instance = getattr(signature_class, field_name)

            if isinstance(field_instance, dspy.InputField):
                analysis["input_fields"] += 1
                field_type = "input"
            elif isinstance(field_instance, dspy.OutputField):
                analysis["output_fields"] += 1
                field_type = "output"
            else:
                field_type = "unknown"

            analysis["field_details"][field_name] = {
                "type": field_type,
                "description": getattr(field_instance, "desc", "No description"),
                "annotation": str(field_obj),
            }

    # Calculate complexity score
    # Simple heuristic: more fields = higher complexity
    analysis["complexity_score"] = (
        analysis["input_fields"] * 1.0 + analysis["output_fields"] * 1.5
    )

    # Classify complexity
    if analysis["complexity_score"] <= 3:
        analysis["complexity_level"] = "Simple"
    elif analysis["complexity_score"] <= 6:
        analysis["complexity_level"] = "Moderate"
    else:
        analysis["complexity_level"] = "Complex"

    return analysis


# Example usage and testing functions
def demo_dspy_helpers():
    """Demonstrate DSPy helper functions"""
    print("🔧 DSPy Helper Utilities Demo")
    print("=" * 50)

    # Setup environment
    print("Setting up DSPy environment...")
    setup_success = setup_dspy_environment()
    print(f"Setup successful: {setup_success}")

    # Create test signature
    print("\nCreating test signature...")
    test_sig = create_test_signature(
        input_fields={"question": "Question to answer"},
        output_fields={
            "answer": "Answer to question",
            "confidence": "Confidence level",
        },
        signature_name="TestQA",
    )

    # Analyze signature complexity
    print("\nAnalyzing signature complexity...")
    complexity = analyze_signature_complexity(test_sig)
    print(
        f"Complexity: {complexity['complexity_level']} "
        f"(Score: {complexity['complexity_score']:.1f})"
    )

    # Run signature test
    print("\nRunning signature test...")
    test_result = run_signature_test(
        test_sig, {"question": "What is the capital of France?"}
    )

    if test_result["success"]:
        print(f"Test successful! Execution time: {test_result['execution_time']:.2f}s")
        print(f"Answer: {test_result['outputs'].get('answer', 'N/A')}")
    else:
        print(f"Test failed: {test_result['error']}")

    # Create signature from template
    print("\nCreating signature from template...")
    template_sig = create_signature_from_template("sentiment_analysis")
    template_complexity = analyze_signature_complexity(template_sig)
    print(f"Template signature complexity: {template_complexity['complexity_level']}")

    print("\n✅ DSPy Helpers Demo Complete!")


if __name__ == "__main__":
    demo_dspy_helpers()
