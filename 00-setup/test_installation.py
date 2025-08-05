#!/usr/bin/env python3
"""
Installation test script for DSPy Zero-to-Expert learning repository.

This script performs comprehensive testing of the DSPy and Marimo installation,
including API connectivity tests and basic functionality verification.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common import (
    DSPyParameterPanel,
    ExactMatchMetric,
    SignatureTester,
    get_config,
    setup_dspy_environment,
    setup_logging,
    validate_environment,
)

# Set up logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def test_basic_imports() -> Dict[str, bool]:
    """Test basic package imports."""
    print_header("Basic Import Tests")

    results = {}

    # Test core packages
    packages = [
        ("dspy", "DSPy framework"),
        ("marimo", "Marimo reactive notebooks"),
        ("openai", "OpenAI client"),
        ("anthropic", "Anthropic client"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
    ]

    for package, description in packages:
        try:
            __import__(package)
            print_success(f"{description}")
            results[package] = True
        except ImportError:
            print_error(f"{description} - MISSING")
            results[package] = False

    # Test common utilities
    try:
        from common import DSPyParameterPanel, ReactiveModule, get_config

        print_success("Common utilities")
        results["common"] = True
    except ImportError as e:
        print_error(f"Common utilities - FAILED: {e}")
        results["common"] = False

    return results


def test_dspy_functionality() -> Dict[str, bool]:
    """Test DSPy functionality in detail."""
    print_header("DSPy Functionality Tests")

    results = {}

    try:
        import dspy

        print_success("DSPy import")
        results["dspy_import"] = True

        # Test signature creation
        class TestSignature(dspy.Signature):
            """Test signature for installation verification."""

            input_text = dspy.InputField(desc="Input text to process")
            output_text = dspy.OutputField(desc="Processed output text")

        print_success("DSPy Signature creation")
        results["signature_creation"] = True

        # Test module creation
        predictor = dspy.Predict(TestSignature)
        print_success("DSPy Predict module creation")
        results["module_creation"] = True

        # Test ChainOfThought
        cot_predictor = dspy.ChainOfThought(TestSignature)
        print_success("DSPy ChainOfThought module creation")
        results["cot_creation"] = True

        # Test configuration (without making API calls)
        config = get_config()
        available_providers = config.get_available_llm_providers()

        if available_providers:
            print_success(f"LLM providers available: {', '.join(available_providers)}")
            results["llm_config"] = True

            # Test DSPy configuration
            try:
                setup_dspy_environment(provider=available_providers[0])
                print_success("DSPy environment configuration")
                results["dspy_config"] = True
            except Exception as e:
                print_warning(f"DSPy configuration warning: {e}")
                results["dspy_config"] = False
        else:
            print_warning("No LLM providers configured")
            results["llm_config"] = False
            results["dspy_config"] = False

    except Exception as e:
        print_error(f"DSPy functionality test failed: {e}")
        results["dspy_import"] = False
        results["signature_creation"] = False
        results["module_creation"] = False
        results["cot_creation"] = False
        results["llm_config"] = False
        results["dspy_config"] = False

    return results


def test_marimo_functionality() -> Dict[str, bool]:
    """Test Marimo functionality in detail."""
    print_header("Marimo Functionality Tests")

    results = {}

    try:
        import marimo as mo

        print_success("Marimo import")
        results["marimo_import"] = True

        # Test UI elements
        slider = mo.ui.slider(0, 100, value=50, label="Test Slider")
        text_input = mo.ui.text(value="test", label="Test Input")
        dropdown = mo.ui.dropdown(
            ["option1", "option2"], value="option1", label="Test Dropdown"
        )

        print_success("Marimo UI elements creation")
        results["ui_elements"] = True

        # Test our custom Marimo components
        try:
            panel = DSPyParameterPanel(show_temperature=True, show_max_tokens=True)
            print_success("Custom DSPy parameter panel")
            results["custom_components"] = True
        except Exception as e:
            print_warning(f"Custom components warning: {e}")
            results["custom_components"] = False

        # Test Marimo CLI availability
        import subprocess

        try:
            result = subprocess.run(
                ["marimo", "--version"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                print_success(f"Marimo CLI: {version}")
                results["marimo_cli"] = True
            else:
                print_warning("Marimo CLI not working properly")
                results["marimo_cli"] = False

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_warning("Marimo CLI not available")
            results["marimo_cli"] = False

    except Exception as e:
        print_error(f"Marimo functionality test failed: {e}")
        results["marimo_import"] = False
        results["ui_elements"] = False
        results["custom_components"] = False
        results["marimo_cli"] = False

    return results


def test_integration() -> Dict[str, bool]:
    """Test DSPy-Marimo integration."""
    print_header("DSPy-Marimo Integration Tests")

    results = {}

    try:
        import dspy
        import marimo as mo

        # Test signature tester integration
        class IntegrationTestSignature(dspy.Signature):
            """Integration test signature."""

            question = dspy.InputField()
            answer = dspy.OutputField()

        # Test our signature tester
        tester = SignatureTester(IntegrationTestSignature)
        print_success("Signature tester integration")
        results["signature_tester"] = True

        # Test evaluation utilities
        metric = ExactMatchMetric()
        test_result = metric.evaluate("hello", "hello")

        if test_result.score == 1.0:
            print_success("Evaluation utilities integration")
            results["evaluation_utils"] = True
        else:
            print_error("Evaluation utilities failed")
            results["evaluation_utils"] = False

        # Test reactive module wrapper (without API calls)
        from common import ReactiveModule

        predictor = dspy.Predict(IntegrationTestSignature)
        reactive_predictor = ReactiveModule(predictor, name="test_predictor")

        print_success("Reactive module wrapper")
        results["reactive_wrapper"] = True

    except Exception as e:
        print_error(f"Integration test failed: {e}")
        logger.exception("Integration test error")
        results["signature_tester"] = False
        results["evaluation_utils"] = False
        results["reactive_wrapper"] = False

    return results


def test_api_connectivity() -> Dict[str, bool]:
    """Test API connectivity (optional, only if user wants to)."""
    print_header("API Connectivity Tests (Optional)")

    results = {}
    config = get_config()

    print_info("API connectivity tests require making actual API calls.")
    print_info("This will consume a small amount of API credits.")

    # For now, just check configuration without making calls
    providers = [
        ("openai", config.has_openai_config()),
        ("anthropic", config.has_anthropic_config()),
        ("cohere", config.has_cohere_config()),
    ]

    for provider, is_configured in providers:
        if is_configured:
            print_success(f"{provider.title()} API key configured")
            results[f"{provider}_config"] = True
        else:
            print_info(f"{provider.title()} API key not configured")
            results[f"{provider}_config"] = False

    # Note: Actual API calls would be implemented here if needed
    print_info("Skipping actual API calls to avoid consuming credits during testing")
    print_info("API connectivity will be tested when you run your first DSPy example")

    return results


def run_performance_test() -> Dict[str, Any]:
    """Run basic performance tests."""
    print_header("Performance Tests")

    results = {}

    try:
        # Test import speed
        start_time = time.time()
        import dspy
        import marimo as mo

        from common import DSPyParameterPanel, get_config

        import_time = time.time() - start_time

        print_success(f"Import speed: {import_time:.3f} seconds")
        results["import_time"] = import_time

        # Test component creation speed
        start_time = time.time()
        panel = DSPyParameterPanel()
        config = get_config()
        creation_time = time.time() - start_time

        print_success(f"Component creation: {creation_time:.3f} seconds")
        results["creation_time"] = creation_time

        # Test signature creation speed
        start_time = time.time()

        class PerfTestSignature(dspy.Signature):
            input_field = dspy.InputField()
            output_field = dspy.OutputField()

        predictor = dspy.Predict(PerfTestSignature)
        signature_time = time.time() - start_time

        print_success(f"Signature creation: {signature_time:.3f} seconds")
        results["signature_time"] = signature_time

        # Overall performance assessment
        total_time = import_time + creation_time + signature_time
        if total_time < 2.0:
            print_success("Overall performance: Excellent")
        elif total_time < 5.0:
            print_success("Overall performance: Good")
        else:
            print_warning("Overall performance: Slow (may indicate issues)")

        results["total_time"] = total_time

    except Exception as e:
        print_error(f"Performance test failed: {e}")
        results["error"] = str(e)

    return results


def generate_test_report(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Generate a comprehensive test report."""
    print_header("Installation Test Report")

    # Count successes and failures
    total_tests = 0
    passed_tests = 0

    for category, results in all_results.items():
        if category == "performance":
            continue  # Skip performance results in pass/fail counting

        for test_name, result in results.items():
            if isinstance(result, bool):
                total_tests += 1
                if result:
                    passed_tests += 1

    # Print summary
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    print(
        f"Test Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)"
    )

    if success_rate >= 90:
        print_success("🎉 Excellent! Your installation is working perfectly.")
        print_info("You're ready to start learning DSPy!")
    elif success_rate >= 75:
        print_success("✅ Good! Your installation is mostly working.")
        print_info("You can proceed with learning, but some features may be limited.")
    elif success_rate >= 50:
        print_warning("⚠️ Partial installation detected.")
        print_info("Some features may not work. Consider reinstalling dependencies.")
    else:
        print_error("❌ Installation has significant issues.")
        print_info("Please check your installation and dependencies.")

    # Detailed breakdown
    print("\nDetailed Results:")
    for category, results in all_results.items():
        if category == "performance":
            continue

        print(f"\n{category.replace('_', ' ').title()}:")
        for test_name, result in results.items():
            if isinstance(result, bool):
                status = "✓" if result else "✗"
                print(f"  {status} {test_name.replace('_', ' ').title()}")

    # Performance summary
    if "performance" in all_results:
        perf = all_results["performance"]
        if "total_time" in perf:
            print(f"\nPerformance: {perf['total_time']:.3f}s total startup time")

    # Next steps
    print("\nNext Steps:")
    if success_rate >= 75:
        print_info("1. Run: uv run 00-setup/setup_environment.py")
        print_info("2. Start learning: uv run marimo run 00-setup/hello_dspy_marimo.py")
    else:
        print_info("1. Check the error messages above")
        print_info("2. Reinstall dependencies: uv sync")
        print_info("3. Check your .env file configuration")
        print_info("4. Run this test again")


def main():
    """Main test function."""
    print_header("DSPy Zero-to-Expert Installation Test")
    print("This script will test your DSPy and Marimo installation.\n")

    all_results = {}

    try:
        # Run all tests
        all_results["basic_imports"] = test_basic_imports()
        all_results["dspy_functionality"] = test_dspy_functionality()
        all_results["marimo_functionality"] = test_marimo_functionality()
        all_results["integration"] = test_integration()
        all_results["api_connectivity"] = test_api_connectivity()
        all_results["performance"] = run_performance_test()

        # Generate report
        generate_test_report(all_results)

        # Determine exit code
        total_critical_tests = (
            len(all_results["basic_imports"])
            + len(all_results["dspy_functionality"])
            + len(all_results["marimo_functionality"])
            + len(all_results["integration"])
        )

        passed_critical_tests = sum(
            sum(1 for result in results.values() if isinstance(result, bool) and result)
            for category, results in all_results.items()
            if category
            in [
                "basic_imports",
                "dspy_functionality",
                "marimo_functionality",
                "integration",
            ]
        )

        success_rate = (
            (passed_critical_tests / total_critical_tests * 100)
            if total_critical_tests > 0
            else 0
        )

        sys.exit(0 if success_rate >= 75 else 1)

    except KeyboardInterrupt:
        print_error("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Test failed with unexpected error: {e}")
        logger.exception("Unexpected test error")
        sys.exit(1)


if __name__ == "__main__":
    main()
