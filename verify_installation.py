#!/usr/bin/env python3
"""
Installation verification script for DSPy Zero-to-Expert learning repository.

This script verifies that all required dependencies are properly installed
and configured for the learning environment.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


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


def check_python_version() -> bool:
    """Check if Python version is 3.11 or higher."""
    print_header("Python Version Check")

    version = sys.version_info
    required_major, required_minor = 3, 11

    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"Required Python version: {required_major}.{required_minor}+")

    if version.major >= required_major and version.minor >= required_minor:
        print_success("Python version is compatible")
        return True
    else:
        print_error(f"Python {required_major}.{required_minor}+ is required")
        return False


def check_package_installation() -> Tuple[List[str], List[str]]:
    """Check if required packages are installed."""
    print_header("Package Installation Check")

    # Core packages that must be installed
    core_packages = [
        ("dspy", "dspy-ai"),
        ("marimo", "marimo"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("sklearn", "scikit-learn"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("faiss", "faiss-cpu"),
        ("chromadb", "chromadb"),
        ("qdrant_client", "qdrant-client"),
        ("pytest", "pytest"),
        ("jupyter", "jupyter"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "pydantic"),
        ("rich", "rich"),
        ("tqdm", "tqdm"),
    ]

    # Optional packages
    optional_packages = [
        ("mlflow", "mlflow"),
        ("langfuse", "langfuse"),
        ("tavily", "tavily-python"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence-transformers"),
    ]

    installed = []
    missing = []

    print("Checking core packages...")
    for import_name, package_name in core_packages:
        try:
            importlib.import_module(import_name)
            print_success(f"{package_name}")
            installed.append(package_name)
        except ImportError:
            print_error(f"{package_name} - MISSING")
            missing.append(package_name)

    print("\nChecking optional packages...")
    for import_name, package_name in optional_packages:
        try:
            importlib.import_module(import_name)
            print_success(f"{package_name} (optional)")
            installed.append(package_name)
        except ImportError:
            print_warning(f"{package_name} - MISSING (optional)")

    return installed, missing


def check_dspy_functionality() -> bool:
    """Test basic DSPy functionality."""
    print_header("DSPy Functionality Check")

    try:
        import dspy

        print_success("DSPy imported successfully")

        # Test signature creation
        class TestSignature(dspy.Signature):
            """Test signature for verification."""

            input_text = dspy.InputField()
            output_text = dspy.OutputField()

        print_success("DSPy Signature creation works")

        # Test module creation (without LLM call)
        predictor = dspy.Predict(TestSignature)
        print_success("DSPy Predict module creation works")

        return True

    except Exception as e:
        print_error(f"DSPy functionality test failed: {e}")
        return False


def check_marimo_functionality() -> bool:
    """Test basic Marimo functionality."""
    print_header("Marimo Functionality Check")

    try:
        import marimo as mo

        print_success("Marimo imported successfully")

        # Test UI element creation
        slider = mo.ui.slider(0, 100, value=50)
        print_success("Marimo UI elements work")

        # Check if marimo command is available
        result = subprocess.run(
            ["marimo", "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print_success(f"Marimo CLI available: {result.stdout.strip()}")
            return True
        else:
            print_error("Marimo CLI not available")
            return False

    except subprocess.TimeoutExpired:
        print_error("Marimo CLI check timed out")
        return False
    except Exception as e:
        print_error(f"Marimo functionality test failed: {e}")
        return False


def check_environment_configuration() -> bool:
    """Check environment configuration."""
    print_header("Environment Configuration Check")

    # Check if .env file exists
    env_file = Path(".env")
    env_template = Path(".env.template")

    if not env_template.exists():
        print_error(".env.template file not found")
        return False
    else:
        print_success(".env.template file exists")

    if not env_file.exists():
        print_warning(
            ".env file not found - you'll need to create it from .env.template"
        )
        print_info("Copy .env.template to .env and configure your API keys")
        return True
    else:
        print_success(".env file exists")

    # Load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv()

        # Check for common API keys (without revealing them)
        api_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "COHERE_API_KEY",
            "TAVILY_API_KEY",
        ]

        configured_keys = []
        for key in api_keys:
            value = os.getenv(key)
            if value and value != f"your_{key.lower()}_here":
                configured_keys.append(key)
                print_success(f"{key} is configured")
            else:
                print_warning(f"{key} not configured")

        if configured_keys:
            print_info(f"Found {len(configured_keys)} configured API keys")
        else:
            print_warning("No API keys configured - some features may not work")

        return True

    except Exception as e:
        print_error(f"Environment configuration check failed: {e}")
        return False


def check_directory_structure() -> bool:
    """Check if the project directory structure is correct."""
    print_header("Directory Structure Check")

    required_dirs = [
        "common",
        "00-setup",
        "01-foundations",
        "02-advanced-modules",
        "03-retrieval-rag",
        "04-optimization-teleprompters",
        "05-evaluation-metrics",
        "06-datasets-examples",
        "07-tracing-debugging",
        "08-custom-modules",
        "09-production-deployment",
        "10-advanced-projects",
        "assets",
        "docs",
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"{dir_name}/ directory exists")
        else:
            print_error(f"{dir_name}/ directory missing")
            missing_dirs.append(dir_name)

    if missing_dirs:
        print_error(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print_success("All required directories exist")
        return True


def run_comprehensive_check() -> Dict[str, bool]:
    """Run all verification checks."""
    print_header("DSPy Zero-to-Expert Installation Verification")
    print("This script will verify your installation and configuration.\n")

    results = {}

    # Run all checks
    results["python_version"] = check_python_version()

    installed, missing = check_package_installation()
    results["packages"] = len(missing) == 0

    results["dspy_functionality"] = check_dspy_functionality()
    results["marimo_functionality"] = check_marimo_functionality()
    results["environment_config"] = check_environment_configuration()
    results["directory_structure"] = check_directory_structure()

    return results


def print_summary(results: Dict[str, bool]) -> None:
    """Print verification summary."""
    print_header("Verification Summary")

    passed = sum(results.values())
    total = len(results)

    for check, status in results.items():
        check_name = check.replace("_", " ").title()
        if status:
            print_success(f"{check_name}")
        else:
            print_error(f"{check_name}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} checks passed{Colors.END}")

    if passed == total:
        print_success(
            "\n🎉 All checks passed! Your environment is ready for DSPy learning."
        )
        print_info("You can now start with Module 00: Environment Setup & Introduction")
        print_info("Run: uv run marimo run 00-setup/hello_dspy_marimo.py")
    else:
        print_error(
            f"\n❌ {total - passed} checks failed. Please address the issues above."
        )
        print_info("Refer to the troubleshooting guide: docs/TROUBLESHOOTING.md")

        if not results.get("packages", True):
            print_info("To install missing packages, run: uv sync")

        if not results.get("environment_config", True):
            print_info("Copy .env.template to .env and configure your API keys")


def main():
    """Main verification function."""
    try:
        results = run_comprehensive_check()
        print_summary(results)

        # Exit with error code if any checks failed
        if not all(results.values()):
            sys.exit(1)

    except KeyboardInterrupt:
        print_error("\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Verification failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
