#!/usr/bin/env python3
"""
Environment setup script for DSPy Zero-to-Expert learning repository.

This script automates the environment configuration process, including:
- API key setup and validation
- DSPy configuration
- Marimo setup
- Dependency verification
"""

import logging
import os
import sys
from inspect import cleandoc
from pathlib import Path

import dspy

from common import (
    configure_dspy_lm,
    get_config,
    setup_dspy_environment,
    setup_logging,
    validate_environment,
)

# Add the project root to the path so we can import common utilities
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def check_env_file() -> bool:
    """Check if .env file exists and guide user through creation if needed."""
    print_header("Environment File Setup")

    env_file = Path(".env")
    env_template = Path(".env.template")

    if not env_template.exists():
        print_error(".env.template file not found!")
        print_info("This file should have been created during project setup.")
        return False

    if env_file.exists():
        print_success(".env file already exists")
        return True

    print_warning(".env file not found")
    print_info("Creating .env file from template...")

    try:
        # Copy template to .env
        with open(env_template, "r") as template:
            content = template.read()

        with open(env_file, "w") as env:
            env.write(content)

        print_success(".env file created from template")
        print_info("Please edit .env file and add your API keys before continuing")
        print_info(
            "Required: At least one LLM provider API key (OpenAI, Anthropic, or Cohere)"
        )

        return True

    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


def setup_api_keys() -> dict[str, bool]:
    """Guide user through API key setup and validation."""
    print_header("API Key Configuration")

    config = get_config()
    api_status = {}

    # Check each provider
    providers = [
        ("OpenAI", "OPENAI_API_KEY", config.has_openai_config()),
        ("Anthropic", "ANTHROPIC_API_KEY", config.has_anthropic_config()),
        ("Cohere", "COHERE_API_KEY", config.has_cohere_config()),
    ]

    for provider_name, env_var, is_configured in providers:
        if is_configured:
            print_success(f"{provider_name} API key configured")
            api_status[provider_name.lower()] = True
        else:
            print_warning(f"{provider_name} API key not configured")
            print_info(f"Set {env_var} in your .env file to use {provider_name}")
            api_status[provider_name.lower()] = False

    # Check if at least one provider is configured
    if not any(api_status.values()):
        print_error("No LLM provider API keys configured!")
        print_info("You need at least one API key to use DSPy")
        print_info("Edit your .env file and add an API key for:")
        print_info("  - OpenAI: https://platform.openai.com/api-keys")
        print_info("  - Anthropic: https://console.anthropic.com/")
        print_info("  - Cohere: https://dashboard.cohere.ai/api-keys")
        return api_status

    print_success(f"Found {sum(api_status.values())} configured LLM provider(s)")

    # Check optional services
    optional_services = [
        ("Tavily Search", config.has_tavily_config()),
        ("Langfuse Observability", config.has_langfuse_config()),
    ]

    print_info("\nOptional services:")
    for service_name, is_configured in optional_services:
        if is_configured:
            print_success(f"{service_name} configured")
        else:
            print_info(f"{service_name} not configured (optional)")

    return api_status


def test_dspy_setup() -> bool:
    """Test DSPy configuration and basic functionality."""
    print_header("DSPy Configuration Test")

    try:
        config = get_config()

        # Get available providers
        available_providers = config.get_available_llm_providers()
        if not available_providers:
            print_error("No LLM providers available")
            return False

        print_info(f"Available providers: {', '.join(available_providers)}")

        # Try to configure DSPy with the default provider
        default_provider = config.default_provider
        if default_provider not in available_providers:
            default_provider = available_providers[0]
            print_info(f"Using {default_provider} instead of configured default")

        print_info(f"Configuring DSPy with {default_provider}...")

        # Set up DSPy environment
        setup_dspy_environment(provider=default_provider)
        print_success("DSPy configured successfully")

        # Test basic DSPy functionality
        print_info("Testing basic DSPy functionality...")

        # Create a simple signature
        class TestSignature(dspy.Signature):
            """Test signature for environment verification."""

            question = dspy.InputField()
            answer = dspy.OutputField()

        # Create a predictor (but don't call it to avoid API costs during setup)
        predictor = dspy.Predict(TestSignature)
        print_success("DSPy signature and predictor creation successful")

        return True

    except Exception as e:
        print_error(f"DSPy setup failed: {e}")
        logger.exception("DSPy setup error")
        return False


def test_marimo_setup() -> bool:
    """Test Marimo installation and basic functionality."""
    print_header("Marimo Setup Test")

    try:
        import marimo as mo

        print_success("Marimo imported successfully")

        # Test basic UI element creation
        test_slider = mo.ui.slider(0, 100, value=50)
        test_text = mo.ui.text(value="test")
        print_success("Marimo UI elements created successfully")

        # Check if marimo CLI is available
        import subprocess

        result = subprocess.run(
            ["marimo", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Marimo CLI available: {version}")
        else:
            print_warning("Marimo CLI not available or not working properly")
            return False

        return True

    except ImportError:
        print_error("Marimo not installed or not importable")
        return False
    except subprocess.TimeoutExpired:
        print_error("Marimo CLI test timed out")
        return False
    except Exception as e:
        print_error(f"Marimo setup test failed: {e}")
        return False


def create_sample_notebook() -> bool:
    """Create a sample Marimo notebook for testing."""
    print_header("Sample Notebook Creation")

    sample_notebook_content = cleandoc(
        '''import marimo as mo
        import dspy
        from common import get_config, DSPyParameterPanel

        # Configure DSPy
        config = get_config()
        available_providers = config.get_available_llm_providers()

        if available_providers:
            from common import setup_dspy_environment
            setup_dspy_environment()
            
            mo.md(f"""
            # DSPy Environment Test
            
            ✅ **Environment configured successfully!**
            
            - Available LLM providers: {', '.join(available_providers)}
            - Default provider: {config.default_provider}
            - Default model: {config.default_model}
            
            You can now proceed with the DSPy learning modules.
            """)
        else:
            mo.md("""
            # ⚠️ Configuration Required
            
            Please configure at least one LLM provider in your `.env` file:
            
            - OpenAI: Set `OPENAI_API_KEY`
            - Anthropic: Set `ANTHROPIC_API_KEY`  
            - Cohere: Set `COHERE_API_KEY`
            
            Then restart this notebook.
            """)
        '''
    )

    try:
        sample_file = Path("00-setup/environment_test.py")
        with open(sample_file, "w") as f:
            f.write(sample_notebook_content)

        print_success(f"Sample notebook created: {sample_file}")
        print_info("Run with: uv run marimo run 00-setup/environment_test.py")
        return True

    except Exception as e:
        print_error(f"Failed to create sample notebook: {e}")
        return False


def run_comprehensive_setup() -> bool:
    """Run the complete environment setup process."""
    print_header("DSPy Zero-to-Expert Environment Setup")
    print("This script will help you set up your development environment.\n")

    success_count = 0
    total_steps = 5

    # Step 1: Check/create .env file
    if check_env_file():
        success_count += 1

    # Step 2: Setup and validate API keys
    api_status = setup_api_keys()
    if any(api_status.values()):
        success_count += 1

    # Step 3: Test DSPy setup
    if test_dspy_setup():
        success_count += 1

    # Step 4: Test Marimo setup
    if test_marimo_setup():
        success_count += 1

    # Step 5: Create sample notebook
    if create_sample_notebook():
        success_count += 1

    # Final summary
    print_header("Setup Summary")

    if success_count == total_steps:
        print_success(f"🎉 All {total_steps} setup steps completed successfully!")
        print_info("Your environment is ready for DSPy learning.")
        print_info("Next steps:")
        print_info(
            "  1. Test your setup: uv run marimo run 00-setup/environment_test.py"
        )
        print_info(
            "  2. Start learning: uv run marimo run 00-setup/hello_dspy_marimo.py"
        )
        return True
    else:
        print_warning(f"⚠️ {success_count}/{total_steps} setup steps completed")
        print_info("Please address the issues above before proceeding.")

        if success_count == 0:
            print_error("Setup failed completely. Please check your installation.")
        elif success_count < 3:
            print_warning("Critical issues found. Environment may not work properly.")
        else:
            print_info("Minor issues found. Environment should work with limitations.")

        return False


def main():
    """Main setup function."""
    try:
        success = run_comprehensive_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed with unexpected error: {e}")
        logger.exception("Unexpected setup error")
        sys.exit(1)


if __name__ == "__main__":
    main()
