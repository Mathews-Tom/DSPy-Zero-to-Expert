#!/usr/bin/env python3
"""
DSPy Configuration Module

This module handles automatic DSPy language model configuration using environment variables.
It supports multiple providers and automatically selects the best available model.

Author: DSPy Learning Framework
"""

import logging
import os
from pathlib import Path

import dspy

# Set up logging
logger = logging.getLogger(__name__)


def load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from .env file"""
    if env_path is None:
        # Look for .env file in current directory and parent directories
        current_dir = Path(__file__).parent
        for parent in [current_dir] + list(current_dir.parents):
            env_file = parent / ".env"
            if env_file.exists():
                env_path = str(env_file)
                break

    if env_path and os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        logger.info("Loaded environment variables from %s", env_path)


def configure_dspy_lm(provider: str = "auto") -> bool:
    """
    Configure DSPy with the best available language model.

    Args:
        provider: "auto", "openai", "anthropic", or "fallback"

    Returns:
        bool: True if successfully configured, False if using fallback
    """
    # Load environment variables
    load_env_file()

    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    try:
        if provider == "auto":
            # Auto-select best available provider
            if openai_key and openai_key != "your_openai_api_key_here":
                provider = "openai"
            elif anthropic_key and anthropic_key != "your_anthropic_api_key_here":
                provider = "anthropic"
            else:
                provider = "fallback"

        if provider == "openai" and openai_key:
            # Configure with latest OpenAI model using direct LM
            os.environ["OPENAI_API_KEY"] = openai_key  # Ensure it's in environment
            lm = dspy.LM(model="openai/gpt-4.1")
            dspy.configure(lm=lm)
            logger.info("✅ DSPy configured with OpenAI GPT-4.1")
            return True

        elif provider == "anthropic" and anthropic_key:
            # Configure with latest Claude model using direct LM
            os.environ["ANTHROPIC_API_KEY"] = (
                anthropic_key  # Ensure it's in environment
            )
            lm = dspy.LM(model="anthropic/claude-3-7-sonnet-20250219")
            dspy.configure(lm=lm)
            logger.info("✅ DSPy configured with Claude 3.7 Sonnet")
            return True

        else:
            logger.info("ℹ️ No valid API keys found, using fallback implementations")
            return False

    except Exception as e:
        logger.warning("Failed to configure DSPy LM: %s", e)
        logger.info("ℹ️  Using fallback implementations")
        return False


def is_dspy_configured() -> bool:
    """Check if DSPy is properly configured and ready to use"""
    try:
        # Check for direct LM configuration (primary method)
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            return True
        # Check for TwoStepAdapter configuration (fallback)
        elif hasattr(dspy.settings, "adapter") and dspy.settings.adapter is not None:
            return True
        else:
            return False
    except Exception as _:
        return False


def get_configured_model_info() -> dict:
    """Get information about the currently configured model"""
    try:
        # Check for TwoStepAdapter configuration
        if hasattr(dspy.settings, "adapter") and dspy.settings.adapter is not None:
            adapter = dspy.settings.adapter
            if hasattr(adapter, "lm") and adapter.lm is not None:
                lm = adapter.lm
                return {
                    "configured": True,
                    "model": getattr(lm, "model", "unknown"),
                    "provider": type(lm).__name__,
                    "adapter": type(adapter).__name__,
                }
            else:
                return {
                    "configured": True,
                    "model": "unknown",
                    "provider": type(adapter).__name__,
                    "adapter": type(adapter).__name__,
                }
        # Fallback to check for direct LM configuration
        elif hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            lm = dspy.settings.lm
            return {
                "configured": True,
                "model": getattr(lm, "model", "unknown"),
                "provider": type(lm).__name__,
            }
        else:
            return {
                "configured": False,
                "model": "fallback",
                "provider": "fallback",
            }
    except Exception as e:
        return {
            "configured": False,
            "model": "fallback",
            "provider": "fallback",
            "error": str(e),
        }


# Auto-configure when module is imported
if __name__ != "__main__":
    configure_dspy_lm()

# Test configuration when run directly
if __name__ == "__main__":
    print("=== DSPy Configuration Test ===")

    success = configure_dspy_lm()
    model_info = get_configured_model_info()

    print(f"Configuration successful: {success}")
    print(f"Model info: {model_info}")

    if success:
        print("✅ DSPy is properly configured and ready to use!")
    else:
        print("ℹ️  DSPy not configured, fallback mechanisms will be used")
