#!/usr/bin/env python3
# pylint: disable=import-error

"""Test script to validate LLM API keys using litellm and common/config."""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import litellm

from common.config import get_config


def main():
    config = get_config()
    providers = config.get_available_llm_providers()

    if not providers:
        print("No LLM providers configured. Please set your API keys in .env.")
        sys.exit(1)

    success = True

    for provider in providers:
        print(f"\nTesting {provider} API key validity...")
        llm_cfg = config.get_llm_config(provider)
        # Skip non-default providers to only test default model
        if provider != config.default_provider:
            print(
                f"⏭️ Skipping {provider}; default provider is {config.default_provider}"
            )
            continue

        # Skip tests for non-default providers
        if provider != config.default_provider:
            print(
                f"⏭️ Skipping tests for {provider}; default provider is {config.default_provider}"
            )
            continue

        # Completion test only
        test_model = config.default_model
        try:
            response = litellm.completion(
                model=test_model,
                messages=[{"role": "user", "content": "Say hello!"}],
                **llm_cfg,
            )
            msg = (
                response["choices"][0]["message"]["content"]
                if isinstance(response, dict) and "choices" in response
                else str(response)
            )
            print(
                f"✅ {provider} completion with model {test_model} succeeded. Response: {msg}"
            )
        except Exception as e:
            print(f"❌ {provider} completion failed for model {test_model}. Error: {e}")
            print(f"   API key: {llm_cfg.get('api_key')}")
            success = False

    if success:
        print("\nAll API key validations succeeded.")
        sys.exit(0)
    else:
        print("\nSome API key validations failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
