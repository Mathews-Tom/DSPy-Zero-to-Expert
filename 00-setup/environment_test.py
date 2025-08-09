# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import dspy
import marimo as mo

from common import get_config

# Configure DSPy
config = get_config()
available_providers = config.get_available_llm_providers()

if available_providers:
    from common import setup_dspy_environment

    setup_dspy_environment()

    mo.md(
        f"""
    # DSPy Environment Test

    ✅ **Environment configured successfully!**

    - Available LLM providers: {', '.join(available_providers)}
    - Default provider: {config.default_provider}
    - Default model: {config.default_model}

    You can now proceed with the DSPy learning modules.
    """
    )
else:
    mo.md(
        """
    # ⚠️ Configuration Required

    Please configure at least one LLM provider in your `.env` file:

    - OpenAI: Set `OPENAI_API_KEY`
    - Anthropic: Set `ANTHROPIC_API_KEY`
    - Cohere: Set `COHERE_API_KEY`

    Then restart this notebook.
    """
    )
