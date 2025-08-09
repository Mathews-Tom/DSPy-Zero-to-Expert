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

    import marimo as mo
    from marimo import output

    from common import get_config, validate_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return cleandoc, get_config, mo, output, validate_environment


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # 🧪 Exercise 1: Environment Check

            **Objective**: Verify that your DSPy learning environment is properly set up.

            ## What You'll Do

            1. Check your configuration settings
            2. Validate API key setup
            3. Test basic functionality
            4. Troubleshoot any issues

            ## Instructions

            Work through each section below and ensure all checks pass before proceeding to the next module.
            """
        )
    )
    output.replace(cell1_out)
    return


@app.cell
def _(cleandoc, get_config, mo, output):
    # Load and display configuration
    config = get_config()

    cell2_out = mo.md(
        cleandoc(
            f"""
            ## 📋 Configuration Check

            **Current Configuration:**
            - Environment: {config.environment}
            - Debug Mode: {config.debug}
            - Default LLM Provider: {config.default_provider}
            - Default Model: {config.default_model}
            - Cache Enabled: {config.enable_cache}
            - Log Level: {config.log_level}
            """
        )
    )
    output.replace(cell2_out)
    return (config,)


@app.cell
def _(cleandoc, config, mo, output):
    # Check API key configuration
    providers_status = {
        "OpenAI": config.has_openai_config(),
        "Anthropic": config.has_anthropic_config(),
        "Cohere": config.has_cohere_config(),
        "Tavily": config.has_tavily_config(),
        "Langfuse": config.has_langfuse_config(),
    }

    configured_count = sum(providers_status.values())

    model_status_text = []
    for provider, is_configured in providers_status.items():
        model_status = "✅" if is_configured else "❌"
        model_status_text.append(f"- {model_status} {provider}  ")

    cell3_out = mo.md(
        cleandoc(
            f"""
            ## 🔑 API Keys Status

            **Configured Services ({configured_count}/5):**

            {chr(10).join(model_status_text)}

            **Required:** At least one LLM provider (OpenAI, Anthropic, or Cohere)  
            **Optional:** Tavily (for search), Langfuse (for observability)  
            """
        )
    )
    output.replace(cell3_out)
    return


@app.cell
def _(cleandoc, mo, output, validate_environment):
    # Run environment validation
    validation_results = validate_environment()

    validation_text = []
    for check, result in validation_results.items():
        if check == "missing_packages":
            continue  # Handle separately
        status = "✅" if result else "❌"
        check_name = check.replace("_", " ").title()
        validation_text.append(f"- {status} {check_name}  ")

    # Handle missing packages
    if (
        "missing_packages" in validation_results
        and validation_results["missing_packages"]
    ):
        missing = ", ".join(validation_results["missing_packages"])
        validation_text.append(f"- ❌ Missing Packages: {missing}")

    cell4_out = mo.md(
        cleandoc(
            f"""
            ## 🔍 Environment Validation

            **System Checks:**  

            {chr(10).join(validation_text)}  
            """
        )
    )
    output.replace(cell4_out)
    return


@app.cell
def _(mo, output):
    # Interactive troubleshooting section
    cell5_issue_selector = mo.ui.dropdown(
        options=[
            "No issues - everything works!",
            "Missing API keys",
            "Package import errors",
            "Configuration problems",
            "Other issue",
        ],
        label="What issue are you experiencing?",
    )
    cell5_description = mo.ui.text_area(
        placeholder="Describe your issue in detail...",
        label="Issue Description (optional)",
    )

    cell5_content = mo.vstack(
        [mo.md("## 🛠️ Troubleshooting"), cell5_issue_selector, cell5_description]
    )

    output.replace(cell5_content)
    return (cell5_issue_selector,)


@app.cell
def _(cell5_issue_selector, cleandoc, mo, output):
    # Provide troubleshooting guidance based on selected issue
    if cell5_issue_selector is not None and cell5_issue_selector.value:
        cell6_issue = cell5_issue_selector.value

        if cell6_issue == "No issues - everything works!":
            guidance = cleandoc(
                """
                🎉 **Excellent!** Your environment is ready.

                **Next Steps:**
                1. Continue to Module 01: DSPy Foundations
                2. Run: `uv run marimo run 01-foundations/signatures_basics.py`
                """
            )

        elif cell6_issue == "Missing API keys":
            guidance = cleandoc(
                """
                🔑 **API Key Setup Help**

                **Steps to fix:**
                1. Edit your `.env` file in the project root
                2. Add at least one of these API keys:
                    - `OPENAI_API_KEY=your_key_here`
                    - `ANTHROPIC_API_KEY=your_key_here`
                    - `COHERE_API_KEY=your_key_here`
                3. Restart this notebook

                **Where to get API keys:**
                    - OpenAI: https://platform.openai.com/api-keys
                    - Anthropic: https://console.anthropic.com/
                    - Cohere: https://dashboard.cohere.ai/api-keys
                """
            )

        elif cell6_issue == "Package import errors":
            guidance = cleandoc(
                """
                📦 **Package Installation Help**

                **Steps to fix:**
                1. Run: `uv sync` to install all dependencies
                2. Check that you're in the correct directory
                3. Verify Python version is 3.11+
                4. Try: `uv run verify_installation.py`

                **If problems persist:**
                    - Delete `.venv` folder and run `uv sync` again
                    - Check for conflicting Python installations
                """
            )

        elif cell6_issue == "Configuration problems":
            guidance = cleandoc(
                """
                ⚙️ **Configuration Help**

                **Steps to fix:**
                1. Run: `uv run 00-setup/setup_environment.py`
                2. Check your `.env` file exists and has correct format
                3. Verify file permissions
                4. Try copying from `.env.template` again

                **Common issues:**
                    - Typos in environment variable names
                    - Missing quotes around API keys with special characters
                    - Wrong file encoding (should be UTF-8)
                """
            )

        else:  # Other issue
            guidance = cleandoc(
                """
                🆘 **General Troubleshooting**

                **Try these steps:**
                1. Run the full installation test: `uv run verify_installation.py`
                2. Check the troubleshooting guide: `docs/TROUBLESHOOTING.md`
                3. Review the setup script output: `uv run 00-setup/setup_environment.py`

                **Get Help:**
                - Check GitHub issues for similar problems
                - Review the documentation in the `docs/` folder
                - Ensure you're using the latest version
                """
            )

        cell6_out = mo.md(f"### 💡 Troubleshooting Guidance\n{guidance}")
    else:
        cell6_out = mo.md("*Select an issue above to get troubleshooting guidance.*")
    output.replace(cell6_out)
    return


@app.cell
def _(mo, output):
    # Exercise completion checklist
    cell7_config_check = mo.ui.checkbox(label="✅ Configuration loaded successfully")
    cell7_api_keys = mo.ui.checkbox(label="✅ At least one LLM provider configured")
    cell7_packages = mo.ui.checkbox(label="✅ All required packages installed")
    cell7_environment = mo.ui.checkbox(label="✅ Environment validation passed")
    cell7_troubleshooting = mo.ui.checkbox(label="✅ Any issues resolved")

    cell7_out = mo.vstack(
        [
            mo.md("## ✅ Exercise Completion Checklist"),
            mo.md("Check off each item as you complete it:"),
            cell7_config_check,
            cell7_api_keys,
            cell7_packages,
            cell7_environment,
            cell7_troubleshooting,
        ]
    )
    output.replace(cell7_out)
    return (
        cell7_api_keys,
        cell7_config_check,
        cell7_environment,
        cell7_packages,
        cell7_troubleshooting,
    )


@app.cell
def _(
    cell7_api_keys,
    cell7_config_check,
    cell7_environment,
    cell7_packages,
    cell7_troubleshooting,
    cleandoc,
    mo,
    output,
):
    # Validate exercise completion
    cell8_checks = [
        cell7_config_check.value,
        cell7_api_keys.value,
        cell7_packages.value,
        cell7_environment.value,
        cell7_troubleshooting.value,
    ]
    cell8_completed_count = sum(cell8_checks)
    cell8_total_checks = len(cell8_checks)

    if cell8_completed_count == cell8_total_checks:
        cell8_out = mo.md(
            cleandoc(
                """
                    ## 🎉 Exercise Complete!

                    **Congratulations!** You've successfully completed the environment setup exercise.

                    **What you've accomplished:**
                    - ✅ Verified your DSPy learning environment
                    - ✅ Confirmed API key configuration
                    - ✅ Validated package installation
                    - ✅ Resolved any setup issues

                    **You're now ready for Module 01!**

                    Run the next module with:
                    ```bash
                    uv run marimo run 01-foundations/signatures_basics.py
                    ```
                    """
            )
        )
    elif cell8_completed_count > 0:
        cell8_out = mo.md(
            cleandoc(
                f"""
                ## 🔄 Progress: {cell8_completed_count}/{cell8_total_checks} Complete

                You're making good progress! Complete the remaining items above before proceeding to Module 01.
                """
            )
        )
    else:
        cell8_out = mo.md(
            "*Complete the checklist items above to finish this exercise.*"
        )
    output.replace(cell8_out)
    return


@app.cell
def _(cleandoc, mo, output):
    cell9_out = mo.md(
        cleandoc(
            """
            ## 📚 Additional Resources

            **If you need more help:**

            - **Setup Script**: `uv run 00-setup/setup_environment.py`
            - **Installation Test**: `uv run verify_installation.py`
            - **Troubleshooting Guide**: `docs/TROUBLESHOOTING.md`
            - **Best Practices**: `docs/BEST_PRACTICES.md`

            **Quick Commands:**  
            - Test installation: `uv run 00-setup/test_installation.py`  
            - Environment test: `uv run marimo run 00-setup/environment_test.py`  
            - Main intro: `uv run marimo run 00-setup/hello_dspy_marimo.py`  
            """
        )
    )
    output.replace(cell9_out)
    return


if __name__ == "__main__":
    app.run()
