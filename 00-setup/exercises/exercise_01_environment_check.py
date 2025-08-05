import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    from pathlib import Path

    import marimo as mo

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from common import get_config, validate_environment

    return Path, get_config, mo, project_root, sys, validate_environment


@app.cell
def __(mo):
    mo.md(
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
    return


@app.cell
def __(get_config, mo):
    # Load and display configuration
    config = get_config()

    mo.md(
        f"""
    ## 📋 Configuration Check
    
    **Current Configuration:**
    - Environment: {config.environment}
    - Debug Mode: {config.debug}
    - Default LLM Provider: {config.default_llm_provider}
    - Default Model: {config.default_model}
    - Cache Enabled: {config.enable_cache}
    - Log Level: {config.log_level}
    """
    )
    return (config,)


@app.cell
def __(config, mo):
    # Check API key configuration
    providers_status = {
        "OpenAI": config.has_openai_config(),
        "Anthropic": config.has_anthropic_config(),
        "Cohere": config.has_cohere_config(),
        "Tavily": config.has_tavily_config(),
        "Langfuse": config.has_langfuse_config(),
    }

    configured_count = sum(providers_status.values())

    status_text = []
    for provider, is_configured in providers_status.items():
        status = "✅" if is_configured else "❌"
        status_text.append(f"- {status} {provider}")

    mo.md(
        f"""
    ## 🔑 API Keys Status
    
    **Configured Services ({configured_count}/5):**
    
    {chr(10).join(status_text)}
    
    **Required:** At least one LLM provider (OpenAI, Anthropic, or Cohere)
    **Optional:** Tavily (for search), Langfuse (for observability)
    """
    )
    return configured_count, providers_status, status_text


@app.cell
def __(mo, validate_environment):
    # Run environment validation
    validation_results = validate_environment()

    validation_text = []
    for check, result in validation_results.items():
        if check == "missing_packages":
            continue  # Handle separately
        status = "✅" if result else "❌"
        check_name = check.replace("_", " ").title()
        validation_text.append(f"- {status} {check_name}")

    # Handle missing packages
    if (
        "missing_packages" in validation_results
        and validation_results["missing_packages"]
    ):
        missing = ", ".join(validation_results["missing_packages"])
        validation_text.append(f"- ❌ Missing Packages: {missing}")

    mo.md(
        f"""
    ## 🔍 Environment Validation
    
    **System Checks:**
    
    {chr(10).join(validation_text)}
    """
    )
    return (
        check,
        check_name,
        missing,
        result,
        status,
        validation_results,
        validation_text,
    )


@app.cell
def __(mo):
    # Interactive troubleshooting section
    troubleshooting_form = mo.ui.form(
        {
            "issue": mo.ui.dropdown(
                options=[
                    "No issues - everything works!",
                    "Missing API keys",
                    "Package import errors",
                    "Configuration problems",
                    "Other issue",
                ],
                label="What issue are you experiencing?",
            ),
            "description": mo.ui.text_area(
                placeholder="Describe your issue in detail...",
                label="Issue Description (optional)",
            ),
        }
    )

    mo.vstack([mo.md("## 🛠️ Troubleshooting"), troubleshooting_form])
    return (troubleshooting_form,)


@app.cell
def __(mo, troubleshooting_form):
    # Provide troubleshooting guidance based on selected issue
    if troubleshooting_form.value:
        issue = troubleshooting_form.value["issue"]

        if issue == "No issues - everything works!":
            guidance = """
            🎉 **Excellent!** Your environment is ready.
            
            **Next Steps:**
            1. Continue to Module 01: DSPy Foundations
            2. Run: `uv run marimo run 01-foundations/signatures_basics.py`
            """

        elif issue == "Missing API keys":
            guidance = """
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

        elif issue == "Package import errors":
            guidance = """
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

        elif issue == "Configuration problems":
            guidance = """
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

        else:  # Other issue
            guidance = """
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

        mo.md(f"### 💡 Troubleshooting Guidance\n{guidance}")
    else:
        mo.md("*Select an issue above to get troubleshooting guidance.*")
    return guidance, issue


@app.cell
def __(mo):
    # Exercise completion checklist
    completion_checklist = mo.ui.form(
        {
            "config_check": mo.ui.checkbox(
                label="✅ Configuration loaded successfully"
            ),
            "api_keys": mo.ui.checkbox(label="✅ At least one LLM provider configured"),
            "packages": mo.ui.checkbox(label="✅ All required packages installed"),
            "environment": mo.ui.checkbox(label="✅ Environment validation passed"),
            "troubleshooting": mo.ui.checkbox(label="✅ Any issues resolved"),
        }
    )

    mo.vstack(
        [
            mo.md("## ✅ Exercise Completion Checklist"),
            mo.md("Check off each item as you complete it:"),
            completion_checklist,
        ]
    )
    return (completion_checklist,)


@app.cell
def __(completion_checklist, mo):
    # Validate exercise completion
    if completion_checklist.value:
        checks = completion_checklist.value
        completed_count = sum(checks.values())
        total_checks = len(checks)

        if completed_count == total_checks:
            mo.md(
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
        elif completed_count > 0:
            mo.md(
                f"""
            ## 🔄 Progress: {completed_count}/{total_checks} Complete
            
            You're making good progress! Complete the remaining items above before proceeding to Module 01.
            """
            )
        else:
            mo.md("*Complete the checklist items above to finish this exercise.*")
    else:
        mo.md("*Complete the checklist items above to finish this exercise.*")
    return checks, completed_count, total_checks


@app.cell
def __(mo):
    mo.md(
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
    return


if __name__ == "__main__":
    app.run()
