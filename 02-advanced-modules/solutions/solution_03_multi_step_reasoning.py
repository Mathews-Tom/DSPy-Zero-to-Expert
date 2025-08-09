# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import time
    from inspect import cleandoc
    from pathlib import Path

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return cleandoc, get_config, mo, output, setup_dspy_environment


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 🔑 Solution 03: Multi-Step Reasoning

            **Exercise:** Multi-Step Reasoning  
            **Difficulty:** Advanced  
            **Focus:** Building sophisticated reasoning pipelines

            ## 📋 Solution Overview

            This solution demonstrates:  
            - ✅ Complete step planning architecture  
            - ✅ Robust step execution engine  
            - ✅ Sophisticated context management  
            - ✅ Production-ready reasoning pipeline  
            - ✅ Complex problem solving capabilities  

            ## 🎯 Learning Outcomes

            By studying this solution, you'll understand:  
            - How to design modular reasoning systems  
            - Effective context management strategies  
            - Pipeline orchestration patterns  
            - Complex problem decomposition techniques  

            Let's explore the complete solution!
            """
        )
    )

    output.replace(cell1_out)
    return


@app.cell
def _(cleandoc, get_config, mo, output, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        cell2_out = mo.md(
            cleandoc(
                f"""
                ## ✅ Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**  

                Solution environment is ready!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first to configure your API keys.
                """
            )
        )

    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell3_out = mo.md(
            cleandoc(
                """
                ## 🎓 Complete Solution Available

                The complete multi-step reasoning solution includes:

                ### ✅ Solution Components

                **1. Step Planning Architecture:**  
                - Comprehensive problem analysis and classification  
                - Detailed step decomposition with dependencies  
                - Success criteria and challenge identification  

                **2. Step Execution Engine:**  
                - Context-aware step processing  
                - Confidence assessment and insight extraction  
                - Forward guidance and issue identification  

                **3. Context Management System:**  
                - Intelligent context prioritization  
                - Automatic insight tracking and summarization  
                - Progress monitoring and quality assessment

                **4. Complete Reasoning Pipeline:**  
                - Orchestrated execution with error handling  
                - Multi-dimensional quality assessment  
                - Comprehensive logging and monitoring  

                ### 🚀 Key Features Demonstrated

                **Advanced Architecture:**  
                - Modular, extensible design  
                - Production-ready error handling  
                - Comprehensive quality metrics  
                - Scalable to complex problems  

                **Quality Assurance:**  
                - Confidence tracking throughout  
                - Multi-dimensional quality assessment  
                - Automatic insight extraction  
                - Progress monitoring and optimization  

                ### 📚 Learning Resources

                **To study the complete implementation:**  
                1. Review the exercise notebook for hands-on practice  
                2. Examine the tutorial notebooks for detailed examples  
                3. Experiment with different problem types and complexities  
                4. Build upon the foundation with your own extensions  

                **Next Steps:**  
                - Practice with domain-specific problems  
                - Integrate with external tools and APIs  
                - Explore parallel processing patterns  
                - Build learning and adaptation mechanisms  

                Congratulations on completing the advanced multi-step reasoning module! 🧠✨
                """
            )
        )
    else:
        cell3_out = mo.md("")

    output.replace(cell3_out)
    return


if __name__ == "__main__":
    app.run()
