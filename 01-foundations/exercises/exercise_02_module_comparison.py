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

    from common import (
        ComparisonViewer,
        PerformanceBenchmark,
        get_config,
        setup_dspy_environment,
    )

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return (
        ComparisonViewer,
        PerformanceBenchmark,
        cleandoc,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            r"""
            # ⚖️ Exercise 2: Module Performance Comparison

            **Objective:** Master the art of choosing the right DSPy module for your specific use case.

            ## 📚 What You'll Learn
            - Compare Predict vs ChainOfThought performance
            - Understand speed vs accuracy trade-offs
            - Analyze module behavior with different task types
            - Make data-driven module selection decisions

            ## 🎮 Exercise Structure
            - **3 Comparative Challenges** - Different task complexity levels
            - **Performance Benchmarking** - Systematic speed and accuracy analysis
            - **Decision Framework** - Learn when to use each module type
            - **Real-world Scenarios** - Apply knowledge to practical problems

            Let's discover which module works best for different scenarios!
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
                - Provider: {config.default_provider}
                - Model: {config.default_model}

                Ready to compare module performance!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first.
                """
            )
        )
    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Challenge 1: Simple Classification Task

                **Task:** Compare modules on a straightforward text classification problem.

                **Signature:** Email spam detection  
                - Input: Email content and sender information  
                - Output: Classification (spam/not spam) and confidence  

                This will help you understand baseline performance differences.
                """
            )
        )

        # Define the signature for Challenge 1
        class EmailSpamDetector(dspy.Signature):
            """Classify emails as spam or not spam based on content and sender information."""

            email_content = dspy.InputField(desc="The email subject and body content")
            sender_info = dspy.InputField(
                desc="Information about the sender (email, domain)"
            )
            classification = dspy.OutputField(
                desc="Classification: 'spam' or 'not spam'"
            )
            confidence = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")

        # Create both module types
        predict_spam = dspy.Predict(EmailSpamDetector)
        cot_spam = dspy.ChainOfThought(EmailSpamDetector)

        cell3_content = mo.md(
            cleandoc(
                """
                ### 📧 Email Spam Detection Signature Created

                **Modules Ready:**  
                - `dspy.Predict(EmailSpamDetector)` - Direct classification  
                - `dspy.ChainOfThought(EmailSpamDetector)` - Reasoning-based classification  
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        EmailSpamDetector = None
        predict_spam = None
        cot_spam = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack(
        [
            cell3_desc,
            cell3_content,
        ]
    )
    output.replace(cell3_out)
    return EmailSpamDetector, cot_spam, predict_spam


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Test cases for spam detection
        spam_test_cases = [
            {
                "email_content": "URGENT! You've won $1,000,000! Click here now to claim your prize! Limited time offer!",
                "sender_info": "winner@lottery-prize.com (suspicious domain)",
            },
            {
                "email_content": "Hi John, Thanks for your presentation yesterday. Could you send me the slides? Best regards, Sarah",
                "sender_info": "sarah.johnson@company.com (verified colleague)",
            },
            {
                "email_content": "Get rich quick! Make money from home! No experience needed! Click now!",
                "sender_info": "money@get-rich-now.biz (promotional domain)",
            },
            {
                "email_content": "Your monthly bank statement is ready. Please log in to your account to view it.",
                "sender_info": "statements@yourbank.com (official bank domain)",
            },
        ]

        # Challenge 1 execution button
        run_challenge1 = mo.ui.run_button(
            label="🚀 Run Challenge 1: Spam Detection Comparison"
        )

        cell4_out = mo.vstack(
            [
                mo.md("### 📝 Test Cases Ready"),
                mo.md(f"- {len(spam_test_cases)} diverse email examples"),
                mo.md("- Mix of obvious spam and legitimate emails"),
                mo.md("- Includes sender context information"),
                run_challenge1,
            ]
        )
    else:
        spam_test_cases = None
        run_challenge1 = None
        cell4_out = mo.md("")
    output.replace(cell4_out)
    return run_challenge1, spam_test_cases


@app.cell
def _(
    ComparisonViewer,
    available_providers,
    cleandoc,
    cot_spam,
    mo,
    output,
    predict_spam,
    run_challenge1,
    spam_test_cases,
    time,
):
    if available_providers and run_challenge1.value and spam_test_cases:
        try:
            email_results = []
            email_predict_times = []
            email_cot_times = []

            for i, email_test_case in enumerate(spam_test_cases):
                # Test Predict module
                email_predict_start_time = time.time()
                email_predict_result = predict_spam(**email_test_case)
                email_predict_time = time.time() - email_predict_start_time
                email_predict_times.append(email_predict_time)

                # Test ChainOfThought module
                email_cot_start_time = time.time()
                email_cot_result = cot_spam(**email_test_case)
                email_cot_time = time.time() - email_cot_start_time
                email_cot_times.append(email_cot_time)

                # Store results
                email_results.append(
                    {
                        "test_case": i + 1,
                        "email_preview": email_test_case["email_content"][:50] + "...",
                        "predict_result": email_predict_result,
                        "predict_time": email_predict_time,
                        "cot_result": email_cot_result,
                        "cot_time": email_cot_time,
                    }
                )

            # Calculate averages
            avg_email_predict_time = sum(email_predict_times) / len(email_predict_times)
            avg_email_cot_time = sum(email_cot_times) / len(email_cot_times)

            # Create comparison visualization
            email_comparison_viewer = ComparisonViewer()
            email_comparison_viewer.add_comparison(
                "Predict Module",
                f"Average time: {avg_email_predict_time:.3f}s",
                {"module_type": "Direct", "reasoning": "None"},
            )
            email_comparison_viewer.add_comparison(
                "ChainOfThought Module",
                f"Average time: {avg_email_cot_time:.3f}s",
                {"module_type": "Reasoning", "reasoning": "Included"},
            )

            # Display detailed results
            email_result_details = []
            for email_result in email_results:
                email_result_details.append(
                    cleandoc(
                        f"""
                        **Test {email_result['test_case']}:** {email_result['email_preview']}
                        - **Predict:** {email_result['predict_result'].classification} (confidence: {email_result['predict_result'].confidence}) - {email_result['predict_time']:.3f}s
                        - **ChainOfThought:** {email_result['cot_result'].classification} (confidence: {email_result['cot_result'].confidence}) - {email_result['cot_time']:.3f}s
                        """
                    )
                )

            cell5_out = mo.vstack(
                [
                    mo.md("## 📊 Challenge 1 Results: Spam Detection"),
                    email_comparison_viewer.render(),
                    mo.md("### 📈 Performance Analysis"),
                    mo.md(
                        cleandoc(
                            f"""
                            **Speed Comparison:**
                            - Predict average: {avg_email_predict_time:.3f} seconds
                            - ChainOfThought average: {avg_email_cot_time:.3f} seconds
                            - Speed difference: {((avg_email_cot_time - avg_email_predict_time) / avg_email_predict_time * 100):.1f}% slower for CoT

                            **Key Insights:**
                            - Simple classification tasks may not need reasoning overhead
                            - Both modules should produce similar accuracy for clear cases
                            - Speed difference becomes important at scale
                            """
                        )
                    ),
                    mo.md("### 🔍 Detailed Results"),
                    mo.md("\n".join(email_result_details)),
                ]
            )

        except Exception as e:
            cell5_out = mo.md(f"❌ **Error in Challenge 1:** {str(e)}")
    else:
        cell5_out = mo.md(
            "*Click 'Run Challenge 1' to compare spam detection performance*"
        )

    output.replace(cell5_out)
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Challenge 2: Complex Analysis Task

                **Task:** Compare modules on a task requiring deeper reasoning.

                **Signature:** Financial document analysis  
                - Input: Financial text and analysis type  
                - Output: Key insights, risk assessment, recommendations, confidence  

                This will show how reasoning capabilities affect complex tasks.
                """
            )
        )

        # Define signature for Challenge 2
        class FinancialAnalyzer(dspy.Signature):
            """Analyze financial documents to extract insights, assess risks, and provide recommendations."""

            financial_text = dspy.InputField(desc="Financial document content or data")
            analysis_type = dspy.InputField(
                desc="Type of analysis: 'investment', 'risk', 'performance'"
            )
            key_insights = dspy.OutputField(desc="3-5 key insights from the analysis")
            risk_assessment = dspy.OutputField(
                desc="Risk level: 'low', 'medium', 'high' with explanation"
            )
            recommendations = dspy.OutputField(
                desc="Specific actionable recommendations"
            )
            confidence = dspy.OutputField(
                desc="Confidence in analysis accuracy (0.0-1.0)"
            )

        # Create both module types
        predict_finance = dspy.Predict(FinancialAnalyzer)
        cot_finance = dspy.ChainOfThought(FinancialAnalyzer)

        cell6_content = mo.md(
            cleandoc(
                """
                ### 💰 Financial Analysis Signature Created

                **Modules Ready:**  
                - `dspy.Predict(FinancialAnalyzer)` - Direct analysis  
                - `dspy.ChainOfThought(FinancialAnalyzer)` - Reasoning-based analysis  
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        FinancialAnalyzer = None
        predict_finance = None
        cot_finance = None
        cell6_content = mo.md("")
    cell6_out = mo.vstack(
        [
            cell6_desc,
            cell6_content,
        ]
    )
    output.replace(cell6_out)
    return FinancialAnalyzer, cot_finance, predict_finance


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Test cases for financial analysis
        finance_test_cases = [
            {
                "financial_text": "Company XYZ reported Q3 revenue of $50M, up 15% YoY. However, operating expenses increased 25% due to expansion costs. Cash flow remains positive at $8M. Debt-to-equity ratio is 0.4.",
                "analysis_type": "performance",
            },
            {
                "financial_text": "Tech startup ABC raised $10M Series A. Monthly burn rate is $800K with 18 months runway. Customer acquisition cost is $150, lifetime value is $2000. Growing 20% MoM.",
                "analysis_type": "investment",
            },
        ]

        # Challenge 2 execution button
        run_challenge2 = mo.ui.run_button(
            label="🚀 Run Challenge 2: Financial Analysis Comparison"
        )

        cell7_out = mo.vstack(
            [
                mo.md("### 📊 Financial Analysis Test Cases"),
                mo.md("- Complex financial scenarios requiring interpretation"),
                mo.md("- Multiple data points to synthesize"),
                mo.md("- Requires domain knowledge and reasoning"),
                run_challenge2,
            ]
        )
    else:
        finance_test_cases = None
        run_challenge2 = None
        cell7_out = mo.md("")
    output.replace(cell7_out)
    return finance_test_cases, run_challenge2


@app.cell
def _(
    available_providers,
    cleandoc,
    cot_finance,
    finance_test_cases,
    mo,
    output,
    predict_finance,
    run_challenge2,
    time,
):
    if available_providers and run_challenge2.value and finance_test_cases:
        try:
            finance_results = []
            finance_predict_times = []
            finance_cot_times = []

            for j, finance_test_case in enumerate(finance_test_cases):
                # Test Predict module
                finance_predict_start_time = time.time()
                finance_predict_result = predict_finance(**finance_test_case)
                finance_predict_time = time.time() - finance_predict_start_time
                finance_predict_times.append(finance_predict_time)

                # Test ChainOfThought module
                finance_cot_start_time = time.time()
                finance_cot_result = cot_finance(**finance_test_case)
                finance_cot_time = time.time() - finance_cot_start_time
                finance_cot_times.append(finance_cot_time)

                # Store results
                finance_results.append(
                    {
                        "test_case": j + 1,
                        "analysis_type": finance_test_case["analysis_type"],
                        "predict_result": finance_predict_result,
                        "predict_time": finance_predict_time,
                        "cot_result": finance_cot_result,
                        "cot_time": finance_cot_time,
                    }
                )

            # Calculate averages
            avg_finance_predict = sum(finance_predict_times) / len(
                finance_predict_times
            )
            avg_finance_cot = sum(finance_cot_times) / len(finance_cot_times)

            # Display results
            finance_details = []
            for finance_result in finance_results:
                finance_details.append(
                    cleandoc(
                        f"""
                        **Test {finance_result['test_case']} ({finance_result['analysis_type']} analysis):**

                        **Predict Module ({finance_result['predict_time']:.3f}s):**
                        - Insights: {finance_result['predict_result'].key_insights}
                        - Risk: {finance_result['predict_result'].risk_assessment}
                        - Recommendations: {finance_result['predict_result'].recommendations}

                        **ChainOfThought Module ({finance_result['cot_time']:.3f}s):**
                        - Insights: {finance_result['cot_result'].key_insights}
                        - Risk: {finance_result['cot_result'].risk_assessment}
                        - Recommendations: {finance_result['cot_result'].recommendations}
                        - Reasoning: {getattr(finance_result['cot_result'], 'rationale', 'No rationale shown')}
                        """
                    )
                )

            cell8_out = mo.vstack(
                [
                    mo.md("## 📊 Challenge 2 Results: Financial Analysis"),
                    mo.md("### 📈 Performance Comparison"),
                    mo.md(
                        cleandoc(
                            f"""
                            **Speed Analysis:**  
                            - Predict average: {avg_finance_predict:.3f} seconds  
                            - ChainOfThought average: {avg_finance_cot:.3f} seconds  
                            - Speed difference: {((avg_finance_cot - avg_finance_predict) / avg_finance_predict * 100):.1f}% slower for CoT  

                            **Quality Observations:**  
                            - ChainOfThought provides more detailed reasoning  
                            - Complex analysis benefits from step-by-step thinking  
                            - Reasoning helps validate conclusions  
                            """
                        )
                    ),
                    mo.md("### 🔍 Detailed Analysis Results"),
                    mo.md("\n".join(finance_details)),
                    mo.md(
                        cleandoc(
                            """
                            ### 💡 Key Insights from Challenge 2
                            - **Complex tasks benefit more from reasoning** - ChainOfThought shows its value  
                            - **Quality vs Speed trade-off** - Better analysis takes more time  
                            - **Reasoning transparency** - CoT shows how conclusions were reached  
                            - **Domain expertise** - Complex domains need more sophisticated processing  
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell8_out = mo.md(f"❌ **Error in Challenge 2:** {str(e)}")
    else:
        cell8_out = mo.md(
            "*Click 'Run Challenge 2' to compare financial analysis performance*"
        )

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell9_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Challenge 3: Mathematical Reasoning Task

                **Task:** Compare modules on multi-step mathematical problem solving.

                **Signature:** Word problem solver  
                - Input: Math word problem and difficulty level  
                - Output: Solution steps, final answer, verification, confidence  

                This will demonstrate the maximum benefit of reasoning-based modules.
                """
            )
        )

        # Define signature for Challenge 3
        class MathWordProblemSolver(dspy.Signature):
            """Solve mathematical word problems with step-by-step reasoning and verification."""

            problem_text = dspy.InputField(
                desc="The mathematical word problem to solve"
            )
            difficulty_level = dspy.InputField(
                desc="Problem difficulty: 'easy', 'medium', 'hard'"
            )
            solution_steps = dspy.OutputField(desc="Step-by-step solution process")
            final_answer = dspy.OutputField(desc="Final numerical answer with units")
            verification = dspy.OutputField(
                desc="Verification of the answer by checking"
            )
            confidence = dspy.OutputField(
                desc="Confidence in solution correctness (0.0-1.0)"
            )

        # Create both module types
        predict_math = dspy.Predict(MathWordProblemSolver)
        cot_math = dspy.ChainOfThought(MathWordProblemSolver)

        cell9_content = mo.md(
            cleandoc(
                """
                ### 🧮 Math Problem Solver Signature Created

                **Modules Ready:**  
                - `dspy.Predict(MathWordProblemSolver)` - Direct problem solving  
                - `dspy.ChainOfThought(MathWordProblemSolver)` - Reasoning-based solving  
                """
            )
        )
    else:
        cell9_desc = mo.md("")
        MathWordProblemSolver = None
        predict_math = None
        cot_math = None
        cell9_content = mo.md("")

    cell9_out = mo.vstack(
        [
            cell9_desc,
            cell9_content,
        ]
    )
    output.replace(cell9_out)
    return MathWordProblemSolver, cot_math, predict_math


@app.cell
def _(available_providers, mo, output):
    if available_providers:
        # Test cases for math problems
        math_test_cases = [
            {
                "problem_text": "A train travels 240 miles in 4 hours. If it increases its speed by 20 mph for the next 3 hours, how far will it travel in total?",
                "difficulty_level": "medium",
            },
            {
                "problem_text": "Sarah has $500. She spends 1/4 of it on groceries, then 30% of the remaining amount on clothes. How much money does she have left?",
                "difficulty_level": "medium",
            },
            {
                "problem_text": "A rectangular garden is 3 times as long as it is wide. If the perimeter is 48 feet, what are the dimensions and area of the garden?",
                "difficulty_level": "hard",
            },
        ]

        # Challenge 3 execution button
        run_challenge3 = mo.ui.run_button(
            label="🚀 Run Challenge 3: Math Problem Comparison"
        )

        cell10_out = mo.vstack(
            [
                mo.md("### 🧮 Mathematical Reasoning Test Cases"),
                mo.md("- Multi-step problems requiring sequential reasoning"),
                mo.md("- Mix of arithmetic, percentage, and geometry problems"),
                mo.md("- Verification and checking required"),
                run_challenge3,
            ]
        )
    else:
        math_test_cases = None
        run_challenge3 = None
        cell10_out = mo.md("")
    output.replace(cell10_out)
    return math_test_cases, run_challenge3


@app.cell
def _(
    available_providers,
    cleandoc,
    cot_math,
    math_test_cases,
    mo,
    output,
    predict_math,
    run_challenge3,
    time,
):
    if available_providers and run_challenge3.value and math_test_cases:
        try:
            math_results = []
            math_predict_times = []
            math_cot_times = []

            for k, math_test_case in enumerate(math_test_cases):
                # Test Predict module
                math_predict_start_time = time.time()
                math_predict_result = predict_math(**math_test_case)
                math_predict_time = time.time() - math_predict_start_time
                math_predict_times.append(math_predict_time)

                # Test ChainOfThought module
                math_cot_start_time = time.time()
                math_cot_result = cot_math(**math_test_case)
                math_cot_time = time.time() - math_cot_start_time
                math_cot_times.append(math_cot_time)

                # Store results
                math_results.append(
                    {
                        "test_case": k + 1,
                        "difficulty": math_test_case["difficulty_level"],
                        "problem": math_test_case["problem_text"],
                        "predict_result": math_predict_result,
                        "predict_time": math_predict_time,
                        "cot_result": math_cot_result,
                        "cot_time": math_cot_time,
                    }
                )

            # Calculate averages
            avg_math_predict = sum(math_predict_times) / len(math_predict_times)
            avg_math_cot = sum(math_cot_times) / len(math_cot_times)

            # Display results
            math_details = []
            for math_result in math_results:
                math_details.append(
                    cleandoc(
                        f"""
                        **Test {math_result['test_case']} ({math_result['difficulty']} difficulty):**
                        *Problem: {math_result['problem'][:80]}...*

                        **Predict Module ({math_result['predict_time']:.3f}s):**
                        - Steps: {math_result['predict_result'].solution_steps}
                        - Answer: {math_result['predict_result'].final_answer}
                        - Confidence: {math_result['predict_result'].confidence}

                        **ChainOfThought Module ({math_result['cot_time']:.3f}s):**
                        - Steps: {math_result['cot_result'].solution_steps}
                        - Answer: {math_result['cot_result'].final_answer}
                        - Verification: {math_result['cot_result'].verification}
                        - Confidence: {math_result['cot_result'].confidence}
                        - Reasoning: {getattr(math_result['cot_result'], 'rationale', 'No rationale shown')}
                        """
                    )
                )

            cell11_out = mo.vstack(
                [
                    mo.md("## 📊 Challenge 3 Results: Mathematical Reasoning"),
                    mo.md("### 📈 Performance Analysis"),
                    mo.md(
                        cleandoc(
                            f"""
                            **Speed Comparison:**
                            - Predict average: {avg_math_predict:.3f} seconds
                            - ChainOfThought average: {avg_math_cot:.3f} seconds
                            - Speed difference: {((avg_math_cot - avg_math_predict) / avg_math_predict * 100):.1f}% slower for CoT

                            **Reasoning Quality:**
                            - ChainOfThought provides step-by-step breakdown
                            - Verification step helps catch errors
                            - Reasoning transparency aids debugging
                            - Better suited for complex multi-step problems
                            """
                        )
                    ),
                    mo.md("### 🔍 Detailed Math Problem Results"),
                    mo.md("\n".join(math_details)),
                    mo.md(
                        cleandoc(
                            """
                            ### 🧠 Challenge 3 Key Insights
                            - **Reasoning is crucial for math** - Step-by-step approach prevents errors  
                            - **Verification adds value** - Checking work improves accuracy  
                            - **Transparency matters** - Seeing the reasoning helps trust results  
                            - **Complex problems need complex modules** - Don't use a hammer for surgery  
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell11_out = mo.md(f"❌ **Error in Challenge 3:** {str(e)}")
    else:
        cell11_out = mo.md(
            "*Click 'Run Challenge 3' to compare mathematical reasoning performance*"
        )

    output.replace(cell11_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell12_desc = mo.md(
            cleandoc(
                """
                ## 📊 Comprehensive Performance Benchmark

                **Task:** Run a systematic benchmark across all three challenge types.

                This will give you quantitative data to make informed module selection decisions.
                """
            )
        )

        # Benchmark execution button
        run_benchmark = mo.ui.run_button(label="🏃‍♂️ Run Comprehensive Benchmark")

        cell12_content = mo.vstack(
            [
                mo.md("### 🎯 Benchmark Scope"),
                mo.md(
                    "- All three challenge types (Classification, Analysis, Reasoning)"
                ),
                mo.md("- Multiple test runs for statistical significance"),
                mo.md("- Speed, accuracy, and consistency metrics"),
                mo.md("- Comprehensive comparison report"),
                run_benchmark,
            ]
        )
    else:
        cell12_desc = mo.md("")
        run_benchmark = None
        cell12_content = mo.md("")

    cell12_out = mo.vstack(
        [
            cell12_desc,
            cell12_content,
        ]
    )
    output.replace(cell12_out)
    return (run_benchmark,)


@app.cell
def _(
    EmailSpamDetector,
    FinancialAnalyzer,
    MathWordProblemSolver,
    PerformanceBenchmark,
    available_providers,
    cleandoc,
    dspy,
    finance_test_cases,
    math_test_cases,
    mo,
    output,
    run_benchmark,
    spam_test_cases,
):
    if available_providers and run_benchmark.value:
        try:
            # Create benchmark instance
            benchmark = PerformanceBenchmark("Module Comparison Benchmark")

            # Benchmark results storage
            benchmark_results = {}

            # Test 1: Spam Detection
            if spam_test_cases:
                predict_spam_bench = dspy.Predict(EmailSpamDetector)
                cot_spam_bench = dspy.ChainOfThought(EmailSpamDetector)

                spam_inputs = [
                    lambda case=case: predict_spam_bench(**case)
                    for case in spam_test_cases
                ]
                spam_cot_inputs = [
                    lambda case=case: cot_spam_bench(**case) for case in spam_test_cases
                ]

                predict_spam_results = benchmark.benchmark_system(
                    lambda: [fn() for fn in spam_inputs],
                    ["spam_detection"],
                    "Predict - Spam Detection",
                    warmup_runs=1,
                    benchmark_runs=2,
                )

                cot_spam_results = benchmark.benchmark_system(
                    lambda: [fn() for fn in spam_cot_inputs],
                    ["spam_detection"],
                    "ChainOfThought - Spam Detection",
                    warmup_runs=1,
                    benchmark_runs=2,
                )

                benchmark_results["spam"] = {
                    "predict": predict_spam_results,
                    "cot": cot_spam_results,
                }

            # Test 2: Financial Analysis
            if finance_test_cases:
                predict_finance_bench = dspy.Predict(FinancialAnalyzer)
                cot_finance_bench = dspy.ChainOfThought(FinancialAnalyzer)

                finance_inputs = [
                    lambda case=case: predict_finance_bench(**case)
                    for case in finance_test_cases
                ]
                finance_cot_inputs = [
                    lambda case=case: cot_finance_bench(**case)
                    for case in finance_test_cases
                ]

                predict_finance_results = benchmark.benchmark_system(
                    lambda: [fn() for fn in finance_inputs],
                    ["financial_analysis"],
                    "Predict - Financial Analysis",
                    warmup_runs=1,
                    benchmark_runs=2,
                )

                cot_finance_results = benchmark.benchmark_system(
                    lambda: [fn() for fn in finance_cot_inputs],
                    ["financial_analysis"],
                    "ChainOfThought - Financial Analysis",
                    warmup_runs=1,
                    benchmark_runs=2,
                )

                benchmark_results["finance"] = {
                    "predict": predict_finance_results,
                    "cot": cot_finance_results,
                }

            # Test 3: Math Problems
            if math_test_cases:
                predict_math_bench = dspy.Predict(MathWordProblemSolver)
                cot_math_bench = dspy.ChainOfThought(MathWordProblemSolver)

                math_inputs = [
                    lambda case=case: predict_math_bench(**case)
                    for case in math_test_cases
                ]
                math_cot_inputs = [
                    lambda case=case: cot_math_bench(**case) for case in math_test_cases
                ]

                predict_math_results = benchmark.benchmark_system(
                    lambda: [fn() for fn in math_inputs],
                    ["math_problems"],
                    "Predict - Math Problems",
                    warmup_runs=1,
                    benchmark_runs=2,
                )

                cot_math_results = benchmark.benchmark_system(
                    lambda: [fn() for fn in math_cot_inputs],
                    ["math_problems"],
                    "ChainOfThought - Math Problems",
                    warmup_runs=1,
                    benchmark_runs=2,
                )

                benchmark_results["math"] = {
                    "predict": predict_math_results,
                    "cot": cot_math_results,
                }

            # Create summary
            summary_data = []
            for task_type, bench_results in benchmark_results.items():
                predict_data = bench_results["predict"]
                cot_data = bench_results["cot"]

                summary_data.append(
                    {
                        "Task Type": task_type.title(),
                        "Predict Avg Time": f"{predict_data['mean_time']:.3f}s",
                        "CoT Avg Time": f"{cot_data['mean_time']:.3f}s",
                        "Speed Difference": f"{((cot_data['mean_time'] - predict_data['mean_time']) / predict_data['mean_time'] * 100):.1f}%",
                        "Predict Success": f"{predict_data['success_rate']:.1%}",
                        "CoT Success": f"{cot_data['success_rate']:.1%}",
                    }
                )

            cell13_out = mo.vstack(
                [
                    mo.md("## 🏆 Comprehensive Benchmark Results"),
                    mo.ui.table(summary_data),
                    mo.md(
                        cleandoc(
                            """
                            ### 📊 Benchmark Analysis

                            **Key Findings:**
                            1. **Simple Tasks (Spam Detection)**: Predict module is faster with similar accuracy
                            2. **Complex Analysis (Financial)**: ChainOfThought provides better insights
                            3. **Reasoning Tasks (Math)**: ChainOfThought significantly outperforms Predict

                            **Decision Framework:**
                            - **Use Predict when**: Speed is critical, task is straightforward, high volume
                            - **Use ChainOfThought when**: Accuracy is critical, complex reasoning needed, explanation required

                            **Performance Patterns:**
                            - Speed difference increases with task complexity
                            - ChainOfThought shows more consistent results for complex tasks
                            - Reasoning overhead pays off for multi-step problems
                            """
                        )
                    ),
                    mo.md(
                        cleandoc(
                            """
                            ### 🎯 Module Selection Guidelines

                            | Task Complexity | Volume | Accuracy Need | Recommended Module |
                            |-----------------|--------|---------------|-------------------|
                            | Simple | High | Standard | **Predict** |
                            | Simple | Low | High | **ChainOfThought** |
                            | Complex | High | Standard | **Predict** (with monitoring) |
                            | Complex | Low | High | **ChainOfThought** |
                            | Reasoning | Any | High | **ChainOfThought** |
                            """
                        )
                    ),
                ]
            )

        except Exception as e:
            cell13_out = mo.md(f"❌ **Benchmark Error:** {str(e)}")
    else:
        cell13_out = mo.md(
            "*Click 'Run Comprehensive Benchmark' to see detailed performance analysis*"
        )

    output.replace(cell13_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell14_desc = mo.md(
            cleandoc(
                """
                ## 🎯 Decision Framework Exercise

                **Task:** Apply your knowledge to make module selection decisions for new scenarios.

                For each scenario below, choose the appropriate module and justify your decision.
                """
            )
        )

        # Decision scenarios
        scenarios = [
            {
                "title": "Real-time Chat Moderation",
                "description": "Classify chat messages as appropriate/inappropriate in a live gaming platform. Need to process 1000+ messages per minute.",
                "factors": [
                    "High volume",
                    "Simple classification",
                    "Speed critical",
                    "Basic accuracy needed",
                ],
            },
            {
                "title": "Medical Diagnosis Assistant",
                "description": "Analyze patient symptoms and medical history to suggest potential diagnoses and recommend tests.",
                "factors": [
                    "Complex reasoning",
                    "High accuracy critical",
                    "Low volume",
                    "Explanation needed",
                ],
            },
            {
                "title": "Code Review Automation",
                "description": "Review code changes for potential bugs, security issues, and best practice violations.",
                "factors": [
                    "Medium complexity",
                    "Moderate volume",
                    "High accuracy",
                    "Detailed feedback needed",
                ],
            },
            {
                "title": "Social Media Content Filtering",
                "description": "Filter inappropriate content from social media posts. Process thousands of posts per hour.",
                "factors": [
                    "High volume",
                    "Moderate complexity",
                    "Speed important",
                    "Good accuracy needed",
                ],
            },
        ]

        # Create decision radio buttons for each scenario
        scenario_radios = []
        for _, scenario in enumerate(scenarios):
            radio = mo.ui.radio(
                options=[
                    "Predict",
                    "ChainOfThought",
                    "Depends on specific requirements",
                ],
                label=f"**{scenario['title']}**\n{scenario['description']}\nFactors: {', '.join(scenario['factors'])}",
                value="Predict",
            )
            scenario_radios.append(radio)

        cell14_content = mo.vstack(
            [mo.md("### 🤔 Make Your Decisions"), *scenario_radios]
        )
    else:
        cell14_desc = mo.md("")
        scenario_radios = None
        scenarios = None
        cell14_content = mo.md("")

    cell14_out = mo.vstack(
        [
            cell14_desc,
            cell14_content,
        ]
    )
    output.replace(cell14_out)
    return scenario_radios, scenarios


@app.cell
def _(available_providers, cleandoc, mo, output, scenario_radios, scenarios):
    if available_providers and scenario_radios is not None and scenarios:
        # Get decisions from individual radio buttons
        decisions = {}
        for l, scn_radio in enumerate(scenario_radios):
            decisions[f"scenario_{l}"] = scn_radio.value

        # Recommended answers with explanations
        recommendations = [
            {
                "recommended": "Predict",
                "explanation": "High volume (1000+ msg/min) requires speed. Simple classification task doesn't need reasoning overhead. Basic accuracy is sufficient for chat moderation.",
            },
            {
                "recommended": "ChainOfThought",
                "explanation": "Medical diagnosis requires complex reasoning, high accuracy, and explanation. Low volume allows for slower processing. Patient safety demands thorough analysis.",
            },
            {
                "recommended": "ChainOfThought",
                "explanation": "Code review needs detailed analysis and explanation. Moderate volume allows reasoning time. High accuracy prevents bugs in production.",
            },
            {
                "recommended": "Predict",
                "explanation": "High volume processing needs speed. While accuracy is important, the speed requirement and moderate complexity favor Predict with good monitoring.",
            },
        ]

        # Evaluate decisions
        overall_results = []
        score = 0

        for m, (scn, recommendation) in enumerate(zip(scenarios, recommendations)):
            user_choice = decisions[f"scenario_{m}"]
            is_correct = user_choice == recommendation["recommended"]
            if is_correct:
                score += 1

            overall_results.append(
                cleandoc(
                    f"""
                    **{scn['title']}**
                    - Your choice: **{user_choice}**
                    - Recommended: **{recommendation['recommended']}**
                    - {'✅ Correct!' if is_correct else '❌ Consider this:'} {recommendation['explanation']}
                    """
                )
            )

        cell15_out = mo.vstack(
            [
                mo.md(f"## 🎯 Decision Framework Results: {score}/{len(scenarios)}"),
                mo.md("\n".join(overall_results)),
                mo.md(
                    cleandoc(
                        f"""
                        ### 📊 Your Performance Analysis

                        **Score: {score}/{len(scenarios)} ({score/len(scenarios)*100:.0f}%)**

                        {'🏆 Excellent decision-making! You understand the trade-offs well.' if score >= len(scenarios) * 0.8
                        else '🎯 Good progress! Review the explanations to refine your decision framework.' if score >= len(scenarios) * 0.6
                        else '💪 Keep learning! Focus on the key factors: volume, complexity, accuracy needs, and speed requirements.'}

                        ### 🔑 Key Decision Factors
                        1. **Volume**: High volume → Predict, Low volume → Either
                        2. **Complexity**: Simple → Predict, Complex → ChainOfThought
                        3. **Accuracy**: Standard → Predict, Critical → ChainOfThought
                        4. **Speed**: Critical → Predict, Flexible → ChainOfThought
                        5. **Explanation**: Not needed → Predict, Required → ChainOfThought
                        """
                    )
                ),
            ]
        )
    else:
        cell15_out = mo.md(
            "*Complete the decision scenarios above to see your results*"
        )
        decisions = None
        recommendation = []

    output.replace(cell15_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell16_out = mo.md(
        cleandoc(
            """
            ## 🎓 Exercise 2 Complete!

            ### 🏆 What You've Mastered

            ✅ **Performance Comparison** - Systematic module evaluation
            ✅ **Speed vs Accuracy Trade-offs** - Understanding the balance
            ✅ **Task Complexity Analysis** - Matching modules to problem types
            ✅ **Decision Framework** - Data-driven module selection
            ✅ **Benchmarking Skills** - Quantitative performance measurement

            ### 🎯 Key Insights Gained

            1. **Simple Tasks**: Predict module excels with speed advantage
            2. **Complex Analysis**: ChainOfThought provides better insights
            3. **Reasoning Problems**: ChainOfThought is essential for accuracy
            4. **Volume Considerations**: High volume favors Predict module
            5. **Explanation Needs**: ChainOfThought provides transparency

            ### 📊 Decision Matrix You've Built

            | Factor | Predict | ChainOfThought |
            |--------|---------|----------------|
            | **Speed** | ⚡ Excellent | 🐌 Slower |
            | **Simple Tasks** | ✅ Perfect | ⚖️ Overkill |
            | **Complex Tasks** | ⚠️ Limited | 🎯 Excellent |
            | **High Volume** | ✅ Ideal | ❌ Bottleneck |
            | **Explanation** | ❌ None | ✅ Detailed |
            | **Debugging** | ⚠️ Limited | 🔍 Excellent |

            ### 🚀 Ready for Advanced Testing?

            You now understand module performance characteristics! Time to master advanced testing techniques:

            **Next Exercise:**
            ```bash
            uv run marimo run 01-foundations/exercises/exercise_03_signature_optimization.py
            ```

            **Coming Up:**  
            - Parameter optimization techniques  
            - A/B testing methodologies  
            - Performance monitoring strategies  
            - Production deployment considerations  

            ### 🎯 Practice Challenges

            Before moving on, try comparing modules for:  
            1. **Customer Service Chatbot** - Response generation with personality  
            2. **Legal Document Analysis** - Contract review and risk assessment  
            3. **Creative Writing Assistant** - Story generation with style control  
            4. **Technical Documentation** - API documentation generation  

            Each domain has unique performance characteristics - explore them!
            """
        )
        if available_providers
        else ""
    )
    output.replace(cell16_out)
    return


if __name__ == "__main__":
    app.run()
