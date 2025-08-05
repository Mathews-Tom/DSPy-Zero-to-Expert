import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    import time
    from pathlib import Path

    import dspy
    import marimo as mo

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from common import (
        ComparisonViewer,
        PerformanceBenchmark,
        get_config,
        setup_dspy_environment,
    )

    return (
        ComparisonViewer,
        Path,
        PerformanceBenchmark,
        dspy,
        get_config,
        mo,
        project_root,
        setup_dspy_environment,
        sys,
        time,
    )


@app.cell
def __(mo):
    mo.md(
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
    return


@app.cell
def __(get_config, mo, setup_dspy_environment):
    # Setup DSPy environment
    config = get_config()
    available_providers = config.get_available_llm_providers()

    if available_providers:
        setup_dspy_environment()
        mo.md(
            f"""
        ## ✅ Environment Ready
        
        **Configuration:**
        - Provider: {config.default_llm_provider}
        - Model: {config.default_model}
        
        Ready to compare module performance!
        """
        )
    else:
        mo.md(
            """
        ## ⚠️ Setup Required
        
        Please complete Module 00 setup first.
        """
        )
    return available_providers, config


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 1: Simple Classification Task
        
        **Task:** Compare modules on a straightforward text classification problem.
        
        **Signature:** Email spam detection
        - Input: Email content and sender information
        - Output: Classification (spam/not spam) and confidence
        
        This will help you understand baseline performance differences.
        """
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

        mo.md(
            """
        ### 📧 Email Spam Detection Signature Created
        
        **Modules Ready:**
        - `dspy.Predict(EmailSpamDetector)` - Direct classification
        - `dspy.ChainOfThought(EmailSpamDetector)` - Reasoning-based classification
        """
        )
    else:
        EmailSpamDetector = None
        predict_spam = None
        cot_spam = None
    return EmailSpamDetector, cot_spam, predict_spam


@app.cell
def __(available_providers, mo):
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
        run_challenge1 = mo.ui.button(
            label="🚀 Run Challenge 1: Spam Detection Comparison"
        )

        mo.vstack(
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
    return run_challenge1, spam_test_cases


@app.cell
def __(
    ComparisonViewer,
    available_providers,
    cot_spam,
    mo,
    predict_spam,
    run_challenge1,
    spam_test_cases,
    time,
):
    if available_providers and run_challenge1.value and spam_test_cases:
        try:
            results = []
            predict_times = []
            cot_times = []

            for i, test_case in enumerate(spam_test_cases):
                # Test Predict module
                start_time = time.time()
                predict_result = predict_spam(**test_case)
                predict_time = time.time() - start_time
                predict_times.append(predict_time)

                # Test ChainOfThought module
                start_time = time.time()
                cot_result = cot_spam(**test_case)
                cot_time = time.time() - start_time
                cot_times.append(cot_time)

                # Store results
                results.append(
                    {
                        "test_case": i + 1,
                        "email_preview": test_case["email_content"][:50] + "...",
                        "predict_result": predict_result,
                        "predict_time": predict_time,
                        "cot_result": cot_result,
                        "cot_time": cot_time,
                    }
                )

            # Calculate averages
            avg_predict_time = sum(predict_times) / len(predict_times)
            avg_cot_time = sum(cot_times) / len(cot_times)

            # Create comparison visualization
            comparison_viewer = ComparisonViewer()
            comparison_viewer.add_comparison(
                "Predict Module",
                f"Average time: {avg_predict_time:.3f}s",
                {"module_type": "Direct", "reasoning": "None"},
            )
            comparison_viewer.add_comparison(
                "ChainOfThought Module",
                f"Average time: {avg_cot_time:.3f}s",
                {"module_type": "Reasoning", "reasoning": "Included"},
            )

            # Display detailed results
            result_details = []
            for result in results:
                result_details.append(
                    f"""
**Test {result['test_case']}:** {result['email_preview']}
- **Predict:** {result['predict_result'].classification} (confidence: {result['predict_result'].confidence}) - {result['predict_time']:.3f}s
- **ChainOfThought:** {result['cot_result'].classification} (confidence: {result['cot_result'].confidence}) - {result['cot_time']:.3f}s
"""
                )

            mo.vstack(
                [
                    mo.md("## 📊 Challenge 1 Results: Spam Detection"),
                    comparison_viewer.render(),
                    mo.md("### 📈 Performance Analysis"),
                    mo.md(
                        f"""
**Speed Comparison:**
- Predict average: {avg_predict_time:.3f} seconds
- ChainOfThought average: {avg_cot_time:.3f} seconds
- Speed difference: {((avg_cot_time - avg_predict_time) / avg_predict_time * 100):.1f}% slower for CoT

**Key Insights:**
- Simple classification tasks may not need reasoning overhead
- Both modules should produce similar accuracy for clear cases
- Speed difference becomes important at scale
                """
                    ),
                    mo.md("### 🔍 Detailed Results"),
                    mo.md("\n".join(result_details)),
                ]
            )

        except Exception as e:
            mo.md(f"❌ **Error in Challenge 1:** {str(e)}")
    else:
        mo.md("*Click 'Run Challenge 1' to compare spam detection performance*")
    return (
        avg_cot_time,
        avg_predict_time,
        comparison_viewer,
        cot_result,
        cot_time,
        cot_times,
        predict_result,
        predict_time,
        predict_times,
        result,
        result_details,
        results,
        start_time,
    )


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 2: Complex Analysis Task
        
        **Task:** Compare modules on a task requiring deeper reasoning.
        
        **Signature:** Financial document analysis
        - Input: Financial text and analysis type
        - Output: Key insights, risk assessment, recommendations, confidence
        
        This will show how reasoning capabilities affect complex tasks.
        """
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

        mo.md(
            """
        ### 💰 Financial Analysis Signature Created
        
        **Modules Ready:**
        - `dspy.Predict(FinancialAnalyzer)` - Direct analysis
        - `dspy.ChainOfThought(FinancialAnalyzer)` - Reasoning-based analysis
        """
        )
    else:
        FinancialAnalyzer = None
        predict_finance = None
        cot_finance = None
    return FinancialAnalyzer, cot_finance, predict_finance


@app.cell
def __(available_providers, mo):
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
        run_challenge2 = mo.ui.button(
            label="🚀 Run Challenge 2: Financial Analysis Comparison"
        )

        mo.vstack(
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
    return finance_test_cases, run_challenge2


@app.cell
def __(
    available_providers,
    cot_finance,
    finance_test_cases,
    mo,
    predict_finance,
    run_challenge2,
    time,
):
    if available_providers and run_challenge2.value and finance_test_cases:
        try:
            finance_results = []
            finance_predict_times = []
            finance_cot_times = []

            for i, test_case in enumerate(finance_test_cases):
                # Test Predict module
                start_time = time.time()
                predict_result = predict_finance(**test_case)
                predict_time = time.time() - start_time
                finance_predict_times.append(predict_time)

                # Test ChainOfThought module
                start_time = time.time()
                cot_result = cot_finance(**test_case)
                cot_time = time.time() - start_time
                finance_cot_times.append(cot_time)

                # Store results
                finance_results.append(
                    {
                        "test_case": i + 1,
                        "analysis_type": test_case["analysis_type"],
                        "predict_result": predict_result,
                        "predict_time": predict_time,
                        "cot_result": cot_result,
                        "cot_time": cot_time,
                    }
                )

            # Calculate averages
            avg_finance_predict = sum(finance_predict_times) / len(
                finance_predict_times
            )
            avg_finance_cot = sum(finance_cot_times) / len(finance_cot_times)

            # Display results
            finance_details = []
            for result in finance_results:
                finance_details.append(
                    f"""
**Test {result['test_case']} ({result['analysis_type']} analysis):**

**Predict Module ({result['predict_time']:.3f}s):**
- Insights: {result['predict_result'].key_insights}
- Risk: {result['predict_result'].risk_assessment}
- Recommendations: {result['predict_result'].recommendations}

**ChainOfThought Module ({result['cot_time']:.3f}s):**
- Insights: {result['cot_result'].key_insights}
- Risk: {result['cot_result'].risk_assessment}  
- Recommendations: {result['cot_result'].recommendations}
- Reasoning: {getattr(result['cot_result'], 'rationale', 'No rationale shown')}
"""
                )

            mo.vstack(
                [
                    mo.md("## 📊 Challenge 2 Results: Financial Analysis"),
                    mo.md("### 📈 Performance Comparison"),
                    mo.md(
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
                    ),
                    mo.md("### 🔍 Detailed Analysis Results"),
                    mo.md("\n".join(finance_details)),
                    mo.md(
                        """
### 💡 Key Insights from Challenge 2
- **Complex tasks benefit more from reasoning** - ChainOfThought shows its value
- **Quality vs Speed trade-off** - Better analysis takes more time
- **Reasoning transparency** - CoT shows how conclusions were reached
- **Domain expertise** - Complex domains need more sophisticated processing
                """
                    ),
                ]
            )

        except Exception as e:
            mo.md(f"❌ **Error in Challenge 2:** {str(e)}")
    else:
        mo.md("*Click 'Run Challenge 2' to compare financial analysis performance*")
    return (
        avg_finance_cot,
        avg_finance_predict,
        cot_result,
        cot_time,
        finance_cot_times,
        finance_details,
        finance_predict_times,
        finance_results,
        predict_result,
        predict_time,
        result,
        start_time,
    )


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 3: Mathematical Reasoning Task
        
        **Task:** Compare modules on multi-step mathematical problem solving.
        
        **Signature:** Word problem solver
        - Input: Math word problem and difficulty level
        - Output: Solution steps, final answer, verification, confidence
        
        This will demonstrate the maximum benefit of reasoning-based modules.
        """
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

        mo.md(
            """
        ### 🧮 Math Problem Solver Signature Created
        
        **Modules Ready:**
        - `dspy.Predict(MathWordProblemSolver)` - Direct problem solving
        - `dspy.ChainOfThought(MathWordProblemSolver)` - Reasoning-based solving
        """
        )
    else:
        MathWordProblemSolver = None
        predict_math = None
        cot_math = None
    return MathWordProblemSolver, cot_math, predict_math


@app.cell
def __(available_providers, mo):
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
        run_challenge3 = mo.ui.button(
            label="🚀 Run Challenge 3: Math Problem Comparison"
        )

        mo.vstack(
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
    return math_test_cases, run_challenge3


@app.cell
def __(
    available_providers,
    cot_math,
    math_test_cases,
    mo,
    predict_math,
    run_challenge3,
    time,
):
    if available_providers and run_challenge3.value and math_test_cases:
        try:
            math_results = []
            math_predict_times = []
            math_cot_times = []

            for i, test_case in enumerate(math_test_cases):
                # Test Predict module
                start_time = time.time()
                predict_result = predict_math(**test_case)
                predict_time = time.time() - start_time
                math_predict_times.append(predict_time)

                # Test ChainOfThought module
                start_time = time.time()
                cot_result = cot_math(**test_case)
                cot_time = time.time() - start_time
                math_cot_times.append(cot_time)

                # Store results
                math_results.append(
                    {
                        "test_case": i + 1,
                        "difficulty": test_case["difficulty_level"],
                        "problem": test_case["problem_text"],
                        "predict_result": predict_result,
                        "predict_time": predict_time,
                        "cot_result": cot_result,
                        "cot_time": cot_time,
                    }
                )

            # Calculate averages
            avg_math_predict = sum(math_predict_times) / len(math_predict_times)
            avg_math_cot = sum(math_cot_times) / len(math_cot_times)

            # Display results
            math_details = []
            for result in math_results:
                math_details.append(
                    f"""
**Test {result['test_case']} ({result['difficulty']} difficulty):**
*Problem: {result['problem'][:80]}...*

**Predict Module ({result['predict_time']:.3f}s):**
- Steps: {result['predict_result'].solution_steps}
- Answer: {result['predict_result'].final_answer}
- Confidence: {result['predict_result'].confidence}

**ChainOfThought Module ({result['cot_time']:.3f}s):**
- Steps: {result['cot_result'].solution_steps}
- Answer: {result['cot_result'].final_answer}
- Verification: {result['cot_result'].verification}
- Confidence: {result['cot_result'].confidence}
- Reasoning: {getattr(result['cot_result'], 'rationale', 'No rationale shown')}
"""
                )

            mo.vstack(
                [
                    mo.md("## 📊 Challenge 3 Results: Mathematical Reasoning"),
                    mo.md("### 📈 Performance Analysis"),
                    mo.md(
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
                    ),
                    mo.md("### 🔍 Detailed Math Problem Results"),
                    mo.md("\n".join(math_details)),
                    mo.md(
                        """
### 🧠 Challenge 3 Key Insights
- **Reasoning is crucial for math** - Step-by-step approach prevents errors
- **Verification adds value** - Checking work improves accuracy
- **Transparency matters** - Seeing the reasoning helps trust results
- **Complex problems need complex modules** - Don't use a hammer for surgery
                """
                    ),
                ]
            )

        except Exception as e:
            mo.md(f"❌ **Error in Challenge 3:** {str(e)}")
    else:
        mo.md("*Click 'Run Challenge 3' to compare mathematical reasoning performance*")
    return (
        avg_math_cot,
        avg_math_predict,
        cot_result,
        cot_time,
        math_cot_times,
        math_details,
        math_predict_times,
        math_results,
        predict_result,
        predict_time,
        result,
        start_time,
    )


@app.cell
def __(PerformanceBenchmark, available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 📊 Comprehensive Performance Benchmark
        
        **Task:** Run a systematic benchmark across all three challenge types.
        
        This will give you quantitative data to make informed module selection decisions.
        """
        )

        # Benchmark execution button
        run_benchmark = mo.ui.button(label="🏃‍♂️ Run Comprehensive Benchmark")

        mo.vstack(
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
        run_benchmark = None
    return (run_benchmark,)


@app.cell
def __(
    EmailSpamDetector,
    FinancialAnalyzer,
    MathWordProblemSolver,
    PerformanceBenchmark,
    available_providers,
    dspy,
    finance_test_cases,
    math_test_cases,
    mo,
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
            for task_type, results in benchmark_results.items():
                predict_data = results["predict"]
                cot_data = results["cot"]

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

            mo.vstack(
                [
                    mo.md("## 🏆 Comprehensive Benchmark Results"),
                    mo.ui.table(summary_data),
                    mo.md(
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
                    ),
                    mo.md(
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
                    ),
                ]
            )

        except Exception as e:
            mo.md(f"❌ **Benchmark Error:** {str(e)}")
    else:
        mo.md(
            "*Click 'Run Comprehensive Benchmark' to see detailed performance analysis*"
        )
    return (
        benchmark,
        benchmark_results,
        cot_data,
        cot_finance_bench,
        cot_finance_results,
        cot_math_bench,
        cot_math_results,
        cot_spam_bench,
        cot_spam_results,
        finance_cot_inputs,
        finance_inputs,
        math_cot_inputs,
        math_inputs,
        predict_data,
        predict_finance_bench,
        predict_finance_results,
        predict_math_bench,
        predict_math_results,
        predict_spam_bench,
        predict_spam_results,
        spam_cot_inputs,
        spam_inputs,
        summary_data,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Decision Framework Exercise
        
        **Task:** Apply your knowledge to make module selection decisions for new scenarios.
        
        For each scenario below, choose the appropriate module and justify your decision.
        """
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

        # Create decision form
        decision_form = mo.ui.form(
            {
                f"scenario_{i}": mo.ui.radio(
                    options=[
                        "Predict",
                        "ChainOfThought",
                        "Depends on specific requirements",
                    ],
                    label=f"**{scenario['title']}**\n{scenario['description']}\nFactors: {', '.join(scenario['factors'])}",
                    value="Predict",
                )
                for i, scenario in enumerate(scenarios)
            }
        )

        mo.vstack([mo.md("### 🤔 Make Your Decisions"), decision_form])
    else:
        decision_form = None
        scenarios = None
    return decision_form, scenarios


@app.cell
def __(available_providers, decision_form, mo, scenarios):
    if available_providers and decision_form.value and scenarios:
        decisions = decision_form.value

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
        results = []
        score = 0

        for i, (scenario, recommendation) in enumerate(zip(scenarios, recommendations)):
            user_choice = decisions[f"scenario_{i}"]
            is_correct = user_choice == recommendation["recommended"]
            if is_correct:
                score += 1

            results.append(
                f"""
**{scenario['title']}**
- Your choice: **{user_choice}**
- Recommended: **{recommendation['recommended']}**
- {'✅ Correct!' if is_correct else '❌ Consider this:'} {recommendation['explanation']}
"""
            )

        mo.vstack(
            [
                mo.md(f"## 🎯 Decision Framework Results: {score}/{len(scenarios)}"),
                mo.md("\n".join(results)),
                mo.md(
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
                ),
            ]
        )
    else:
        mo.md("*Complete the decision scenarios above to see your results*")
    return (
        decisions,
        is_correct,
        recommendation,
        recommendations,
        results,
        score,
        user_choice,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
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
    return


if __name__ == "__main__":
    app.run()
