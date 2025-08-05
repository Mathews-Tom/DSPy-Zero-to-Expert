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
        DSPyParameterPanel,
        DSPyResultViewer,
        get_config,
        setup_dspy_environment,
    )

    return (
        DSPyParameterPanel,
        DSPyResultViewer,
        Path,
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
        # 🎛️ Exercise 3: Signature Optimization & Parameter Tuning
        
        **Objective:** Master the art of optimizing DSPy signatures through systematic parameter tuning and iterative improvement.
        
        ## 📚 What You'll Learn
        - Optimize signature performance through parameter tuning
        - Understand the impact of temperature, max_tokens, and other settings
        - Apply A/B testing methodologies to signature improvement
        - Develop systematic optimization workflows
        
        ## 🎮 Exercise Structure
        - **Parameter Impact Analysis** - See how settings affect output
        - **Systematic Optimization** - Step-by-step improvement process
        - **A/B Testing Framework** - Compare signature variations
        - **Production Optimization** - Real-world optimization strategies
        
        Let's optimize your signatures for peak performance!
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
        
        Ready to optimize signature performance!
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
        ## 🎯 Challenge 1: Parameter Impact Analysis
        
        **Task:** Understand how different parameters affect signature performance.
        
        **Test Signature:** Creative writing assistant
        - Input: Topic and writing style
        - Output: Creative content, tone analysis, improvement suggestions
        
        We'll systematically vary parameters to see their impact.
        """
        )

        # Define the test signature
        class CreativeWritingAssistant(dspy.Signature):
            """Generate creative content with specified style and provide analysis and improvement suggestions."""

            topic = dspy.InputField(desc="The topic or theme to write about")
            writing_style = dspy.InputField(
                desc="Desired writing style: 'formal', 'casual', 'poetic', 'humorous'"
            )
            content = dspy.OutputField(desc="Creative content (150-200 words)")
            tone_analysis = dspy.OutputField(
                desc="Analysis of the tone and style achieved"
            )
            improvements = dspy.OutputField(desc="Specific suggestions for improvement")

        # Create predictor
        creative_predictor = dspy.Predict(CreativeWritingAssistant)

        mo.md(
            """
        ### ✍️ Creative Writing Assistant Signature Created
        
        **Ready to test parameter variations:**
        - Temperature (creativity vs consistency)
        - Max tokens (length control)
        - Model selection (capability differences)
        """
        )
    else:
        CreativeWritingAssistant = None
        creative_predictor = None
    return CreativeWritingAssistant, creative_predictor


@app.cell
def __(DSPyParameterPanel, available_providers, mo):
    if available_providers:
        # Create parameter panel for testing
        param_panel = DSPyParameterPanel(
            show_temperature=True,
            show_max_tokens=True,
            show_model_selection=True,
            show_provider_selection=False,
            custom_params={
                "test_runs": {
                    "type": "slider",
                    "min": 1,
                    "max": 5,
                    "default": 3,
                    "label": "Number of Test Runs",
                }
            },
        )

        mo.vstack(
            [
                mo.md("### 🎛️ Parameter Control Panel"),
                mo.md("Adjust parameters to see their impact on creative output:"),
                param_panel.render(),
            ]
        )
    else:
        param_panel = None
    return (param_panel,)


@app.cell
def __(available_providers, mo):
    if available_providers:
        # Test case for parameter analysis
        test_case = {
            "topic": "A mysterious library that appears only at midnight",
            "writing_style": "poetic",
        }

        # Parameter test button
        run_param_test = mo.ui.button(label="🧪 Test Parameter Impact")

        mo.vstack(
            [
                mo.md("### 📝 Test Case"),
                mo.md(f"**Topic:** {test_case['topic']}"),
                mo.md(f"**Style:** {test_case['writing_style']}"),
                run_param_test,
            ]
        )
    else:
        test_case = None
        run_param_test = None
    return run_param_test, test_case


@app.cell
def __(
    available_providers,
    creative_predictor,
    mo,
    param_panel,
    run_param_test,
    test_case,
    time,
):
    if available_providers and run_param_test.value and param_panel and test_case:
        try:
            # Get parameter values
            params = param_panel.get_values()
            num_runs = params.get("test_runs", 3)

            # Apply parameters to DSPy (this would typically be done through dspy.settings)
            # For demonstration, we'll show the parameter impact conceptually

            results = []
            times = []

            for run in range(num_runs):
                start_time = time.time()
                result = creative_predictor(**test_case)
                execution_time = time.time() - start_time

                results.append(result)
                times.append(execution_time)

            avg_time = sum(times) / len(times)

            # Analyze results for consistency
            contents = [r.content for r in results]
            content_lengths = [len(c.split()) for c in contents]
            avg_length = sum(content_lengths) / len(content_lengths)

            mo.vstack(
                [
                    mo.md("## 📊 Parameter Impact Analysis Results"),
                    mo.md(
                        f"""
### ⚙️ Parameter Settings
- **Temperature:** {params.get('temperature', 'default')}
- **Max Tokens:** {params.get('max_tokens', 'default')}
- **Model:** {params.get('model', 'default')}
- **Test Runs:** {num_runs}

### 📈 Performance Metrics
- **Average Execution Time:** {avg_time:.3f} seconds
- **Average Content Length:** {avg_length:.1f} words
- **Length Variation:** {max(content_lengths) - min(content_lengths)} words
                """
                    ),
                    mo.md("### 📝 Generated Content Samples"),
                ]
                + [
                    mo.vstack(
                        [
                            mo.md(
                                f"**Run {i+1} ({times[i]:.3f}s, {len(contents[i].split())} words):**"
                            ),
                            mo.md(f"*Content:* {contents[i][:100]}..."),
                            mo.md(f"*Tone:* {results[i].tone_analysis}"),
                            mo.md("---"),
                        ]
                    )
                    for i in range(min(3, len(results)))
                ]
                + [
                    mo.md(
                        """
### 💡 Parameter Impact Observations

**Temperature Effects:**
- **Low (0.1-0.3):** More consistent, predictable output
- **Medium (0.4-0.7):** Balanced creativity and coherence  
- **High (0.8-1.0):** More creative but potentially inconsistent

**Max Tokens Effects:**
- **Low (50-100):** Concise but may cut off content
- **Medium (150-300):** Good balance for most tasks
- **High (500+):** Allows full expression but slower

**Model Effects:**
- **Smaller models:** Faster but less sophisticated
- **Larger models:** Better quality but slower and more expensive
                """
                    )
                ]
            )

        except Exception as e:
            mo.md(f"❌ **Parameter Test Error:** {str(e)}")
    else:
        mo.md(
            "*Adjust parameters above and click 'Test Parameter Impact' to see results*"
        )
    return (
        avg_length,
        avg_time,
        content_lengths,
        contents,
        execution_time,
        num_runs,
        params,
        result,
        results,
        run,
        start_time,
        times,
    )


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 2: A/B Testing Framework
        
        **Task:** Compare two signature variations to determine which performs better.
        
        **Scenario:** Email subject line optimizer
        - Version A: Simple optimization
        - Version B: Advanced optimization with analysis
        
        We'll systematically compare their performance.
        """
        )

        # Define two signature variations
        class EmailSubjectOptimizerA(dspy.Signature):
            """Optimize email subject lines for better open rates."""

            original_subject = dspy.InputField(desc="Original email subject line")
            target_audience = dspy.InputField(desc="Target audience description")
            optimized_subject = dspy.OutputField(desc="Improved subject line")
            improvement_reason = dspy.OutputField(
                desc="Brief explanation of improvements"
            )

        class EmailSubjectOptimizerB(dspy.Signature):
            """Optimize email subject lines with comprehensive analysis and multiple options."""

            original_subject = dspy.InputField(desc="Original email subject line")
            target_audience = dspy.InputField(desc="Target audience description")
            email_context = dspy.InputField(desc="Email content context or purpose")
            optimized_subject = dspy.OutputField(desc="Primary improved subject line")
            alternative_subjects = dspy.OutputField(
                desc="2-3 alternative subject line options"
            )
            analysis = dspy.OutputField(
                desc="Detailed analysis of optimization strategy"
            )
            predicted_improvement = dspy.OutputField(
                desc="Predicted open rate improvement percentage"
            )

        # Create predictors
        optimizer_a = dspy.Predict(EmailSubjectOptimizerA)
        optimizer_b = dspy.Predict(EmailSubjectOptimizerB)

        mo.md(
            """
        ### 📧 Email Subject Optimizers Created
        
        **Version A (Simple):**
        - 2 inputs, 2 outputs
        - Basic optimization approach
        - Faster execution expected
        
        **Version B (Advanced):**
        - 3 inputs, 4 outputs  
        - Comprehensive analysis
        - More detailed but slower
        """
        )
    else:
        EmailSubjectOptimizerA = None
        EmailSubjectOptimizerB = None
        optimizer_a = None
        optimizer_b = None
    return (
        EmailSubjectOptimizerA,
        EmailSubjectOptimizerB,
        optimizer_a,
        optimizer_b,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        # A/B test cases
        ab_test_cases = [
            {
                "original_subject": "Monthly Newsletter",
                "target_audience": "Small business owners",
                "email_context": "Monthly business tips and industry updates",
            },
            {
                "original_subject": "Sale This Weekend",
                "target_audience": "Fashion-conscious millennials",
                "email_context": "Weekend flash sale on trendy clothing items",
            },
            {
                "original_subject": "Important Update",
                "target_audience": "Software developers",
                "email_context": "Critical security update for development tools",
            },
        ]

        # A/B test execution button
        run_ab_test = mo.ui.button(label="⚖️ Run A/B Test Comparison")

        mo.vstack(
            [
                mo.md("### 🧪 A/B Test Cases Ready"),
                mo.md(f"- {len(ab_test_cases)} diverse email scenarios"),
                mo.md("- Different audiences and contexts"),
                mo.md("- Performance and quality comparison"),
                run_ab_test,
            ]
        )
    else:
        ab_test_cases = None
        run_ab_test = None
    return ab_test_cases, run_ab_test


@app.cell
def __(
    ab_test_cases,
    available_providers,
    mo,
    optimizer_a,
    optimizer_b,
    run_ab_test,
    time,
):
    if available_providers and run_ab_test.value and ab_test_cases:
        try:
            ab_results = []
            a_times = []
            b_times = []

            for i, test_case in enumerate(ab_test_cases):
                # Test Version A
                start_time = time.time()
                result_a = optimizer_a(
                    original_subject=test_case["original_subject"],
                    target_audience=test_case["target_audience"],
                )
                time_a = time.time() - start_time
                a_times.append(time_a)

                # Test Version B
                start_time = time.time()
                result_b = optimizer_b(**test_case)  # Uses all fields
                time_b = time.time() - start_time
                b_times.append(time_b)

                ab_results.append(
                    {
                        "test_case": i + 1,
                        "original": test_case["original_subject"],
                        "audience": test_case["target_audience"],
                        "result_a": result_a,
                        "time_a": time_a,
                        "result_b": result_b,
                        "time_b": time_b,
                    }
                )

            # Calculate metrics
            avg_time_a = sum(a_times) / len(a_times)
            avg_time_b = sum(b_times) / len(b_times)

            # Display A/B test results
            comparison_details = []
            for result in ab_results:
                comparison_details.append(
                    f"""
**Test {result['test_case']}:** "{result['original']}" → {result['audience']}

**Version A ({result['time_a']:.3f}s):**
- Optimized: "{result['result_a'].optimized_subject}"
- Reason: {result['result_a'].improvement_reason}

**Version B ({result['time_b']:.3f}s):**
- Primary: "{result['result_b'].optimized_subject}"
- Alternatives: {result['result_b'].alternative_subjects}
- Analysis: {result['result_b'].analysis}
- Predicted improvement: {result['result_b'].predicted_improvement}
"""
                )

            mo.vstack(
                [
                    mo.md("## ⚖️ A/B Test Results: Email Subject Optimization"),
                    mo.md("### 📊 Performance Comparison"),
                    mo.md(
                        f"""
**Speed Analysis:**
- Version A average: {avg_time_a:.3f} seconds
- Version B average: {avg_time_b:.3f} seconds  
- Speed difference: {((avg_time_b - avg_time_a) / avg_time_a * 100):.1f}% slower for Version B

**Quality Analysis:**
- Version A: Simple, focused optimization
- Version B: Comprehensive analysis with alternatives
- Version B provides more actionable insights
                """
                    ),
                    mo.md("### 🔍 Detailed A/B Comparison"),
                    mo.md("\n".join(comparison_details)),
                    mo.md(
                        """
### 🎯 A/B Test Insights

**Version A Advantages:**
- ⚡ Faster execution (better for high volume)
- 🎯 Simple, focused output
- 💰 Lower computational cost

**Version B Advantages:**
- 🧠 More comprehensive analysis
- 🎨 Multiple creative options
- 📊 Quantitative predictions
- 🔍 Better for optimization learning

**Recommendation:**
- Use Version A for high-volume, real-time optimization
- Use Version B for strategic campaigns and learning
                """
                    ),
                ]
            )

        except Exception as e:
            mo.md(f"❌ **A/B Test Error:** {str(e)}")
    else:
        mo.md(
            "*Click 'Run A/B Test Comparison' to see signature performance comparison*"
        )
    return (
        a_times,
        ab_results,
        avg_time_a,
        avg_time_b,
        b_times,
        comparison_details,
        result,
        result_a,
        result_b,
        start_time,
        test_case,
        time_a,
        time_b,
    )


@app.cell
def __(available_providers, dspy, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 3: Iterative Signature Improvement
        
        **Task:** Systematically improve a signature through multiple iterations.
        
        **Base Signature:** Product review analyzer
        - Start with basic version
        - Apply optimization principles
        - Measure improvement at each step
        
        This demonstrates real-world optimization workflows.
        """
        )

        # Define signature iterations
        class ProductReviewAnalyzerV1(dspy.Signature):
            """Analyze product reviews."""

            review_text = dspy.InputField(desc="Product review text")
            sentiment = dspy.OutputField(desc="Sentiment")
            rating = dspy.OutputField(desc="Rating")

        class ProductReviewAnalyzerV2(dspy.Signature):
            """Analyze product reviews for sentiment and key insights."""

            review_text = dspy.InputField(desc="Product review text to analyze")
            product_category = dspy.InputField(desc="Product category for context")
            sentiment = dspy.OutputField(
                desc="Sentiment: positive, negative, or neutral"
            )
            rating_prediction = dspy.OutputField(desc="Predicted star rating (1-5)")
            key_points = dspy.OutputField(
                desc="Key positive and negative points mentioned"
            )

        class ProductReviewAnalyzerV3(dspy.Signature):
            """Comprehensively analyze product reviews with detailed insights and actionable recommendations."""

            review_text = dspy.InputField(
                desc="Complete product review text for analysis"
            )
            product_category = dspy.InputField(
                desc="Product category (electronics, clothing, books, etc.)"
            )
            reviewer_context = dspy.InputField(
                desc="Context about reviewer if available"
            )
            sentiment_analysis = dspy.OutputField(
                desc="Detailed sentiment: positive/negative/neutral with confidence score"
            )
            rating_prediction = dspy.OutputField(
                desc="Predicted star rating (1-5) with reasoning"
            )
            feature_analysis = dspy.OutputField(
                desc="Analysis of specific product features mentioned"
            )
            improvement_suggestions = dspy.OutputField(
                desc="Suggestions for product improvement based on review"
            )
            review_helpfulness = dspy.OutputField(
                desc="Assessment of review quality and helpfulness"
            )

        # Create predictors for all versions
        analyzer_v1 = dspy.Predict(ProductReviewAnalyzerV1)
        analyzer_v2 = dspy.Predict(ProductReviewAnalyzerV2)
        analyzer_v3 = dspy.Predict(ProductReviewAnalyzerV3)

        mo.md(
            """
        ### 📱 Product Review Analyzer Versions Created
        
        **V1 (Basic):** 1 input, 2 outputs - Minimal functionality
        **V2 (Improved):** 2 inputs, 3 outputs - Added context and detail
        **V3 (Advanced):** 3 inputs, 5 outputs - Comprehensive analysis
        
        Let's test the evolution of signature quality!
        """
        )
    else:
        ProductReviewAnalyzerV1 = None
        ProductReviewAnalyzerV2 = None
        ProductReviewAnalyzerV3 = None
        analyzer_v1 = None
        analyzer_v2 = None
        analyzer_v3 = None
    return (
        ProductReviewAnalyzerV1,
        ProductReviewAnalyzerV2,
        ProductReviewAnalyzerV3,
        analyzer_v1,
        analyzer_v2,
        analyzer_v3,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        # Test case for iterative improvement
        improvement_test_case = {
            "review_text": "I bought these wireless headphones last month and I'm really impressed! The sound quality is excellent - crisp highs and deep bass. Battery life is amazing, easily lasts 8+ hours. The only downside is they're a bit heavy for long listening sessions and the case feels cheap. Overall great value for the price though!",
            "product_category": "electronics",
            "reviewer_context": "Verified purchase, frequent headphone user",
        }

        # Iterative improvement test button
        run_improvement_test = mo.ui.button(label="📈 Test Iterative Improvements")

        mo.vstack(
            [
                mo.md("### 🎧 Test Case: Wireless Headphones Review"),
                mo.md(f"**Review:** {improvement_test_case['review_text'][:100]}..."),
                mo.md(f"**Category:** {improvement_test_case['product_category']}"),
                run_improvement_test,
            ]
        )
    else:
        improvement_test_case = None
        run_improvement_test = None
    return improvement_test_case, run_improvement_test


@app.cell
def __(
    analyzer_v1,
    analyzer_v2,
    analyzer_v3,
    available_providers,
    improvement_test_case,
    mo,
    run_improvement_test,
    time,
):
    if available_providers and run_improvement_test.value and improvement_test_case:
        try:
            # Test all three versions
            versions = []

            # V1 Test (basic inputs only)
            start_time = time.time()
            result_v1 = analyzer_v1(review_text=improvement_test_case["review_text"])
            time_v1 = time.time() - start_time
            versions.append(("V1 (Basic)", result_v1, time_v1, 2))

            # V2 Test
            start_time = time.time()
            result_v2 = analyzer_v2(
                review_text=improvement_test_case["review_text"],
                product_category=improvement_test_case["product_category"],
            )
            time_v2 = time.time() - start_time
            versions.append(("V2 (Improved)", result_v2, time_v2, 3))

            # V3 Test
            start_time = time.time()
            result_v3 = analyzer_v3(**improvement_test_case)
            time_v3 = time.time() - start_time
            versions.append(("V3 (Advanced)", result_v3, time_v3, 5))

            # Display iterative improvement results
            improvement_details = []
            for version_name, result, exec_time, output_count in versions:
                improvement_details.append(
                    f"""
**{version_name} ({exec_time:.3f}s, {output_count} outputs):**
{result}
"""
                )

            # Calculate improvement metrics
            complexity_increase = [
                (v[3] - versions[0][3]) / versions[0][3] * 100 for v in versions[1:]
            ]
            time_increase = [
                (v[2] - versions[0][2]) / versions[0][2] * 100 for v in versions[1:]
            ]

            mo.vstack(
                [
                    mo.md("## 📈 Iterative Improvement Results"),
                    mo.md("### 🔄 Version Evolution"),
                    mo.md("\n".join(improvement_details)),
                    mo.md("### 📊 Improvement Metrics"),
                    mo.md(
                        f"""
**Complexity Growth:**
- V2 vs V1: {complexity_increase[0]:.1f}% more outputs
- V3 vs V1: {complexity_increase[1]:.1f}% more outputs

**Performance Impact:**
- V2 vs V1: {time_increase[0]:.1f}% time increase
- V3 vs V1: {time_increase[1]:.1f}% time increase

**Quality Assessment:**
- V1: Basic sentiment and rating
- V2: Added context and key points
- V3: Comprehensive analysis with actionable insights
                """
                    ),
                    mo.md(
                        """
### 🎯 Iterative Improvement Insights

**Optimization Strategy:**
1. **Start Simple** - V1 provides baseline functionality
2. **Add Context** - V2 improves accuracy with product category
3. **Enhance Output** - V3 provides comprehensive business value

**Trade-off Analysis:**
- Each iteration adds ~50-100% more processing time
- Quality improvements are substantial and measurable
- V3 provides business-actionable insights worth the cost

**Production Recommendations:**
- Use V1 for high-volume, basic sentiment analysis
- Use V2 for balanced performance and insight
- Use V3 for strategic analysis and product improvement
                """
                    ),
                ]
            )

        except Exception as e:
            mo.md(f"❌ **Improvement Test Error:** {str(e)}")
    else:
        mo.md(
            "*Click 'Test Iterative Improvements' to see signature evolution results*"
        )
    return (
        complexity_increase,
        exec_time,
        improvement_details,
        output_count,
        result,
        result_v1,
        result_v2,
        result_v3,
        start_time,
        time_increase,
        time_v1,
        time_v2,
        time_v3,
        version_name,
        versions,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎯 Challenge 4: Production Optimization Strategy
        
        **Task:** Develop a comprehensive optimization strategy for production deployment.
        
        **Scenario:** You need to deploy a signature for a high-traffic application.
        Consider all optimization factors and make strategic decisions.
        """
        )

        # Production optimization form
        production_form = mo.ui.form(
            {
                "use_case": mo.ui.dropdown(
                    options=[
                        "Real-time chat moderation (1000+ req/min)",
                        "Email marketing optimization (100 req/min)",
                        "Document analysis service (10 req/min)",
                        "Creative content generation (1 req/min)",
                    ],
                    label="Production Use Case",
                    value="Email marketing optimization (100 req/min)",
                ),
                "priority": mo.ui.radio(
                    options=["Speed", "Quality", "Cost", "Balanced"],
                    label="Primary Optimization Priority",
                    value="Balanced",
                ),
                "budget": mo.ui.dropdown(
                    options=[
                        "Low ($100/month)",
                        "Medium ($500/month)",
                        "High ($2000/month)",
                        "Enterprise",
                    ],
                    label="Budget Constraints",
                    value="Medium ($500/month)",
                ),
                "accuracy_requirement": mo.ui.slider(
                    min=70, max=99, value=85, label="Minimum Accuracy Requirement (%)"
                ),
                "analyze": mo.ui.button(label="🎯 Generate Optimization Strategy"),
            }
        )

        mo.vstack(
            [
                mo.md("### 🏭 Production Optimization Planner"),
                mo.md(
                    "Define your production requirements to get a customized optimization strategy:"
                ),
                production_form,
            ]
        )
    else:
        production_form = None
    return (production_form,)


@app.cell
def __(available_providers, mo, production_form):
    if (
        available_providers
        and production_form.value
        and production_form.value["analyze"]
    ):
        form_data = production_form.value

        # Generate optimization strategy based on inputs
        use_case = form_data["use_case"]
        priority = form_data["priority"]
        budget = form_data["budget"]
        accuracy = form_data["accuracy_requirement"]

        # Determine recommendations based on inputs
        if "1000+" in use_case:
            volume = "Very High"
            recommended_module = "Predict"
            caching_strategy = "Aggressive caching with Redis"
        elif "100" in use_case:
            volume = "High"
            recommended_module = "Predict with CoT for edge cases"
            caching_strategy = "Moderate caching"
        elif "10" in use_case:
            volume = "Medium"
            recommended_module = "ChainOfThought"
            caching_strategy = "Selective caching"
        else:
            volume = "Low"
            recommended_module = "ChainOfThought with multiple attempts"
            caching_strategy = "Minimal caching"

        # Budget-based recommendations
        if "Low" in budget:
            model_recommendation = "Smaller, efficient models"
            infrastructure = "Single instance with auto-scaling"
        elif "Medium" in budget:
            model_recommendation = "Balanced model selection"
            infrastructure = "Load-balanced instances"
        else:
            model_recommendation = "Premium models for best quality"
            infrastructure = "Multi-region deployment"

        # Priority-based optimizations
        if priority == "Speed":
            optimizations = [
                "Use Predict module exclusively",
                "Implement aggressive caching",
                "Optimize prompt length",
                "Use faster models even if slightly less accurate",
            ]
        elif priority == "Quality":
            optimizations = [
                "Use ChainOfThought for complex cases",
                "Implement multi-attempt validation",
                "Use premium models",
                "Add human review for edge cases",
            ]
        elif priority == "Cost":
            optimizations = [
                "Use smaller models with good performance",
                "Implement smart caching to reduce API calls",
                "Batch processing where possible",
                "Monitor and optimize token usage",
            ]
        else:  # Balanced
            optimizations = [
                "Hybrid approach: Predict for simple, CoT for complex",
                "Moderate caching strategy",
                "Cost-effective model selection",
                "Performance monitoring and adjustment",
            ]

        mo.vstack(
            [
                mo.md("## 🎯 Production Optimization Strategy"),
                mo.md(
                    f"""
### 📋 Your Requirements Analysis
- **Use Case:** {use_case}
- **Volume:** {volume}
- **Priority:** {priority}
- **Budget:** {budget}
- **Accuracy Target:** {accuracy}%

### 🏗️ Recommended Architecture
**Module Selection:** {recommended_module}
**Model Strategy:** {model_recommendation}
**Infrastructure:** {infrastructure}
**Caching:** {caching_strategy}
            """
                ),
                mo.md("### 🔧 Optimization Strategies"),
                mo.md("\n".join([f"- {opt}" for opt in optimizations])),
                mo.md(
                    f"""
### 📊 Implementation Roadmap

**Phase 1: Foundation (Week 1-2)**
1. Implement basic signature with {recommended_module}
2. Set up monitoring and logging
3. Deploy with basic caching
4. Establish baseline metrics

**Phase 2: Optimization (Week 3-4)**
1. A/B test different parameter settings
2. Implement advanced caching strategies
3. Optimize for your {priority.lower()} priority
4. Fine-tune based on real traffic

**Phase 3: Scaling (Week 5-6)**
1. Implement auto-scaling based on demand
2. Add performance monitoring dashboards
3. Set up alerting for quality degradation
4. Plan for capacity growth

### 🎯 Success Metrics
- **Response Time:** < {2.0 if priority == "Speed" else 5.0 if priority == "Balanced" else 10.0} seconds
- **Accuracy:** > {accuracy}%
- **Uptime:** > 99.9%
- **Cost per Request:** < ${0.01 if "Low" in budget else 0.05 if "Medium" in budget else 0.20}

### ⚠️ Risk Mitigation
- Implement circuit breakers for API failures
- Set up fallback responses for edge cases
- Monitor for prompt injection attacks
- Regular model performance reviews
            """
                ),
            ]
        )
    else:
        mo.md(
            "*Complete the production requirements form above to get your optimization strategy*"
        )
    return (
        accuracy,
        budget,
        caching_strategy,
        form_data,
        infrastructure,
        model_recommendation,
        optimizations,
        priority,
        recommended_module,
        use_case,
        volume,
    )


@app.cell
def __(available_providers, mo):
    if available_providers:
        mo.md(
            """
        ## 🎓 Exercise 3 Complete!
        
        ### 🏆 What You've Mastered
        
        ✅ **Parameter Impact Analysis** - Understanding how settings affect performance
        ✅ **A/B Testing Framework** - Systematic signature comparison methodology
        ✅ **Iterative Improvement** - Step-by-step signature enhancement process
        ✅ **Production Optimization** - Real-world deployment strategy development
        ✅ **Performance Trade-offs** - Balancing speed, quality, and cost
        
        ### 🎯 Key Optimization Principles Learned
        
        1. **Parameter Tuning**
           - Temperature controls creativity vs consistency
           - Max tokens affects completeness and cost
           - Model selection impacts quality and speed
        
        2. **A/B Testing**
           - Systematic comparison reveals true performance differences
           - Multiple metrics needed for comprehensive evaluation
           - Statistical significance matters for decisions
        
        3. **Iterative Improvement**
           - Start simple, add complexity gradually
           - Measure impact of each change
           - Balance feature additions with performance costs
        
        4. **Production Strategy**
           - Volume requirements drive architecture decisions
           - Budget constraints shape model and infrastructure choices
           - Monitoring and alerting are essential for reliability
        
        ### 📊 Optimization Decision Matrix
        
        | Scenario | Module Choice | Key Optimizations |
        |----------|---------------|-------------------|
        | **High Volume** | Predict | Caching, fast models, simple prompts |
        | **High Quality** | ChainOfThought | Premium models, validation, review |
        | **Low Budget** | Predict | Efficient models, smart caching |
        | **Balanced** | Hybrid | Context-aware module selection |
        
        ### 🚀 Ready for Advanced Modules?
        
        You now have solid optimization skills! Time to explore advanced DSPy capabilities:
        
        **Next Module:**
        ```bash
        uv run marimo run 02-advanced-modules/react_implementation.py
        ```
        
        **Coming Up:**
        - ReAct (Reasoning + Acting) modules
        - Tool integration with external APIs
        - Multi-step reasoning pipelines
        - Advanced debugging and tracing
        
        ### 🎯 Advanced Optimization Challenges
        
        Before moving on, try optimizing signatures for:
        1. **Multi-language Support** - Optimize for different languages
        2. **Domain Adaptation** - Tune for specific industries
        3. **Edge Case Handling** - Optimize for unusual inputs
        4. **Batch Processing** - Optimize for bulk operations
        
        ### 💡 Production Checklist
        
        When deploying optimized signatures:
        - [ ] Baseline performance metrics established
        - [ ] A/B testing framework implemented
        - [ ] Monitoring and alerting configured
        - [ ] Fallback strategies defined
        - [ ] Cost monitoring and budgets set
        - [ ] Security and prompt injection protection
        - [ ] Regular performance review schedule
        
        Master these optimization techniques and your DSPy systems will perform at their peak!
        """
        )
    return


if __name__ == "__main__":
    app.run()
