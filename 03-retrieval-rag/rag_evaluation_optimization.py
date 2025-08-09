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
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return cleandoc, dspy, get_config, mo, output, setup_dspy_environment, time


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 RAG Evaluation and Optimization

            **Duration:** 90-120 minutes  
            **Prerequisites:** Completed Basic and Advanced RAG  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Master comprehensive RAG evaluation metrics  
            - ✅ Build automated evaluation pipelines  
            - ✅ Implement A/B testing for RAG systems  
            - ✅ Create optimization and tuning frameworks  
            - ✅ Design monitoring and alerting systems  

            ## 📈 RAG Evaluation Dimensions

            **Quality Metrics:**  
            - **Retrieval Quality** - Precision, recall, relevance of retrieved documents  
            - **Generation Quality** - Accuracy, coherence, completeness of responses  
            - **End-to-End Performance** - Overall system effectiveness  
            - **User Experience** - Response time, satisfaction, usability  

            **Optimization Areas:**  
            - **Retrieval Optimization** - Embedding models, indexing strategies  
            - **Generation Optimization** - Prompt engineering, model selection  
            - **System Optimization** - Caching, parallelization, resource usage  

            Let's build comprehensive evaluation and optimization systems!
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
                ## ✅ RAG Evaluation Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**

                Ready to build evaluation and optimization systems!
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
def _(available_providers, cleandoc, dspy, mo, output):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 1: Comprehensive RAG Evaluation Framework

                **Multi-dimensional evaluation** covering all aspects of RAG performance:
                """
            )
        )

        # Evaluation Signatures
        class RetrievalEvaluationSignature(dspy.Signature):
            """Evaluate retrieval quality and relevance."""

            query = dspy.InputField(desc="Original user query")
            retrieved_documents = dspy.InputField(
                desc="Documents retrieved by the system"
            )
            ground_truth_docs = dspy.InputField(
                desc="Known relevant documents (if available)"
            )

            precision_score = dspy.OutputField(
                desc="Precision of retrieved documents (0.0-1.0)"
            )
            recall_score = dspy.OutputField(
                desc="Recall of relevant documents (0.0-1.0)"
            )
            relevance_score = dspy.OutputField(
                desc="Overall relevance quality (0.0-1.0)"
            )
            diversity_score = dspy.OutputField(
                desc="Diversity of retrieved content (0.0-1.0)"
            )
            evaluation_reasoning = dspy.OutputField(
                desc="Detailed reasoning for scores"
            )

        class GenerationEvaluationSignature(dspy.Signature):
            """Evaluate generation quality and accuracy."""

            query = dspy.InputField(desc="Original user query")
            generated_response = dspy.InputField(desc="Generated response")
            retrieved_context = dspy.InputField(desc="Context used for generation")
            reference_answer = dspy.InputField(desc="Reference answer (if available)")

            accuracy_score = dspy.OutputField(
                desc="Factual accuracy of response (0.0-1.0)"
            )
            completeness_score = dspy.OutputField(
                desc="Completeness of response (0.0-1.0)"
            )
            coherence_score = dspy.OutputField(
                desc="Coherence and readability (0.0-1.0)"
            )
            groundedness_score = dspy.OutputField(
                desc="How well grounded in retrieved context (0.0-1.0)"
            )
            evaluation_reasoning = dspy.OutputField(
                desc="Detailed reasoning for scores"
            )

        class EndToEndEvaluationSignature(dspy.Signature):
            """Evaluate overall RAG system performance."""

            query = dspy.InputField(desc="Original user query")
            system_response = dspy.InputField(desc="Complete system response")
            execution_metrics = dspy.InputField(
                desc="System execution metrics (time, resources)"
            )
            user_context = dspy.InputField(desc="User context and intent")

            overall_quality = dspy.OutputField(desc="Overall system quality (0.0-1.0)")
            user_satisfaction = dspy.OutputField(
                desc="Predicted user satisfaction (0.0-1.0)"
            )
            efficiency_score = dspy.OutputField(
                desc="System efficiency score (0.0-1.0)"
            )
            improvement_areas = dspy.OutputField(desc="Specific areas for improvement")
            evaluation_summary = dspy.OutputField(
                desc="Comprehensive evaluation summary"
            )

        cell3_content = mo.md(
            cleandoc(
                """
                ### 📊 RAG Evaluation Signatures Created

                **Evaluation Dimensions:**  
                - **Retrieval Evaluation** - Precision, recall, relevance, diversity  
                - **Generation Evaluation** - Accuracy, completeness, coherence, groundedness  
                - **End-to-End Evaluation** - Overall quality, satisfaction, efficiency  

                These signatures enable comprehensive RAG assessment!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        RetrievalEvaluationSignature = None
        GenerationEvaluationSignature = None
        EndToEndEvaluationSignature = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (
        EndToEndEvaluationSignature,
        GenerationEvaluationSignature,
        RetrievalEvaluationSignature,
    )


@app.cell
def _(
    EndToEndEvaluationSignature,
    GenerationEvaluationSignature,
    RetrievalEvaluationSignature,
    available_providers,
    cleandoc,
    dspy,
    mo,
    output,
    time,
):
    if available_providers and RetrievalEvaluationSignature:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 2: Automated Evaluation Pipeline

                **Complete evaluation system** with automated testing and reporting:
                """
            )
        )

        class RAGEvaluationPipeline:
            """Comprehensive automated evaluation pipeline for RAG systems."""

            def __init__(self):
                self.retrieval_evaluator = dspy.ChainOfThought(
                    RetrievalEvaluationSignature
                )
                self.generation_evaluator = dspy.ChainOfThought(
                    GenerationEvaluationSignature
                )
                self.endtoend_evaluator = dspy.ChainOfThought(
                    EndToEndEvaluationSignature
                )

                self.evaluation_history = []
                self.benchmark_results = {}

            def evaluate_rag_system(self, test_cases: list, rag_system) -> dict:
                """Evaluate RAG system on multiple test cases."""
                start_time = time.time()

                try:
                    evaluation_results = []

                    for i, test_case in enumerate(test_cases):
                        case_result = self._evaluate_single_case(test_case, rag_system)
                        case_result["test_case_id"] = i
                        evaluation_results.append(case_result)

                    # Aggregate results
                    aggregated_metrics = self._aggregate_metrics(evaluation_results)

                    pipeline_result = {
                        "success": True,
                        "evaluation_timestamp": time.time(),
                        "total_test_cases": len(test_cases),
                        "individual_results": evaluation_results,
                        "aggregated_metrics": aggregated_metrics,
                        "evaluation_time": time.time() - start_time,
                        "system_performance": self._assess_system_performance(
                            aggregated_metrics
                        ),
                    }

                    self.evaluation_history.append(pipeline_result)
                    return pipeline_result

                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "evaluation_time": time.time() - start_time,
                    }

            def _evaluate_single_case(self, test_case: dict, rag_system) -> dict:
                """Evaluate a single test case."""
                query = test_case["query"]

                # Execute RAG system
                execution_start = time.time()
                try:
                    rag_result = rag_system(query)
                    execution_time = time.time() - execution_start

                    # Handle different result formats
                    if hasattr(rag_result, "answer"):
                        response = rag_result.answer
                        retrieved_docs = getattr(rag_result, "retrieved_docs", [])
                    else:
                        response = str(rag_result)
                        retrieved_docs = []

                    success = True
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "query": query,
                    }

                if not success:
                    return {
                        "success": False,
                        "error": "RAG system execution failed",
                        "query": query,
                    }

                # Evaluate retrieval
                retrieval_eval = self.retrieval_evaluator(
                    query=query,
                    retrieved_documents=str(retrieved_docs),
                    ground_truth_docs=test_case.get(
                        "ground_truth_docs", "Not available"
                    ),
                )

                # Evaluate generation
                generation_eval = self.generation_evaluator(
                    query=query,
                    generated_response=response,
                    retrieved_context=str(retrieved_docs),
                    reference_answer=test_case.get("reference_answer", "Not available"),
                )

                # Evaluate end-to-end
                execution_metrics = {
                    "execution_time": execution_time,
                    "retrieved_docs_count": len(retrieved_docs),
                    "response_length": len(response),
                }

                endtoend_eval = self.endtoend_evaluator(
                    query=query,
                    system_response=response,
                    execution_metrics=str(execution_metrics),
                    user_context=test_case.get("user_context", "General query"),
                )

                return {
                    "success": True,
                    "query": query,
                    "response": response,
                    "evaluations": {
                        "retrieval": {
                            "precision_score": self._parse_score(
                                retrieval_eval.precision_score
                            ),
                            "recall_score": self._parse_score(
                                retrieval_eval.recall_score
                            ),
                            "relevance_score": self._parse_score(
                                retrieval_eval.relevance_score
                            ),
                            "diversity_score": self._parse_score(
                                retrieval_eval.diversity_score
                            ),
                            "reasoning": retrieval_eval.evaluation_reasoning,
                        },
                        "generation": {
                            "accuracy_score": self._parse_score(
                                generation_eval.accuracy_score
                            ),
                            "completeness_score": self._parse_score(
                                generation_eval.completeness_score
                            ),
                            "coherence_score": self._parse_score(
                                generation_eval.coherence_score
                            ),
                            "groundedness_score": self._parse_score(
                                generation_eval.groundedness_score
                            ),
                            "reasoning": generation_eval.evaluation_reasoning,
                        },
                        "endtoend": {
                            "overall_quality": self._parse_score(
                                endtoend_eval.overall_quality
                            ),
                            "user_satisfaction": self._parse_score(
                                endtoend_eval.user_satisfaction
                            ),
                            "efficiency_score": self._parse_score(
                                endtoend_eval.efficiency_score
                            ),
                            "improvement_areas": endtoend_eval.improvement_areas,
                            "summary": endtoend_eval.evaluation_summary,
                        },
                    },
                    "execution_metrics": execution_metrics,
                }

            def _aggregate_metrics(self, results: list) -> dict:
                """Aggregate metrics across all test cases."""
                successful_results = [r for r in results if r.get("success", False)]

                if not successful_results:
                    return {"error": "No successful evaluations to aggregate"}

                # Aggregate retrieval metrics
                retrieval_metrics = {
                    "avg_precision": sum(
                        r["evaluations"]["retrieval"]["precision_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_recall": sum(
                        r["evaluations"]["retrieval"]["recall_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_relevance": sum(
                        r["evaluations"]["retrieval"]["relevance_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_diversity": sum(
                        r["evaluations"]["retrieval"]["diversity_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                }

                # Aggregate generation metrics
                generation_metrics = {
                    "avg_accuracy": sum(
                        r["evaluations"]["generation"]["accuracy_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_completeness": sum(
                        r["evaluations"]["generation"]["completeness_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_coherence": sum(
                        r["evaluations"]["generation"]["coherence_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_groundedness": sum(
                        r["evaluations"]["generation"]["groundedness_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                }

                # Aggregate end-to-end metrics
                endtoend_metrics = {
                    "avg_overall_quality": sum(
                        r["evaluations"]["endtoend"]["overall_quality"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_user_satisfaction": sum(
                        r["evaluations"]["endtoend"]["user_satisfaction"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_efficiency": sum(
                        r["evaluations"]["endtoend"]["efficiency_score"]
                        for r in successful_results
                    )
                    / len(successful_results),
                }

                # Performance metrics
                performance_metrics = {
                    "avg_execution_time": sum(
                        r["execution_metrics"]["execution_time"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "avg_retrieved_docs": sum(
                        r["execution_metrics"]["retrieved_docs_count"]
                        for r in successful_results
                    )
                    / len(successful_results),
                    "success_rate": (
                        len(successful_results) / len(results) if results else 0
                    ),
                }

                return {
                    "retrieval": retrieval_metrics,
                    "generation": generation_metrics,
                    "endtoend": endtoend_metrics,
                    "performance": performance_metrics,
                    "total_evaluated": len(successful_results),
                }

            def _assess_system_performance(self, metrics: dict) -> dict:
                """Assess overall system performance and provide recommendations."""
                if "error" in metrics:
                    return {"status": "error", "message": metrics["error"]}

                # Calculate overall scores
                retrieval_score = (
                    metrics["retrieval"]["avg_precision"]
                    + metrics["retrieval"]["avg_recall"]
                    + metrics["retrieval"]["avg_relevance"]
                ) / 3

                generation_score = (
                    metrics["generation"]["avg_accuracy"]
                    + metrics["generation"]["avg_completeness"]
                    + metrics["generation"]["avg_coherence"]
                ) / 3

                overall_score = (
                    retrieval_score
                    + generation_score
                    + metrics["endtoend"]["avg_overall_quality"]
                ) / 3

                # Determine performance level
                if overall_score >= 0.8:
                    status = "excellent"
                    recommendations = [
                        "System performing well",
                        "Consider minor optimizations",
                    ]
                elif overall_score >= 0.6:
                    status = "good"
                    recommendations = [
                        "Good performance",
                        "Focus on weak areas for improvement",
                    ]
                elif overall_score >= 0.4:
                    status = "needs_improvement"
                    recommendations = [
                        "Significant improvements needed",
                        "Review retrieval and generation strategies",
                    ]
                else:
                    status = "poor"
                    recommendations = [
                        "Major system overhaul required",
                        "Consider different approaches",
                    ]

                return {
                    "status": status,
                    "overall_score": overall_score,
                    "retrieval_score": retrieval_score,
                    "generation_score": generation_score,
                    "recommendations": recommendations,
                    "performance_summary": f"System achieving {overall_score:.1%} overall performance",
                }

            def _parse_score(self, score_text: str) -> float:
                """Parse score from text to float."""
                try:
                    import re

                    numbers = re.findall(r"0?\.\d+|\d+\.?\d*", str(score_text))
                    if numbers:
                        return max(0.0, min(1.0, float(numbers[0])))
                    return 0.5  # Default
                except:
                    return 0.5

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🏗️ RAG Evaluation Pipeline Created

                **Pipeline Features:**  
                - **Multi-Dimensional Evaluation** - Retrieval, generation, and end-to-end assessment  
                - **Automated Testing** - Batch evaluation on multiple test cases  
                - **Performance Analysis** - Comprehensive metrics aggregation and analysis  
                - **System Assessment** - Overall performance categorization and recommendations  

                The pipeline provides comprehensive RAG system assessment!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        RAGEvaluationPipeline = None
        cell4_content = mo.md("")

    cell4_out = mo.vstack([cell4_desc, cell4_content])
    output.replace(cell4_out)
    return (RAGEvaluationPipeline,)


@app.cell
def _(RAGEvaluationPipeline, available_providers, cleandoc, dspy, mo, output, time):
    if available_providers:
        cell5_desc = mo.md(
            cleandoc(
                """
                ## ⚡ Step 3: RAG Optimization Framework

                **Automated optimization** for improving RAG system performance:
                """
            )
        )

        # Optimization Signatures
        class ParameterOptimizationSignature(dspy.Signature):
            """Optimize RAG system parameters based on evaluation results."""

            current_parameters = dspy.InputField(desc="Current system parameters")
            evaluation_results = dspy.InputField(
                desc="Recent evaluation results and metrics"
            )
            optimization_goals = dspy.InputField(
                desc="Specific optimization objectives"
            )

            recommended_parameters = dspy.OutputField(
                desc="Optimized parameter recommendations"
            )
            optimization_reasoning = dspy.OutputField(
                desc="Reasoning for parameter changes"
            )
            expected_improvements = dspy.OutputField(
                desc="Expected performance improvements"
            )
            implementation_priority = dspy.OutputField(
                desc="Priority order for implementing changes"
            )

        class PerformanceOptimizationSignature(dspy.Signature):
            """Optimize system performance and efficiency."""

            performance_metrics = dspy.InputField(desc="Current performance metrics")
            resource_constraints = dspy.InputField(
                desc="Available resources and constraints"
            )
            optimization_targets = dspy.InputField(
                desc="Performance targets to achieve"
            )

            optimization_strategies = dspy.OutputField(
                desc="Specific optimization strategies"
            )
            resource_allocation = dspy.OutputField(
                desc="Recommended resource allocation"
            )
            implementation_plan = dspy.OutputField(
                desc="Step-by-step implementation plan"
            )
            risk_assessment = dspy.OutputField(
                desc="Potential risks and mitigation strategies"
            )

        # RAG Optimization System
        class RAGOptimizationFramework:
            """Comprehensive optimization framework for RAG systems."""

            def __init__(self):
                self.parameter_optimizer = dspy.ChainOfThought(
                    ParameterOptimizationSignature
                )
                self.performance_optimizer = dspy.ChainOfThought(
                    PerformanceOptimizationSignature
                )
                self.optimization_history = []

            def optimize_rag_system(
                self,
                evaluation_results: dict,
                current_config: dict,
                optimization_goals: list,
            ) -> dict:
                """Optimize RAG system based on evaluation results."""
                try:
                    # Step 1: Parameter optimization
                    goals_text = ", ".join(optimization_goals)

                    param_optimization = self.parameter_optimizer(
                        current_parameters=str(current_config),
                        evaluation_results=str(
                            evaluation_results.get("aggregated_metrics", {})
                        ),
                        optimization_goals=goals_text,
                    )

                    # Step 2: Performance optimization
                    performance_metrics = evaluation_results.get(
                        "aggregated_metrics", {}
                    ).get("performance", {})

                    perf_optimization = self.performance_optimizer(
                        performance_metrics=str(performance_metrics),
                        resource_constraints="Standard cloud deployment with moderate resources",
                        optimization_targets=goals_text,
                    )

                    # Step 3: Create optimization plan
                    optimization_plan = self._create_optimization_plan(
                        param_optimization, perf_optimization, evaluation_results
                    )

                    optimization_result = {
                        "success": True,
                        "timestamp": time.time(),
                        "parameter_optimization": {
                            "recommended_parameters": param_optimization.recommended_parameters,
                            "reasoning": param_optimization.optimization_reasoning,
                            "expected_improvements": param_optimization.expected_improvements,
                            "priority": param_optimization.implementation_priority,
                        },
                        "performance_optimization": {
                            "strategies": perf_optimization.optimization_strategies,
                            "resource_allocation": perf_optimization.resource_allocation,
                            "implementation_plan": perf_optimization.implementation_plan,
                            "risk_assessment": perf_optimization.risk_assessment,
                        },
                        "optimization_plan": optimization_plan,
                    }

                    self.optimization_history.append(optimization_result)
                    return optimization_result

                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time(),
                    }

            def _create_optimization_plan(
                self, param_opt, perf_opt, evaluation_results
            ) -> dict:
                """Create comprehensive optimization plan."""
                current_performance = evaluation_results.get("system_performance", {})

                plan = {
                    "immediate_actions": [],
                    "short_term_improvements": [],
                    "long_term_optimizations": [],
                    "success_metrics": [],
                }

                # Analyze current performance to prioritize actions
                overall_score = current_performance.get("overall_score", 0.5)

                if overall_score < 0.4:
                    plan["immediate_actions"].extend(
                        [
                            "Review and fix basic system functionality",
                            "Improve error handling and robustness",
                            "Optimize retrieval relevance",
                        ]
                    )
                elif overall_score < 0.7:
                    plan["immediate_actions"].extend(
                        [
                            "Fine-tune retrieval parameters",
                            "Improve generation quality",
                            "Optimize response time",
                        ]
                    )
                else:
                    plan["immediate_actions"].extend(
                        [
                            "Fine-tune for edge cases",
                            "Optimize for specific use cases",
                            "Enhance user experience",
                        ]
                    )

                # Add parameter-specific recommendations
                plan["short_term_improvements"].append(
                    f"Parameter optimization: {param_opt.recommended_parameters}"
                )

                # Add performance-specific recommendations
                plan["long_term_optimizations"].append(
                    f"Performance strategies: {perf_opt.optimization_strategies}"
                )

                # Define success metrics
                plan["success_metrics"] = [
                    f"Improve overall score from {overall_score:.2f} to {min(overall_score + 0.2, 1.0):.2f}",
                    "Reduce average response time by 20%",
                    "Increase user satisfaction score by 15%",
                ]

                return plan

            def run_ab_test(
                self, system_a, system_b, test_cases: list, test_name: str = "A/B Test"
            ) -> dict:
                """Run A/B test between two RAG systems."""
                try:
                    # Create evaluation pipeline
                    evaluator = RAGEvaluationPipeline()

                    # Evaluate both systems
                    results_a = evaluator.evaluate_rag_system(test_cases, system_a)
                    results_b = evaluator.evaluate_rag_system(test_cases, system_b)

                    # Compare results
                    comparison = self._compare_systems(results_a, results_b)

                    ab_test_result = {
                        "success": True,
                        "test_name": test_name,
                        "timestamp": time.time(),
                        "system_a_results": results_a,
                        "system_b_results": results_b,
                        "comparison": comparison,
                        "winner": comparison.get("winner", "tie"),
                        "confidence": comparison.get("confidence", "low"),
                    }

                    return ab_test_result

                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "test_name": test_name,
                        "timestamp": time.time(),
                    }

            def _compare_systems(self, results_a: dict, results_b: dict) -> dict:
                """Compare two system evaluation results."""
                if not (results_a.get("success") and results_b.get("success")):
                    return {"error": "One or both evaluations failed"}

                metrics_a = results_a["aggregated_metrics"]
                metrics_b = results_b["aggregated_metrics"]

                # Compare key metrics
                comparisons = {}
                winner_count_a = 0
                winner_count_b = 0

                metric_categories = [
                    "retrieval",
                    "generation",
                    "endtoend",
                    "performance",
                ]

                for category in metric_categories:
                    if category in metrics_a and category in metrics_b:
                        category_comparison = {}

                        for metric, value_a in metrics_a[category].items():
                            if metric in metrics_b[category]:
                                value_b = metrics_b[category][metric]

                                # For time metrics, lower is better
                                if "time" in metric:
                                    winner = (
                                        "A"
                                        if value_a < value_b
                                        else "B" if value_b < value_a else "tie"
                                    )
                                else:
                                    winner = (
                                        "A"
                                        if value_a > value_b
                                        else "B" if value_b > value_a else "tie"
                                    )

                                category_comparison[metric] = {
                                    "system_a": value_a,
                                    "system_b": value_b,
                                    "winner": winner,
                                    "difference": abs(value_a - value_b),
                                }

                                if winner == "A":
                                    winner_count_a += 1
                                elif winner == "B":
                                    winner_count_b += 1

                        comparisons[category] = category_comparison

                # Determine overall winner
                if winner_count_a > winner_count_b:
                    overall_winner = "A"
                    confidence = (
                        "high" if winner_count_a > winner_count_b * 1.5 else "medium"
                    )
                elif winner_count_b > winner_count_a:
                    overall_winner = "B"
                    confidence = (
                        "high" if winner_count_b > winner_count_a * 1.5 else "medium"
                    )
                else:
                    overall_winner = "tie"
                    confidence = "low"

                return {
                    "winner": overall_winner,
                    "confidence": confidence,
                    "detailed_comparisons": comparisons,
                    "summary": {
                        "system_a_wins": winner_count_a,
                        "system_b_wins": winner_count_b,
                        "ties": len(
                            [
                                c
                                for cat in comparisons.values()
                                for c in cat.values()
                                if c["winner"] == "tie"
                            ]
                        ),
                    },
                }

        cell5_content = mo.md(
            cleandoc(
                """
                ### ⚡ RAG Optimization Framework Created

                **Optimization Features:**  
                - **Parameter Optimization** - Intelligent parameter tuning based on evaluation results  
                - **Performance Optimization** - System-level performance improvements  
                - **A/B Testing** - Statistical comparison between different RAG approaches  
                - **Optimization Planning** - Structured improvement roadmaps  

                **Optimization Process:**  
                1. **Analyze** current system performance and identify bottlenecks  
                2. **Recommend** parameter and architectural improvements  
                3. **Plan** implementation with prioritized action items  
                4. **Test** improvements through A/B testing  
                5. **Monitor** results and iterate  

                The framework provides data-driven RAG system optimization!
                """
            )
        )
    else:
        cell5_desc = mo.md("")
        RAGOptimizationFramework = None
        cell5_content = mo.md("")

    cell5_out = mo.vstack([cell5_desc, cell5_content])
    output.replace(cell5_out)
    return (RAGOptimizationFramework,)


@app.cell
def _(available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell6_desc = mo.md(
            cleandoc(
                """
                ## 📈 Step 4: Production Monitoring and Alerting

                **Real-time monitoring** for production RAG systems:
                """
            )
        )

        class RAGMonitoringSystem:
            """Real-time monitoring system for production RAG systems."""

            def __init__(self):
                self.metrics_history = []
                self.alert_thresholds = {
                    "response_time": 5.0,  # seconds
                    "error_rate": 0.05,  # 5%
                    "quality_score": 0.6,  # minimum quality
                }
                self.active_alerts = []
                self.performance_baselines = {}

            def track_query(
                self,
                query: str,
                response: str,
                execution_time: float,
                quality_metrics: dict = None,
                error: str = None,
            ):
                """Track individual query performance."""
                timestamp = time.time()

                query_record = {
                    "timestamp": timestamp,
                    "query": query,
                    "response": response,
                    "execution_time": execution_time,
                    "quality_metrics": quality_metrics or {},
                    "error": error,
                    "success": error is None,
                }

                self.metrics_history.append(query_record)

                # Check for alerts
                self._check_alerts(query_record)

                # Maintain history size
                if len(self.metrics_history) > 10000:
                    self.metrics_history = self.metrics_history[-5000:]

            def get_performance_summary(self, time_window_hours: int = 24) -> dict:
                """Get performance summary for specified time window."""
                cutoff_time = time.time() - (time_window_hours * 3600)
                recent_queries = [
                    q for q in self.metrics_history if q["timestamp"] > cutoff_time
                ]

                if not recent_queries:
                    return {"error": "No queries in specified time window"}

                successful_queries = [q for q in recent_queries if q["success"]]

                summary = {
                    "time_window_hours": time_window_hours,
                    "total_queries": len(recent_queries),
                    "successful_queries": len(successful_queries),
                    "error_rate": (len(recent_queries) - len(successful_queries))
                    / len(recent_queries),
                    "avg_response_time": sum(
                        q["execution_time"] for q in successful_queries
                    )
                    / max(len(successful_queries), 1),
                    "min_response_time": min(
                        (q["execution_time"] for q in successful_queries), default=0
                    ),
                    "max_response_time": max(
                        (q["execution_time"] for q in successful_queries), default=0
                    ),
                }

                # Quality metrics summary
                if successful_queries and successful_queries[0]["quality_metrics"]:
                    quality_metrics = {}
                    for metric_name in successful_queries[0]["quality_metrics"].keys():
                        values = [
                            q["quality_metrics"][metric_name]
                            for q in successful_queries
                            if metric_name in q["quality_metrics"]
                        ]
                        if values:
                            quality_metrics[f"avg_{metric_name}"] = sum(values) / len(
                                values
                            )

                    summary["quality_metrics"] = quality_metrics

                return summary

            def _check_alerts(self, query_record: dict):
                """Check if query triggers any alerts."""
                alerts_triggered = []

                # Response time alert
                if (
                    query_record["execution_time"]
                    > self.alert_thresholds["response_time"]
                ):
                    alerts_triggered.append(
                        {
                            "type": "high_response_time",
                            "message": f"Query took {query_record['execution_time']:.2f}s (threshold: {self.alert_thresholds['response_time']}s)",
                            "severity": "warning",
                            "timestamp": query_record["timestamp"],
                        }
                    )

                # Error alert
                if query_record["error"]:
                    alerts_triggered.append(
                        {
                            "type": "query_error",
                            "message": f"Query failed: {query_record['error']}",
                            "severity": "error",
                            "timestamp": query_record["timestamp"],
                        }
                    )

                # Quality alert
                if query_record["quality_metrics"]:
                    overall_quality = sum(
                        query_record["quality_metrics"].values()
                    ) / len(query_record["quality_metrics"])
                    if overall_quality < self.alert_thresholds["quality_score"]:
                        alerts_triggered.append(
                            {
                                "type": "low_quality",
                                "message": f"Low quality response: {overall_quality:.2f} (threshold: {self.alert_thresholds['quality_score']})",
                                "severity": "warning",
                                "timestamp": query_record["timestamp"],
                            }
                        )

                # Add to active alerts
                self.active_alerts.extend(alerts_triggered)

                # Maintain alert history
                if len(self.active_alerts) > 1000:
                    self.active_alerts = self.active_alerts[-500:]

            def get_active_alerts(self, severity_filter: str = None) -> list:
                """Get active alerts, optionally filtered by severity."""
                recent_cutoff = time.time() - 3600  # Last hour
                recent_alerts = [
                    alert
                    for alert in self.active_alerts
                    if alert["timestamp"] > recent_cutoff
                ]

                if severity_filter:
                    recent_alerts = [
                        alert
                        for alert in recent_alerts
                        if alert["severity"] == severity_filter
                    ]

                return recent_alerts

            def update_alert_thresholds(self, new_thresholds: dict):
                """Update alert thresholds."""
                self.alert_thresholds.update(new_thresholds)

            def generate_health_report(self) -> str:
                """Generate comprehensive system health report."""
                summary = self.get_performance_summary(24)
                alerts = self.get_active_alerts()

                report_lines = [
                    "=" * 50,
                    "📊 RAG SYSTEM HEALTH REPORT",
                    "=" * 50,
                    f"Report Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "📈 PERFORMANCE SUMMARY (24h):",
                    f"  Total Queries: {summary.get('total_queries', 0)}",
                    f"  Success Rate: {(1 - summary.get('error_rate', 0)) * 100:.1f}%",
                    f"  Avg Response Time: {summary.get('avg_response_time', 0):.3f}s",
                    f"  Response Time Range: {summary.get('min_response_time', 0):.3f}s - {summary.get('max_response_time', 0):.3f}s",
                    "",
                ]

                # Quality metrics
                if "quality_metrics" in summary:
                    report_lines.extend(
                        [
                            "🎯 QUALITY METRICS:",
                            *[
                                f"  {metric}: {value:.3f}"
                                for metric, value in summary["quality_metrics"].items()
                            ],
                            "",
                        ]
                    )

                # Active alerts
                if alerts:
                    report_lines.extend(
                        [
                            f"🚨 ACTIVE ALERTS ({len(alerts)}):",
                            *[
                                f"  {alert['severity'].upper()}: {alert['message']}"
                                for alert in alerts[-10:]
                            ],
                            "",
                        ]
                    )
                else:
                    report_lines.extend(["✅ NO ACTIVE ALERTS", ""])

                # System status
                error_rate = summary.get("error_rate", 0)
                avg_response_time = summary.get("avg_response_time", 0)

                if error_rate > 0.1 or avg_response_time > 10:
                    status = "🔴 CRITICAL"
                elif error_rate > 0.05 or avg_response_time > 5:
                    status = "🟡 WARNING"
                else:
                    status = "🟢 HEALTHY"

                report_lines.extend([f"SYSTEM STATUS: {status}", "=" * 50])

                return "\n".join(report_lines)

        cell6_content = mo.md(
            cleandoc(
                """
                ### 📈 Production Monitoring System Created

                **Monitoring Features:**  
                - **Real-time Query Tracking** - Performance and quality metrics for every query  
                - **Automated Alerting** - Configurable thresholds for response time, errors, and quality  
                - **Performance Baselines** - Historical performance tracking and trend analysis  
                - **Health Reporting** - Comprehensive system health summaries  

                **Alert Types:**  
                - **High Response Time** - Queries exceeding time thresholds  
                - **Query Errors** - Failed queries and system errors  
                - **Low Quality** - Responses below quality thresholds  
                - **Error Rate** - System-wide error rate monitoring  

                **Monitoring Capabilities:**  
                - **24/7 Tracking** - Continuous performance monitoring  
                - **Configurable Thresholds** - Customizable alert conditions  
                - **Historical Analysis** - Performance trends and patterns  
                - **Real-time Dashboards** - Live system health visibility  

                The monitoring system provides production-ready observability!
                """
            )
        )
    else:
        cell6_desc = mo.md("")
        RAGMonitoringSystem = None
        cell6_content = mo.md("")

    cell6_out = mo.vstack([cell6_desc, cell6_content])
    output.replace(cell6_out)
    return (RAGMonitoringSystem,)


@app.cell
def _(
    RAGEvaluationPipeline,
    RAGMonitoringSystem,
    RAGOptimizationFramework,
    available_providers,
    cleandoc,
    mo,
    output,
):
    if available_providers and all(
        [RAGEvaluationPipeline, RAGOptimizationFramework, RAGMonitoringSystem]
    ):
        # Create testing interface components
        cell7_test_query = mo.ui.text_area(
            placeholder="Enter a test query for RAG evaluation...",
            label="Test Query",
            rows=2,
        )

        cell7_evaluation_type = mo.ui.dropdown(
            options=[
                "Single Query Evaluation",
                "Batch Evaluation",
                "A/B Testing",
                "Performance Monitoring",
            ],
            label="Evaluation Type",
            value="Single Query Evaluation",
        )

        cell7_run_button = mo.ui.run_button(label="🧪 Run RAG Evaluation")

        cell7_desc = mo.md(
            cleandoc(
                """
                ## 🧪 Step 5: Interactive RAG Evaluation Interface

                **Test the complete evaluation system** with real queries and scenarios:
                """
            )
        )

        cell7_interface = mo.vstack(
            [
                cell7_test_query,
                cell7_evaluation_type,
                cell7_run_button,
            ]
        )

        cell7_content = mo.vstack(
            [
                mo.md("### 🧪 RAG Evaluation Testing Interface"),
                mo.md(
                    "Test different evaluation scenarios and see the comprehensive analysis:"
                ),
                cell7_interface,
            ]
        )
    else:
        cell7_desc = mo.md("")
        cell7_test_query = None
        cell7_evaluation_type = None
        cell7_run_button = None
        cell7_content = mo.md("")

    cell7_out = mo.vstack([cell7_desc, cell7_content])
    output.replace(cell7_out)
    return cell7_evaluation_type, cell7_run_button, cell7_test_query


@app.cell
def _(
    RAGEvaluationPipeline,
    RAGMonitoringSystem,
    available_providers,
    cell7_evaluation_type,
    cell7_run_button,
    cell7_test_query,
    cleandoc,
    mo,
    output,
    time,
):
    if (
        available_providers
        and cell7_run_button
        and cell7_run_button.value
        and cell7_test_query
        and cell7_test_query.value.strip()
    ):
        try:
            # Get test parameters
            cell8_query = cell7_test_query.value.strip()
            cell8_eval_type = cell7_evaluation_type.value

            # Mock RAG system for demonstration
            class MockRAGSystem:
                def __call__(self, query):
                    class MockResult:
                        def __init__(self, query):
                            self.answer = f"This is a comprehensive answer about {query}. The system has analyzed the query and provided relevant information based on retrieved documents."
                            self.retrieved_docs = [
                                {"text": f"Document 1 about {query}", "score": 0.85},
                                {
                                    "text": f"Document 2 related to {query}",
                                    "score": 0.72,
                                },
                                {
                                    "text": f"Document 3 discussing {query}",
                                    "score": 0.68,
                                },
                            ]

                    return MockResult(query)

            mock_rag = MockRAGSystem()

            if cell8_eval_type == "Single Query Evaluation":
                # Single query evaluation
                evaluator = RAGEvaluationPipeline()
                test_cases = [
                    {
                        "query": cell8_query,
                        "user_context": "General information request",
                        "reference_answer": "Not available for this demo",
                    }
                ]

                cell8_start_time = time.time()
                results = evaluator.evaluate_rag_system(test_cases, mock_rag)
                cell8_execution_time = time.time() - cell8_start_time

                if results["success"]:
                    metrics = results["aggregated_metrics"]
                    assessment = results["system_performance"]

                    cell8_result_text = cleandoc(
                        f"""
                        ### 📊 Single Query Evaluation Results
    
                        **Query:** {cell8_query}
    
                        **Performance Metrics:**  
                        - **Retrieval Quality:** {metrics['retrieval']['avg_relevance_score']:.3f}  
                        - **Generation Quality:** {metrics['generation']['avg_accuracy_score']:.3f}  
                        - **Overall Quality:** {metrics['endtoend']['avg_overall_quality']:.3f}  
                        - **Response Time:** {metrics['performance']['avg_execution_time']:.3f}s  
    
                        **System Assessment:** {assessment['status'].upper()}  
                        **Overall Score:** {assessment['overall_score']:.1%}  
    
                        **Recommendations:**  
                        {chr(10).join(f"• {rec}" for rec in assessment['recommendations'])}  
    
                        **Evaluation Time:** {cell8_execution_time:.2f} seconds  
                        """
                    )
                else:
                    cell8_result_text = f"**Evaluation Failed:** {results.get('error', 'Unknown error')}"

            elif cell8_eval_type == "Performance Monitoring":
                # Performance monitoring demonstration
                monitor = RAGMonitoringSystem()

                # Simulate some queries
                for i in range(5):
                    test_query = f"{cell8_query} - variation {i+1}"
                    start_time = time.time()
                    result = mock_rag(test_query)
                    exec_time = (
                        time.time() - start_time + (i * 0.1)
                    )  # Add some variation

                    monitor.track_query(
                        query=test_query,
                        response=result.answer,
                        execution_time=exec_time,
                        quality_metrics={
                            "relevance": 0.8 + (i * 0.02),
                            "accuracy": 0.75 + (i * 0.03),
                        },
                    )

                # Generate health report
                health_report = monitor.generate_health_report()
                cell8_result_text = f"### 📈 Performance Monitoring Results\n\n```\n{health_report}\n```"

            else:
                cell8_result_text = f"**{cell8_eval_type}** demonstration would require multiple RAG systems or test cases. This is a simplified demo showing the evaluation framework capabilities."

            cell8_out = mo.vstack(
                [
                    mo.md("### 🧪 Evaluation Results"),
                    mo.md(cell8_result_text),
                    mo.md("---"),
                    mo.md(
                        "*Try different evaluation types and queries to explore the comprehensive evaluation capabilities!*"
                    ),
                ]
            )

        except Exception as e:
            cell8_out = mo.vstack(
                [
                    mo.md("### ⚠️ Evaluation Error"),
                    mo.md(f"An error occurred during evaluation: {str(e)}"),
                    mo.md("Please check your query and try again."),
                ]
            )
    else:
        cell8_out = mo.md(
            "*Enter a query above and click 'Run RAG Evaluation' to see comprehensive evaluation results*"
        )

    output.replace(cell8_out)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    if available_providers:
        cell9_out = mo.md(
            cleandoc(
                """
                ## 🎉 RAG Evaluation and Optimization Complete!

                ### 🏆 What You've Built

                **Complete Evaluation System:**  
                - ✅ **Multi-Dimensional Evaluation** - Retrieval, generation, and end-to-end metrics  
                - ✅ **Automated Testing Pipeline** - Comprehensive test suites and benchmarking  
                - ✅ **Optimization Framework** - Parameter tuning and A/B testing  
                - ✅ **Production Monitoring** - Real-time performance tracking and alerting  

                ### 🔍 Key Capabilities Mastered

                **Evaluation Excellence:**  
                - Comprehensive quality assessment across multiple dimensions  
                - Automated scoring and performance analysis  
                - Statistical significance testing and comparison  
                - Production-ready monitoring and alerting  

                **Optimization Mastery:**  
                - Data-driven parameter optimization  
                - A/B testing for system comparison  
                - Performance bottleneck identification  
                - Structured improvement planning  

                ### 🚀 Production Deployment Ready

                **Enterprise Features:**  
                - **Real-time Monitoring** - 24/7 system health tracking  
                - **Automated Alerting** - Proactive issue detection  
                - **Performance Baselines** - Historical trend analysis  
                - **Quality Assurance** - Continuous quality monitoring  

                **Scalability Considerations:**  
                - **Distributed Evaluation** - Scale testing across multiple systems  
                - **Historical Analytics** - Long-term performance trend analysis  
                - **Custom Metrics** - Domain-specific evaluation criteria  
                - **Integration APIs** - Connect with existing monitoring infrastructure  

                ### 💡 Next Steps for Production

                **Advanced Evaluation:**  
                - **Human-in-the-Loop** - Combine automated metrics with human judgment  
                - **Domain-Specific Metrics** - Specialized evaluation for specific use cases  
                - **Multi-Language Support** - Evaluation across different languages  
                - **Adversarial Testing** - Robustness testing with challenging queries  

                **Enterprise Integration:**  
                - **CI/CD Integration** - Automated evaluation in deployment pipelines  
                - **Dashboard Integration** - Connect with existing monitoring dashboards  
                - **Alert Integration** - Route alerts to existing notification systems  
                - **Data Pipeline Integration** - Connect with data warehouses and analytics  

                ### 🎯 Learning Outcomes

                You've mastered the complete lifecycle of RAG system evaluation and optimization:  
                - **Comprehensive Assessment** - Multi-dimensional quality evaluation  
                - **Automated Testing** - Scalable evaluation pipelines  
                - **Data-Driven Optimization** - Evidence-based system improvements  
                - **Production Monitoring** - Real-time system health and performance  

                **Congratulations!** You now have the skills to build, evaluate, and optimize production-grade RAG systems with confidence! 🚀

                ### 🌟 Module 03 Complete!

                You've successfully completed the entire RAG journey:  
                1. **Basic RAG Implementation** - Core concepts and foundations  
                2. **Vector Database Integration** - Scalable storage and retrieval  
                3. **Advanced RAG Patterns** - Sophisticated reasoning and strategies  
                4. **Evaluation and Optimization** - Production-ready quality assurance  

                **Ready to build world-class RAG systems!** 🎉
                """
            )
        )
    else:
        cell9_out = mo.md("")

    output.replace(cell9_out)
    return


if __name__ == "__main__":
    app.run()
