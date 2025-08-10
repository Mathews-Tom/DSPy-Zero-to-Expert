# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import statistics
    import sys
    import time
    from collections import defaultdict
    from dataclasses import asdict
    from datetime import datetime, timedelta
    from inspect import cleandoc
    from pathlib import Path
    from typing import Any, Optional

    import dspy
    import marimo as mo
    from marimo import output

    from common import get_config, setup_dspy_environment

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    return (
        Any,
        cleandoc,
        defaultdict,
        dspy,
        get_config,
        mo,
        output,
        setup_dspy_environment,
        statistics,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 Interactive Evaluation Dashboard

            **Duration:** 2-3 hours  
            **Prerequisites:** Completed evaluation framework and custom metrics modules  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will master:  
            - **Real-time Evaluation** - Monitor evaluation progress as it happens  
            - **Interactive Configuration** - Dynamically configure evaluation parameters  
            - **Visual Analytics** - Create compelling visualizations of evaluation results  
            - **Comparative Analysis** - Compare multiple systems and metrics side-by-side  
            - **Export and Reporting** - Generate comprehensive evaluation reports  

            ## 📈 Dashboard Overview

            **Why Interactive Dashboards Matter:**  
            - **Real-time Insights** - Get immediate feedback on system performance  
            - **Parameter Exploration** - Experiment with different evaluation configurations  
            - **Visual Understanding** - See patterns and trends in evaluation data  
            - **Decision Support** - Make data-driven decisions about system improvements  

            **Dashboard Components:**  
            - **Configuration Panel** - Interactive controls for evaluation setup  
            - **Real-time Monitoring** - Live progress tracking and status updates  
            - **Results Visualization** - Charts, graphs, and statistical summaries  
            - **Comparison Tools** - Side-by-side analysis of different approaches  

            Let's build a comprehensive evaluation dashboard!
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
        # Use the first available provider
        provider = available_providers[0]
        setup_dspy_environment(provider=provider)
        cell2_out = mo.md(
            cleandoc(
                f"""
                ## ✅ Dashboard Environment Ready

                **Configuration:**
                - Provider: **{provider}**
                - Model: **{config.default_model}**
                - Available Providers: **{', '.join(available_providers)}**

                Ready to build interactive evaluation dashboards!
                """
            )
        )
    else:
        cell2_out = mo.md(
            cleandoc(
                """
                ## ⚠️ Setup Required

                Please complete Module 00 setup first to configure your API keys.

                **Required:** At least one of the following API keys:
                - `OPENAI_API_KEY` for OpenAI models
                - `ANTHROPIC_API_KEY` for Anthropic models  
                - `COHERE_API_KEY` for Cohere models

                Add your API key to the `.env` file in the project root.
                """
            )
        )

    output.replace(cell2_out)
    return (available_providers,)


@app.cell
def _(
    Any,
    available_providers,
    cleandoc,
    defaultdict,
    dspy,
    mo,
    output,
    statistics,
    time,
):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Dashboard Core Engine

                **Building the foundation** for interactive evaluation dashboards:
                """
            )
        )

        class EvaluationDashboard:
            """Comprehensive dashboard for evaluation monitoring and analysis."""

            def __init__(self):
                self.evaluation_sessions = {}
                self.active_evaluations = {}
                self.dashboard_config = {
                    "refresh_interval": 2.0,  # seconds
                    "max_history": 50,
                    "auto_save": True,
                    "visualization_theme": "default",
                }
                self.session_counter = 0

            def create_evaluation_session(
                self, name: str, description: str = ""
            ) -> str:
                """Create a new evaluation session."""
                session_id = f"session_{self.session_counter}_{int(time.time())}"
                self.session_counter += 1

                session = {
                    "id": session_id,
                    "name": name,
                    "description": description,
                    "created_at": time.time(),
                    "status": "created",
                    "metrics": {},
                    "systems": {},
                    "results": {},
                    "progress": 0.0,
                    "current_step": "initialization",
                    "logs": [],
                    "metadata": {},
                }

                self.evaluation_sessions[session_id] = session
                return session_id

            def add_system_to_session(
                self, session_id: str, system_name: str, system: Any
            ) -> bool:
                """Add a system to evaluate in the session."""
                if session_id not in self.evaluation_sessions:
                    return False

                session = self.evaluation_sessions[session_id]
                session["systems"][system_name] = {
                    "system": system,
                    "added_at": time.time(),
                    "status": "ready",
                }

                self._log_session_event(session_id, f"Added system: {system_name}")
                return True

            def add_metric_to_session(
                self, session_id: str, metric_name: str, metric: Any
            ) -> bool:
                """Add a metric to the evaluation session."""
                if session_id not in self.evaluation_sessions:
                    return False

                session = self.evaluation_sessions[session_id]
                session["metrics"][metric_name] = {
                    "metric": metric,
                    "added_at": time.time(),
                    "weight": 1.0,
                    "enabled": True,
                }

                self._log_session_event(session_id, f"Added metric: {metric_name}")
                return True

            def start_evaluation(
                self, session_id: str, test_examples: list[dspy.Example]
            ) -> bool:
                """Start evaluation for a session."""
                if session_id not in self.evaluation_sessions:
                    return False

                session = self.evaluation_sessions[session_id]

                if not session["systems"] or not session["metrics"]:
                    self._log_session_event(
                        session_id, "Cannot start: No systems or metrics configured"
                    )
                    return False

                session["status"] = "running"
                session["current_step"] = "evaluation"
                session["test_examples"] = test_examples
                session["start_time"] = time.time()

                self.active_evaluations[session_id] = {
                    "total_combinations": len(session["systems"])
                    * len(session["metrics"]),
                    "completed_combinations": 0,
                    "current_system": None,
                    "current_metric": None,
                    "start_time": time.time(),
                }

                self._log_session_event(
                    session_id, f"Started evaluation with {len(test_examples)} examples"
                )

                # Simulate evaluation process
                self._run_evaluation_simulation(session_id)

                return True

            def _run_evaluation_simulation(self, session_id: str):
                """Simulate evaluation process (in real implementation, this would be async)."""
                session = self.evaluation_sessions[session_id]
                active_eval = self.active_evaluations[session_id]
                test_examples = session["test_examples"]

                results = {}

                for system_name, system_info in session["systems"].items():
                    system = system_info["system"]
                    results[system_name] = {}

                    active_eval["current_system"] = system_name

                    for metric_name, metric_info in session["metrics"].items():
                        if not metric_info["enabled"]:
                            continue

                        active_eval["current_metric"] = metric_name
                        metric = metric_info["metric"]

                        # Simulate evaluation
                        try:
                            # Generate predictions
                            predictions = []
                            for example in test_examples:
                                if hasattr(system, "forward"):
                                    inputs = example.inputs()
                                    pred = system.forward(**inputs)
                                else:
                                    pred = system(example)
                                predictions.append(pred)

                            # Evaluate with metric
                            if hasattr(metric, "batch_evaluate"):
                                metric_results = metric.batch_evaluate(
                                    test_examples, predictions
                                )
                            else:
                                # Fallback for simple metrics
                                metric_results = []
                                for ex, pred in zip(
                                    test_examples, predictions, strict=False
                                ):
                                    score = (
                                        metric.evaluate(ex, pred)
                                        if hasattr(metric, "evaluate")
                                        else metric(ex, pred)
                                    )
                                    if hasattr(score, "score"):
                                        metric_results.append(score)
                                    else:
                                        # Simple score
                                        from dataclasses import dataclass

                                        @dataclass
                                        class SimpleResult:
                                            score: float
                                            metadata: dict = None

                                        metric_results.append(
                                            SimpleResult(score=float(score))
                                        )

                            # Aggregate results
                            scores = [r.score for r in metric_results]

                            results[system_name][metric_name] = {
                                "individual_scores": scores,
                                "mean_score": (
                                    statistics.mean(scores) if scores else 0.0
                                ),
                                "std_dev": (
                                    statistics.stdev(scores) if len(scores) > 1 else 0.0
                                ),
                                "min_score": min(scores) if scores else 0.0,
                                "max_score": max(scores) if scores else 0.0,
                                "total_examples": len(scores),
                                "evaluation_time": time.time()
                                - active_eval["start_time"],
                                "detailed_results": metric_results,
                            }

                        except Exception as e:
                            results[system_name][metric_name] = {
                                "error": str(e),
                                "mean_score": 0.0,
                                "total_examples": 0,
                            }

                        active_eval["completed_combinations"] += 1
                        session["progress"] = (
                            active_eval["completed_combinations"]
                            / active_eval["total_combinations"]
                        )

                # Complete evaluation
                session["results"] = results
                session["status"] = "completed"
                session["end_time"] = time.time()
                session["total_time"] = session["end_time"] - session["start_time"]

                self._log_session_event(session_id, "Evaluation completed successfully")

                if session_id in self.active_evaluations:
                    del self.active_evaluations[session_id]

            def get_session_status(self, session_id: str) -> dict[str, Any]:
                """Get current status of an evaluation session."""
                if session_id not in self.evaluation_sessions:
                    return {"error": "Session not found"}

                session = self.evaluation_sessions[session_id]
                status = {
                    "session_id": session_id,
                    "name": session["name"],
                    "status": session["status"],
                    "progress": session["progress"],
                    "current_step": session["current_step"],
                    "systems_count": len(session["systems"]),
                    "metrics_count": len(
                        [m for m in session["metrics"].values() if m["enabled"]]
                    ),
                    "created_at": session["created_at"],
                }

                if session_id in self.active_evaluations:
                    active = self.active_evaluations[session_id]
                    status.update(
                        {
                            "current_system": active["current_system"],
                            "current_metric": active["current_metric"],
                            "completed_combinations": active["completed_combinations"],
                            "total_combinations": active["total_combinations"],
                        }
                    )

                return status

            def get_session_results(self, session_id: str) -> dict[str, Any]:
                """Get results from a completed evaluation session."""
                if session_id not in self.evaluation_sessions:
                    return {"error": "Session not found"}

                session = self.evaluation_sessions[session_id]

                if session["status"] != "completed":
                    return {
                        "error": "Evaluation not completed",
                        "status": session["status"],
                    }

                return {
                    "session_id": session_id,
                    "name": session["name"],
                    "results": session["results"],
                    "total_time": session.get("total_time", 0),
                    "systems": list(session["systems"].keys()),
                    "metrics": list(session["metrics"].keys()),
                    "summary": self._generate_results_summary(session["results"]),
                }

            def _generate_results_summary(
                self, results: dict[str, Any]
            ) -> dict[str, Any]:
                """Generate summary statistics from evaluation results."""
                summary = {
                    "best_performers": {},
                    "metric_rankings": {},
                    "overall_rankings": {},
                    "insights": [],
                }

                # Find best performer for each metric
                for system_name, system_results in results.items():
                    for metric_name, metric_data in system_results.items():
                        if "error" in metric_data:
                            continue

                        if metric_name not in summary["best_performers"]:
                            summary["best_performers"][metric_name] = {
                                "system": system_name,
                                "score": metric_data["mean_score"],
                            }
                        elif (
                            metric_data["mean_score"]
                            > summary["best_performers"][metric_name]["score"]
                        ):
                            summary["best_performers"][metric_name] = {
                                "system": system_name,
                                "score": metric_data["mean_score"],
                            }

                # Calculate overall rankings
                system_scores = defaultdict(list)
                for system_name, system_results in results.items():
                    for _, metric_data in system_results.items():
                        if "error" not in metric_data:
                            system_scores[system_name].append(metric_data["mean_score"])

                overall_scores = {}
                for system_name, scores in system_scores.items():
                    overall_scores[system_name] = (
                        statistics.mean(scores) if scores else 0.0
                    )

                summary["overall_rankings"] = sorted(
                    overall_scores.items(), key=lambda x: x[1], reverse=True
                )

                # Generate insights
                if summary["overall_rankings"]:
                    best_system = summary["overall_rankings"][0]
                    summary["insights"].append(
                        f"Best overall performer: {best_system[0]} (avg score: {best_system[1]:.3f})"
                    )

                return summary

            def _log_session_event(self, session_id: str, message: str):
                """Log an event for a session."""
                if session_id in self.evaluation_sessions:
                    self.evaluation_sessions[session_id]["logs"].append(
                        {"timestamp": time.time(), "message": message}
                    )

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Dashboard Core Engine Created

                **Core Features:**  
                - **Session Management** - Create and manage multiple evaluation sessions  
                - **System Integration** - Add and configure multiple systems for evaluation  
                - **Metric Configuration** - Add and configure evaluation metrics with weights  
                - **Progress Tracking** - Real-time monitoring of evaluation progress  
                - **Results Aggregation** - Comprehensive results collection and analysis  

                **Key Capabilities:**  
                - **Multi-System Evaluation** - Compare multiple systems simultaneously  
                - **Multi-Metric Assessment** - Use multiple evaluation criteria  
                - **Real-time Status** - Live updates on evaluation progress  
                - **Error Handling** - Graceful handling of evaluation failures  
                - **Results Summary** - Automatic generation of insights and rankings  

                Ready to build the interactive interface!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        EvaluationDashboard = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (EvaluationDashboard,)


@app.cell
def _(EvaluationDashboard, available_providers, cleandoc, dspy, mo, output):
    if available_providers and EvaluationDashboard:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 🎛️ Step 2: Interactive Dashboard Interface

                **Build and configure** evaluation sessions with interactive controls:
                """
            )
        )

        # Create dashboard instance
        dashboard = EvaluationDashboard()

        # Sample systems for demonstration
        class SimpleQASystem:
            """Simple QA system for demonstration."""

            def __init__(self, name: str, accuracy_bias: float = 0.0):
                self.name = name
                self.accuracy_bias = accuracy_bias
                self.answers = {
                    "What is the capital of France?": "Paris",
                    "Who wrote Romeo and Juliet?": "William Shakespeare",
                    "What is 2 + 2?": "4",
                    "What is the largest planet?": "Jupiter",
                }

            def forward(self, question):
                answer = self.answers.get(question, "I don't know")
                # Add some variation based on accuracy bias
                if self.accuracy_bias > 0 and question in self.answers:
                    if self.accuracy_bias > 0.5:
                        answer = answer.upper()  # Different format
                    elif self.accuracy_bias > 0.3:
                        answer = f"The answer is {answer}"  # Verbose
                return dspy.Prediction(answer=answer)

        # Sample metrics
        class SimpleExactMatchMetric:
            """Simple exact match metric."""

            def __init__(self, name: str = "exact_match"):
                self.name = name

            def evaluate(self, example, prediction, trace=None):
                expected = getattr(example, "answer", "").lower().strip()
                predicted = (
                    getattr(prediction, "answer", str(prediction)).lower().strip()
                )
                score = 1.0 if expected == predicted else 0.0

                from dataclasses import dataclass

                @dataclass
                class Result:
                    score: float
                    metadata: dict = None

                return Result(
                    score=score, metadata={"expected": expected, "predicted": predicted}
                )

        class SimpleFuzzyMatchMetric:
            """Simple fuzzy match metric."""

            def __init__(self, name: str = "fuzzy_match", threshold: float = 0.8):
                self.name = name
                self.threshold = threshold

            def evaluate(self, example, prediction, trace=None):
                import difflib

                expected = str(getattr(example, "answer", "")).lower().strip()
                predicted = (
                    str(getattr(prediction, "answer", str(prediction))).lower().strip()
                )

                if not expected or not predicted:
                    similarity = 0.0
                else:
                    similarity = difflib.SequenceMatcher(
                        None, expected, predicted
                    ).ratio()

                score = 1.0 if similarity >= self.threshold else similarity

                from dataclasses import dataclass

                @dataclass
                class Result:
                    score: float
                    metadata: dict = None

                return Result(
                    score=score,
                    metadata={"similarity": similarity, "threshold": self.threshold},
                )

        # Sample test data
        sample_qa_examples = [
            dspy.Example(
                question="What is the capital of France?", answer="Paris"
            ).with_inputs("question"),
            dspy.Example(
                question="Who wrote Romeo and Juliet?", answer="William Shakespeare"
            ).with_inputs("question"),
            dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
            dspy.Example(
                question="What is the largest planet?", answer="Jupiter"
            ).with_inputs("question"),
        ]

        # Dashboard controls
        session_name_input = mo.ui.text(
            value="QA Evaluation Session", label="Session Name"
        )

        session_description_input = mo.ui.text_area(
            value="Comparative evaluation of QA systems using multiple metrics",
            label="Session Description",
        )

        create_session_button = mo.ui.run_button(
            label="🆕 Create Session", kind="success"
        )

        # System configuration
        system_selection = mo.ui.multiselect(
            options=["accurate_system", "verbose_system", "random_system"],
            value=["accurate_system", "verbose_system"],
            label="Select Systems to Evaluate",
        )

        # Metric configuration
        metric_selection = mo.ui.multiselect(
            options=["exact_match", "fuzzy_match"],
            value=["exact_match", "fuzzy_match"],
            label="Select Evaluation Metrics",
        )

        fuzzy_threshold_slider = mo.ui.slider(
            start=0.5,
            stop=1.0,
            value=0.8,
            step=0.05,
            label="Fuzzy Match Threshold",
            show_value=True,
        )

        start_evaluation_button = mo.ui.run_button(
            label="🚀 Start Evaluation", kind="info"
        )

        refresh_status_button = mo.ui.run_button(
            label="🔄 Refresh Status", kind="neutral"
        )

        view_results_button = mo.ui.run_button(label="📊 View Results", kind="info")

        dashboard_controls = mo.vstack(
            [
                mo.md("### 🎛️ Evaluation Dashboard Controls"),
                mo.md("**Session Configuration:**"),
                session_name_input,
                session_description_input,
                create_session_button,
                mo.md("---"),
                mo.md("**System & Metric Configuration:**"),
                system_selection,
                metric_selection,
                fuzzy_threshold_slider,
                mo.md("---"),
                mo.md("**Evaluation Actions:**"),
                mo.hstack(
                    [
                        start_evaluation_button,
                        refresh_status_button,
                        view_results_button,
                    ]
                ),
            ]
        )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 🎛️ Interactive Dashboard Interface Created

                **Interface Components:**  
                - **Session Management** - Create and configure evaluation sessions  
                - **System Selection** - Choose which systems to evaluate  
                - **Metric Configuration** - Select and configure evaluation metrics  
                - **Real-time Controls** - Start evaluations and monitor progress  

                **Sample Systems:**  
                - **Accurate System** - High-quality responses with exact matches  
                - **Verbose System** - Correct but verbose responses  
                - **Random System** - Random responses for comparison baseline  

                **Available Metrics:**  
                - **Exact Match** - Strict string matching  
                - **Fuzzy Match** - Similarity-based matching with configurable threshold  

                Configure your evaluation session and start the evaluation!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        dashboard_controls = mo.md("")
        cell4_content = mo.md("")
        dashboard = None
        create_session_button = None
        start_evaluation_button = None
        refresh_status_button = None
        view_results_button = None
        sample_qa_examples = None

    cell4_out = mo.vstack([cell4_desc, dashboard_controls, cell4_content])
    output.replace(cell4_out)
    return (
        SimpleExactMatchMetric,
        SimpleFuzzyMatchMetric,
        SimpleQASystem,
        create_session_button,
        dashboard,
        fuzzy_threshold_slider,
        metric_selection,
        refresh_status_button,
        sample_qa_examples,
        session_description_input,
        session_name_input,
        start_evaluation_button,
        system_selection,
        view_results_button,
    )


@app.cell
def _(
    SimpleExactMatchMetric,
    SimpleFuzzyMatchMetric,
    SimpleQASystem,
    cleandoc,
    create_session_button,
    dashboard,
    fuzzy_threshold_slider,
    metric_selection,
    mo,
    output,
    refresh_status_button,
    sample_qa_examples,
    session_description_input,
    session_name_input,
    start_evaluation_button,
    system_selection,
    view_results_button,
):
    # Handle dashboard interactions
    import __main__

    dashboard_display = mo.md("")

    if create_session_button.value:
        # Create new evaluation session
        session_name = session_name_input.value
        session_description = session_description_input.value

        __main__.current_session_id = dashboard.create_evaluation_session(
            session_name, session_description
        )

        # Add selected systems
        selected_systems = system_selection.value
        systems_added = []

        if "accurate_system" in selected_systems:
            system = SimpleQASystem("Accurate System", accuracy_bias=0.0)
            dashboard.add_system_to_session(
                __main__.current_session_id, "accurate_system", system
            )
            systems_added.append("Accurate System")

        if "verbose_system" in selected_systems:
            system = SimpleQASystem("Verbose System", accuracy_bias=0.4)
            dashboard.add_system_to_session(
                __main__.current_session_id, "verbose_system", system
            )
            systems_added.append("Verbose System")

        if "random_system" in selected_systems:
            system = SimpleQASystem("Random System", accuracy_bias=0.8)
            dashboard.add_system_to_session(
                __main__.current_session_id, "random_system", system
            )
            systems_added.append("Random System")

        # Add selected metrics
        selected_metrics = metric_selection.value
        metrics_added = []

        if "exact_match" in selected_metrics:
            metric = SimpleExactMatchMetric("exact_match")
            dashboard.add_metric_to_session(
                __main__.current_session_id, "exact_match", metric
            )
            metrics_added.append("Exact Match")

        if "fuzzy_match" in selected_metrics:
            metric = SimpleFuzzyMatchMetric(
                "fuzzy_match", threshold=fuzzy_threshold_slider.value
            )
            dashboard.add_metric_to_session(
                __main__.current_session_id, "fuzzy_match", metric
            )
            metrics_added.append("Fuzzy Match")

        dashboard_display = mo.md(
            cleandoc(
                f"""
                ## ✅ Evaluation Session Created

                **Session ID:** {__main__.current_session_id}  
                **Name:** {session_name}  
                **Description:** {session_description}  

                ### 🔧 Configuration

                **Systems Added:** {', '.join(systems_added)}  
                **Metrics Added:** {', '.join(metrics_added)}  
                **Fuzzy Threshold:** {fuzzy_threshold_slider.value:.2f}  

                ### 📋 Next Steps

                1. Click "🚀 Start Evaluation" to begin the evaluation process
                2. Use "🔄 Refresh Status" to monitor progress
                3. Click "📊 View Results" when evaluation is complete

                Your evaluation session is ready to run!
                """
            )
        )

    elif start_evaluation_button.value and __main__.current_session_id:
        # Start evaluation
        success = dashboard.start_evaluation(
            __main__.current_session_id, sample_qa_examples
        )

        if success:
            dashboard_display = mo.md(
                cleandoc(
                    f"""
                    ## 🚀 Evaluation Started

                    **Session ID:** {__main__.current_session_id}  
                    **Test Examples:** {len(sample_qa_examples)}  

                    ### ⏳ Evaluation in Progress

                    The evaluation is now running. This process will:
                    1. Generate predictions from all configured systems
                    2. Evaluate predictions using all configured metrics
                    3. Aggregate results and generate insights

                    Use "🔄 Refresh Status" to monitor progress or "📊 View Results" when complete.
                    """
                )
            )
        else:
            dashboard_display = mo.md(
                cleandoc(
                    """
                    ## ❌ Evaluation Failed to Start

                    Please ensure you have:
                    - Created a session first
                    - Added at least one system
                    - Added at least one metric

                    Try creating a new session with the required components.
                    """
                )
            )

    elif refresh_status_button.value and __main__.current_session_id:
        # Refresh status
        status = dashboard.get_session_status(__main__.current_session_id)

        if "error" not in status:
            progress_bar = "█" * int(status["progress"] * 20) + "░" * (
                20 - int(status["progress"] * 20)
            )

            dashboard_display = mo.md(
                cleandoc(
                    f"""
                    ## 📊 Evaluation Status

                    **Session:** {status["name"]}  
                    **Status:** {status["status"].title()}  
                    **Progress:** {status["progress"]:.1%}  

                    ### 📈 Progress Bar
                    ```
                    {progress_bar} {status["progress"]:.1%}
                    ```

                    ### 🔧 Configuration
                    - **Systems:** {status["systems_count"]}  
                    - **Metrics:** {status["metrics_count"]}  
                    - **Current Step:** {status["current_step"]}  

                    {f'**Current System:** {status.get("current_system", "N/A")}' if status.get("current_system") else ""}  
                    {f'**Current Metric:** {status.get("current_metric", "N/A")}' if status.get("current_metric") else ""}  

                    {'✅ Evaluation Complete! Click "📊 View Results" to see the results.' if status["status"] == "completed" else '⏳ Evaluation in progress...'}
                    """
                )
            )
        else:
            dashboard_display = mo.md(f"## ❌ Error: {status['error']}")

    elif view_results_button.value and __main__.current_session_id:
        # View results
        results = dashboard.get_session_results(__main__.current_session_id)

        if "error" not in results:
            # Format results for display
            results_text = []

            for system_name, system_results in results["results"].items():
                results_text.append(f"### 🤖 {system_name.replace('_', ' ').title()}")

                for metric_name, metric_data in system_results.items():
                    if "error" in metric_data:
                        results_text.append(
                            f"- **{metric_name}**: Error - {metric_data['error']}"
                        )
                    else:
                        results_text.append(
                            f"- **{metric_name}**: {metric_data['mean_score']:.3f} ± {metric_data['std_dev']:.3f}"
                        )
                        results_text.append(
                            f"  - Range: {metric_data['min_score']:.3f} - {metric_data['max_score']:.3f}"
                        )
                        results_text.append(
                            f"  - Examples: {metric_data['total_examples']}"
                        )

                results_text.append("")

            # Best performers
            best_performers_text = []
            for metric_name, best_data in results["summary"]["best_performers"].items():
                best_performers_text.append(
                    f"- **{metric_name}**: {best_data['system'].replace('_', ' ').title()} ({best_data['score']:.3f})"
                )

            # Overall rankings
            rankings_text = []
            for i, (system_name, score) in enumerate(
                results["summary"]["overall_rankings"], 1
            ):
                rankings_text.append(
                    f"{i}. **{system_name.replace('_', ' ').title()}**: {score:.3f}"
                )

            dashboard_display = mo.md(
                cleandoc(
                    f"""
                    ## 📊 Evaluation Results

                    **Session:** {results["name"]}  
                    **Total Time:** {results["total_time"]:.2f} seconds  
                    **Systems Evaluated:** {len(results["systems"])}  
                    **Metrics Used:** {len(results["metrics"])}  

                    ### 📈 Detailed Results

                    {chr(10).join(results_text)}

                    ### 🏆 Best Performers by Metric

                    {chr(10).join(best_performers_text)}

                    ### 🥇 Overall Rankings

                    {chr(10).join(rankings_text)}

                    ### 💡 Insights

                    {chr(10).join(f"- {insight}" for insight in results["summary"]["insights"])}

                    ### 📋 Summary

                    The evaluation successfully compared {len(results["systems"])} systems using {len(results["metrics"])} metrics. 
                    Results show clear performance differences across different evaluation dimensions.
                    """
                )
            )
        else:
            dashboard_display = mo.md(f"## ❌ Error: {results['error']}")

    # Error handling for buttons clicked without session
    elif start_evaluation_button.value and not __main__.current_session_id:
        dashboard_display = mo.md(
            "## ⚠️ No session created yet. Please create a session first by clicking 'Create Session'."
        )

    elif refresh_status_button.value and not __main__.current_session_id:
        dashboard_display = mo.md(
            "## ⚠️ No session available. Please create a session first."
        )

    elif view_results_button.value and not __main__.current_session_id:
        dashboard_display = mo.md(
            "## ⚠️ No session results available. Please create and run a session first."
        )

    output.replace(dashboard_display)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell6_out = mo.md(
        cleandoc(
            """
            ## 🎯 Advanced Dashboard Features

            ### 🏆 Dashboard Design Best Practices

            **User Experience Principles:**  
            - **Intuitive Navigation** - Clear workflow from configuration to results  
            - **Real-time Feedback** - Immediate response to user actions  
            - **Progressive Disclosure** - Show information when relevant  
            - **Error Prevention** - Validate inputs and provide clear guidance  

            **Visual Design Guidelines:**  
            - **Consistent Layout** - Maintain consistent spacing and alignment  
            - **Color Coding** - Use colors to convey status and importance  
            - **Typography Hierarchy** - Clear information hierarchy with headings  
            - **Responsive Design** - Adapt to different screen sizes and contexts  

            ### ⚡ Advanced Dashboard Capabilities

            **Real-time Monitoring:**  
            - **Live Progress Tracking** - Real-time updates on evaluation progress  
            - **Performance Metrics** - Monitor evaluation speed and resource usage  
            - **Error Alerting** - Immediate notification of evaluation failures  
            - **Status Indicators** - Clear visual indicators of system status  

            **Interactive Analytics:**  
            - **Drill-down Analysis** - Click to explore detailed results  
            - **Comparative Visualization** - Side-by-side system comparisons  
            - **Trend Analysis** - Track performance changes over time  
            - **Statistical Testing** - Significance testing for system comparisons  

            ### 🔧 Customization and Configuration

            **Dashboard Personalization:**  
            - **Custom Layouts** - Arrange dashboard components to user preference  
            - **Saved Configurations** - Save and reuse evaluation configurations  
            - **Theme Selection** - Choose visual themes and color schemes  
            - **Widget Selection** - Add/remove dashboard widgets as needed  

            **Advanced Configuration:**  
            - **Metric Weighting** - Assign different weights to evaluation metrics  
            - **Threshold Settings** - Configure performance thresholds and alerts  
            - **Export Options** - Customize report formats and content  
            - **Integration Settings** - Connect with external tools and systems  

            ### 🚀 Production Dashboard Features

            **Scalability:**  
            - **Distributed Evaluation** - Scale evaluation across multiple machines  
            - **Queue Management** - Handle multiple concurrent evaluations  
            - **Resource Monitoring** - Track and optimize resource usage  
            - **Load Balancing** - Distribute evaluation workload efficiently  

            **Enterprise Features:**  
            - **User Management** - Role-based access control and permissions  
            - **Audit Logging** - Complete audit trail of all evaluation activities  
            - **API Integration** - RESTful APIs for programmatic access  
            - **Backup and Recovery** - Automated backup of evaluation data  

            ### 💡 Next Steps

            **Enhance Your Dashboard:**  
            1. **Add Custom Visualizations** - Create charts and graphs for better insights  
            2. **Implement Export Features** - Generate PDF reports and data exports  
            3. **Build Alerting System** - Set up automated alerts for performance issues  
            4. **Create Templates** - Build reusable evaluation templates  

            Excellent work building a comprehensive evaluation dashboard! 🎉
            """
        )
        if available_providers
        else ""
    )

    output.replace(cell6_out)
    return


if __name__ == "__main__":
    app.run()
