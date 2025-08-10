# pylint: disable=import-error,import-outside-toplevel,reimported
# cSpell:ignore dspy marimo

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import random
    import sys
    import time
    from collections.abc import Callable
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
        get_config,
        mo,
        output,
        random,
        setup_dspy_environment,
        time,
    )


@app.cell
def _(cleandoc, mo, output):
    cell1_out = mo.md(
        cleandoc(
            """
            # 📊 Optimization Progress Dashboard

            **Duration:** 90-120 minutes  
            **Prerequisites:** Completed all previous optimization modules  
            **Difficulty:** Advanced

            ## 🎯 Learning Objectives

            By the end of this module, you will:  
            - ✅ Build real-time optimization progress visualization  
            - ✅ Create optimization strategy comparison interfaces  
            - ✅ Implement optimization result export and analysis tools  
            - ✅ Design comprehensive monitoring and alerting systems  
            - ✅ Master advanced dashboard development patterns  

            ## 📈 Dashboard Overview

            **Why Optimization Dashboards Matter:**  
            - **Real-time Monitoring** - Track optimization progress as it happens  
            - **Strategy Comparison** - Compare different optimization approaches  
            - **Performance Analysis** - Identify trends and patterns in results  
            - **Decision Support** - Make data-driven optimization decisions  

            **Dashboard Components:**  
            - **Progress Tracking** - Real-time optimization status and metrics  
            - **Performance Visualization** - Charts and graphs of optimization results  
            - **Strategy Comparison** - Side-by-side analysis of different approaches  
            - **Export Tools** - Save and share optimization results  

            Let's build a comprehensive optimization dashboard!
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
                ## ✅ Dashboard Environment Ready

                **Configuration:**  
                - Provider: **{config.default_provider}**  
                - Model: **{config.default_model}**

                Ready to build optimization dashboards!
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
def _(Any, available_providers, cleandoc, mo, output, time):
    if available_providers:
        cell3_desc = mo.md(
            cleandoc(
                """
                ## 🏗️ Step 1: Optimization Dashboard Engine

                **Building the core dashboard system** for tracking and visualizing optimization progress:
                """
            )
        )

        class OptimizationDashboard:
            """Comprehensive dashboard for optimization tracking and analysis."""

            def __init__(self):
                self.optimization_runs = []
                self.active_optimizations = {}
                self.dashboard_config = {
                    "refresh_interval": 5.0,  # seconds
                    "max_history": 100,
                    "auto_export": False,
                    "alert_thresholds": {
                        "min_improvement": 0.05,
                        "max_time": 300.0,  # 5 minutes
                        "min_success_rate": 0.8,
                    },
                }

            def start_optimization_tracking(
                self, optimization_id: str, config: dict[str, Any]
            ) -> dict[str, Any]:
                """Start tracking a new optimization run."""
                optimization_run = {
                    "id": optimization_id,
                    "start_time": time.time(),
                    "config": config.copy(),
                    "status": "running",
                    "progress": 0.0,
                    "current_stage": "initialization",
                    "metrics": {
                        "baseline_score": 0.0,
                        "current_score": 0.0,
                        "best_score": 0.0,
                        "improvement": 0.0,
                        "iterations": 0,
                        "time_elapsed": 0.0,
                    },
                    "stages": [],
                    "alerts": [],
                    "logs": [],
                }

                self.active_optimizations[optimization_id] = optimization_run
                self.optimization_runs.append(optimization_run)

                # Limit history
                if len(self.optimization_runs) > self.dashboard_config["max_history"]:
                    self.optimization_runs = self.optimization_runs[
                        -self.dashboard_config["max_history"] :
                    ]

                return {
                    "success": True,
                    "optimization_id": optimization_id,
                    "message": "Optimization tracking started",
                }

            def update_optimization_progress(
                self, optimization_id: str, update_data: dict[str, Any]
            ) -> dict[str, Any]:
                """Update progress for an active optimization."""
                if optimization_id not in self.active_optimizations:
                    return {"success": False, "error": "Optimization not found"}

                run = self.active_optimizations[optimization_id]

                # Update basic metrics
                if "progress" in update_data:
                    run["progress"] = min(1.0, max(0.0, update_data["progress"]))

                if "current_stage" in update_data:
                    run["current_stage"] = update_data["current_stage"]

                if "status" in update_data:
                    run["status"] = update_data["status"]

                # Update metrics
                if "metrics" in update_data:
                    for key, value in update_data["metrics"].items():
                        if key in run["metrics"]:
                            run["metrics"][key] = value

                # Calculate derived metrics
                run["metrics"]["time_elapsed"] = time.time() - run["start_time"]
                run["metrics"]["improvement"] = (
                    run["metrics"]["current_score"] - run["metrics"]["baseline_score"]
                )

                # Add stage information
                if "stage_info" in update_data:
                    stage_info = update_data["stage_info"].copy()
                    stage_info["timestamp"] = time.time()
                    run["stages"].append(stage_info)

                # Add logs
                if "log_message" in update_data:
                    log_entry = {
                        "timestamp": time.time(),
                        "message": update_data["log_message"],
                        "level": update_data.get("log_level", "info"),
                    }
                    run["logs"].append(log_entry)

                # Check for alerts
                self._check_alerts(optimization_id)

                return {"success": True, "message": "Progress updated"}

            def complete_optimization(
                self, optimization_id: str, final_results: dict[str, Any]
            ) -> dict[str, Any]:
                """Mark an optimization as complete and store final results."""
                if optimization_id not in self.active_optimizations:
                    return {"success": False, "error": "Optimization not found"}

                run = self.active_optimizations[optimization_id]

                # Update final status
                run["status"] = "completed"
                run["progress"] = 1.0
                run["end_time"] = time.time()
                run["total_time"] = run["end_time"] - run["start_time"]

                # Store final results
                run["final_results"] = final_results.copy()

                # Update final metrics
                if "final_score" in final_results:
                    run["metrics"]["current_score"] = final_results["final_score"]
                    run["metrics"]["best_score"] = max(
                        run["metrics"]["best_score"], final_results["final_score"]
                    )

                # Remove from active optimizations
                del self.active_optimizations[optimization_id]

                return {
                    "success": True,
                    "message": "Optimization completed",
                    "total_time": run["total_time"],
                    "final_score": run["metrics"]["current_score"],
                }

            def get_dashboard_data(self) -> dict[str, Any]:
                """Get comprehensive dashboard data for visualization."""
                current_time = time.time()

                # Active optimizations summary
                active_summary = {
                    "total_active": len(self.active_optimizations),
                    "running": len(
                        [
                            r
                            for r in self.active_optimizations.values()
                            if r["status"] == "running"
                        ]
                    ),
                    "avg_progress": sum(
                        r["progress"] for r in self.active_optimizations.values()
                    )
                    / max(1, len(self.active_optimizations)),
                }

                # Recent runs summary
                recent_runs = [
                    r
                    for r in self.optimization_runs
                    if current_time - r["start_time"] < 3600
                ]  # Last hour
                completed_runs = [
                    r for r in self.optimization_runs if r["status"] == "completed"
                ]

                recent_summary = {
                    "total_runs": len(self.optimization_runs),
                    "recent_runs": len(recent_runs),
                    "completed_runs": len(completed_runs),
                    "success_rate": len(
                        [r for r in completed_runs if r["metrics"]["improvement"] > 0]
                    )
                    / max(1, len(completed_runs)),
                    "avg_improvement": sum(
                        r["metrics"]["improvement"] for r in completed_runs
                    )
                    / max(1, len(completed_runs)),
                    "avg_time": sum(r.get("total_time", 0) for r in completed_runs)
                    / max(1, len(completed_runs)),
                }

                # Performance trends
                performance_data = []
                for run in completed_runs[-20:]:  # Last 20 runs
                    performance_data.append(
                        {
                            "run_id": run["id"],
                            "start_time": run["start_time"],
                            "baseline_score": run["metrics"]["baseline_score"],
                            "final_score": run["metrics"]["current_score"],
                            "improvement": run["metrics"]["improvement"],
                            "total_time": run.get("total_time", 0),
                            "optimization_type": run["config"].get(
                                "optimization_type", "unknown"
                            ),
                        }
                    )

                # Active alerts
                all_alerts = []
                for run in self.active_optimizations.values():
                    all_alerts.extend(run["alerts"])

                return {
                    "timestamp": current_time,
                    "active_summary": active_summary,
                    "recent_summary": recent_summary,
                    "performance_data": performance_data,
                    "active_optimizations": list(self.active_optimizations.values()),
                    "alerts": all_alerts[-10:],  # Last 10 alerts
                    "config": self.dashboard_config,
                }

            def _check_alerts(self, optimization_id: str):
                """Check for alert conditions and add alerts if needed."""
                run = self.active_optimizations[optimization_id]
                alerts = run["alerts"]
                thresholds = self.dashboard_config["alert_thresholds"]

                current_time = time.time()
                elapsed_time = current_time - run["start_time"]

                # Time-based alerts
                if elapsed_time > thresholds["max_time"] and run["status"] == "running":
                    alert = {
                        "timestamp": current_time,
                        "type": "time_exceeded",
                        "message": f"Optimization {optimization_id} has been running for {elapsed_time:.1f}s",
                        "severity": "warning",
                    }
                    if not any(a["type"] == "time_exceeded" for a in alerts):
                        alerts.append(alert)

                # Improvement-based alerts
                if run["metrics"]["improvement"] < 0 and run["progress"] > 0.5:
                    alert = {
                        "timestamp": current_time,
                        "type": "negative_improvement",
                        "message": f"Optimization {optimization_id} showing negative improvement: {run['metrics']['improvement']:.3f}",
                        "severity": "warning",
                    }
                    if not any(a["type"] == "negative_improvement" for a in alerts):
                        alerts.append(alert)

            def export_results(
                self, format_type: str = "json", optimization_ids: list[str] = None
            ) -> dict[str, Any]:
                """Export optimization results in various formats."""
                if optimization_ids is None:
                    runs_to_export = self.optimization_runs
                else:
                    runs_to_export = [
                        r for r in self.optimization_runs if r["id"] in optimization_ids
                    ]

                export_data = {
                    "export_timestamp": time.time(),
                    "export_format": format_type,
                    "total_runs": len(runs_to_export),
                    "runs": runs_to_export,
                }

                if format_type == "json":
                    return {"success": True, "data": export_data, "format": "json"}
                elif format_type == "summary":
                    summary = {
                        "total_runs": len(runs_to_export),
                        "completed_runs": len(
                            [r for r in runs_to_export if r["status"] == "completed"]
                        ),
                        "avg_improvement": sum(
                            r["metrics"]["improvement"] for r in runs_to_export
                        )
                        / max(1, len(runs_to_export)),
                        "best_run": (
                            max(
                                runs_to_export,
                                key=lambda r: r["metrics"]["improvement"],
                            )
                            if runs_to_export
                            else None
                        ),
                        "optimization_types": list(
                            {
                                r["config"].get("optimization_type", "unknown")
                                for r in runs_to_export
                            }
                        ),
                    }
                    return {"success": True, "data": summary, "format": "summary"}

                return {"success": False, "error": "Unsupported format"}

            def compare_optimizations(
                self, optimization_ids: list[str]
            ) -> dict[str, Any]:
                """Compare multiple optimization runs."""
                runs_to_compare = [
                    r for r in self.optimization_runs if r["id"] in optimization_ids
                ]

                if len(runs_to_compare) < 2:
                    return {
                        "success": False,
                        "error": "Need at least 2 runs to compare",
                    }

                comparison = {
                    "runs_compared": len(runs_to_compare),
                    "comparison_timestamp": time.time(),
                    "metrics_comparison": {},
                    "best_performer": None,
                    "insights": [],
                }

                # Compare metrics
                metrics_to_compare = [
                    "baseline_score",
                    "current_score",
                    "improvement",
                    "time_elapsed",
                ]
                for metric in metrics_to_compare:
                    values = [r["metrics"][metric] for r in runs_to_compare]
                    comparison["metrics_comparison"][metric] = {
                        "values": values,
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "best_run_id": runs_to_compare[values.index(max(values))]["id"],
                    }

                # Determine best performer
                best_run = max(
                    runs_to_compare, key=lambda r: r["metrics"]["improvement"]
                )
                comparison["best_performer"] = {
                    "run_id": best_run["id"],
                    "improvement": best_run["metrics"]["improvement"],
                    "optimization_type": best_run["config"].get(
                        "optimization_type", "unknown"
                    ),
                }

                # Generate insights
                improvements = [r["metrics"]["improvement"] for r in runs_to_compare]
                times = [r["metrics"]["time_elapsed"] for r in runs_to_compare]

                if max(improvements) - min(improvements) > 0.1:
                    comparison["insights"].append(
                        "Significant variation in improvement across runs"
                    )

                if max(times) / min(times) > 2:
                    comparison["insights"].append(
                        "Large variation in optimization times"
                    )

                return {"success": True, "comparison": comparison}

        cell3_content = mo.md(
            cleandoc(
                """
                ### 🏗️ Optimization Dashboard Engine Created

                **Core Features:**  
                - **Real-time Tracking** - Monitor active optimizations with live updates  
                - **Progress Visualization** - Track optimization stages and metrics  
                - **Alert System** - Automated alerts for time limits and performance issues  
                - **Export Tools** - Save results in multiple formats  
                - **Comparison Engine** - Side-by-side analysis of optimization runs  

                **Dashboard Components:**  
                - **Active Monitoring** - Track running optimizations  
                - **Historical Analysis** - Analyze past optimization performance  
                - **Performance Trends** - Visualize improvement patterns over time  
                - **Alert Management** - Monitor and respond to optimization issues  

                Ready to build interactive dashboard interfaces!
                """
            )
        )
    else:
        cell3_desc = mo.md("")
        OptimizationDashboard = None
        cell3_content = mo.md("")

    cell3_out = mo.vstack([cell3_desc, cell3_content])
    output.replace(cell3_out)
    return (OptimizationDashboard,)


@app.cell
def _(
    OptimizationDashboard,
    available_providers,
    cleandoc,
    mo,
    output,
    random,
):
    if available_providers and OptimizationDashboard:
        cell4_desc = mo.md(
            cleandoc(
                """
                ## 📊 Step 2: Interactive Dashboard Interface

                **Real-time dashboard** with live optimization monitoring and controls:
                """
            )
        )

        # Create dashboard instance
        dashboard = OptimizationDashboard()

        # Simulate some optimization runs for demonstration
        def simulate_optimization_runs():
            """Create sample optimization data for dashboard demonstration."""
            optimization_types = ["BootstrapFewShot", "MIPRO", "Custom"]

            for i in range(5):
                opt_id = f"opt_{i+1}"
                opt_type = random.choice(optimization_types)

                config = {
                    "optimization_type": opt_type,
                    "max_demos": random.randint(4, 16),
                    "metric_type": random.choice(
                        ["exact_match", "fuzzy_match", "composite"]
                    ),
                }

                # Start tracking
                dashboard.start_optimization_tracking(opt_id, config)

                # Simulate progress updates
                baseline_score = random.uniform(0.3, 0.6)
                final_score = baseline_score + random.uniform(0.1, 0.4)

                dashboard.update_optimization_progress(
                    opt_id,
                    {
                        "progress": 0.3,
                        "current_stage": "stage_1",
                        "metrics": {
                            "baseline_score": baseline_score,
                            "current_score": baseline_score + 0.05,
                        },
                    },
                )

                dashboard.update_optimization_progress(
                    opt_id,
                    {
                        "progress": 0.7,
                        "current_stage": "stage_2",
                        "metrics": {"current_score": baseline_score + 0.15},
                    },
                )

                # Complete optimization
                dashboard.complete_optimization(
                    opt_id,
                    {"final_score": final_score, "optimization_successful": True},
                )

        # Initialize with sample data
        simulate_optimization_runs()

        # Dashboard controls
        refresh_button = mo.ui.run_button(
            label="🔄 Refresh Dashboard",
            kind="info",
        )

        start_demo_button = mo.ui.run_button(
            label="🚀 Start Demo Optimization",
            kind="success",
        )

        export_button = mo.ui.run_button(
            label="📤 Export Results",
            kind="neutral",
        )

        compare_button = mo.ui.run_button(
            label="📊 Compare Runs",
            kind="neutral",
        )

        # Dashboard configuration
        auto_refresh_checkbox = mo.ui.checkbox(
            value=False,
            label="Auto Refresh (5s)",
        )

        alert_threshold_slider = mo.ui.slider(
            start=0.01,
            stop=0.2,
            value=0.05,
            step=0.01,
            label="Min Improvement Alert Threshold",
            show_value=True,
        )

        dashboard_controls = mo.vstack(
            [
                mo.md("### 🎛️ Dashboard Controls"),
                mo.hstack([refresh_button, start_demo_button]),
                mo.hstack([export_button, compare_button]),
                mo.md("---"),
                mo.md("**Configuration:**"),
                auto_refresh_checkbox,
                alert_threshold_slider,
            ]
        )

        cell4_content = mo.md(
            cleandoc(
                """
                ### 📊 Interactive Dashboard Interface Created

                **Interface Features:**  
                - **Real-time Controls** - Refresh, start optimizations, export data  
                - **Configuration Options** - Auto-refresh and alert thresholds  
                - **Demo Mode** - Simulate optimizations for testing  
                - **Export Tools** - Save dashboard data and results  

                Use the controls above to interact with the optimization dashboard!
                """
            )
        )
    else:
        cell4_desc = mo.md("")
        dashboard = None
        dashboard_controls = mo.md("")
        cell4_content = mo.md("")
        refresh_button = None
        start_demo_button = None
        export_button = None
        compare_button = None
        auto_refresh_checkbox = None
        alert_threshold_slider = None

    cell4_out = mo.vstack([cell4_desc, dashboard_controls, cell4_content])
    output.replace(cell4_out)
    return (
        compare_button,
        dashboard,
        export_button,
        refresh_button,
        start_demo_button,
    )


@app.cell
def _(
    cleandoc,
    dashboard,
    mo,
    output,
    random,
    refresh_button,
    start_demo_button,
    time,
):
    # Handle dashboard interactions
    dashboard_display = mo.md("")

    if refresh_button is not None and refresh_button.value and dashboard:
        # Get current dashboard data
        dashboard_data = dashboard.get_dashboard_data()

        active_summary = dashboard_data["active_summary"]
        recent_summary = dashboard_data["recent_summary"]
        alerts = dashboard_data["alerts"]

        dashboard_display = mo.md(
            cleandoc(
                f"""
                ## 📊 Optimization Dashboard

                **Last Updated:** {time.strftime('%H:%M:%S')}

                ### 🔄 Active Optimizations
                - **Total Active:** {active_summary['total_active']}  
                - **Currently Running:** {active_summary['running']}  
                - **Average Progress:** {active_summary['avg_progress']:.1%}  

                ### 📈 Recent Performance
                - **Total Runs:** {recent_summary['total_runs']}  
                - **Completed Runs:** {recent_summary['completed_runs']}  
                - **Success Rate:** {recent_summary['success_rate']:.1%}  
                - **Average Improvement:** +{recent_summary['avg_improvement']:.3f}  
                - **Average Time:** {recent_summary['avg_time']:.1f}s  

                ### 🚨 Recent Alerts
                {f"**{len(alerts)} alerts**" if alerts else "**No recent alerts**"}
                """
            )
        )

        # Add performance data visualization
        if dashboard_data["performance_data"]:
            perf_data = dashboard_data["performance_data"][-10:]  # Last 10 runs

            performance_summary = mo.md(
                cleandoc(
                    f"""
                    ### 📊 Performance Trends (Last 10 Runs)

                    **Optimization Types:**  
                    {", ".join({r["optimization_type"] for r in perf_data})}

                    **Performance Range:**  
                    - Best Improvement: +{max(r["improvement"] for r in perf_data):.3f}  
                    - Worst Improvement: +{min(r["improvement"] for r in perf_data):.3f}  
                    - Time Range: {min(r["total_time"] for r in perf_data):.1f}s - {max(r["total_time"] for r in perf_data):.1f}s  
                    """
                )
            )

            dashboard_display = mo.vstack([dashboard_display, performance_summary])

    elif start_demo_button is not None and start_demo_button.value and dashboard:
        # Start a demo optimization
        demo_id = f"demo_{int(time.time())}"
        demo_config = {
            "optimization_type": random.choice(["BootstrapFewShot", "MIPRO"]),
            "max_demos": random.randint(4, 12),
            "metric_type": "comprehensive",
        }

        result = dashboard.start_optimization_tracking(demo_id, demo_config)

        # Simulate some progress
        baseline_score = random.uniform(0.4, 0.7)
        dashboard.update_optimization_progress(
            demo_id,
            {
                "progress": 0.2,
                "current_stage": "initialization",
                "metrics": {
                    "baseline_score": baseline_score,
                    "current_score": baseline_score,
                },
                "log_message": "Starting optimization process",
            },
        )

        dashboard_display = mo.md(
            cleandoc(
                f"""
                ## 🚀 Demo Optimization Started

                **Optimization ID:** {demo_id}  
                **Type:** {demo_config['optimization_type']}  
                **Status:** {result['message']}  

                **Configuration:**  
                - Max Demos: {demo_config['max_demos']}  
                - Metric Type: {demo_config['metric_type']}  
                - Baseline Score: {baseline_score:.3f}  

                Use the refresh button to see updated progress!
                """
            )
        )

    output.replace(dashboard_display)
    return


@app.cell
def _(cleandoc, compare_button, dashboard, export_button, mo, output, time):
    # Handle export and comparison
    export_display = mo.md("")

    if export_button is not None and export_button.value and dashboard:
        # Export dashboard results
        export_result = dashboard.export_results(format_type="summary")

        if export_result["success"]:
            summary_data = export_result["data"]

            export_display = mo.md(
                cleandoc(
                    f"""
                    ## 📤 Export Results

                    **Export Format:** Summary  
                    **Export Time:** {time.strftime('%H:%M:%S')}  

                    ### 📊 Summary Statistics
                    - **Total Runs:** {summary_data['total_runs']}  
                    - **Completed Runs:** {summary_data['completed_runs']}  
                    - **Average Improvement:** +{summary_data['avg_improvement']:.3f}  
                    - **Optimization Types:** {', '.join(summary_data['optimization_types'])}  

                    ### 🏆 Best Run
                    {f"**Run ID:** {summary_data['best_run']['id']}" if summary_data['best_run'] else "No completed runs"}  
                    {f"**Improvement:** +{summary_data['best_run']['metrics']['improvement']:.3f}" if summary_data['best_run'] else ""}  
                    {f"**Type:** {summary_data['best_run']['config'].get('optimization_type', 'Unknown')}" if summary_data['best_run'] else ""}  

                    Export completed successfully!
                    """
                )
            )

    elif compare_button is not None and compare_button.value and dashboard:
        # Compare recent optimization runs
        recent_runs = [
            r for r in dashboard.optimization_runs if r["status"] == "completed"
        ][-3:]

        if len(recent_runs) >= 2:
            run_ids = [r["id"] for r in recent_runs]
            comparison_result = dashboard.compare_optimizations(run_ids)

            if comparison_result["success"]:
                comparison = comparison_result["comparison"]
                best_performer = comparison["best_performer"]

                export_display = mo.md(
                    cleandoc(
                        f"""
                        ## 📊 Optimization Comparison

                        **Runs Compared:** {comparison['runs_compared']}  
                        **Comparison Time:** {time.strftime('%H:%M:%S')}  

                        ### 🏆 Best Performer
                        - **Run ID:** {best_performer['run_id']}  
                        - **Improvement:** +{best_performer['improvement']:.3f}  
                        - **Type:** {best_performer['optimization_type']}  

                        ### 📈 Metrics Comparison
                        - **Improvement Range:** {comparison['metrics_comparison']['improvement']['min']:.3f} - {comparison['metrics_comparison']['improvement']['max']:.3f}  
                        - **Time Range:** {comparison['metrics_comparison']['time_elapsed']['min']:.1f}s - {comparison['metrics_comparison']['time_elapsed']['max']:.1f}s  

                        ### 💡 Insights
                        {chr(10).join(f"- {insight}" for insight in comparison['insights']) if comparison['insights'] else "- No significant patterns detected"}

                        Comparison completed successfully!
                        """
                    )
                )
        else:
            export_display = mo.md(
                cleandoc(
                    """
                    ## ⚠️ Comparison Not Available

                    Need at least 2 completed optimization runs for comparison.  
                    Run more optimizations and try again.
                    """
                )
            )

    output.replace(export_display)
    return


@app.cell
def _(available_providers, cleandoc, mo, output):
    cell6_out = mo.md(
        cleandoc(
            """
                ## 🎯 Advanced Dashboard Features

                ### 🏆 Dashboard Best Practices

                **Real-time Monitoring:**  
                - **Live Updates** - Refresh data automatically or on-demand  
                - **Progress Tracking** - Visual progress bars and stage indicators  
                - **Performance Metrics** - Key metrics displayed prominently  
                - **Alert System** - Automated notifications for important events  

                **Data Visualization:**  
                - **Trend Charts** - Show performance improvements over time  
                - **Comparison Views** - Side-by-side analysis of different runs  
                - **Distribution Plots** - Understand metric distributions  
                - **Timeline Views** - Track optimization stages and timing  

                ### ⚡ Advanced Features

                **Automated Analysis:**  
                - **Pattern Detection** - Identify successful optimization patterns  
                - **Anomaly Detection** - Flag unusual performance or behavior  
                - **Recommendation Engine** - Suggest optimal parameters based on history  
                - **Predictive Analytics** - Estimate optimization completion times  

                **Integration Capabilities:**  
                - **Export Formats** - JSON, CSV, summary reports  
                - **API Integration** - Connect with external monitoring systems  
                - **Webhook Support** - Real-time notifications to external services  
                - **Database Storage** - Persistent storage for long-term analysis  

                ### 🔧 Production Deployment

                **Scalability:**  
                - **Concurrent Monitoring** - Track multiple optimizations simultaneously  
                - **Resource Management** - Monitor system resource usage  
                - **Load Balancing** - Distribute monitoring across multiple instances  
                - **Caching Strategy** - Optimize dashboard performance  

                **Security:**  
                - **Access Control** - Role-based dashboard access  
                - **Data Privacy** - Secure handling of optimization data  
                - **Audit Logging** - Track dashboard usage and changes  
                - **Secure Export** - Protected data export mechanisms  

                ### 🚀 Next Steps

                **Enhance Your Dashboard:**  
                1. **Add Custom Visualizations** - Create domain-specific charts  
                2. **Implement Alerting** - Set up email/SMS notifications  
                3. **Build Reports** - Generate automated optimization reports  
                4. **Scale for Production** - Deploy with proper monitoring infrastructure  

                **Integration Opportunities:**  
                - **CI/CD Pipelines** - Integrate optimization into deployment workflows  
                - **MLOps Platforms** - Connect with ML experiment tracking  
                - **Business Intelligence** - Feed data into BI dashboards  
                - **Monitoring Systems** - Integrate with APM and logging platforms  

                ### 💡 Module 04 Complete!

                **What You've Built:**  
                - **BootstrapFewShot Optimization** - Interactive optimization with parameter tuning  
                - **MIPRO Implementation** - Advanced multi-stage optimization  
                - **Custom Metrics System** - Domain-specific evaluation metrics  
                - **Optimization Dashboard** - Comprehensive monitoring and analysis  

                **Key Skills Mastered:**  
                - **Optimization Strategies** - Understanding when and how to use different approaches  
                - **Metric Design** - Creating effective evaluation functions  
                - **Performance Analysis** - Analyzing and comparing optimization results  
                - **Dashboard Development** - Building monitoring and visualization systems  

                Congratulations on completing the DSPy Optimization module! 🎉

                You now have the skills to:  
                - Optimize DSPy modules for maximum performance  
                - Design custom metrics for your specific use cases  
                - Monitor and analyze optimization progress  
                - Make data-driven decisions about optimization strategies  

                Ready to move on to Module 05: Evaluation & Metrics! 🚀
                """
        )
        if available_providers
        else ""
    )

    output.replace(cell6_out)
    return


if __name__ == "__main__":
    app.run()
